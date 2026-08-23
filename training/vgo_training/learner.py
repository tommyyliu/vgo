from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterable, Iterator, Mapping, TextIO

import numpy as np
import torch
from torch import nn

from .dataset import (
    PreparedRasterDataset,
    RasterDataset,
    file_sha256,
    load_dataset,
)
from .model import MODEL_ARCHITECTURES, build_model
from .packed_states import is_packable, pack as pack_states
from .packed_policy import (
    is_packable as is_policy_packable,
    pack as pack_policy,
)
from .recency import row_weights
from .serve import load_model
from .train_demo import (
    DIHEDRAL_TRANSFORMS,
    apply_dihedral,
    atomic_write_text,
    build_scheduler,
    policy_cross_entropy,
    prepare_policy_supervision,
)


POLICY_TARGET = "progressive_empirical_importance_v1"
POLICY_DENOMINATOR = "full_legal_raster_v1"
PREPARATION_VERSION = "importance-full-legal-v1"
PROTOCOL_SCHEMA = "vgo.learner.protocol.v1"
# Share of the loss carried by the batch-normalized heads, following KataGo.
# They dominate optimization so the trunk gains nothing from inflating weights;
# the remainder trains the unnormalized heads that inference actually uses.
NORMED_HEAD_WEIGHT = 0.8

# Ownership loss weight, relative to policy at 1.0. KataGo uses 1.5 against a
# value weight of 1.2 -- comparable magnitude, because ownership is dense
# supervision worth about as much as the outcome itself, unlike its score
# belief head at 0.0015. Ours is scaled the same way against our value weight.
OWNERSHIP_WEIGHT = 1.5


@dataclass(frozen=True)
class LearnerConfig:
    """All tunable state for one learner update.

    A service accepts these fields on every update. Parameters which do not
    affect model identity (epochs, learning rate, reporting, and sampling) may
    change freely. A device, compilation, architecture, or shape change causes
    a deliberate model reinitialization unless an explicit checkpoint supplies
    the new model.
    """

    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 3e-3
    value_weight: float = 1.0
    model_width: int = 32
    blocks: int = 3
    architecture: str = "flat"
    # Which planes the network reads. A property of the *model*, not of the
    # data: a position shard stores the game, and the raster is rendered at load
    # time, so two runs over the same shards can train different encodings.
    #
    # It has to be configured rather than inferred. The shard header records
    # what generation happened to be set to, which stopped identifying a layout
    # the moment two shared a width -- `compact-pass` and `compact-dead-zone`
    # are both six planes and differ only in which capture predicate they
    # carry, so the count cannot tell them apart and a wrong guess feeds a model
    # a plane meaning something else. None keeps the old header-derived
    # behaviour, for runs that predate the question.
    raster_kind: str | None = None
    # Fixed-variance init plus He-scale convs (ddrnet). Changes the computed
    # function, so it is recorded in the checkpoint and cannot be toggled on a
    # warm start.
    variance_scaled: bool = False
    # GroupNorm groups per residual block; None leaves the block unnormalized.
    # Supersedes variance_scaled, which stands in the same place.
    norm_groups: int | None = None
    # Weight on the auxiliary ownership loss, relative to policy at 1.0. Zero
    # disables the head's supervision *and* stops the window holding its
    # targets, which is 0.063 MB per sample -- a fifth of a packed sample. The
    # head still exists and still exports as nothing, so this is reversible
    # without touching model identity.
    ownership_weight: float = OWNERSHIP_WEIGHT
    # Per-shard sampling decay: 1.0 samples the whole window uniformly, 0.9
    # makes each older shard 10% less likely than its successor. Lets a long
    # window stay diverse while the gradient follows recent play. Identity
    # config -- it changes what the model trains on.
    recency_decay: float = 1.0
    # Trailing residual blocks in each ddrnet context stage to replace with
    # transformer blocks. Identity config at 0, which is byte-identical to a
    # net built without it. Attention is the one part of this model that is
    # not resolution-agnostic -- rotary tables are built per board size -- so a
    # checkpoint carrying it is fixed to the raster it was constructed for.
    context_attention_blocks: int = 0
    attention_heads: int = 8
    # Muon on the conv/linear trunk, Adam on heads, norms and biases.
    # Measured on the 25-shard window, the same w96 model reached policy_kl
    # 0.845 at epoch 1 under plain Adam against 0.736 under Muon, and the
    # architecture sweep that chose w64 ran entirely under Muon -- so a run
    # comparing itself to those numbers has to use it. `full_adam` opts out
    # and puts every parameter on Adam at `learning_rate`.
    muon_learning_rate: float = 0.01
    full_adam: bool = False
    # Overrides the `full_adam` pair when set, so every existing recipe keeps
    # its meaning. "ranger21" is AdamW plus lookahead, gradient centralization,
    # adaptive gradient clipping, norm loss and stable weight decay -- a bundle,
    # so a win by it does not isolate which of those did the work. Its own
    # warmup and warmdown are switched off in `_build_optimizer`, which leaves
    # `schedule` driving every arm and makes the comparison about the optimizer
    # rather than about two different learning-rate curves.
    optimizer: str | None = None
    threads: int = 4
    device: str = "cuda"
    precision: str = "float32"
    seed: int = 7
    compile: bool = True
    restore_optimizer: bool = True
    schedule: str = "wsd"
    # Fractional values are meaningful and sometimes necessary: warmup is
    # converted to steps as warmup_epochs * steps_per_epoch, so with a short
    # epoch count an integer floor of 1 makes warmup consume the whole update
    # and the decay phase never runs.
    warmup_epochs: float = 5
    decay_fraction: float = 0.2
    final_learning_rate_fraction: float = 0.01
    report_every: int = 20
    validation_fraction: float = 0.1
    augment: bool = True

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.value_weight) or self.value_weight < 0.0:
            raise ValueError("value weight must be finite and nonnegative")
        if self.model_width <= 0 or self.blocks <= 0:
            raise ValueError("model width and blocks must be positive")
        if self.architecture not in MODEL_ARCHITECTURES:
            raise ValueError(f"unknown model architecture: {self.architecture!r}")
        if self.threads <= 0:
            raise ValueError("thread count must be positive")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError(f"unknown training precision: {self.precision!r}")
        if self.schedule not in ("wsd", "cosine"):
            raise ValueError(f"unknown learning-rate schedule: {self.schedule!r}")
        if self.optimizer is not None and self.optimizer not in (
            "adam",
            "muon",
            "ranger21",
        ):
            raise ValueError(f"unknown optimizer: {self.optimizer!r}")
        if self.warmup_epochs < 0:
            raise ValueError("warmup epochs must be nonnegative")
        if not 0.0 <= self.decay_fraction <= 1.0:
            raise ValueError("decay fraction must be in [0, 1]")
        if not 0.0 < self.final_learning_rate_fraction <= 1.0:
            raise ValueError("final learning-rate fraction must be in (0, 1]")
        if self.report_every <= 0:
            raise ValueError("report interval must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation fraction must be in [0, 1)")
        if not math.isfinite(self.ownership_weight) or self.ownership_weight < 0.0:
            raise ValueError("ownership weight must be finite and nonnegative")
        if not 0.0 < self.recency_decay <= 1.0:
            raise ValueError("recency decay must be in (0, 1]")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        defaults: LearnerConfig | None = None,
    ) -> LearnerConfig:
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown learner options: {sorted(unknown)}")
        merged = asdict(defaults or cls())
        merged.update(values)
        config = cls(**merged)
        config.validate()
        return config


@dataclass(frozen=True)
class LearnerUpdate:
    datasets: tuple[Path, ...]
    output: Path
    initial_checkpoint: Path | None
    config: LearnerConfig

    @classmethod
    def from_mapping(
        cls,
        message: Mapping[str, object],
        *,
        defaults: LearnerConfig | None = None,
    ) -> LearnerUpdate:
        if "datasets" not in message or "output" not in message:
            raise ValueError("update requires datasets and output")
        raw_datasets = message["datasets"]
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError("datasets must be a non-empty JSON list")
        nested = message.get("config", {})
        if not isinstance(nested, dict):
            raise ValueError("config must be a JSON object")
        option_names = {field.name for field in fields(LearnerConfig)}
        options = dict(nested)
        options.update(
            {
                key: value
                for key, value in message.items()
                if key in option_names
            }
        )
        allowed = {
            "command",
            "request_id",
            "datasets",
            "output",
            "initial_checkpoint",
            "config",
            *option_names,
        }
        unknown = set(message) - allowed
        if unknown:
            raise ValueError(f"unknown update fields: {sorted(unknown)}")
        initial = message.get("initial_checkpoint")
        return cls(
            datasets=tuple(Path(str(path)) for path in raw_datasets),
            output=Path(str(message["output"])),
            initial_checkpoint=None if initial is None else Path(str(initial)),
            config=LearnerConfig.from_mapping(options, defaults=defaults),
        )


@dataclass(frozen=True)
class _FileSignature:
    device: int
    inode: int
    size: int
    modified_ns: int
    manifest_digest: str | None


@dataclass(frozen=True)
class PreparedReplayShard:
    """One validated replay shard after raw search-only fields are consumed."""

    path: Path
    digest: str
    dataset: PreparedRasterDataset
    split_hashes: torch.Tensor
    corrected_samples: int
    signature: _FileSignature

    @property
    def samples(self) -> int:
        return self.dataset.samples


def _manifest_digest(path: Path) -> str | None:
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.get("dataset_sha256")
    return digest if isinstance(digest, str) else None


def _signature(path: Path) -> _FileSignature:
    metadata = path.stat()
    return _FileSignature(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        manifest_digest=_manifest_digest(path),
    )


def _stable_game_hashes(dataset: RasterDataset, digest: str) -> torch.Tensor:
    """Hash games, not rows, so every ply remains on the same side of a split."""

    games = dataset.game_ids.detach().cpu().numpy().astype(np.uint64, copy=True)
    salt = np.uint64(int(digest[:16], 16))
    values = games ^ salt
    # SplitMix64 is cheap, deterministic, and well distributed even when game
    # ids are consecutive small integers.
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    # Signed int64 comparisons are portable across PyTorch CPU builds. Dropping
    # one bit leaves a uniform value in [0, 2**63).
    return torch.from_numpy((values >> np.uint64(1)).astype(np.int64, copy=False))


def _drop_ownership(dataset: PreparedRasterDataset) -> PreparedRasterDataset:
    """Release ownership targets a zero-weighted loss will never read.

    Dense over the policy grid at float32, so they are 0.063 MB per sample --
    a third of a packed sample, resident for every update the shard's window
    spans. The stager already zero-fills the ownership slot for shards that
    have none, so dropping them here takes the same path a pre-`final_stones`
    shard does.
    """
    if dataset.ownerships is None:
        return dataset
    return replace(dataset, ownerships=None)


def _pack_states(dataset: RasterDataset) -> RasterDataset:
    """Replace a shard's dense states with the packed planes, when it can.

    `states` is left in place as a zero-sample view so anything reading its
    dtype or channel count still works; the rows themselves come from
    `packed_states`. A layout the packer does not recognise -- a different
    channel count, a komi plane that varies, a channel that stopped being
    binary -- is returned untouched and keeps its dense states.
    """
    if dataset.packed_states is not None or not is_packable(dataset.states):
        return dataset
    return replace(
        dataset,
        states=dataset.states[:0].clone(),
        packed_states=pack_states(dataset.states),
    )


def _pack_policy(dataset: RasterDataset) -> RasterDataset:
    """Replace a shard's dense policy targets and masks with the packed form.

    These dominate the window once states are packed and ownership is released:
    a dense float32 target that is 99.67% zeros, and a boolean mask at one byte
    per cell. `policies` and `policy_masks` are left as zero-sample views so
    anything reading their width still works -- `apply_dihedral` takes the
    policy resolution from `policies.shape[1]` -- and the stager expands only
    the rows a batch needs.
    """
    if dataset.packed_policy is not None or not is_policy_packable(
        dataset.policies, dataset.policy_masks
    ):
        return dataset
    return replace(
        dataset,
        policies=dataset.policies[:0].clone(),
        policy_masks=dataset.policy_masks[:0].clone(),
        packed_policy=pack_policy(dataset.policies, dataset.policy_masks),
    )


class ReplayCache:
    """In-memory cache of validated, prepared immutable replay shards."""

    def __init__(
        self,
        *,
        loader: Callable[[str | Path], RasterDataset] = load_dataset,
    ) -> None:
        self._loader = loader
        self._entries: dict[Path, PreparedReplayShard] = {}
        self._raster_kind: str | None = None
        self.hits = 0
        self.misses = 0

    def use_raster_kind(self, raster_kind: str | None) -> None:
        """Render with `raster_kind` from here on, dropping anything else.

        The kind belongs to the *update*, not to this object: one learner
        process serves every update of a run, and each arrives with its own
        config. Binding the loader once at construction reads the process
        defaults instead, which are whatever the service started with -- for the
        pipeline, nothing.

        A cached shard was rendered under whichever kind was in force when it was
        loaded, so a change invalidates the cache rather than being applied to
        new entries only. In practice the kind is run identity and never changes
        mid-run, which is exactly why a silent mismatch would go unnoticed.
        """
        if raster_kind == self._raster_kind:
            return
        self._raster_kind = raster_kind
        self._entries.clear()
        self._loader = partial(load_dataset, raster_kind=raster_kind)

    def get(self, path: str | Path, preparation_batch_size: int) -> PreparedReplayShard:
        resolved = Path(path).resolve(strict=True)
        signature = _signature(resolved)
        cached = self._entries.get(resolved)
        if cached is not None and cached.signature == signature:
            self.hits += 1
            return cached

        self.misses += 1
        raw = self._loader(resolved)
        digest = signature.manifest_digest or file_sha256(resolved)
        corrected_samples = int(
            torch.count_nonzero(raw.proposal_counts[:, :-1], dim=1).gt(0).sum().item()
        )
        split_hashes = _stable_game_hashes(raw, digest)
        prepared = prepare_policy_supervision(
            raw,
            preparation_batch_size,
            validate_targets=False,
        )
        # A cached shard stays resident for every update its window spans, so
        # its states are the term that scales. Three of the five compact
        # channels are binary and komi is one value per sample, which packs to
        # 4.2x less; the stager expands only the rows a batch needs, at 0.04%
        # of a training step. Layouts without that structure keep dense states.
        prepared = _pack_states(prepared)
        # The targets are the rest of the window once states are packed: dense
        # float32 that is 99.67% zeros, plus a mask at one byte per bit.
        prepared = _pack_policy(prepared)
        entry = PreparedReplayShard(
            path=resolved,
            digest=digest,
            dataset=prepared,
            split_hashes=split_hashes,
            corrected_samples=corrected_samples,
            signature=signature,
        )
        self._entries[resolved] = entry
        return entry

    def window(
        self, paths: Iterable[str | Path], preparation_batch_size: int
    ) -> ReplayWindow:
        resolved = tuple(Path(path).resolve(strict=True) for path in paths)
        if not resolved:
            raise ValueError("at least one replay dataset is required")
        # The RL window is the ownership boundary. Once a shard leaves it,
        # retaining several gigabytes of tensors cannot produce a future hit.
        # Evict before loading the entering shard so a sliding N-shard window
        # never has an avoidable N+1-shard peak in host memory.
        active = set(resolved)
        self._entries = {
            path: shard for path, shard in self._entries.items() if path in active
        }
        shards = tuple(
            self.get(path, preparation_batch_size) for path in resolved
        )
        return ReplayWindow(shards)

    def status(self) -> dict[str, object]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": [
                {
                    "path": str(shard.path),
                    "digest": shard.digest,
                    "samples": shard.samples,
                    "preparation": PREPARATION_VERSION,
                }
                for shard in self._entries.values()
            ],
        }


@dataclass(frozen=True)
class ShardSelection:
    shard: PreparedReplayShard
    rows: torch.Tensor


@dataclass(frozen=True)
class BatchPart:
    shard: PreparedReplayShard
    rows: torch.Tensor


@dataclass(frozen=True)
class BatchSpec:
    parts: tuple[BatchPart, ...]
    transform: int = 0

    @property
    def samples(self) -> int:
        return sum(int(part.rows.numel()) for part in self.parts)


@dataclass(frozen=True)
class ReplayView:
    selections: tuple[ShardSelection, ...]

    @property
    def samples(self) -> int:
        return sum(int(selection.rows.numel()) for selection in self.selections)

    @property
    def height(self) -> int:
        return self.selections[0].shard.dataset.height

    @property
    def width(self) -> int:
        return self.selections[0].shard.dataset.width

    @property
    def channels(self) -> int:
        return self.selections[0].shard.dataset.channels

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(str(selection.shard.path) for selection in self.selections)

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
        augment: bool = False,
        weights: torch.Tensor | None = None,
    ) -> list[BatchSpec]:
        """Batch specs over this view.

        `weights` is one frequency weight per row of the concatenated view,
        averaging 1.0. Rows are repeated by those weights before shuffling, so
        an epoch keeps its length in expectation while its composition shifts.
        See vgo_training/recency.py.
        """
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        shard_ids = torch.cat(
            [
                torch.full(
                    (selection.rows.numel(),),
                    index,
                    dtype=torch.int64,
                )
                for index, selection in enumerate(self.selections)
            ]
        )
        rows = torch.cat([selection.rows for selection in self.selections])
        if weights is not None:
            if weights.numel() != rows.numel():
                raise ValueError(
                    f"weights cover {weights.numel()} rows, view has {rows.numel()}"
                )
            # floor(w) copies plus one more with the fractional probability,
            # which is unbiased in expectation.
            floor = weights.floor()
            extra = torch.rand(weights.shape, generator=generator) < (weights - floor)
            counts = (floor + extra.to(weights.dtype)).to(torch.long)
            repeat = torch.repeat_interleave(
                torch.arange(counts.numel()), counts
            )
            shard_ids = shard_ids[repeat]
            rows = rows[repeat]
        if shuffle and rows.numel() > 1:
            order = torch.randperm(rows.numel(), generator=generator)
            shard_ids = shard_ids[order]
            rows = rows[order]

        batches: list[BatchSpec] = []
        for start in range(0, rows.numel(), batch_size):
            stop = min(start + batch_size, rows.numel())
            batch_shards = shard_ids[start:stop]
            batch_rows = rows[start:stop]
            parts = tuple(
                BatchPart(
                    selection.shard,
                    batch_rows[batch_shards == index],
                )
                for index, selection in enumerate(self.selections)
                if bool((batch_shards == index).any())
            )
            transform = (
                int(
                    torch.randint(
                        len(DIHEDRAL_TRANSFORMS),
                        (1,),
                        generator=generator,
                    ).item()
                )
                if augment
                else 0
            )
            batches.append(BatchSpec(parts, transform))
        return batches


@dataclass(frozen=True)
class ReplaySplit:
    training: ReplayView
    validation: ReplayView
    validation_samples: int


class ReplayWindow:
    def __init__(self, shards: tuple[PreparedReplayShard, ...]) -> None:
        if not shards:
            raise ValueError("a replay window cannot be empty")
        first = shards[0].dataset
        expected = (
            first.channels,
            first.height,
            first.width,
            first.policies.shape[1],
        )
        for shard in shards[1:]:
            dataset = shard.dataset
            actual = (
                dataset.channels,
                dataset.height,
                dataset.width,
                dataset.policies.shape[1],
            )
            if actual != expected:
                raise ValueError("all replay shards must have the same tensor shape")
        self.shards = shards

    @property
    def samples(self) -> int:
        return sum(shard.samples for shard in self.shards)

    @property
    def channels(self) -> int:
        return self.shards[0].dataset.channels

    @property
    def height(self) -> int:
        return self.shards[0].dataset.height

    @property
    def width(self) -> int:
        return self.shards[0].dataset.width

    @property
    def policy_size(self) -> int:
        return self.shards[0].dataset.policies.shape[1]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(str(shard.path) for shard in self.shards)

    def split(self, validation_fraction: float) -> ReplaySplit:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation fraction must be in [0, 1)")
        cutoff = int(validation_fraction * (1 << 63))
        validation_masks = [
            shard.split_hashes < cutoff for shard in self.shards
        ]
        validation_samples = sum(int(mask.sum().item()) for mask in validation_masks)

        # Preserve whole games while avoiding an empty side when there is more
        # than one game in the window. A one-game fixture intentionally falls
        # back to evaluating the training view, matching the old no-split path.
        if validation_fraction > 0.0 and validation_samples == 0 and self.samples > 1:
            choices = [
                (int(shard.split_hashes.min().item()), index)
                for index, shard in enumerate(self.shards)
            ]
            _, selected = min(choices)
            score = self.shards[selected].split_hashes.min()
            candidate = self.shards[selected].split_hashes == score
            if int(candidate.sum().item()) < self.samples:
                validation_masks[selected] = candidate
                validation_samples = int(candidate.sum().item())
        if validation_samples == self.samples and self.samples > 1:
            choices = [
                (int(shard.split_hashes.max().item()), index)
                for index, shard in enumerate(self.shards)
            ]
            _, selected = max(choices)
            score = self.shards[selected].split_hashes.max()
            candidate = self.shards[selected].split_hashes == score
            if int(candidate.sum().item()) < self.samples:
                validation_masks[selected] = validation_masks[selected] & ~candidate
                validation_samples -= int(candidate.sum().item())

        training = ReplayView(
            tuple(
                ShardSelection(shard, (~mask).nonzero().flatten())
                for shard, mask in zip(self.shards, validation_masks, strict=True)
                if int((~mask).sum().item()) > 0
            )
        )
        if validation_samples:
            validation = ReplayView(
                tuple(
                    ShardSelection(shard, mask.nonzero().flatten())
                    for shard, mask in zip(self.shards, validation_masks, strict=True)
                    if int(mask.sum().item()) > 0
                )
            )
        else:
            validation = training
        if training.samples == 0:
            raise ValueError("replay split contains no training samples")
        return ReplaySplit(training, validation, validation_samples)


@dataclass
class _StagingSlot:
    host: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    device: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    copy_complete: torch.cuda.Event | None = None
    consumed: torch.cuda.Event | None = None


def ownership_loss(
    predicted: torch.Tensor, targets: torch.Tensor, present: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy of the predicted ownership map.

    Each cell's output is read as a logit for "the mover owns this cell", which
    is exactly a two-class softmax with the redundant second logit dropped --
    softmax over two classes depends only on their difference, so one logit
    carries it. Verified identical to `cross_entropy` on the two-logit form.

    BCE rather than MSE because MSE has no finite optimum per cell. Targets are
    exactly +/-1, so MSE keeps pulling the raw value toward the target even once
    the sign is long settled: at logit +8 on a correct cell it still applies a
    gradient of 14, against BCE's 0.0003. That wasted pull is what drove 16.6%
    of cells past +/-1, a magnitude that means nothing since ownership is
    bounded. BCE spends gradient only where the model is wrong or unsure, and
    `sigmoid(logit)` then reads as a per-cell confidence.

    MSE does give a larger gradient on a confidently wrong cell (18 against 1),
    but that was the same argument that favoured MSE for the value head and it
    was the wrong frame there too: what drives learning is the ratio between
    wrong and settled cells, not the absolute magnitude.

    `present` masks samples whose game has no stored final board -- shards
    written before the field existed -- so a mixed window trains on the ones
    that have it rather than on zeros pretending to be a target.
    """
    if not bool(present.any()):
        return predicted.sum() * 0.0
    # Targets arrive as +/-1 from the mover's view; BCE wants 0/1.
    return nn.functional.binary_cross_entropy_with_logits(
        predicted[present], (targets[present] > 0).to(predicted.dtype)
    )


def value_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy of win/loss logits against a +/-1 outcome.

    The head is categorical now, so the loss is too. tanh + MSE carried a
    (1 - v^2) factor that vanished exactly where the model was most wrong: on a
    confidently mistaken prediction the gradient was 1.9e-6 against 1.0 here, a
    factor of half a million. Measured on the trained model, the median position
    saw 0.0004 of the gradient a non-saturating loss would give.

    Accepts the scalar target the shards already store, so no replay version
    changes: +1 means the mover won, -1 that it lost, 0 a tie.

    A tie becomes 0.5/0.5 rather than a third class. Ties need
    black - white - komi inside f64::EPSILON on continuous areas and there were
    zero in 1400 games, so a draw logit would be a class that never fires --
    but a hard `target <= 0` would have trained the rare tie as an outright
    loss, which is a full-strength wrong gradient on a position that was even.
    Soft targets cost nothing and say what actually happened.
    """
    if logits.dim() == 1:
        # A checkpoint from before the head became categorical.
        raise ValueError(
            "value head emitted a scalar; this checkpoint predates the "
            "win/loss head and its value semantics are incompatible"
        )
    win = (targets.to(logits.dtype) + 1.0) / 2.0
    return nn.functional.cross_entropy(
        logits, torch.stack((win, 1.0 - win), dim=1)
    )


class BatchStager:
    """Double-buffered shard gather and asynchronous host-to-device pipeline."""

    def __init__(
        self,
        *,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        policy_size: int,
        device: torch.device,
        state_dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        # index_select writes straight into the staging buffer and requires the
        # same scalar type, so this follows the dataset rather than assuming
        # one. Shards load half; a caller holding float32 states says so.
        self.state_dtype = state_dtype
        self.shape = (channels, height, width, policy_size)
        self.device_name = str(device)
        self.device = device
        pin = device.type == "cuda"
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vgo-replay-prefetch"
        )
        self._copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._slots = [
            self._new_slot(batch_size, channels, height, width, policy_size, pin)
            for _ in range(2)
        ]
        self._closed = False

    def _new_slot(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        policy_size: int,
        pin: bool,
    ) -> _StagingSlot:
        host = (
            torch.empty(
                (batch, channels, height, width),
                dtype=self.state_dtype,
                pin_memory=pin,
            ),
            torch.empty((batch, policy_size), dtype=torch.float32, pin_memory=pin),
            torch.empty((batch, policy_size), dtype=torch.bool, pin_memory=pin),
            torch.empty((batch,), dtype=torch.float32, pin_memory=pin),
            # Ownership: policy_size - 1 cells, the placement block without the
            # pass slot. Always allocated; shards lacking `final_stones` leave
            # it zero and the loss masks those rows.
            torch.empty((batch, policy_size - 1), dtype=torch.float32, pin_memory=pin),
        )
        device_buffers = None
        if self.device.type == "cuda":
            device_buffers = tuple(
                torch.empty_like(tensor, device=self.device) for tensor in host
            )
        return _StagingSlot(host=host, device=device_buffers)

    def compatible(
        self,
        *,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        policy_size: int,
        device: torch.device,
        state_dtype: torch.dtype,
    ) -> bool:
        return (
            self.batch_size == batch_size
            and self.shape == (channels, height, width, policy_size)
            and self.device_name == str(device)
            # A cached stager whose buffers are the wrong scalar type would
            # fail on the first index_select rather than be rebuilt.
            and self.state_dtype == state_dtype
            and not self._closed
        )

    @staticmethod
    def _fill(slot: _StagingSlot, spec: BatchSpec) -> int:
        if slot.copy_complete is not None:
            # A pinned buffer can be recycled as soon as its asynchronous copy
            # finishes; it need not wait for the GPU computation consuming the
            # separate device buffer.
            slot.copy_complete.synchronize()
        count = spec.samples
        offset = 0
        for part in spec.parts:
            part_count = int(part.rows.numel())
            dataset = part.shard.dataset
            # States may be held packed: three of the five compact channels are
            # binary and komi is one value per sample, so the window keeps 4.2x
            # less and expands the rows this batch actually needs. See
            # vgo_training/packed_states.py.
            packed = getattr(dataset, "packed_states", None)
            if packed is None:
                torch.index_select(
                    dataset.states,
                    0,
                    part.rows,
                    out=slot.host[0][offset : offset + part_count],
                )
            else:
                packed.expand(
                    part.rows, out=slot.host[0][offset : offset + part_count]
                )
            # Targets may be held packed for the same reason as states: the
            # dense form is a float32 tensor that is 99.67% zeros plus a mask
            # at one byte per bit. See vgo_training/packed_policy.py.
            packed_targets = getattr(dataset, "packed_policy", None)
            if packed_targets is None:
                sources = (dataset.policies, dataset.policy_masks)
                for source, target in zip(sources, slot.host[1:3], strict=False):
                    torch.index_select(
                        source,
                        0,
                        part.rows,
                        out=target[offset : offset + part_count],
                    )
            else:
                packed_targets.expand_policies(
                    part.rows, slot.host[1][offset : offset + part_count]
                )
                packed_targets.expand_masks(
                    part.rows, slot.host[2][offset : offset + part_count]
                )
            torch.index_select(
                dataset.values,
                0,
                part.rows,
                out=slot.host[3][offset : offset + part_count],
            )
            # Ownership separately: a shard predating `final_stones` has none,
            # and zeroing the destination slice is far cheaper than materialising
            # a zero source of `samples x policy_size` to index into.
            destination = slot.host[4][offset : offset + part_count]
            if dataset.ownerships is None:
                destination.zero_()
            else:
                torch.index_select(dataset.ownerships, 0, part.rows, out=destination)
            offset += part_count
        if spec.transform:
            dataset = spec.parts[0].shard.dataset
            states, policies, masks = apply_dihedral(
                slot.host[0][:count],
                slot.host[1][:count],
                slot.host[2][:count],
                spec.transform,
                dataset.height,
                dataset.width,
                # `policies` may be a zero-sample view when the targets are
                # packed; its width survives, which is all this needs.
                int(round((dataset.policies.shape[1] - 1) ** 0.5)),
                int(round((dataset.policies.shape[1] - 1) ** 0.5)),
            )
            slot.host[0][:count].copy_(states)
            slot.host[1][:count].copy_(policies)
            slot.host[2][:count].copy_(masks)
            # Ownership is a spatial field over the same grid as the policy's
            # placement block, so it needs the identical reindexing -- without
            # this an augmented batch pairs a rotated board with an unrotated
            # target. `apply_dihedral` takes policy vectors, which carry a
            # trailing pass slot, so a dummy one is padded on and dropped again.
            # Verified equivalent to rotating the map as an image for transforms
            # 1, 3 and 5.
            side = int(round((dataset.policies.shape[1] - 1) ** 0.5))
            owned = slot.host[4][:count]
            padded = torch.cat(
                (owned, torch.zeros((count, 1), dtype=owned.dtype)), dim=1
            )
            _, rotated, _ = apply_dihedral(
                slot.host[0][:count],
                padded,
                slot.host[2][:count],
                spec.transform,
                dataset.height,
                dataset.width,
                side,
                side,
            )
            owned.copy_(rotated[:, :-1])
        return count

    def _transfer(
        self, slot: _StagingSlot, count: int
    ) -> tuple[torch.Tensor, ...]:
        if self.device.type != "cuda":
            return tuple(tensor[:count] for tensor in slot.host)  # type: ignore[return-value]
        assert self._copy_stream is not None
        assert slot.device is not None
        with torch.cuda.stream(self._copy_stream):
            if slot.consumed is not None:
                self._copy_stream.wait_event(slot.consumed)
            for host, device in zip(slot.host, slot.device, strict=True):
                device[:count].copy_(host[:count], non_blocking=True)
            slot.copy_complete = torch.cuda.Event()
            slot.copy_complete.record(self._copy_stream)
        torch.cuda.current_stream(self.device).wait_event(slot.copy_complete)
        return tuple(tensor[:count] for tensor in slot.device)  # type: ignore[return-value]

    def _mark_consumed(self, slot: _StagingSlot) -> None:
        if self.device.type == "cuda":
            slot.consumed = torch.cuda.Event()
            slot.consumed.record(torch.cuda.current_stream(self.device))

    def batches(
        self, specs: Iterable[BatchSpec]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._closed:
            raise RuntimeError("batch stager is closed")
        iterator = iter(specs)
        pending: list[tuple[_StagingSlot, Future[int]]] = []
        for slot in self._slots:
            try:
                spec = next(iterator)
            except StopIteration:
                break
            pending.append((slot, self._executor.submit(self._fill, slot, spec)))
        while pending:
            slot, future = pending.pop(0)
            count = future.result()
            try:
                yield self._transfer(slot, count)
            finally:
                # The generator resumes (or closes after an exception) only
                # after the caller launched its work for this batch, so this
                # event precisely protects device-buffer reuse.
                self._mark_consumed(slot)
            try:
                spec = next(iterator)
            except StopIteration:
                continue
            pending.append((slot, self._executor.submit(self._fill, slot, spec)))

    def close(self) -> None:
        if self._closed:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._executor.shutdown(wait=True)
        self._closed = True


@torch.no_grad()
def evaluate(
    model: nn.Module,
    view: ReplayView,
    stager: BatchStager,
    *,
    batch_size: int,
    value_weight: float,
    precision: str,
) -> dict[str, float]:
    """Evaluate with one device-to-host synchronization for the entire split."""

    if view.samples <= 0:
        raise ValueError("cannot compute metrics over an empty dataset")
    model.eval()
    # One slot per metric summed below; keep in step with the torch.stack call
    # and with the totals[...] indices after it.
    accumulator = torch.zeros(6, dtype=torch.float64, device=stager.device)
    # Five, not four: the stager carries ownership alongside the rest. Metrics
    # do not use it -- the ownership head is training-only -- but the tuple
    # still has to be unpacked.
    for states, targets, masks, values, _ownership in stager.batches(
        view.batches(batch_size, shuffle=False)
    ):
        with torch.autocast(
            device_type=stager.device.type,
            dtype=torch.bfloat16,
            enabled=stager.device.type == "cuda" and precision == "bfloat16",
        ):
            logits, predictions = model(states.float())
            masked_logits = logits.masked_fill(
                ~masks, torch.finfo(logits.dtype).min
            )
            cross_entropy = -(
                targets * torch.log_softmax(masked_logits, dim=1)
            ).sum(dim=1)
            target_entropy = -(
                targets * targets.clamp_min(1e-12).log()
            ).sum(dim=1)
            squared_error = (predictions - values).square()
            # Sign agreement: for a categorical head this is the metric that
            # matters -- did it pick the right outcome -- where MAE mixes
            # correctness with confidence.
            value_correct = (
                (predictions.sign() == values.sign()) | (values == 0)
            ).to(dtype=logits.dtype)
            top1 = (
                masked_logits.argmax(dim=1) == targets.argmax(dim=1)
            ).to(dtype=logits.dtype)
            absolute_error = (predictions - values).abs()
        accumulator += torch.stack(
            (
                cross_entropy.sum(),
                target_entropy.sum(),
                squared_error.sum(),
                top1.sum(),
                absolute_error.sum(),
                value_correct.sum(),
            )
        ).to(dtype=torch.float64)

    # This is intentionally the only CUDA-to-Python synchronization in the
    # metric pass.
    totals = accumulator.cpu().tolist()
    cross_entropy = totals[0] / view.samples
    target_entropy = totals[1] / view.samples
    value_mse = totals[2] / view.samples
    return {
        "loss": cross_entropy + value_weight * value_mse,
        "policy_cross_entropy": cross_entropy,
        "policy_target_entropy": target_entropy,
        "policy_kl": cross_entropy - target_entropy,
        "policy_top1": totals[3] / view.samples,
        "value_mae": totals[4] / view.samples,
        # The head predicts a category now, so report how often it picks the
        # right one. 0.5 is chance; value_mae can look respectable while sign
        # accuracy sits at chance, which is what memorisation looks like.
        "value_sign_accuracy": totals[5] / view.samples,
    }


def _cpu_clone(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_signature(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _atomic_torch_save(value: object, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    if os.name != "nt":
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _build_optimizer(
    model: nn.Module,
    config: "LearnerConfig",
    log: "Callable[[str], None]",
) -> torch.optim.Optimizer:
    """Adam, or Muon on the trunk with Adam on everything else.

    `full_adam` puts every parameter on Adam, which is what every run before
    Muon landed used. Otherwise 2D+ weights that are not an output head go to
    Muon: the heads are 1x1 convs and thin linears, which are rank-degenerate
    and so meaningless to orthogonalize, and norm weights are 1D.

    `config.optimizer`, when set, overrides that pair by name so an A/B can
    select an arm without recipes having to know about `full_adam`.
    """
    choice = config.optimizer or ("adam" if config.full_adam else "muon")

    if choice == "ranger21":
        # Ranger21 schedules its own warmup and warmdown from num_epochs and
        # num_batches_per_epoch, and refuses to construct without them. Both are
        # switched off here so `schedule` still drives the rate, which is what
        # keeps an optimizer A/B from silently comparing two different curves;
        # with scheduling off the counts are unused, so the epoch count is
        # passed for its logging and the batch count is nominal.
        from ranger21 import Ranger21

        return Ranger21(
            model.parameters(),
            lr=config.learning_rate,
            num_epochs=max(1, config.epochs),
            num_batches_per_epoch=1,
            use_warmup=False,
            warmdown_active=False,
        )

    if choice == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    from .muon import HybridMuon

    trunk, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = any(
            token in name
            for token in ("policy_map", "pass_head", "value_head", "ownership_map")
        )
        (trunk if parameter.ndim >= 2 and not head else rest).append(parameter)
    log(
        f"muon: {sum(p.numel() for p in trunk):,} trunk params @ lr "
        f"{config.muon_learning_rate}, {sum(p.numel() for p in rest):,} on Adam "
        f"@ lr {config.learning_rate}"
    )
    return HybridMuon(
        [
            {"params": trunk, "lr": config.muon_learning_rate, "use_muon": True},
            {"params": rest, "lr": config.learning_rate, "use_muon": False},
        ]
    )


class PersistentLearner:
    """Long-lived model, optimizer, replay cache, and staging-buffer owner."""

    def __init__(
        self,
        *,
        defaults: LearnerConfig | None = None,
        replay_cache: ReplayCache | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.defaults = defaults or LearnerConfig()
        self.defaults.validate()
        self.replay_cache = replay_cache or ReplayCache()
        # Only a seed. Each update carries its own config and calls
        # `use_raster_kind` with it; see the note there.
        self.replay_cache.use_raster_kind(self.defaults.raster_kind)
        self._log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._model_metadata: dict[str, object] | None = None
        self._device: torch.device | None = None
        self._compiled = False
        self._stager: BatchStager | None = None
        self.current_checkpoint: Path | None = None
        self._current_checkpoint_signature: tuple[int, int, int, int] | None = None
        self._current_checkpoint_digest: str | None = None
        self.updates = 0
        self._closed = False

    def _resolve_device(self, name: str) -> torch.device:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training requested, but torch.cuda.is_available() is false"
            )
        return device

    def _runtime_matches_checkpoint(
        self,
        path: Path,
        config: LearnerConfig,
        window: ReplayWindow,
        policy_resolution: int,
    ) -> bool:
        metadata = self._model_metadata or {}
        return (
            self.model is not None
            and self.current_checkpoint == path
            and self._current_checkpoint_signature == _checkpoint_signature(path)
            and self._current_checkpoint_digest is not None
            and self._device == self._resolve_device(config.device)
            and self._compiled
            == bool(
                config.compile and torch.device(config.device).type == "cuda"
            )
            and (
                int(metadata.get("channels", -1)),
                int(metadata.get("height", -1)),
                int(metadata.get("width", -1)),
                int(metadata.get("policy_resolution", -1)),
            )
            == (
                window.channels,
                window.height,
                window.width,
                policy_resolution,
            )
        )

    def _initialize_runtime(
        self,
        window: ReplayWindow,
        config: LearnerConfig,
        initial_checkpoint: Path | None,
    ) -> tuple[bool, str | None, str | None]:
        device = self._resolve_device(config.device)
        if (
            device.type == "cuda"
            and config.precision == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "bfloat16 training is not supported by this CUDA device; "
                "select precision=float32"
            )
        policy_cells = window.policy_size - 1
        policy_resolution = int(round(policy_cells**0.5))
        if policy_resolution * policy_resolution != policy_cells:
            raise ValueError(
                f"replay policy vector {window.policy_size} is not a square grid plus pass"
            )
        decoupled = (
            policy_resolution
            if (policy_resolution, policy_resolution) != (window.height, window.width)
            else None
        )

        parent_checkpoint: str | None = None
        parent_checkpoint_digest: str | None = None
        reused = False
        if initial_checkpoint is not None:
            resolved_initial = initial_checkpoint.resolve(strict=True)
            parent_checkpoint = str(resolved_initial)
            reused = self._runtime_matches_checkpoint(
                resolved_initial, config, window, policy_resolution
            )
            parent_checkpoint_digest = (
                self._current_checkpoint_digest
                if reused
                else file_sha256(resolved_initial)
            )
        elif (
            self.model is not None
            and self.current_checkpoint is not None
            and self._current_checkpoint_signature is not None
            and self._current_checkpoint_digest is not None
            and self.current_checkpoint.exists()
            and self._current_checkpoint_signature
            == _checkpoint_signature(self.current_checkpoint)
        ):
            metadata = self._model_metadata or {}
            reused = (
                self._device == device
                and self._compiled == bool(config.compile and device.type == "cuda")
                and (
                    int(metadata.get("channels", -1)),
                    int(metadata.get("height", -1)),
                    int(metadata.get("width", -1)),
                    int(metadata.get("policy_resolution", -1)),
                )
                == (
                    window.channels,
                    window.height,
                    window.width,
                    policy_resolution,
                )
                and str(metadata.get("architecture")) == config.architecture
                and int(metadata.get("model_width", -1)) == config.model_width
                and int(metadata.get("blocks", -1)) == config.blocks
            )
            if reused:
                parent_checkpoint = str(self.current_checkpoint)
                parent_checkpoint_digest = self._current_checkpoint_digest

        optimizer_restored = False
        if not reused:
            # Until a new checkpoint is atomically published, a reconstructed
            # runtime has no reusable artifact identity. This also prevents an
            # exception during initial evaluation from associating freshly
            # loaded weights with the previously resident candidate.
            self.current_checkpoint = None
            self._current_checkpoint_signature = None
            self._current_checkpoint_digest = None
            if initial_checkpoint is not None:
                model, checkpoint = load_model(initial_checkpoint.resolve(strict=True))
                if (
                    int(checkpoint["channels"]),
                    int(checkpoint["height"]),
                    int(checkpoint["width"]),
                    int(checkpoint.get("policy_resolution", checkpoint["height"])),
                ) != (
                    window.channels,
                    window.height,
                    window.width,
                    policy_resolution,
                ):
                    raise ValueError("initial checkpoint does not match replay tensor shape")
                # The shape check above does not cover this. Two layouts can
                # share a width and differ in what a plane *means* --
                # `compact-pass` and `compact-dead-zone` are both six planes and
                # disagree only on the capture predicate -- so a swap between
                # them passes every dimension test and quietly feeds the loaded
                # weights a channel trained to mean something else.
                parent_kind = checkpoint.get("raster_kind")
                if (
                    config.raster_kind is not None
                    and parent_kind is not None
                    and parent_kind != config.raster_kind
                ):
                    raise ValueError(
                        f"initial checkpoint was trained on raster kind "
                        f"{parent_kind!r}, cannot warm start into "
                        f"{config.raster_kind!r}"
                    )
                metadata = {
                    "channels": window.channels,
                    "height": window.height,
                    "width": window.width,
                    "policy_resolution": policy_resolution,
                    "model_width": int(checkpoint["model_width"]),
                    "blocks": int(checkpoint["blocks"]),
                    "architecture": str(checkpoint.get("architecture", "flat")),
                    "raster_kind": config.raster_kind or parent_kind,
                    # Follows the parent, not the config: the K constants are
                    # part of the function the loaded weights were trained for,
                    # so a warm start cannot switch this on or off.
                    "variance_scaled": bool(
                        checkpoint.get("variance_scaled", False)
                    ),
                    "norm_groups": checkpoint.get("norm_groups"),
                    # Follows the parent for the same reason as the K
                    # constants: the attention blocks are part of the function
                    # the loaded weights were trained for.
                    "context_attention_blocks": int(
                        checkpoint.get("context_attention_blocks", 0)
                    ),
                    "attention_heads": int(checkpoint.get("attention_heads", 8)),
                }
            else:
                model = build_model(
                    architecture=config.architecture,
                    channels=window.channels,
                    width=config.model_width,
                    blocks=config.blocks,
                    policy_resolution=decoupled,
                    variance_scaled=config.variance_scaled,
                    norm_groups=config.norm_groups,
                    context_attention_blocks=config.context_attention_blocks,
                    attention_heads=config.attention_heads,
                    raster_resolution=window.height,
                )
                checkpoint = {}
                metadata = {
                    "channels": window.channels,
                    "height": window.height,
                    "width": window.width,
                    "policy_resolution": policy_resolution,
                    "model_width": config.model_width,
                    "blocks": config.blocks,
                    "architecture": config.architecture,
                    "raster_kind": config.raster_kind,
                    "variance_scaled": config.variance_scaled,
                    "norm_groups": config.norm_groups,
                    "context_attention_blocks": config.context_attention_blocks,
                    "attention_heads": config.attention_heads,
                }
            model = model.to(device)
            compiled = bool(config.compile and device.type == "cuda")
            if compiled:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.benchmark = True
                # An optimisation, not a correctness requirement, so losing it
                # should cost speed rather than the run -- this reports and
                # carries on instead of raising hours into a job.
                #
                # It is worth checking *which* torch raised before working
                # around a refusal. Torch declines outright on interpreters it
                # has not caught up to, and this box has two installs: 2.9.1 on
                # the system interpreter, which refuses Python 3.14, and 2.13.0
                # in training/.venv, which compiles on it fine. A script run
                # through its shebang finds the first.
                try:
                    model.compile()
                except (RuntimeError, AttributeError) as error:
                    self._log(f"[learner] torch.compile unavailable, continuing: {error}")
                    compiled = False
            optimizer = _build_optimizer(model, config, self._log)
            if config.restore_optimizer and checkpoint.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    optimizer_restored = True
                except (ValueError, KeyError) as error:
                    self._log(f"optimizer state not restored: {error}")
            self.model = model
            self.optimizer = optimizer
            self._model_metadata = metadata
            self._device = device
            self._compiled = compiled
        else:
            # Parent lineage was copied into local immutable strings above. From
            # this point onward even resetting Adam or its learning rate means
            # the resident runtime no longer exactly represents that artifact.
            # Clear identity before the first such mutation so every later
            # exception takes the discard path in update().
            self.current_checkpoint = None
            self._current_checkpoint_signature = None
            self._current_checkpoint_digest = None
            assert self.optimizer is not None
            optimizer_restored = config.restore_optimizer
            if not config.restore_optimizer:
                assert self.model is not None
                self.optimizer = _build_optimizer(self.model, config, self._log)

        assert self.optimizer is not None
        for group in self.optimizer.param_groups:
            # A Muon group keeps its own rate. Stamping the Adam rate over every
            # group would silently drop the trunk from 0.01 to 1e-3, and the
            # scheduler multiplies from `initial_lr`, so both have to survive.
            rate = (
                config.muon_learning_rate
                if group.get("use_muon")
                else config.learning_rate
            )
            group["lr"] = rate
            group["initial_lr"] = rate
        return optimizer_restored, parent_checkpoint, parent_checkpoint_digest

    def _ensure_stager(
        self, window: ReplayWindow, config: LearnerConfig
    ) -> tuple[BatchStager, bool]:
        assert self._device is not None
        arguments = {
            "batch_size": config.batch_size,
            "channels": window.channels,
            "height": window.height,
            "width": window.width,
            "policy_size": window.policy_size,
            "device": self._device,
            # The window is a set of shards, not one tensor; every shard is
            # rendered the same way, so the first one's dtype speaks for all.
            "state_dtype": window.shards[0].dataset.states.dtype,
        }
        if self._stager is not None and self._stager.compatible(**arguments):
            return self._stager, True
        if self._stager is not None:
            self._stager.close()
        self._stager = BatchStager(**arguments)
        return self._stager, False

    def _discard_unpublished_runtime(self) -> None:
        """Drop resident state which no longer identifies a published artifact."""

        device = self._device
        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                # Preserve the update's original exception. The references below
                # are sufficient to prevent any later request from reusing this
                # optimizer even when cleanup itself encounters a poisoned CUDA
                # context.
                pass
        self.model = None
        self.optimizer = None
        self._model_metadata = None
        self._device = None
        self._compiled = False
        self.current_checkpoint = None
        self._current_checkpoint_signature = None
        self._current_checkpoint_digest = None
        if device is not None and device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def update(self, request: LearnerUpdate) -> dict[str, object]:
        try:
            return self._update_once(request)
        except BaseException:
            # Once checkpoint identity is cleared, resident parameters may have
            # received only a prefix of the intended optimizer steps. Never let
            # an implicit-parent retry mistake that state for a valid starting
            # point. Failures before mutation leave a published identity intact
            # and can safely retain the compiled runtime.
            if self.current_checkpoint is None:
                self._discard_unpublished_runtime()
            raise

    def _update_once(self, request: LearnerUpdate) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("learner is closed")
        update_started = time.perf_counter()
        config = request.config
        config.validate()
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        hits_before = self.replay_cache.hits
        misses_before = self.replay_cache.misses
        self.replay_cache.use_raster_kind(config.raster_kind)
        window = self.replay_cache.window(request.datasets, config.batch_size)
        if config.ownership_weight == 0.0:
            # Release the targets the loss will not read. Done here rather than
            # in the cache because only the config knows the weight, and a
            # cached shard may outlive a run that changes it.
            for shard in window.shards:
                object.__setattr__(
                    shard, "dataset", _drop_ownership(shard.dataset)
                )
        split = window.split(config.validation_fraction)
        (
            optimizer_restored,
            parent_checkpoint,
            parent_checkpoint_digest,
        ) = self._initialize_runtime(window, config, request.initial_checkpoint)
        stager, stager_reused = self._ensure_stager(window, config)
        assert self.model is not None
        assert self.optimizer is not None
        assert self._model_metadata is not None
        assert self._device is not None

        generator = torch.Generator().manual_seed(config.seed)
        # The schedule advances per optimizer step, not per epoch. With ten
        # epochs the difference was cosmetic; at one epoch a per-epoch schedule
        # never moves at all -- warmup would consume the whole update and the
        # decay phase would never run. Expressing the horizon in steps keeps
        # the same curve shape whatever the epoch count is.
        steps_per_epoch = max(
            1, len(split.training.batches(config.batch_size, shuffle=False))
        )
        scheduler_arguments = argparse.Namespace(**asdict(config))
        scheduler_arguments.epochs = config.epochs * steps_per_epoch
        scheduler_arguments.warmup_epochs = round(
            config.warmup_epochs * steps_per_epoch
        )
        scheduler = build_scheduler(self.optimizer, scheduler_arguments)
        initial_training = evaluate(
            self.model,
            split.training,
            stager,
            batch_size=config.batch_size,
            value_weight=config.value_weight,
            precision=config.precision,
        )
        initial_validation = evaluate(
            self.model,
            split.validation,
            stager,
            batch_size=config.batch_size,
            value_weight=config.value_weight,
            precision=config.precision,
        )

        def selection_score(current: dict[str, float]) -> float:
            return current["policy_kl"] + config.value_weight * current["value_mae"]

        best_epoch = 0
        best = initial_validation
        best_score = selection_score(best)
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        best_optimizer_state = _cpu_clone(self.optimizer.state_dict())
        optimization_started = time.perf_counter()

        # One weight per training row, from shard age. The window arrives
        # oldest-first, so the last shard is the newest.
        training_weights = None
        if config.recency_decay < 1.0:
            # row_weights expects newest-first; the window is oldest-first, so
            # build it reversed and flip the result back into window order.
            sizes = [
                int(selection.rows.numel())
                for selection in reversed(split.training.selections)
            ]
            training_weights = torch.flip(
                row_weights(sizes, config.recency_decay), dims=(0,)
            )

        self.model.train()
        for epoch in range(1, config.epochs + 1):
            specs = split.training.batches(
                config.batch_size,
                shuffle=True,
                generator=generator,
                augment=config.augment,
                weights=training_weights,
            )
            for (
                states,
                policy_targets,
                policy_masks,
                value_targets,
                ownership_targets,
            ) in stager.batches(specs):
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.bfloat16,
                    enabled=(
                        self._device.type == "cuda"
                        and config.precision == "bfloat16"
                    ),
                ):
                    # Rasters are stored half; the model runs in its own
                    # precision, so widen at the boundary rather than storing
                    # four bytes a pixel for the whole window.
                    outputs = self.model(states.float())
                    logits, values = outputs[0], outputs[1]
                    policy_loss = policy_cross_entropy(
                        logits, policy_targets, policy_masks
                    )
                    value_loss = value_cross_entropy(values, value_targets)
                    loss = policy_loss + config.value_weight * value_loss
                    # Ownership is dense spatial supervision: one target per
                    # cell rather than one scalar per position, which is what
                    # forces the trunk to represent *where* a group lives
                    # instead of only who won. Absent for shards written before
                    # `final_stones`, whose rows arrive zeroed and are masked.
                    # Zero weight drops the targets from the window too, so
                    # `present` would be all-false; compute it once here and
                    # let both head sets read it rather than deriving it inside
                    # one branch that the other silently depends on.
                    train_ownership = config.ownership_weight > 0.0
                    present = (
                        ownership_targets.abs().sum(dim=1) > 0
                        if train_ownership
                        else None
                    )
                    if len(outputs) >= 5 and train_ownership:
                        loss = loss + config.ownership_weight * ownership_loss(
                            outputs[4], ownership_targets, present
                        )
                    # KataGo's one-batch-norm split: the normalized heads take
                    # most of the weight so they drive the trunk, while the
                    # inference heads above learn the same targets without
                    # normalization. Architectures that emit only two outputs
                    # are unaffected.
                    # Dispatch on presence, not on tuple length: this was
                    # `len(outputs) == 4` and stopped matching the moment the
                    # ownership head made it six, which silently dropped the
                    # 80% of the loss that shapes the trunk.
                    if len(outputs) >= 4:
                        normed_logits, normed_values = outputs[2], outputs[3]
                        normed_loss = policy_cross_entropy(
                            normed_logits, policy_targets, policy_masks
                        ) + config.value_weight * value_cross_entropy(
                            normed_values, value_targets
                        )
                        if len(outputs) >= 6 and train_ownership:
                            normed_loss = normed_loss + config.ownership_weight * ownership_loss(
                                outputs[5], ownership_targets, present
                            )
                        loss = (
                            NORMED_HEAD_WEIGHT * normed_loss
                            + (1.0 - NORMED_HEAD_WEIGHT) * loss
                        )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                scheduler.step()
            if (
                epoch == 1
                or epoch % config.report_every == 0
                or epoch == config.epochs
            ):
                current = evaluate(
                    self.model,
                    split.validation,
                    stager,
                    batch_size=config.batch_size,
                    value_weight=config.value_weight,
                    precision=config.precision,
                )
                score = selection_score(current)
                if score < best_score:
                    best_epoch = epoch
                    best = current
                    best_score = score
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
                    best_optimizer_state = _cpu_clone(self.optimizer.state_dict())
                self._log(
                    f"epoch={epoch:4d} policy_kl={current['policy_kl']:.5f} "
                    f"top1={current['policy_top1']:.3f} "
                    f"value_mae={current['value_mae']:.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )
                self.model.train()

        # Snapshot after the loop so this is the weights training actually ended
        # on. config.epochs is always evaluated, so `current` above is the final
        # epoch's metrics, but taking the state here does not depend on that.
        final_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        final_optimizer_state = _cpu_clone(self.optimizer.state_dict())
        # epochs=0 would leave `current` unbound; fall back to the initial
        # measurement so the field always describes the published weights.
        final_validation_metrics = current if config.epochs else initial_validation

        optimization_seconds = time.perf_counter() - optimization_started
        # Publish the last epoch, not the best-scoring one.
        #
        # Selection compares candidates that differ by ~0.07 on
        # policy_kl + value_weight * value_mae, using a validation set of ~38
        # *games* -- the split hashes games rather than rows, and every ply of a
        # game shares one terminal value, so value_mae is estimated from ~38
        # binary outcomes. Its standard error is ~0.16, doubled to ~0.32 by
        # value_weight=2. The signal is roughly a quarter of its own noise, so
        # picking the argmin mostly selects the luckiest measurement.
        #
        # That shows up in the best_epoch distribution across two runs: heavy
        # mass at both 0 (kept nothing: 38% of ddrnet-pipe's updates) and at the
        # final epoch, with a thin middle -- the shape of a max over a random
        # walk, not of a curve that improves then flattens.
        #
        # The last epoch is at least unbiased. best_* stay in the report as
        # diagnostics: their disagreement with the final epoch is the running
        # measure of how noisy selection is, and a validation set large enough
        # to make selection meaningful would show them converging.
        self.model.load_state_dict(final_state)
        self.optimizer.load_state_dict(final_optimizer_state)
        final_training = evaluate(
            self.model,
            split.training,
            stager,
            batch_size=config.batch_size,
            value_weight=config.value_weight,
            precision=config.precision,
        )
        final_validation = evaluate(
            self.model,
            split.validation,
            stager,
            batch_size=config.batch_size,
            value_weight=config.value_weight,
            precision=config.precision,
        )
        # Both halves have always been stored in the checkpoint metadata, but
        # only validation was ever printed, so the generalization gap -- the
        # thing that says whether an update memorized its window -- was
        # invisible without reading the JSON afterwards. Log it per update.
        self._log(
            "train/val  "
            f"value_mae {final_training['value_mae']:.5f}/"
            f"{final_validation['value_mae']:.5f} "
            f"gap {final_validation['value_mae'] - final_training['value_mae']:+.5f}  "
            f"policy_kl {final_training['policy_kl']:.5f}/"
            f"{final_validation['policy_kl']:.5f} "
            f"gap {final_validation['policy_kl'] - final_training['policy_kl']:+.5f}"
        )

        output = request.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema": "vgo.raster-policy-value.v1",
            **self._model_metadata,
            "state_dict": final_state,
            "optimizer_state_dict": final_optimizer_state,
            "parent_checkpoint": parent_checkpoint,
            "parent_checkpoint_sha256": parent_checkpoint_digest,
            "replay_sources": list(window.sources),
            "policy_target": POLICY_TARGET,
            "policy_denominator": POLICY_DENOMINATOR,
        }
        _atomic_torch_save(checkpoint, output)

        self.current_checkpoint = output
        self._current_checkpoint_signature = _checkpoint_signature(output)
        self._current_checkpoint_digest = file_sha256(output)
        self.updates += 1
        corrected_samples = sum(shard.corrected_samples for shard in window.shards)
        report = {
            "schema": "vgo.training-run.v3",
            "datasets": list(window.sources),
            "dataset_digests": [shard.digest for shard in window.shards],
            "checkpoint": str(output),
            "checkpoint_sha256": self._current_checkpoint_digest,
            "parent_checkpoint": parent_checkpoint,
            "parent_checkpoint_sha256": parent_checkpoint_digest,
            "device": str(self._device),
            "precision": (
                config.precision
                if self._device.type == "cuda"
                else "float32"
            ),
            "samples": window.samples,
            "training_samples": split.training.samples,
            "validation_samples": split.validation_samples,
            "shape": [
                window.samples,
                window.channels,
                window.height,
                window.width,
            ],
            "parameters": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "schedule": config.schedule,
            "compiled": self._compiled,
            "optimizer_restored": optimizer_restored,
            "stager_reused": stager_reused,
            "replay_cache_hits": self.replay_cache.hits - hits_before,
            "replay_cache_misses": self.replay_cache.misses - misses_before,
            "value_weight": config.value_weight,
            "policy_target": POLICY_TARGET,
            "policy_denominator": POLICY_DENOMINATOR,
            "importance_corrected_samples": corrected_samples,
            "uncorrected_samples": window.samples - corrected_samples,
            # The published weights are the final epoch's. best_* are kept as
            # diagnostics: selection_regret is how much better the best-scoring
            # epoch measured than the one actually published, and it is the
            # running estimate of how much of that gap is noise. On a validation
            # set large enough for selection to mean something, best_epoch would
            # concentrate near the end and this would go to roughly zero.
            "published_epoch": "final",
            "selection_metric": "policy_kl + value_weight * value_mae",
            "selection_regret": (
                selection_score(final_validation_metrics) - best_score
            ),
            "wall_seconds": time.perf_counter() - update_started,
            "optimization_seconds": optimization_seconds,
            "best_epoch": best_epoch,
            "initial_training": initial_training,
            "initial_validation": initial_validation,
            "best_validation": best,
            "final_training": final_training,
            "final_validation": final_validation,
        }
        report_path = output.with_suffix(output.suffix + ".json")
        atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
        # The service keeps weights, Adam moments, and staging buffers resident,
        # but no completed update needs parameter gradients or transient
        # activation blocks. Release those before the coordinator publishes the
        # response so an overlapping TensorRT actor can reclaim the headroom.
        self.optimizer.zero_grad(set_to_none=True)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return report

    def update_from_mapping(self, message: Mapping[str, object]) -> dict[str, object]:
        return self.update(
            LearnerUpdate.from_mapping(message, defaults=self.defaults)
        )

    def status(self) -> dict[str, object]:
        metadata = None if self._model_metadata is None else dict(self._model_metadata)
        return {
            "schema": "vgo.learner.status.v1",
            "updates": self.updates,
            "current_checkpoint": (
                None if self.current_checkpoint is None else str(self.current_checkpoint)
            ),
            "current_checkpoint_sha256": self._current_checkpoint_digest,
            "model": metadata,
            "device": None if self._device is None else str(self._device),
            "compiled": self._compiled,
            "replay_cache": self.replay_cache.status(),
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._stager is not None:
            self._stager.close()
        self._closed = True


def _write_response(stream: TextIO, response: Mapping[str, object]) -> None:
    stream.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def serve_json_lines(
    learner: PersistentLearner,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
) -> None:
    """Serve one JSON response line per command; all progress stays on stderr."""

    _write_response(
        output_stream,
        {
            "schema": PROTOCOL_SCHEMA,
            "event": "ready",
            "status": "ready",
            "pid": os.getpid(),
        },
    )
    try:
        for line in input_stream:
            if not line.strip():
                continue
            request_id: object = None
            command: object = None
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("request must be a JSON object")
                request_id = message.get("request_id")
                command = message.get("command")
                if command == "update":
                    result: object = learner.update_from_mapping(message)
                elif command == "status":
                    result = learner.status()
                elif command == "shutdown":
                    learner.close()
                    _write_response(
                        output_stream,
                        {
                            "schema": PROTOCOL_SCHEMA,
                            "status": "ok",
                            "ok": True,
                            "command": command,
                            "request_id": request_id,
                            "result": learner.status(),
                        },
                    )
                    return
                else:
                    raise ValueError(f"unknown learner command: {command!r}")
                response = {
                    "schema": PROTOCOL_SCHEMA,
                    "status": "ok",
                    "ok": True,
                    "command": command,
                    "request_id": request_id,
                    "result": result,
                }
                # Pipeline callers historically named this payload `report`.
                # Keep the generic result field for protocol uniformity and the
                # explicit alias so supervisors never mistake the envelope for
                # the atomic training report.
                if command == "update":
                    response["report"] = result
                _write_response(output_stream, response)
            except Exception as error:
                print(
                    f"learner command failed: {type(error).__name__}: {error}",
                    file=error_stream,
                    flush=True,
                )
                _write_response(
                    output_stream,
                    {
                        "schema": PROTOCOL_SCHEMA,
                        "status": "error",
                        "ok": False,
                        "command": command,
                        "request_id": request_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
    finally:
        learner.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent JSON-lines VGO learner service"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="accepted for an explicit service invocation; serving is the default",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--muon-learning-rate", type=float, default=0.01)
    parser.add_argument("--full-adam", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    defaults = LearnerConfig(
        device=arguments.device,
        threads=arguments.threads,
        compile=arguments.compile,
        batch_size=arguments.batch_size,
    )
    serve_json_lines(PersistentLearner(defaults=defaults))
