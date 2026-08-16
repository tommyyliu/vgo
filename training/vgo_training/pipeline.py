from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Collection, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, BinaryIO

from .bradley_terry import fit_ratings
from .model import MODEL_ARCHITECTURES


PIPELINE_SCHEMA = "vgo.pipeline-state.v1"
CONFIG_SCHEMA = "vgo.pipeline-config.v1"
UPDATE_SCHEMA = "vgo.pipeline-update.v1"
TELEMETRY_SCHEMA = "vgo.telemetry-job.v1"
LEARNER_PROTOCOL_SCHEMA = "vgo.learner.protocol.v1"
# The runtime still builds and serves its configured batch (often 32). The
# exported dynamic axis is only a compatibility ceiling, so retaining modest
# headroom here does not consume the memory or compute of a batch of 64. It
# does keep the same checkpoint usable if a later workload can fill 64.
ONNX_EXPORT_BATCH_HEADROOM = 64
OPERATIONAL_CONFIG_FIELDS = {
    "output",
    "updates",
    "maximum_prefetch_shards",
    "actors",
    "writer_queue_games",
    # Request aggregation is serving policy, not learning/search identity. The
    # batch shape can perturb the last bits of accelerator arithmetic, but so
    # can the already-operational precision, driver, and execution-slot count;
    # none changes the intended search. Every shard records the effective batch
    # ceiling for provenance. A configured ceiling must still fit the maximum
    # embedded in the current ONNX artifact, so lowering 64 -> 32 is directly
    # resumable while raising beyond an old export requires re-exporting it.
    "inference_batch",
    "inference_delay_ms",
    "inference_slots",
    "inference_device_id",
    "warm_inference",
    "training_threads",
    "training_device",
    "compile",
    "overlap_actor_learner",
    "arena_actors",
    "telemetry_opponents",
    "telemetry_pairs",
    "arena_komi",
    # Neither changes what is learned: one picks which checkpoints get rated,
    # the other is disk housekeeping. Both must be adjustable on a resume.
    "telemetry_every",
    "retire_shards",
    # Inference precision is an execution choice, not a learning one. It does
    # perturb the numbers a generator sees, so this is a judgement rather than
    # an identity: measured on ddrnet-vs update 14 over 256 real positions,
    # fp16 agrees with fp32 on 100% of policy argmaxes and differs on value by
    # 0.004, which is far below the noise between two searches of the same
    # position. What it does change is speed -- 41.4 ms/batch against 17.5 --
    # and a run that cannot adopt that without starting over pays for the
    # decision twice.
    #
    # The failure this guards against is real -- fp16 overflows when
    # activations approach 65504, which is what ended ddrnet-short -- but a
    # config digest is the wrong place to catch it. Overflow is a property of
    # the checkpoint, which drifts every update, not of the flag, which only
    # changes when someone edits it. What actually catches it is generation
    # failing loudly on a non-finite evaluation, plus checking peak activation
    # against the limit before enabling this (diagnostics in the run's
    # launch.sh). Freezing the flag would only mean a run cannot adopt fp16
    # without starting over.
    "fp16",
    # Removed on 2026-08-16, kept here because old runs stored them.
    #
    # The promotion gate is gone: every candidate is now the next incumbent.
    # It was a measurement with no power that cost real work when it fired.
    # At --arena-pairs 8, scoring 0.55 over 16 games means winning 9, so a
    # candidate exactly as strong as the incumbent promoted 40% of the time and
    # one that truly scored 0.6 was rejected 28% of the time. Across
    # shard-sweep-15000's twenty updates, every arena's 95% interval on the
    # candidate's score contained 0.5 -- not one resolved a difference. Because
    # the incumbent is both the training parent and the generator, each
    # rejection made the next update retrain from the same checkpoint and
    # generate another shard from it: ddrnet-attn-komi rejected 60 of 83
    # overnight, advancing its lineage 23 times in 83 updates.
    #
    # A gate that discards roughly a quarter of real improvements, and burns a
    # shard whenever it does, is worse than no gate. Strength is now measured
    # after the fact by the telemetry tournaments (--telemetry-every), which
    # rate checkpoints without deciding anything.
    #
    # These two names stay in this set so that a run created while they were
    # config fields still resumes: the digest is taken over non-operational
    # fields, and a name absent from the config is simply absent from both
    # sides of that comparison.
    "promotion_arena",
    "promotion_score",
}
# These fields became operational after runs had already stored identity
# digests that included them. Keep the exact historical combinations bounded:
# accepting an arbitrary mismatched digest would let a foreign state file be
# silently relabelled merely because a pipeline-config.json happened to exist.
HISTORICAL_OPERATIONAL_FIELD_ADDITIONS = (
    frozenset({"inference_batch"}),
    frozenset({"promotion_arena", "promotion_score"}),
    frozenset({"inference_batch", "promotion_arena", "promotion_score"}),
)
# Fields that no longer exist. A removed field is the mirror image of a newly
# operational one: the stored pipeline-config.json still carries it, this
# version's config cannot, and a run whose digest was taken while it counted
# toward identity must still resume.
#
# `maximum_truncation_rate` is the one that bites. promotion_arena and
# promotion_score were already operational, so they were never in a digest, but
# the truncation rate was pure identity -- it existed only to let the promotion
# gate reject an arena that lost too many games to truncation, and it went with
# the gate on 2026-08-16.
REMOVED_CONFIG_FIELDS = frozenset(
    {"promotion_arena", "promotion_score", "maximum_truncation_rate"}
)


def _compress_shard(path: Path) -> tuple[Path, int, int]:
    """Compress one shard in place, leaving `<name>.zst` and removing the source.

    Verifies the archive round-trips to the exact original bytes before deleting
    anything: the shard is the only copy of that generation's self-play.
    """
    archive = path.with_suffix(path.suffix + ".zst")
    before = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    expected = digest.hexdigest()

    temporary = archive.with_suffix(archive.suffix + ".tmp")
    subprocess.run(
        ["zstd", "-3", "-T2", "-q", "-f", str(path), "-o", str(temporary)],
        check=True,
    )
    verify = subprocess.run(
        ["zstd", "-dc", str(temporary)], stdout=subprocess.PIPE, check=True
    )
    if hashlib.sha256(verify.stdout).hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"compressed shard does not round-trip: {path}")
    os.replace(temporary, archive)
    after = archive.stat().st_size
    path.unlink()
    return path, before, after


def _log_completed_retirement(path: Path, before: int, after: int) -> None:
    print(
        f"[retire] {path.parent.name}: "
        f"{before / 1e9:.2f} GB -> {after / 1e9:.3f} GB "
        f"({before / max(after, 1):.1f}x)",
        flush=True,
    )


def _log_retirement(task: "asyncio.Task[tuple[Path, int, int]]") -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        print(f"[retire] failed: {error}", flush=True)
        return
    _log_completed_retirement(*task.result())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Make a preceding atomic rename durable where directory fsync exists."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def identity_config(
    value: dict[str, Any],
    operational_fields: Collection[str] = OPERATIONAL_CONFIG_FIELDS,
) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in operational_fields
    }


def compatible_config(value: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults for fields added without changing historical semantics."""

    normalized = value.copy()
    # Runs written before dynamic komi existed used a fixed configured range,
    # exactly the false/default behavior. Backfilling only for comparison keeps
    # those runs resumable while still making `dynamic_komi=True` an identity
    # change once replay exists.
    normalized.setdefault("dynamic_komi", False)
    normalized.setdefault("komi_target_black_win_rate", 0.5)
    normalized.setdefault("komi_recenter_minimum_games", 256)
    normalized.setdefault("komi_recenter_maximum_step", 0.025)
    # Drop fields this version no longer has, so a stored config is compared on
    # the fields both sides actually understand. Bounded to a known list rather
    # than "anything unrecognised": silently ignoring every unknown key would
    # let a genuinely foreign config pass the identity check.
    for removed in REMOVED_CONFIG_FIELDS:
        normalized.pop(removed, None)
    return normalized


def cargo_executable() -> str:
    discovered = shutil.which("cargo")
    if discovered:
        return discovered
    fallback = Path.home() / ".cargo" / "bin" / (
        "cargo.exe" if os.name == "nt" else "cargo"
    )
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("cargo executable was not found")


def runtime_environment() -> dict[str, str]:
    """Environment needed by Rust ONNX Runtime and the Python learner."""

    environment = os.environ.copy()
    candidates: list[Path] = []
    if os.name == "nt":
        packages = Path(sys.prefix) / "Lib" / "site-packages"
        candidates.extend([packages / "tensorrt_libs", packages / "torch" / "lib"])
        cuda = environment.get("CUDA_PATH")
        if cuda:
            candidates.append(Path(cuda) / "bin")
        existing = [str(path) for path in candidates if path.exists()]
        if existing:
            environment["PATH"] = os.pathsep.join(
                existing + [environment.get("PATH", "")]
            )
        return environment

    # A single sys.prefix can split its site-packages across "lib" and "lib64"
    # -- observed with the onnxruntime-gpu wheel, whose pure-Python dist-info
    # lands in one and its native code in the other depending on the platform
    # tags it was built with. Both must be searched; neither predicts the
    # other.
    site_packages = [
        Path(sys.prefix) / libdir / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        for libdir in ("lib", "lib64")
    ]
    # Custom source builds (see docs/RUNNING.md) used these directory names;
    # the prebuilt onnxruntime-gpu wheel installs as plain "onnxruntime". Try
    # the custom names first so a from-source build already on the box is
    # preferred over a prebuilt wheel installed alongside it.
    onnxruntime_dir = next(
        (
            packages / name
            for packages in site_packages
            for name in ("onnxruntime_trt", "onnxruntime_blackwell", "onnxruntime")
            if (packages / name).exists()
        ),
        None,
    )
    library_dirs: list[Path] = []
    if onnxruntime_dir is not None:
        library_dirs.append(onnxruntime_dir / "capi")
        library_dirs.append(onnxruntime_dir)
    for packages in site_packages:
        library_dirs += [
            packages / "tensorrt_libs",
            packages / "nvidia" / "cu13" / "lib",
            packages / "nvidia" / "cudnn" / "lib",
            packages / "torch" / "lib",
        ]
    existing = [str(path) for path in library_dirs if path.exists()]
    if existing:
        prior = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            existing + ([prior] if prior else [])
        )
    if "ORT_DYLIB_PATH" not in environment and onnxruntime_dir is not None:
        for candidate_dir in (onnxruntime_dir / "capi", onnxruntime_dir):
            if not candidate_dir.is_dir():
                continue
            # Unversioned name first (custom builds), then the highest
            # versioned .so the wheel actually ships (prebuilt wheels do not
            # provide an unversioned symlink).
            unversioned = candidate_dir / "libonnxruntime.so"
            if unversioned.exists():
                environment["ORT_DYLIB_PATH"] = str(unversioned)
                break
            versioned = sorted(candidate_dir.glob("libonnxruntime.so.*"))
            if versioned:
                environment["ORT_DYLIB_PATH"] = str(versioned[-1])
                break
    return environment


@dataclass(frozen=True)
class ModelArtifact:
    version: int
    checkpoint: str
    onnx: str
    checkpoint_sha256: str
    onnx_sha256: str
    parent_version: int | None

    @classmethod
    def from_state(cls, value: dict[str, Any]) -> "ModelArtifact":
        """Build from a stored dict, dropping fields this version no longer has.

        State written before 2026-08-16 carries `accepted`, from the promotion
        gate. The gate is gone and every model in `models` was accepted by
        definition, so the key is dropped rather than migrated -- but it must
        not reach the constructor, or resuming any older run raises TypeError.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in known})

    @classmethod
    def from_paths(
        cls,
        *,
        version: int,
        checkpoint: Path,
        onnx: Path,
        parent_version: int | None,
    ) -> "ModelArtifact":
        return cls(
            version=version,
            checkpoint=str(checkpoint.resolve()),
            onnx=str(onnx.resolve()),
            checkpoint_sha256=file_sha256(checkpoint),
            onnx_sha256=file_sha256(onnx),
            parent_version=parent_version,
        )


@dataclass(frozen=True)
class ReplayArtifact:
    sequence: int
    path: str
    manifest: str
    samples: int
    behavior_model_sha256: str | None
    dataset_sha256: str
    seed: int


@dataclass(frozen=True)
class KomiBalanceFit:
    """Recent played-game estimate of the komi giving the requested win rate."""

    target_komi: float
    slope: float
    games: int
    black_equivalents: float


@dataclass(frozen=True)
class KomiRangeDecision:
    low: float
    high: float
    previous_center: float
    fit: KomiBalanceFit | None
    games: int


def fit_komi_balance(
    games: Sequence[tuple[float, float]], target_black_win_rate: float
) -> KomiBalanceFit | None:
    """Fit ``P(Black wins) = sigmoid(a + b * komi)``.

    Outcomes are Black-relative probabilities: 1 for a Black win, 0 for a
    White win, and 0.5 for the vanishingly rare tie. Returning ``None`` is the
    conservative result when the sample does not identify a decreasing komi
    response. The coordinator then keeps its current range unchanged.
    """

    if len(games) < 2 or not 0.0 < target_black_win_rate < 1.0:
        return None
    xs = [komi for komi, _ in games]
    ys = [outcome for _, outcome in games]
    if (
        not all(math.isfinite(value) for value in xs + ys)
        or any(not 0.0 <= value <= 1.0 for value in ys)
        or max(xs) - min(xs) <= 1.0e-9
    ):
        return None
    black_equivalents = sum(ys)
    # A one-sided result can only say "move farther", not how far. Waiting for
    # both outcomes is safer than allowing a separated logistic fit to dictate
    # the next range from its regularizer.
    if black_equivalents <= 0.0 or black_equivalents >= len(games):
        return None

    intercept = 0.0
    slope = 0.0
    for _ in range(200):
        gradient_intercept = 0.0
        gradient_slope = 0.0
        h00 = 1.0e-9
        h01 = 0.0
        h11 = 1.0e-9
        for komi, outcome in games:
            logit = intercept + slope * komi
            probability = (
                1.0 / (1.0 + math.exp(-logit))
                if logit >= 0.0
                else math.exp(logit) / (1.0 + math.exp(logit))
            )
            residual = outcome - probability
            weight = probability * (1.0 - probability)
            gradient_intercept += residual
            gradient_slope += residual * komi
            h00 += weight
            h01 += weight * komi
            h11 += weight * komi * komi
        determinant = h00 * h11 - h01 * h01
        if abs(determinant) < 1.0e-12:
            return None
        step_intercept = (
            h11 * gradient_intercept - h01 * gradient_slope
        ) / determinant
        step_slope = (
            -h01 * gradient_intercept + h00 * gradient_slope
        ) / determinant
        intercept += step_intercept
        slope += step_slope
        if not math.isfinite(intercept) or not math.isfinite(slope):
            return None
        if abs(step_intercept) + abs(step_slope) < 1.0e-10:
            break

    # Higher komi favours White. A flat or increasing fitted response is noise
    # or malformed calibration, not a signal the controller should follow.
    if slope >= -1.0e-6:
        return None
    target_logit = math.log(target_black_win_rate / (1.0 - target_black_win_rate))
    target_komi = (target_logit - intercept) / slope
    if not math.isfinite(target_komi):
        return None
    return KomiBalanceFit(
        target_komi=target_komi,
        slope=slope,
        games=len(games),
        black_equivalents=black_equivalents,
    )


@dataclass(frozen=True)
class PipelineConfig:
    output: str
    updates: int = 10
    samples_per_shard: int = 1024
    shards_per_update: int = 1
    replay_window: int = 8
    maximum_prefetch_shards: int = 1
    # Generator processes running at once. Drain-don't-kill finishes a shard's
    # in-flight games, but that tail runs at falling parallelism -- measured
    # 48.5 of 64 actors on average, so 24% of each shard's wall time is spent
    # winding down. A second process starting while the first drains fills that
    # gap: 4.16 -> 5.49 samples/s, 1.32x. One is the old serial behaviour.
    concurrent_generators: int = 1
    resolution: int = 96
    policy_resolution: int = 32
    radius: float = 1.0 / 18.0
    generation_simulations: int = 256
    maximum_plies: int = 256
    coarse_pool: int = 4
    temperature: float = 1.0
    temperature_plies: int = 30
    actors: int = 64
    writer_queue_games: int = 2
    leaf_batch: int = 1
    inference_batch: int = 16
    inference_delay_ms: int = 1
    inference_slots: int = 2
    provider: str = "tensorrt"
    inference_device_id: int = 0
    fp16: bool = True
    warm_inference: bool = True
    architecture: str = "ddrnet"
    variance_scaled: bool = False
    norm_groups: int | None = None
    # Trailing residual blocks per ddrnet context stage replaced with
    # transformer blocks. 0 is byte-identical to a net built without them.
    context_attention_blocks: int = 0
    attention_heads: int = 8
    muon_learning_rate: float = 0.01
    full_adam: bool = False
    model_width: int = 64
    blocks: int = 8
    training_epochs: int = 10
    training_batch: int = 64
    learning_rate: float = 2e-3
    warm_learning_rate: float = 5e-4
    value_weight: float = 1.0
    # Weight on the auxiliary ownership loss. Zero disables it and releases its
    # targets from the replay window; see LearnerConfig.ownership_weight.
    ownership_weight: float = 1.5
    # Finish in-flight games when a shard fills instead of cancelling them.
    # Every actor has a game running at the boundary, so cutting there discards
    # one partial game per actor -- measured 1.39x more useful work per second
    # with this on. Shards then overshoot their target.
    drain_tail: bool = False
    # Per-shard sampling decay; 1.0 is uniform. Lets a long window stay diverse
    # while the gradient follows recent play. See vgo_training/recency.py.
    recency_decay: float = 1.0
    training_threads: int = 4
    training_device: str = "cuda"
    training_precision: str = "bfloat16"
    schedule: str = "wsd"
    compile: bool = True
    restore_optimizer: bool = True
    # Fractional values allowed; see LearnerConfig.warmup_epochs.
    warmup_epochs: float = 1
    report_every: int = 5
    validation_fraction: float = 0.1
    overlap_actor_learner: bool = True
    # Concede once the side to move has been losing for resign_window
    # consecutive plies. Zero disables it. This belongs to run identity: it
    # changes which positions reach the shard.
    resign_threshold: float = 0.0
    # Target false-positive rate for the resignation rule. When positive the
    # threshold is chosen per update from the trailing calibration rather than
    # fixed: the lowest threshold whose measured error stays under this bound.
    #
    # A fixed threshold cannot follow a model that is still learning. On
    # ddrnet-wl a 0.95 threshold fired on 15 of 1625 games early and 440 of
    # 1686 late -- the same setting was useless and then useful. Simulated over
    # that run, a 5% target would have grown throughput from 0.4 to 24.9 plies
    # saved per game while holding measured error at 3.6-4.9%.
    resign_target_false_positive: float = 0.0
    # Simulations for the remainder of a conceded game, 0 to stop at the
    # concession. Non-zero makes a false positive self-correcting, which is what
    # makes an aggressive target safe.
    resign_soft_simulations: int = 0
    resign_window: int = 5
    resign_minimum_ply: int = 0
    # Initial per-game komi range. Positive favours White: scoring is
    # `black - white - komi > 0`. Generation draws from a truncated normal
    # centred on this range; dynamic_komi recentres it without changing width.
    komi_low: float = 0.0
    komi_high: float = 0.0
    # Fit P(Black wins | komi) over the trailing replay window and move the
    # next shard's range towards the requested win rate. The fit reads exact
    # played-game komi/outcomes, not the coarser manifest display buckets.
    dynamic_komi: bool = False
    komi_target_black_win_rate: float = 0.5
    komi_recenter_minimum_games: int = 256
    komi_recenter_maximum_step: float = 0.025
    raster_kind: str = "semantic"
    resign_disable_fraction: float = 0.1
    arena_pairs: int = 16
    # Komi every telemetry game is played at.
    #
    # vgo-arena's own default is 0.0, and the pipeline never overrode it, so
    # every rating this loop produced before 2026-08-16 was measured at a komi
    # where Black wins about 91% -- P(Black) = sigmoid(2.33 - 24.8*komi) on
    # ddrnet-deep-komi data. Colour-swapped pairs then return one win each
    # whatever the players are worth, which compresses every rating toward its
    # neighbours and buries the signal in noise. With the komi range narrowed to
    # sigma 0.03 it is worse than off-balance: 0.0 is outside the sampled range
    # entirely, so the arena rated models on a game they never trained for.
    #
    # 0.104 is the measured balance point and also what the tournaments use, so
    # telemetry Elo and tournament Elo are on one scale. It is deliberately a
    # constant rather than the controller's live centre: a rating scale that
    # moves with the thing it is measuring is not a scale.
    arena_komi: float = 0.104
    arena_simulations: int = 256
    arena_actors: int = 32
    telemetry_opponents: int = 2
    telemetry_pairs: int = 16
    # Rate an every-Nth checkpoint instead of all of them. The loop's Elo curve
    # is a trend, and a rating every update costs an arena per update while
    # telling you little the neighbouring points did not. 1 rates everything.
    telemetry_every: int = 1
    # Compress shards once they leave the replay window. Housekeeping only: it
    # never blocks an update, and a missed shard is retried after the next one.
    retire_shards: bool = True
    seed: int = 700_001
    arena_seed: int = 900_001
    initial_checkpoint: str | None = None
    initial_onnx: str | None = None
    initial_replay: tuple[str, ...] = ()

    @property
    def output_path(self) -> Path:
        return Path(self.output).resolve()

    def validate(self) -> None:
        positive = {
            "updates": self.updates,
            "samples per shard": self.samples_per_shard,
            "shards per update": self.shards_per_update,
            "replay window": self.replay_window,
            "resolution": self.resolution,
            "policy resolution": self.policy_resolution,
            "generation simulations": self.generation_simulations,
            "maximum plies": self.maximum_plies,
            "actors": self.actors,
            "writer queue games": self.writer_queue_games,
            "leaf batch": self.leaf_batch,
            "inference batch": self.inference_batch,
            "inference slots": self.inference_slots,
            "training epochs": self.training_epochs,
            "training batch": self.training_batch,
            "training threads": self.training_threads,
            "model width": self.model_width,
            "blocks": self.blocks,
            "arena pairs": self.arena_pairs,
            "arena simulations": self.arena_simulations,
            "arena actors": self.arena_actors,
            "telemetry pairs": self.telemetry_pairs,
        }
        invalid = [name for name, count in positive.items() if count <= 0]
        if invalid:
            raise ValueError(f"counts must be positive: {', '.join(invalid)}")
        if self.inference_batch < 2:
            raise ValueError("inference batch must be at least two")
        if self.leaf_batch > self.inference_batch:
            raise ValueError(
                "leaf batch must not exceed the inference batch capacity"
            )
        if self.maximum_prefetch_shards < 0:
            raise ValueError("maximum prefetch shards must be nonnegative")
        if self.concurrent_generators < 1:
            raise ValueError("concurrent generators must be positive")
        if self.inference_delay_ms < 0:
            raise ValueError("inference delay must be nonnegative")
        if self.temperature_plies < 0:
            raise ValueError("temperature plies must be nonnegative")
        if self.telemetry_opponents < 0:
            raise ValueError("telemetry opponents must be nonnegative")
        if self.seed < 0 or self.arena_seed < 0:
            raise ValueError("random seeds must be nonnegative")
        if self.report_every <= 0:
            raise ValueError("report interval must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup epochs must be nonnegative")
        if self.replay_window < self.shards_per_update:
            raise ValueError("replay window must hold at least one update quantum")
        if self.coarse_pool < 0 or self.coarse_pool > self.policy_resolution:
            raise ValueError("coarse pool must fit within the policy resolution")
        if self.policy_resolution > self.resolution:
            raise ValueError("policy resolution must not exceed raster resolution")
        if not 0.0 < self.radius < 0.5:
            raise ValueError("radius must be between zero and one half")
        if self.temperature < 0.0:
            raise ValueError("temperature must be nonnegative")
        if not math.isfinite(self.komi_low) or not math.isfinite(self.komi_high):
            raise ValueError("komi bounds must be finite")
        if self.komi_high < self.komi_low:
            raise ValueError("komi high must not be below komi low")
        if self.dynamic_komi and self.komi_high <= self.komi_low:
            raise ValueError("dynamic komi requires a nonzero initial range")
        if not 0.0 < self.komi_target_black_win_rate < 1.0:
            raise ValueError("komi target Black win rate must be in (0, 1)")
        if self.komi_recenter_minimum_games <= 0:
            raise ValueError("komi recenter minimum games must be positive")
        if (
            not math.isfinite(self.komi_recenter_maximum_step)
            or self.komi_recenter_maximum_step <= 0.0
        ):
            raise ValueError("komi recenter maximum step must be finite and positive")
        if self.learning_rate <= 0.0 or self.warm_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.ownership_weight < 0.0:
            raise ValueError("ownership weight must be nonnegative")
        if not 0.0 < self.recency_decay <= 1.0:
            raise ValueError("recency decay must be in (0, 1]")
        if self.value_weight < 0.0:
            raise ValueError("value weight must be nonnegative")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation fraction must be in [0, 1)")
        if self.provider not in {"cpu", "cuda", "tensorrt"}:
            raise ValueError(f"unsupported inference provider: {self.provider}")
        if self.inference_device_id < 0:
            raise ValueError("inference device id must be nonnegative")
        if self.architecture not in MODEL_ARCHITECTURES:
            raise ValueError(f"unsupported model architecture: {self.architecture}")
        if self.schedule not in {"wsd", "cosine"}:
            raise ValueError(f"unsupported learning-rate schedule: {self.schedule}")
        if self.training_precision not in {"float32", "bfloat16"}:
            raise ValueError(
                f"unsupported training precision: {self.training_precision}"
            )
        if (self.initial_checkpoint is None) != (self.initial_onnx is None):
            raise ValueError(
                "initial checkpoint and initial ONNX model must be supplied together"
            )


@dataclass
class PipelineState:
    config_digest: str
    next_shard: int = 0
    updates_completed: int = 0
    consumed_through_shard: int = -1
    replay: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    rejected_models: list[dict[str, Any]] = field(default_factory=list)
    telemetry_pending: list[dict[str, Any]] = field(default_factory=list)
    telemetry_completed: list[str] = field(default_factory=list)
    started_unix_seconds: float = field(default_factory=time.time)
    active_wall_seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"schema": PIPELINE_SCHEMA, **asdict(self)}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "PipelineState":
        if value.get("schema") != PIPELINE_SCHEMA:
            raise ValueError(f"unsupported pipeline state: {value.get('schema')!r}")
        return cls(
            config_digest=str(value["config_digest"]),
            next_shard=int(value.get("next_shard", 0)),
            updates_completed=int(value.get("updates_completed", 0)),
            consumed_through_shard=int(value.get("consumed_through_shard", -1)),
            replay=list(value.get("replay", [])),
            models=list(value.get("models", [])),
            rejected_models=list(value.get("rejected_models", [])),
            telemetry_pending=list(value.get("telemetry_pending", [])),
            telemetry_completed=list(value.get("telemetry_completed", [])),
            started_unix_seconds=float(value.get("started_unix_seconds", time.time())),
            active_wall_seconds=float(value.get("active_wall_seconds", 0.0)),
        )


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    wall_seconds: float
    stdout_path: Path
    stderr_path: Path

    def json_documents(self) -> list[dict[str, Any]]:
        text = self.stdout_path.read_text(encoding="utf-8")
        documents: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        index = 0
        while True:
            start = text.find("{", index)
            if start < 0:
                return documents
            document, end = decoder.raw_decode(text, start)
            if isinstance(document, dict):
                documents.append(document)
            index = end

    def final_json(self) -> dict[str, Any]:
        documents = self.json_documents()
        if not documents:
            raise ValueError(f"command emitted no JSON document: {self.stdout_path}")
        return documents[-1]


class CommandRunner:
    def __init__(self, environment: dict[str, str]) -> None:
        self.environment = environment

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_prefix: Path,
    ) -> CommandResult:
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        command_path = log_prefix.with_suffix(".command.json")
        stdout_path = log_prefix.with_suffix(".stdout.log")
        stderr_path = log_prefix.with_suffix(".stderr.log")
        atomic_json(
            command_path,
            {
                "command": list(command),
                "cwd": str(cwd),
                "started_unix_seconds": time.time(),
            },
        )
        started = time.perf_counter()
        print(f"[{log_prefix.name}] {' '.join(command)}", flush=True)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=self.environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name != "nt",
            )
            try:
                returncode = await process.wait()
            except asyncio.CancelledError:
                self._terminate(process)
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except TimeoutError:
                    self._kill(process)
                    await process.wait()
                raise
        elapsed = time.perf_counter() - started
        result = CommandResult(
            command=tuple(command),
            returncode=returncode,
            wall_seconds=elapsed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        finished = json.loads(command_path.read_text(encoding="utf-8"))
        finished |= {
            "finished_unix_seconds": time.time(),
            "wall_seconds": elapsed,
            "returncode": returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        atomic_json(command_path, finished)
        if returncode != 0:
            tail = stderr_path.read_text(
                encoding="utf-8", errors="backslashreplace"
            )[-4000:]
            raise RuntimeError(
                f"command failed with exit code {returncode}; see {stderr_path}\n{tail}"
            )
        print(f"[{log_prefix.name}] complete in {elapsed:.1f}s", flush=True)
        return result

    @staticmethod
    def _terminate(process: asyncio.subprocess.Process) -> None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class LearnerService:
    """Supervise the long-lived JSON-lines learner process."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        stderr_task: asyncio.Task[None],
    ) -> None:
        self.process = process
        self.stderr_task = stderr_task
        self._lock = asyncio.Lock()
        self._next_request_id = 1

    @classmethod
    async def start(
        cls,
        *,
        cwd: Path,
        environment: dict[str, str],
        stderr_path: Path,
    ) -> "LearnerService":
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "vgo_training.learner",
            "--serve",
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stderr is not None

        async def drain_stderr() -> None:
            with stderr_path.open("ab") as destination:
                while block := await process.stderr.read(64 * 1024):
                    destination.write(block)
                    destination.flush()

        stderr_task = asyncio.create_task(drain_stderr())
        service = cls(process, stderr_task)
        try:
            ready = await asyncio.wait_for(service._read_response(), timeout=60.0)
        except BaseException:
            await service.close(force=True)
            raise
        if (
            ready.get("schema") != LEARNER_PROTOCOL_SCHEMA
            or ready.get("event") != "ready"
        ):
            await service.close(force=True)
            raise RuntimeError(f"learner did not become ready: {ready}")
        return service

    async def _read_response(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = await self.process.stdout.readline()
        if not line:
            returncode = await self.process.wait()
            raise RuntimeError(
                f"learner stopped before responding (exit {returncode})"
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError(f"learner response is not an object: {response!r}")
        if response.get("schema") != LEARNER_PROTOCOL_SCHEMA:
            raise RuntimeError(f"unsupported learner response: {response!r}")
        if response.get("ok") is False:
            kind = response.get("error_type", "LearnerError")
            detail = response.get("error", "learner update failed")
            raise RuntimeError(f"{kind}: {detail}")
        return response

    async def request(self, value: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self.process.returncode is not None:
                raise RuntimeError(f"learner has stopped with {self.process.returncode}")
            request = dict(value)
            request_id = self._next_request_id
            self._next_request_id += 1
            request["request_id"] = request_id
            assert self.process.stdin is not None
            self.process.stdin.write(
                (json.dumps(request, separators=(",", ":")) + "\n").encode()
            )
            await self.process.stdin.drain()
            response = await self._read_response()
            if response.get("request_id") != request_id:
                raise RuntimeError(
                    "learner response id does not match the request: "
                    f"{response.get('request_id')!r} != {request_id!r}"
                )
            return response

    async def close(self, *, force: bool = False) -> None:
        if self.process.returncode is None:
            if force:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
            else:
                try:
                    await self.request({"command": "shutdown"})
                except (BrokenPipeError, ConnectionError, RuntimeError):
                    try:
                        self.process.terminate()
                    except ProcessLookupError:
                        pass
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=15.0)
                except TimeoutError:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                    await self.process.wait()
        await self.stderr_task


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config
        self.output = config.output_path
        self.training = Path(__file__).resolve().parents[1]
        self.root = self.training.parent
        self.environment = runtime_environment()
        self.runner = CommandRunner(self.environment)
        self.state_path = self.output / "pipeline-state.json"
        # Strong references to in-flight shard compressions. asyncio only holds
        # weak references to tasks, so without this a retirement can be garbage
        # collected mid-run; each task discards itself on completion.
        self._retirements: set[asyncio.Task[Any]] = set()
        # Normalize tuples and Path-like values to their on-disk JSON forms so
        # a freshly constructed config compares equal to the same resumed run.
        config_value = json.loads(
            json.dumps({"schema": CONFIG_SCHEMA, **asdict(config)})
        )
        config_value["output"] = str(self.output)
        self._config_value = config_value
        self.config_digest = canonical_digest(identity_config(config_value))
        self.learner: LearnerService | None = None
        self._gpu_lock = asyncio.Lock()
        self._lease: BinaryIO | None = None
        self.output.mkdir(parents=True, exist_ok=True)
        # Creation itself publishes config/state, so it needs the same lease as
        # execution. The lease is reacquired and state is reloaded at run time;
        # a preconstructed object can therefore never overwrite newer progress.
        with self._run_lease():
            self.state = self._load_or_create_state(self._config_value)

    @classmethod
    def resume(cls, output: Path) -> "Pipeline":
        config_path = output.resolve() / "pipeline-config.json"
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if value.pop("schema", None) != CONFIG_SCHEMA:
            raise ValueError(f"unsupported pipeline configuration: {config_path}")
        value["initial_replay"] = tuple(value.get("initial_replay", ()))
        return cls(PipelineConfig(**value))

    def _load_or_create_state(self, config_value: dict[str, Any]) -> PipelineState:
        self.output.mkdir(parents=True, exist_ok=True)
        config_path = self.output / "pipeline-config.json"
        prior: dict[str, Any] | None = None
        pristine_reconfiguration = False
        state_digest_changed = False
        if config_path.exists():
            prior = json.loads(config_path.read_text(encoding="utf-8"))
            if identity_config(compatible_config(prior)) != identity_config(
                config_value
            ):
                # A launcher writes config/state before generation starts. If
                # that first generator is interrupted, no learned artifact is
                # tied to the identity yet; allowing the empty shell to adopt a
                # corrected recipe is both safe and much less error-prone than
                # asking somebody to hand-edit its digest. Once any shard or
                # model exists, the original immutability rule applies.
                try:
                    candidate = PipelineState.from_json(
                        json.loads(self.state_path.read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    candidate = None
                pristine_reconfiguration = bool(
                    candidate is not None
                    and candidate.next_shard == 0
                    and candidate.updates_completed == 0
                    and not candidate.replay
                    and not candidate.models
                    and not candidate.rejected_models
                )
                if not pristine_reconfiguration:
                    raise ValueError(
                        "pipeline's learning configuration differs from the existing run"
                    )
        if self.state_path.exists():
            state = PipelineState.from_json(
                json.loads(self.state_path.read_text(encoding="utf-8"))
            )
            if pristine_reconfiguration:
                print(
                    "[state] adopting changed learning configuration before the "
                    "first shard",
                    flush=True,
                )
                state.config_digest = self.config_digest
                state_digest_changed = True
            elif state.config_digest != self.config_digest:
                if prior is None:
                    raise ValueError(
                        "pipeline state belongs to a different configuration"
                    )
                # The identity comparison above already passed against this
                # run's own pipeline-config.json, so the two configurations
                # agree on every field that is part of identity *today*. A
                # digest that still disagrees may have been written before a
                # field became operational. Accept only digests produced by
                # those known historical field sets; an arbitrary mismatch may
                # be a foreign state file.
                # `prior` is used as stored, still carrying any removed field,
                # because that is what the old digest was taken over. The empty
                # addition set covers a digest written under today's operational
                # fields but while a since-removed field still counted toward
                # identity, which is what removing one produces.
                historical_digests = {
                    canonical_digest(
                        identity_config(
                            prior,
                            OPERATIONAL_CONFIG_FIELDS - additions,
                        )
                    )
                    for additions in
                    (frozenset(),) + tuple(HISTORICAL_OPERATIONAL_FIELD_ADDITIONS)
                }
                if state.config_digest not in historical_digests:
                    raise ValueError(
                        "pipeline state belongs to a different configuration"
                    )
                print(
                    "[state] config digest was taken under a different set of "
                    f"operational fields ({state.config_digest[:12]} -> "
                    f"{self.config_digest[:12]}); identity matches, refreshing",
                    flush=True,
                )
                state.config_digest = self.config_digest
                state_digest_changed = True
        else:
            state = PipelineState(config_digest=self.config_digest)
            for index, replay in enumerate(self.config.initial_replay):
                path = Path(replay).resolve(strict=True)
                if path.is_dir():
                    path = (path / "dataset.vgo").resolve(strict=True)
                state.replay.append(
                    asdict(
                        self._replay_artifact(
                            path.parent,
                            -len(self.config.initial_replay) + index,
                            verify_dataset_digest=True,
                            expected_samples=None,
                        )
                    )
                )
            if self.config.initial_checkpoint is not None:
                checkpoint = Path(self.config.initial_checkpoint).resolve(strict=True)
                onnx = Path(str(self.config.initial_onnx)).resolve(strict=True)
                state.models.append(
                    asdict(
                        ModelArtifact.from_paths(
                            version=-1,
                            checkpoint=checkpoint,
                            onnx=onnx,
                            parent_version=None,
                        )
                    )
                )
            atomic_json(self.state_path, state.to_json())

        if self.config.updates < state.updates_completed:
            raise ValueError(
                "updates cannot be reduced below already completed work"
            )
        if state_digest_changed:
            atomic_json(self.state_path, state.to_json())
        if prior != config_value:
            history_path = self.output / "pipeline-config-history.json"
            history = (
                json.loads(history_path.read_text(encoding="utf-8"))
                if history_path.exists()
                else []
            )
            history.append(
                {
                    "effective_unix_seconds": time.time(),
                    "updates_completed": state.updates_completed,
                    "next_shard": state.next_shard,
                    "config": config_value,
                }
            )
            atomic_json(history_path, history)
            atomic_json(config_path, config_value)
        return state

    def _save_state(self) -> None:
        atomic_json(self.state_path, self.state.to_json())

    @contextmanager
    def _run_lease(self) -> Iterator[None]:
        """Exclude a second coordinator from mutating the same run."""

        if self._lease is not None:
            yield
            return
        path = self.output / ".pipeline.lock"
        lease = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if lease.seek(0, os.SEEK_END) == 0:
                    lease.write(b"\0")
                    lease.flush()
                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            lease.close()
            raise RuntimeError(
                f"another pipeline owns the run at {self.output}"
            ) from error
        self._lease = lease
        try:
            yield
        finally:
            self._lease = None
            if os.name == "nt":
                import msvcrt

                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    @property
    def incumbent(self) -> ModelArtifact | None:
        if not self.state.models:
            return None
        return ModelArtifact.from_state(self.state.models[-1])

    def _rust_command(self, binary: str) -> list[str]:
        return [
            cargo_executable(),
            "run",
            "--release",
            "-p",
            "vgo-selfplay",
            "--bin",
            binary,
            "--",
        ]

    @staticmethod
    def _manifest_komi_range(manifest: dict[str, Any]) -> tuple[float, float] | None:
        """Effective range recorded by a shard, including pre-controller shards."""

        try:
            low = float(manifest["komi_low"])
            high = float(manifest["komi_high"])
        except (KeyError, TypeError, ValueError):
            # Older manifests record the same endpoints through their display
            # buckets. This lets a dynamic continuation start from the latest
            # range it actually played rather than snapping to its nominal one.
            try:
                buckets = manifest["komi_calibration"]
                low = min(float(bucket["low"]) for bucket in buckets)
                high = max(float(bucket["high"]) for bucket in buckets)
            except (KeyError, TypeError, ValueError):
                return None
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            return None
        return low, high

    def _recent_komi_games(self) -> list[tuple[float, float]]:
        """Exact recent ``(komi, Black outcome)`` rows eligible for fitting."""

        games: list[tuple[float, float]] = []
        for replay in self.state.replay[-self.config.replay_window :]:
            try:
                manifest_path = Path(str(replay["manifest"]))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                games_path = manifest_path.parent / str(manifest["games"])
                lines = games_path.read_text(encoding="utf-8").splitlines()
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    # A hard resignation assigns the winner using the rule
                    # whose calibration is under test. Soft resignation records
                    # `resigned: false` because it plays to an independent end.
                    if bool(row.get("resigned", False)):
                        continue
                    komi = float(row["komi"])
                    utility = float(row["black_utility"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if math.isfinite(komi) and math.isfinite(utility) and -1.0 <= utility <= 1.0:
                    games.append((komi, (utility + 1.0) / 2.0))
        return games

    def _effective_komi_range(self) -> KomiRangeDecision:
        """Range for the next shard, derived only from durable prior replay."""

        config = self.config
        width = config.komi_high - config.komi_low
        configured_center = 0.5 * (config.komi_low + config.komi_high)
        if not config.dynamic_komi:
            return KomiRangeDecision(
                low=config.komi_low,
                high=config.komi_high,
                previous_center=configured_center,
                fit=None,
                games=0,
            )

        current_center = configured_center
        for replay in reversed(self.state.replay):
            try:
                manifest = json.loads(
                    Path(str(replay["manifest"])).read_text(encoding="utf-8")
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if effective := self._manifest_komi_range(manifest):
                current_center = 0.5 * (effective[0] + effective[1])
                break

        recent = self._recent_komi_games()
        fit = (
            fit_komi_balance(recent, config.komi_target_black_win_rate)
            if len(recent) >= config.komi_recenter_minimum_games
            else None
        )
        next_center = current_center
        if fit is not None:
            delta = fit.target_komi - current_center
            maximum = config.komi_recenter_maximum_step
            next_center += max(-maximum, min(maximum, delta))
        return KomiRangeDecision(
            low=next_center - width / 2.0,
            high=next_center + width / 2.0,
            previous_center=current_center,
            fit=fit,
            games=len(recent),
        )

    def _adaptive_resign_threshold(self) -> float:
        """Lowest threshold whose measured error stays under the target.

        Lower fires more often and saves more search, so the cheapest setting
        that clears the error bound is the right one. Returns the configured
        threshold when adaptation is off, when no shard has calibration yet, or
        when nothing clears the bound -- in the last case the configured value
        is the conservative fallback rather than the most aggressive candidate.

        Only the trailing window counts. Calibration describes the model that
        generated those games, and a model still learning invalidates its own
        history: on ddrnet-wl a 0.95 threshold fired on 15 of 1625 games over
        the first fifteen shards and 440 of 1686 over the next sixteen.
        """
        target = float(self.config.resign_target_false_positive)
        if target <= 0.0:
            return float(self.config.resign_threshold)

        totals: dict[float, list[int]] = {}
        for replay in self.state.replay[-self.config.replay_window :]:
            # Seeded shards carry negative sequences, and they calibrate as well
            # as any other: they are real games from this lineage, which is the
            # whole reason for seeding with them. Filtering on the sign left the
            # first shards of a seeded run with no calibration at all, so
            # resignation stayed off exactly when the seed could have paid for
            # it. A shard without a readable manifest is skipped below instead.
            try:
                manifest = json.loads(
                    Path(str(replay["manifest"])).read_text(encoding="utf-8")
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            for entry in manifest.get("resign_calibration", ()):
                # Only the live window: a threshold measured at another window
                # says nothing about how this one behaves.
                if int(entry.get("window", 0)) != int(self.config.resign_window):
                    continue
                bucket = totals.setdefault(float(entry["threshold"]), [0, 0])
                bucket[0] += int(entry["fired"])
                bucket[1] += int(entry["wrong"])

        # A handful of firings can read 0% by luck, so require enough to mean
        # something before trusting a threshold.
        minimum_fires = 30
        for threshold in sorted(totals):
            fired, wrong = totals[threshold]
            if fired >= minimum_fires and wrong / fired <= target:
                return threshold
        # Nothing clears the target, so resignation is switched off for this
        # shard rather than falling back to the configured threshold.
        #
        # The old fallback was backwards: it returned `resign_threshold`, a
        # mid-range constant, which is *more* permissive than the strictest
        # candidate that just failed. On ddrnet-komi3 that silently ran at 4-7%
        # against a 2% target for ten shards, and printed nothing, because the
        # log line only fires when the choice differs from the configured value.
        #
        # A threshold of 1.0 can never be reached -- the head's output is
        # bounded -- so this concedes nothing while the calibration recovers.
        return 1.0

    def generation_command(
        self,
        *,
        output: Path,
        sequence: int,
        model: ModelArtifact | None,
    ) -> list[str]:
        config = self.config
        threshold = self._adaptive_resign_threshold()
        komi = self._effective_komi_range()
        if config.resign_target_false_positive > 0.0:
            # Always log under adaptation: "chose 0.95 because it qualified" and
            # "chose 0.95 because nothing did" were previously indistinguishable,
            # and the second printed nothing at all.
            target = 100 * config.resign_target_false_positive
            if threshold >= 1.0:
                print(
                    f"[resign] shard {sequence}: no threshold met {target:.0f}% "
                    "false positives; resignation disabled for this shard",
                    flush=True,
                )
            else:
                print(
                    f"[resign] threshold {threshold} chosen for shard {sequence} "
                    f"(target FP {target:.0f}%)",
                    flush=True,
                )
        if config.dynamic_komi:
            center = 0.5 * (komi.low + komi.high)
            target = 100.0 * config.komi_target_black_win_rate
            if komi.fit is None:
                print(
                    f"[komi] shard {sequence}: keeping center "
                    f"{komi.previous_center:+.4f}; {komi.games} eligible games "
                    f"(need {config.komi_recenter_minimum_games} and a decreasing fit)",
                    flush=True,
                )
            else:
                print(
                    f"[komi] shard {sequence}: {komi.fit.games} games fit "
                    f"Black {target:.0f}% at {komi.fit.target_komi:+.4f}; "
                    f"center {komi.previous_center:+.4f} -> {center:+.4f}",
                    flush=True,
                )
        command = self._rust_command("vgo-generate-demo") + [
            "--samples",
            str(config.samples_per_shard),
            "--resolution",
            str(config.resolution),
            "--policy-resolution",
            str(config.policy_resolution),
            "--simulations",
            str(config.generation_simulations),
            "--coarse-pool",
            str(config.coarse_pool),
            "--temperature",
            str(config.temperature),
            "--temperature-plies",
            str(config.temperature_plies),
            "--max-plies",
            str(config.maximum_plies),
            "--resign-threshold",
            str(threshold),
            "--resign-window",
            str(config.resign_window),
            "--resign-minimum-ply",
            str(config.resign_minimum_ply),
            "--resign-soft-simulations",
            str(config.resign_soft_simulations),
            # `=` form: a negative komi otherwise parses as a flag, since clap
            # cannot tell `-0.1` from a short option.
            f"--komi-low={komi.low}",
            f"--komi-high={komi.high}",
            "--raster-kind",
            str(config.raster_kind),
            "--model-raster-kind",
            str(config.raster_kind),
            "--resign-disable-fraction",
            str(config.resign_disable_fraction),
            "--radius",
            str(config.radius),
            "--seed",
            str(config.seed + sequence * 1_000_003),
            "--examples",
            "0",
            "--output",
            str(output),
            "--maximum-batch",
            str(config.inference_batch),
            "--delay-ms",
            str(config.inference_delay_ms),
            "--inference-slots",
            str(config.inference_slots),
            "--drain-tail",
            str(config.drain_tail).lower(),
            "--actors",
            str(config.actors),
            "--writer-queue-games",
            str(config.writer_queue_games),
            "--leaf-batch",
            str(config.leaf_batch),
            "--provider",
            config.provider,
            "--device-id",
            str(config.inference_device_id),
            "--fp16",
            str(config.fp16).lower(),
            "--cache-directory",
            str((self.root / "artifacts" / "onnx-cache").resolve()),
        ]
        if model is None:
            command.extend(["--runtime", "naive"])
        else:
            command.extend(["--runtime", "onnx", "--model", model.onnx])
        return command

    def arena_command(
        self,
        *,
        candidate: Path,
        opponents: Sequence[Path],
        seed: int,
        pairs: int,
    ) -> list[str]:
        config = self.config
        command = self._rust_command("vgo-arena") + [
            "--candidate",
            str(candidate),
            # Both seats are this run's own models, so both read its layout.
            "--candidate-raster-kind",
            str(self.config.raster_kind),
            "--pairs",
            str(pairs),
            "--simulations",
            str(config.arena_simulations),
            "--coarse-pool",
            str(config.coarse_pool),
            "--max-plies",
            str(config.maximum_plies),
            "--threads",
            str(config.arena_actors),
            "--leaf-batch",
            str(config.leaf_batch),
            "--resolution",
            str(config.resolution),
            "--policy-resolution",
            str(config.policy_resolution),
            "--radius",
            str(config.radius),
            # vgo-arena defaults this to 0.0, which is not a balanced game and,
            # since the range narrowed, not even inside the training
            # distribution. See the note on PipelineConfig.arena_komi.
            "--komi",
            str(config.arena_komi),
            "--seed",
            str(seed),
            "--maximum-batch",
            str(config.inference_batch),
            "--delay-ms",
            str(config.inference_delay_ms),
            "--provider",
            config.provider,
            "--device-id",
            str(config.inference_device_id),
            "--fp16",
            str(config.fp16).lower(),
            "--cache-directory",
            str((self.root / "artifacts" / "onnx-cache").resolve()),
        ]
        for opponent in opponents:
            command.extend(["--opponent", str(opponent)])
        return command

    @asynccontextmanager
    async def _gpu_lease(self) -> AsyncIterator[None]:
        if self.config.overlap_actor_learner:
            yield
            return
        async with self._gpu_lock:
            yield

    async def _generate_shard(
        self, model: ModelArtifact | None, sequence: int | None = None
    ) -> ReplayArtifact:
        # The caller passes a sequence when several generators run at once;
        # reading `state.next_shard` here would hand both the same number.
        if sequence is None:
            sequence = self.state.next_shard
        replay_root = self.output / "replay"
        final = replay_root / f"shard-{sequence:06d}"
        staging = replay_root / f"shard-{sequence:06d}.staging"
        if final.exists():
            return self._replay_artifact(
                final, sequence, verify_dataset_digest=True
            )
        if staging.exists():
            shutil.rmtree(staging)
        staging.parent.mkdir(parents=True, exist_ok=True)
        command = self.generation_command(
            output=staging, sequence=sequence, model=model
        )
        async with self._gpu_lease():
            result = await self.runner.run(
                command,
                cwd=self.root,
                log_prefix=self.output / "logs" / f"generate-{sequence:06d}",
            )
        report = result.final_json()
        dataset = staging / "dataset.vgo"
        manifest_path = staging / "manifest.json"
        if not dataset.exists() or not manifest_path.exists():
            raise RuntimeError("generator returned success without publishing replay")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("dataset_sha256") != manifest.get("dataset_sha256"):
            raise RuntimeError("generator report and manifest digests disagree")
        behavior_digest = manifest.get(
            "behavior_model_sha256", manifest.get("model_sha256")
        )
        expected_digest = None if model is None else model.onnx_sha256
        if behavior_digest != expected_digest:
            raise RuntimeError(
                "replay behavior-model digest does not match the pinned actor model"
            )
        staging.replace(final)
        sync_directory(final.parent)
        return self._replay_artifact(
            final, sequence, verify_dataset_digest=False
        )

    def _replay_artifact(
        self,
        directory: Path,
        sequence: int,
        *,
        verify_dataset_digest: bool,
        expected_samples: int | None = None,
    ) -> ReplayArtifact:
        dataset = directory / "dataset.vgo"
        manifest_path = directory / "manifest.json"
        if not dataset.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"incomplete replay shard: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "vgo.replay-shard.v1":
            raise RuntimeError(f"unsupported replay manifest: {manifest_path}")
        samples = int(manifest.get("samples", -1))
        expected_samples = (
            self.config.samples_per_shard
            if expected_samples is None and sequence >= 0
            else expected_samples
        )
        # A drained shard overshoots: generation stops starting games at the
        # target but finishes the ones in flight, so the count is a floor rather
        # than an equality. Under-count still fails -- that is a truncated shard.
        if expected_samples is not None:
            short = samples < expected_samples
            over = samples != expected_samples and not self.config.drain_tail
            if short or over:
                raise RuntimeError(
                    f"replay shard has {samples} samples, expected "
                    f"{'at least ' if self.config.drain_tail else ''}"
                    f"{expected_samples}"
                )
        digest = manifest.get("dataset_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("replay manifest has no valid dataset digest")
        if manifest.get("shard_id") not in {None, f"sha256:{digest}"}:
            raise RuntimeError("replay shard identity does not match its digest")
        expected_bytes = manifest.get("dataset_bytes")
        if expected_bytes is not None and dataset.stat().st_size != int(expected_bytes):
            raise RuntimeError("replay dataset size does not match its manifest")
        if verify_dataset_digest and file_sha256(dataset) != digest:
            raise RuntimeError("replay dataset checksum does not match its manifest")
        behavior_digest = manifest.get(
            "behavior_model_sha256", manifest.get("model_sha256")
        )
        if behavior_digest is not None and (
            not isinstance(behavior_digest, str) or len(behavior_digest) != 64
        ):
            raise RuntimeError("replay manifest has an invalid behavior model digest")
        return ReplayArtifact(
            sequence=sequence,
            path=str(dataset.resolve()),
            manifest=str(manifest_path.resolve()),
            samples=samples,
            behavior_model_sha256=behavior_digest,
            dataset_sha256=digest,
            seed=int(manifest.get("seed", self.config.seed)),
        )

    def _commit_replay(self, replay: ReplayArtifact) -> None:
        existing = next(
            (
                value
                for value in self.state.replay
                if int(value["sequence"]) == replay.sequence
            ),
            None,
        )
        if existing is not None:
            if existing["dataset_sha256"] != replay.dataset_sha256:
                raise RuntimeError(
                    f"replay sequence {replay.sequence} changed identity"
                )
            self.state.next_shard = max(
                self.state.next_shard, replay.sequence + 1
            )
            self._save_state()
            return
        self.state.replay.append(asdict(replay))
        self.state.replay.sort(key=lambda value: int(value["sequence"]))
        self.state.next_shard = max(self.state.next_shard, replay.sequence + 1)
        self._save_state()

    def _pending_replay(self) -> list[ReplayArtifact]:
        return [
            ReplayArtifact(**value)
            for value in self.state.replay
            if int(value["sequence"]) > self.state.consumed_through_shard
        ]

    def _active_replay(self, through_sequence: int) -> list[ReplayArtifact]:
        eligible = [
            ReplayArtifact(**value)
            for value in self.state.replay
            if int(value["sequence"]) <= through_sequence
        ]
        return eligible[-self.config.replay_window :]

    def _update_spec(self, update: int) -> tuple[dict[str, Any], Path]:
        update_path = self.output / "updates" / f"update-{update:06d}"
        spec_path = update_path / "update-spec.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            self._validate_update_spec(spec, update, spec_path)
            return spec, update_path
        pending = self._pending_replay()
        if len(pending) < self.config.shards_per_update:
            raise RuntimeError("learner update requested without enough new replay")
        through = pending[self.config.shards_per_update - 1].sequence
        active = self._active_replay(through)
        parent = self.incumbent
        spec = {
            "schema": UPDATE_SCHEMA,
            "update": update,
            "through_shard": through,
            "active_replay": [asdict(replay) for replay in active],
            "parent_model": asdict(parent) if parent is not None else None,
            "created_unix_seconds": time.time(),
        }
        atomic_json(spec_path, spec)
        return spec, update_path

    def _validate_update_spec(
        self, spec: dict[str, Any], update: int, path: Path
    ) -> None:
        """Reconcile durable update intent with the authoritative run state."""

        if spec.get("schema") != UPDATE_SCHEMA:
            raise RuntimeError(f"unsupported update spec: {path}")
        if int(spec.get("update", -1)) != update:
            raise RuntimeError(f"update spec has the wrong sequence: {path}")
        pending = self._pending_replay()
        if len(pending) < self.config.shards_per_update:
            raise RuntimeError(
                "durable update spec has no matching unconsumed replay boundary"
            )
        expected_through = pending[self.config.shards_per_update - 1].sequence
        if int(spec.get("through_shard", -1)) != expected_through:
            raise RuntimeError(
                "durable update spec consumes a different replay boundary"
            )
        expected_replay = [
            asdict(replay) for replay in self._active_replay(expected_through)
        ]
        if spec.get("active_replay") != expected_replay:
            raise RuntimeError(
                "durable update spec names a different replay snapshot"
            )
        parent = self.incumbent
        expected_parent = asdict(parent) if parent is not None else None
        if spec.get("parent_model") != expected_parent:
            raise RuntimeError(
                "durable update spec names a different parent model"
            )

    async def _train(self, spec: dict[str, Any], update_path: Path) -> dict[str, Any]:
        if self.learner is None:
            raise RuntimeError("learner service is not running")
        checkpoint = update_path / "candidate.pt"
        report_path = checkpoint.with_suffix(".pt.json")
        if checkpoint.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self._validate_training_artifact(
                spec, checkpoint, report, verify_checkpoint_digest=True
            )
            atomic_json(report_path, report)
            return report
        parent = spec.get("parent_model")
        request: dict[str, Any] = {
            "command": "update",
            "datasets": [
                replay["path"] for replay in spec["active_replay"]
            ],
            "output": str(checkpoint),
            "epochs": self.config.training_epochs,
            "batch_size": self.config.training_batch,
            "learning_rate": (
                self.config.learning_rate
                if parent is None
                else self.config.warm_learning_rate
            ),
            "value_weight": self.config.value_weight,
            "ownership_weight": self.config.ownership_weight,
            "recency_decay": self.config.recency_decay,
            "model_width": self.config.model_width,
            "blocks": self.config.blocks,
            "architecture": self.config.architecture,
            "variance_scaled": self.config.variance_scaled,
            "norm_groups": self.config.norm_groups,
            "context_attention_blocks": self.config.context_attention_blocks,
            "attention_heads": self.config.attention_heads,
            "muon_learning_rate": self.config.muon_learning_rate,
            "full_adam": self.config.full_adam,
            "threads": self.config.training_threads,
            "device": self.config.training_device,
            "precision": self.config.training_precision,
            "schedule": self.config.schedule,
            "compile": self.config.compile,
            "restore_optimizer": self.config.restore_optimizer,
            "warmup_epochs": self.config.warmup_epochs,
            "seed": self.config.seed + int(spec["update"]),
            "report_every": self.config.report_every,
            "validation_fraction": self.config.validation_fraction,
            "initial_checkpoint": parent["checkpoint"] if parent else None,
        }
        response = await self.learner.request(request)
        report = response.get("result", response.get("report", response))
        if not isinstance(report, dict):
            raise RuntimeError(f"learner returned an invalid report: {response}")
        if not checkpoint.exists():
            raise RuntimeError("learner returned success without a checkpoint")
        self._validate_training_artifact(
            spec, checkpoint, report, verify_checkpoint_digest=False
        )
        # The response is authoritative even if a stale sidecar survived an
        # interrupted prior attempt. Re-publish it durably after validating the
        # new checkpoint rather than leaving a mismatched recovery fast path.
        atomic_json(report_path, report)
        return report

    def _validate_training_artifact(
        self,
        spec: dict[str, Any],
        checkpoint: Path,
        report: dict[str, Any],
        *,
        verify_checkpoint_digest: bool,
    ) -> None:
        if report.get("schema") != "vgo.training-run.v3":
            raise RuntimeError("unsupported learner training report")
        parent = spec.get("parent_model")
        expected_parent_digest = (
            None if parent is None else parent["checkpoint_sha256"]
        )
        if report.get("parent_checkpoint_sha256") != expected_parent_digest:
            raise RuntimeError(
                "learner did not train from the update's authoritative parent"
            )
        expected_datasets = [
            str(Path(replay["path"]).resolve())
            for replay in spec["active_replay"]
        ]
        if report.get("datasets") != expected_datasets:
            raise RuntimeError("training report names a different replay snapshot")
        expected_digests = [
            replay["dataset_sha256"] for replay in spec["active_replay"]
        ]
        if report.get("dataset_digests") != expected_digests:
            raise RuntimeError("training report replay digests do not match the update")
        digest = report.get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("training report has no valid checkpoint digest")
        if str(checkpoint.resolve()) != report.get("checkpoint"):
            raise RuntimeError("training report checkpoint path does not match the update")
        if verify_checkpoint_digest and file_sha256(checkpoint) != digest:
            raise RuntimeError("checkpoint checksum does not match its training report")

    async def _export(self, update_path: Path) -> dict[str, Any]:
        checkpoint = update_path / "candidate.pt"
        onnx = update_path / "candidate.onnx"
        report_path = onnx.with_suffix(".onnx.json")
        if onnx.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self._validate_export_artifact(checkpoint, onnx, report)
            atomic_json(report_path, report)
            return report
        command = [
            sys.executable,
            "-m",
            "vgo_training.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(onnx),
            "--maximum-batch",
            str(max(ONNX_EXPORT_BATCH_HEADROOM, self.config.inference_batch)),
        ]
        result = await self.runner.run(
            command,
            cwd=self.training,
            log_prefix=self.output
            / "logs"
            / f"export-{update_path.name.removeprefix('update-')}",
        )
        report = result.final_json()
        if not onnx.exists():
            raise RuntimeError("export returned success without an ONNX model")
        self._validate_export_artifact(checkpoint, onnx, report)
        atomic_json(report_path, report)
        return report

    async def _warm_inference(
        self, update: int, update_path: Path
    ) -> dict[str, Any] | None:
        """Build the candidate's TensorRT engine during the current actor tail."""

        if not self.config.warm_inference or self.config.provider != "tensorrt":
            return None
        command = [
            cargo_executable(),
            "run",
            "--release",
            "-p",
            "vgo-inference",
            "--bin",
            "vgo-onnx-bench",
            "--",
            "--model",
            str(update_path / "candidate.onnx"),
            "--provider",
            self.config.provider,
            "--device-id",
            str(self.config.inference_device_id),
            "--cache-directory",
            str((self.root / "artifacts" / "onnx-cache").resolve()),
            "--resolution",
            str(self.config.resolution),
            "--policy-resolution",
            str(self.config.policy_resolution),
            "--raster-kind",
            str(self.config.raster_kind),
            "--batch",
            str(self.config.inference_batch),
            "--fp16",
            str(self.config.fp16).lower(),
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--compare-python",
            "false",
        ]
        result = await self.runner.run(
            command,
            cwd=self.root,
            log_prefix=self.output / "logs" / f"warmup-{update:06d}",
        )
        report = result.final_json()
        expected = {
            "provider": self.config.provider,
            "resolution": self.config.resolution,
            "policy_resolution": self.config.policy_resolution,
            "batch": self.config.inference_batch,
            "fp16": self.config.fp16,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "inference warmup did not use the exported model's runtime contract"
            )
        return report

    def _validate_export_artifact(
        self,
        checkpoint: Path,
        onnx: Path,
        report: dict[str, Any],
    ) -> None:
        if report.get("schema") != "vgo.onnx-manifest.v1":
            raise RuntimeError("unsupported ONNX export report")
        if report.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise RuntimeError("ONNX export names a different checkpoint")
        if report.get("checkpoint") != str(checkpoint.resolve()):
            raise RuntimeError("ONNX export checkpoint path does not match the update")
        if report.get("onnx") != str(onnx.resolve()):
            raise RuntimeError("ONNX export path does not match the update")
        digest = report.get("onnx_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("ONNX export report has no valid model digest")
        maximum_batch = report.get("input", {}).get("maximum_batch")
        try:
            maximum_batch = int(maximum_batch)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "ONNX export report has no valid maximum batch"
            ) from error
        if maximum_batch < self.config.inference_batch:
            raise RuntimeError(
                "ONNX export maximum batch is below the configured inference batch"
            )
        if file_sha256(onnx) != digest:
            raise RuntimeError("ONNX checksum does not match its export report")

    def _queue_telemetry(self, model: ModelArtifact) -> None:
        if self.config.telemetry_opponents <= 0:
            return
        # Rating every checkpoint costs an arena per update to resolve a curve
        # that is legible from a fraction of the points. Version 0 is always
        # rated so the trend has an anchor at the origin.
        every = max(1, self.config.telemetry_every)
        if model.version % every != 0:
            return
        opponents = [
            ModelArtifact.from_state(value)
            for value in self.state.models
            if int(value["version"]) < model.version
        ]
        if not opponents:
            return
        # A counter-derived permutation gives a deterministic sample without
        # depending on process-global random state.
        opponents.sort(
            key=lambda value: hashlib.sha256(
                f"{self.config.arena_seed}:{model.version}:{value.version}".encode()
            ).digest()
        )
        selected = opponents[: self.config.telemetry_opponents]
        group_seed = (
            self.config.arena_seed + model.version * 1_000_003
        )
        for order, opponent in enumerate(selected):
            job_id = f"v{model.version:06d}-vs-v{opponent.version:06d}"
            if job_id in self.state.telemetry_completed or any(
                pending["id"] == job_id for pending in self.state.telemetry_pending
            ):
                continue
            self.state.telemetry_pending.append(
                {
                    "schema": TELEMETRY_SCHEMA,
                    "id": job_id,
                    "candidate_version": model.version,
                    "candidate": model.onnx,
                    "opponent_version": opponent.version,
                    "opponent": opponent.onnx,
                    "pairs": self.config.telemetry_pairs,
                    "group_seed": group_seed,
                    "group_index": order,
                    "seed": group_seed + order * 1_000_003,
                }
            )
        self._save_state()

    def _commit_publication(
        self,
        report: dict[str, Any],
        *,
        verify_model_files: bool = False,
    ) -> ModelArtifact | None:
        if report.get("schema") != "vgo.pipeline-publication.v1":
            raise RuntimeError("unsupported model publication")
        update = int(report["update"])
        model = ModelArtifact.from_state(report["model"])
        if model.version != update:
            raise RuntimeError("publication model version does not match its update")
        if verify_model_files:
            if file_sha256(Path(model.checkpoint)) != model.checkpoint_sha256:
                raise RuntimeError("published checkpoint checksum mismatch")
            if file_sha256(Path(model.onnx)) != model.onnx_sha256:
                raise RuntimeError("published ONNX checksum mismatch")
        # `rejected_models` is only read here, never appended to: the promotion
        # gate is gone and every candidate is published. It stays in the state
        # so a run that predates the removal keeps the record of what its gate
        # threw away, and so replaying one of those publications on a resume is
        # recognised as already committed rather than committed a second time.
        existing = next(
            (
                value
                for value in self.state.models + self.state.rejected_models
                if int(value["version"]) == model.version
            ),
            None,
        )
        if existing is not None:
            stored = {k: v for k, v in existing.items() if k != "accepted"}
            if stored != asdict(model):
                raise RuntimeError(
                    f"model version {model.version} changed publication identity"
                )
        else:
            self.state.models.append(asdict(model))
            self._queue_telemetry(model)
        self.state.updates_completed = max(
            self.state.updates_completed, update + 1
        )
        self.state.consumed_through_shard = max(
            self.state.consumed_through_shard,
            int(report["through_shard"]),
        )
        self._save_state()
        self._report_resign_calibration()
        return model

    def _report_resign_calibration(self) -> None:
        """Pool the resignation counterfactual over the active replay window.

        Each shard measures the rule on the games whose true result is known
        independently of it. Under hard resignation that is only the ~10%
        exempted by --resign-disable-fraction, since a conceded game's outcome
        was assigned by the thing under test. Under soft resignation it is every
        completed game: the concession lowers the search budget but the game
        still plays to a real terminal state, so the outcome is the rule's to
        predict, not to assert. The counter below reports whichever set applies
        rather than assuming the holdout.

        Pooling matters in both regimes. Under hard resignation a shard holds
        ~25 measurable games of which ~13 fire, so a single shard's
        false-positive rate is nearly meaningless: measured across 30 shards it
        ranged 0% to 33% with a median of 8%, while the pooled rate was 9.7%.

        Pooling over the window is what makes it readable, and printing it each
        update is what makes it noticed -- the per-shard blocks have been
        written since the run began and nothing ever read them.
        """

        totals: dict[tuple[float, int], list[int]] = {}
        for replay in self.state.replay[-self.config.replay_window :]:
            # Seeded shards count here for the same reason they count in
            # _adaptive_resign_threshold: they are real games from this
            # lineage. Excluding them pooled a seeded run's table over its own
            # shards alone -- 41 exempt games on the first update, which is the
            # single-shard noise this function exists to average away.
            try:
                manifest = json.loads(
                    Path(str(replay["manifest"])).read_text(encoding="utf-8")
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            for entry in manifest.get("resign_calibration", ()):
                try:
                    bucket = totals.setdefault(
                        (float(entry["threshold"]), int(entry.get("window", 0))),
                        [0, 0, 0],
                    )
                    bucket[0] += int(entry["games"])
                    bucket[1] += int(entry["fired"])
                    bucket[2] += int(entry["wrong"])
                except (KeyError, TypeError, ValueError):
                    continue
        if not totals:
            return

        live_threshold = float(self.config.resign_threshold)
        live_window = int(self.config.resign_window)
        # Not "exempt games": that word described the --resign-disable-fraction
        # holdout, and under soft resignation every completed game calibrates.
        # Printing 582 "exempt" games from a run with disable_fraction 0.0 read
        # as soft resign having failed to engage, when it was measuring the full
        # shard exactly as intended.
        calibrated = max(entry[0] for entry in totals.values())
        shards = len(self.state.replay[-self.config.replay_window :])
        print(
            f"[resign] {shards} shards, {calibrated} calibrated games; "
            f"false positives as threshold x window (* is live)",
            flush=True,
        )
        windows = sorted({window for _, window in totals})
        header = "  ".join(f"w{window:<10}" for window in windows)
        print(f"[resign]   thresh  {header}", flush=True)
        for threshold in sorted({value for value, _ in totals}):
            cells = []
            for window in windows:
                games, fired, wrong = totals.get((threshold, window), [0, 0, 0])
                if not fired:
                    cells.append(f"{'-':<12}")
                    continue
                live = (
                    abs(threshold - live_threshold) < 1e-9 and window == live_window
                )
                cell = f"{100 * wrong / fired:.0f}%({wrong}/{fired})"
                cells.append(f"{'*' if live else ' '}{cell:<11}")
            print(f"[resign]   {threshold:<6g}  {''.join(cells)}", flush=True)

    async def _learn_and_publish(
        self,
        update: int,
        spec: dict[str, Any] | None = None,
        update_path: Path | None = None,
    ) -> ModelArtifact | None:
        if spec is None or update_path is None:
            spec, update_path = self._update_spec(update)
        publication_path = update_path / "publication.json"
        if publication_path.exists():
            report = json.loads(publication_path.read_text(encoding="utf-8"))
            if int(report.get("update", -1)) != update:
                raise RuntimeError("publication belongs to a different update")
            if int(report.get("through_shard", -1)) != int(spec["through_shard"]):
                raise RuntimeError("publication consumed a different replay boundary")
            parent = spec.get("parent_model")
            expected_parent = None if parent is None else int(parent["version"])
            if report.get("model", {}).get("parent_version") != expected_parent:
                raise RuntimeError("publication names a different parent model")
            return self._commit_publication(
                report, verify_model_files=True
            )
        started = time.perf_counter()
        # In serialized mode one lease spans the complete publication path, so
        # a waiting actor cannot slip between training and promotion and delay
        # the new incumbent by an entire shard.
        async with self._gpu_lease():
            training_report = await self._train(spec, update_path)
            export_report = await self._export(update_path)
            if (
                training_report["checkpoint_sha256"]
                != export_report["checkpoint_sha256"]
            ):
                raise RuntimeError(
                    "training and export disagree on the candidate checkpoint"
                )
            warmup_report = await self._warm_inference(update, update_path)
        parent = self.incumbent
        model = ModelArtifact(
            version=update,
            checkpoint=str((update_path / "candidate.pt").resolve()),
            onnx=str((update_path / "candidate.onnx").resolve()),
            checkpoint_sha256=str(export_report["checkpoint_sha256"]),
            onnx_sha256=str(export_report["onnx_sha256"]),
            parent_version=parent.version if parent else None,
        )
        report = {
            "schema": "vgo.pipeline-publication.v1",
            "update": update,
            "through_shard": int(spec["through_shard"]),
            "training": training_report,
            "export": export_report,
            "inference_warmup": warmup_report,
            "model": asdict(model),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(publication_path, report)
        published = self._commit_publication(report)
        self._retire_aged_shards(int(spec["through_shard"]))
        return published

    def _retire_aged_shards(self, through_sequence: int) -> None:
        """Compress shards the learner has finished with.

        An update trains on `_active_replay(through_sequence)`, the last
        `replay_window` shards, so anything older is never read again by this
        run. This is the moment that is certainly true: the update has published
        and its replay boundary is durably recorded.

        The shards are still worth keeping -- each one records the model that
        generated it, so the set is a reproducible history -- but a 6.04 GB
        dense shard compresses 30-70x here, because five policy-shaped arrays
        store 16385 float32 slots per sample to hold ~24 nonzero values.

        Compression runs in a thread and is never awaited: it is housekeeping,
        and a slow or failed compression must not delay the next update. A shard
        that is missed simply gets picked up after the following update.
        """
        if not self.config.retire_shards:
            return
        cutoff = through_sequence - self.config.replay_window + 1
        if cutoff <= 0:
            return
        stale = [
            Path(value["path"])
            for value in self.state.replay
            if int(value["sequence"]) < cutoff
        ]
        pending = [path for path in stale if path.exists()]
        if not pending:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Retirement is housekeeping scheduled from the running pipeline. If
            # there is no loop -- a synchronous caller, or a test -- do it inline
            # rather than failing the update that just published.
            for path in pending:
                try:
                    _log_completed_retirement(*_compress_shard(path))
                except Exception as error:  # never fail an update over cleanup
                    print(f"[retire] failed: {error}", flush=True)
            return
        for path in pending:
            # Fire and forget. asyncio.to_thread returns a task we deliberately
            # do not await; the callback exists only so a failure is visible in
            # the log rather than swallowed as a never-retrieved exception.
            task = asyncio.ensure_future(asyncio.to_thread(_compress_shard, path))
            task.add_done_callback(_log_retirement)
            self._retirements.add(task)
            task.add_done_callback(self._retirements.discard)

    def _should_start_generation(
        self,
        pending: Sequence[ReplayArtifact],
        *,
        learning_through: int | None,
        generation_active: bool,
        active_count: int = 0,
        in_flight: int = 0,
    ) -> bool:
        # `generation_active` is the old single-slot signal; `active_count`
        # supersedes it when several generators are allowed.
        if self.config.concurrent_generators <= 1:
            if generation_active:
                return False
        elif active_count >= self.config.concurrent_generators:
            return False
        remaining_updates = self.config.updates - self.state.updates_completed
        remaining_shards = remaining_updates * self.config.shards_per_update
        if len(pending) + in_flight >= remaining_shards:
            return False
        if learning_through is None:
            return len(pending) + in_flight < self.config.shards_per_update
        prefetched = sum(
            replay.sequence > learning_through for replay in pending
        )
        return prefetched + in_flight < self.config.maximum_prefetch_shards

    def report(self) -> dict[str, Any]:
        return {
            "schema": "vgo.pipeline-run.v1",
            "config_digest": self.config_digest,
            "wall_seconds": self.state.active_wall_seconds,
            "calendar_seconds": time.time() - self.state.started_unix_seconds,
            "updates_completed": self.state.updates_completed,
            "replay_shards": len(self.state.replay),
            "models": self.state.models,
            "rejected_models": self.state.rejected_models,
            "telemetry_pending": len(self.state.telemetry_pending),
            "telemetry_completed": len(self.state.telemetry_completed),
            "final_model": self.state.models[-1] if self.state.models else None,
            "utilization": self._utilization_report(),
        }

    def _utilization_report(self) -> dict[str, Any]:
        """Aggregate the counters that explain where pipeline capacity went."""

        generation_wall = 0.0
        summed_game = 0.0
        writer_backpressure = 0.0
        inference_seconds = 0.0
        inference_positions = 0
        inference_batches = 0
        generated_shards = 0
        for replay in self.state.replay:
            if int(replay.get("sequence", -1)) < 0:
                continue
            try:
                manifest = json.loads(
                    Path(str(replay["manifest"])).read_text(encoding="utf-8")
                )
                generation = manifest["generation_metrics"]
                broker = manifest["broker_metrics"]
                generation_wall += float(generation["wall_seconds"])
                summed_game += float(generation["summed_game_seconds"])
                writer_backpressure += float(
                    generation["writer_backpressure_seconds"]
                )
                inference_seconds += float(broker["inference_seconds"])
                inference_positions += int(broker["positions"])
                inference_batches += int(broker["batches"])
                generated_shards += 1
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        learning_wall = 0.0
        learner_wall = 0.0
        optimization_wall = 0.0
        replay_cache_hits = 0
        replay_cache_misses = 0
        measured_updates = 0
        for update in range(self.state.updates_completed):
            publication_path = (
                self.output
                / "updates"
                / f"update-{update:06d}"
                / "publication.json"
            )
            try:
                publication = json.loads(
                    publication_path.read_text(encoding="utf-8")
                )
                training = publication["training"]
                learning_wall += float(publication["wall_seconds"])
                learner_wall += float(training["wall_seconds"])
                optimization_wall += float(training["optimization_seconds"])
                replay_cache_hits += int(training.get("replay_cache_hits", 0))
                replay_cache_misses += int(
                    training.get("replay_cache_misses", 0)
                )
                measured_updates += 1
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        batch_capacity = inference_batches * self.config.inference_batch
        cache_accesses = replay_cache_hits + replay_cache_misses
        stage_work = generation_wall + learning_wall
        active_wall = self.state.active_wall_seconds
        return {
            "measured_generation_shards": generated_shards,
            "measured_updates": measured_updates,
            "generation_wall_seconds": generation_wall,
            "learning_wall_seconds": learning_wall,
            "learner_wall_seconds": learner_wall,
            "learner_optimization_seconds": optimization_wall,
            "stage_work_seconds": stage_work,
            "pipeline_overlap_factor": (
                stage_work / active_wall if active_wall > 0.0 else None
            ),
            "average_active_games": (
                summed_game / generation_wall
                if generation_wall > 0.0
                else None
            ),
            "inference_batch_fill": (
                inference_positions / batch_capacity
                if batch_capacity > 0
                else None
            ),
            "inference_positions": inference_positions,
            "inference_batches": inference_batches,
            "inference_seconds": inference_seconds,
            "writer_backpressure_seconds": writer_backpressure,
            "learner_optimization_fraction": (
                optimization_wall / learner_wall
                if learner_wall > 0.0
                else None
            ),
            "replay_cache_hit_ratio": (
                replay_cache_hits / cache_accesses
                if cache_accesses > 0
                else None
            ),
            "replay_cache_hits": replay_cache_hits,
            "replay_cache_misses": replay_cache_misses,
        }

    async def run(self) -> dict[str, Any]:
        with self._run_lease():
            return await self._run()

    async def _run(self) -> dict[str, Any]:
        self.state = self._load_or_create_state(self._config_value)
        if self.state.updates_completed >= self.config.updates:
            final_path = self.output / "run.json"
            report = self.report()
            atomic_json(final_path, report)
            return report
        active_started = time.perf_counter()
        try:
            self.learner = await LearnerService.start(
                cwd=self.training,
                environment=self.environment,
                stderr_path=self.output / "logs" / "learner.stderr.log",
            )
        except BaseException:
            self.state.active_wall_seconds += (
                time.perf_counter() - active_started
            )
            self._save_state()
            raise
        generations: set[asyncio.Task[ReplayArtifact]] = set()
        # Sequences handed to running generators. `state.next_shard` only moves
        # on commit, so without this two concurrent tasks would claim the same
        # number and race for the same staging directory.
        reserved_sequences: set[int] = set()
        learning: asyncio.Task[ModelArtifact | None] | None = None
        learning_through: int | None = None
        failed = True
        try:
            while self.state.updates_completed < self.config.updates:
                pending = self._pending_replay()

                if (
                    learning is None
                    and len(pending) >= self.config.shards_per_update
                ):
                    update = self.state.updates_completed
                    spec, update_path = self._update_spec(update)
                    learning_through = int(spec["through_shard"])
                    learning = asyncio.create_task(
                        self._learn_and_publish(update, spec, update_path)
                    )

                # Start as many generators as the config allows. Each claims
                # its own sequence up front: `state.next_shard` only advances on
                # commit, so concurrent tasks reading it would collide.
                while self._should_start_generation(
                    pending,
                    learning_through=learning_through,
                    generation_active=bool(generations),
                    active_count=len(generations),
                    in_flight=len(generations),
                ):
                    sequence = max(
                        self.state.next_shard,
                        max(reserved_sequences, default=-1) + 1,
                    )
                    reserved_sequences.add(sequence)
                    # The incumbent is captured once per shard. A publication
                    # can finish while this task runs, but the shard remains a
                    # valid, explicitly stamped sample from the older policy.
                    generations.add(
                        asyncio.create_task(
                            self._generate_shard(self.incumbent, sequence)
                        )
                    )

                tasks = set(generations)
                if learning is not None:
                    tasks.add(learning)
                if not tasks:
                    raise RuntimeError("pipeline has no runnable work")
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                finished = generations & done
                for task in finished:
                    replay = task.result()
                    reserved_sequences.discard(replay.sequence)
                    self._commit_replay(replay)
                generations -= finished
                if learning in done:
                    learning.result()
                    learning = None
                    learning_through = None
                    # Rate the checkpoint now rather than at the end of the
                    # run. Queued telemetry used to wait for --drain-telemetry
                    # after the final update, which was tolerable while the
                    # promotion arena reported on every update; with the gate
                    # removed this is the only strength signal the run
                    # produces, and a 240-update run would have emitted none of
                    # it for eighty hours while the queue grew.
                    #
                    # Under the GPU lease so it cannot run against a generator.
                    # Cost is bounded by --telemetry-every: at every 5 with 2
                    # opponents and 16 pairs it is 64 games per rated point,
                    # against the 16-game arena that used to run on every
                    # single update.
                    if self.state.telemetry_pending:
                        async with self._gpu_lease():
                            await self._run_pending_telemetry()
            failed = False
        finally:
            active = [
                task
                for task in (*generations, learning)
                if task is not None and not task.done()
            ]
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            try:
                await self.learner.close(force=failed)
            finally:
                self.learner = None
                self.state.active_wall_seconds += (
                    time.perf_counter() - active_started
                )
                self._save_state()

        report = self.report()
        atomic_json(self.output / "run.json", report)
        return report

    async def drain_telemetry(self) -> None:
        with self._run_lease():
            await self._drain_telemetry()

    async def _drain_telemetry(self) -> None:
        # Entry point for --telemetry-only, where the pipeline was constructed
        # fresh and its in-memory state is empty. Reloading here would discard
        # anything a live run holds that is not yet on disk, so the loop calls
        # _run_pending_telemetry directly instead.
        self.state = self._load_or_create_state(self._config_value)
        await self._run_pending_telemetry()

    async def _run_pending_telemetry(self) -> None:
        while self.state.telemetry_pending:
            first = self.state.telemetry_pending[0]
            candidate_version = int(first["candidate_version"])
            candidate = str(first["candidate"])
            jobs = [
                job
                for job in self.state.telemetry_pending
                if int(job["candidate_version"]) == candidate_version
                and str(job["candidate"]) == candidate
                and job.get("group_seed") == first.get("group_seed")
            ]
            jobs.sort(key=lambda job: int(job.get("group_index", 0)))
            missing = [
                job
                for job in jobs
                if not (
                    self.output / "telemetry" / f"{job['id']}.json"
                ).exists()
            ]
            if missing:
                pairs = {int(job["pairs"]) for job in missing}
                if len(pairs) != 1:
                    raise RuntimeError("batched telemetry jobs disagree on pair count")
                group_seed = missing[0].get("group_seed")
                first_index = int(missing[0].get("group_index", 0))
                if group_seed is None or any(
                    int(job["seed"])
                    != int(group_seed)
                    + (first_index + index) * 1_000_003
                    for index, job in enumerate(missing)
                ):
                    # Legacy pending jobs have independent seeds. Preserve their
                    # exact experiments rather than silently rebasing them.
                    missing = missing[:1]
                    pairs = {int(missing[0]["pairs"])}
                    group_seed = int(missing[0]["seed"])
                    first_index = 0
                command = self.arena_command(
                    candidate=Path(candidate),
                    opponents=[
                        Path(str(job["opponent"])) for job in missing
                    ],
                    # Arena derives a deterministic per-opponent offset from
                    # this group seed. Job identity remains stable even though
                    # execution is amortized across one model load.
                    seed=int(group_seed) + first_index * 1_000_003,
                    pairs=pairs.pop(),
                )
                result = await self.runner.run(
                    command,
                    cwd=self.root,
                    log_prefix=self.output
                    / "logs"
                    / f"telemetry-v{candidate_version:06d}",
                )
                arenas = result.json_documents()
                if len(arenas) != len(missing):
                    raise RuntimeError(
                        "batched arena emitted an unexpected number of results"
                    )
                for job, arena in zip(missing, arenas, strict=True):
                    opponent = Path(str(job["opponent"])).resolve()
                    reported = arena.get("opponent_model")
                    if reported and Path(str(reported)).resolve() != opponent:
                        raise RuntimeError(
                            "batched arena result order does not match its jobs"
                        )
                    atomic_json(
                        self.output / "telemetry" / f"{job['id']}.json",
                        {
                            "schema": "vgo.telemetry-result.v1",
                            "id": str(job["id"]),
                            "candidate_version": candidate_version,
                            "opponent_version": int(job["opponent_version"]),
                            "arena": arena,
                        },
                    )
            completed_ids = {
                str(job["id"])
                for job in jobs
                if (
                    self.output / "telemetry" / f"{job['id']}.json"
                ).exists()
            }
            self.state.telemetry_pending = [
                job
                for job in self.state.telemetry_pending
                if str(job["id"]) not in completed_ids
            ]
            for job_id in sorted(completed_ids):
                if job_id not in self.state.telemetry_completed:
                    self.state.telemetry_completed.append(job_id)
            self._save_state()
        self._refresh_ratings()
        if self.state.updates_completed >= self.config.updates:
            atomic_json(self.output / "run.json", self.report())

    def _refresh_ratings(self) -> None:
        matches: list[dict[str, Any]] = []
        for path in sorted((self.output / "telemetry").glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("schema") != "vgo.telemetry-result.v1":
                continue
            arena = result["arena"]
            matches.append(
                {
                    "a": int(result["candidate_version"]),
                    "b": int(result["opponent_version"]),
                    "a_wins": int(arena["candidate_wins"]),
                    "b_wins": int(arena["candidate_losses"]),
                    "draws": int(arena.get("draws", 0)),
                }
            )
        if not matches:
            return
        anchor = min(
            min(int(match["a"]), int(match["b"])) for match in matches
        )
        ratings = fit_ratings(matches, anchor=anchor, prior_games=0.25)
        atomic_json(
            self.output / "telemetry" / "ratings.json",
            {str(version): round(rating, 1) for version, rating in ratings.items()},
        )


def config_from_arguments(arguments: argparse.Namespace) -> PipelineConfig:
    values = vars(arguments).copy()
    values.pop("drain_telemetry", None)
    values.pop("telemetry_only", None)
    values["output"] = str(Path(values["output"]).resolve())
    values["initial_checkpoint"] = (
        str(Path(values["initial_checkpoint"]).resolve())
        if values.get("initial_checkpoint") is not None
        else None
    )
    values["initial_onnx"] = (
        str(Path(values["initial_onnx"]).resolve())
        if values.get("initial_onnx") is not None
        else None
    )
    values["initial_replay"] = tuple(
        str(Path(path).resolve()) for path in values.get("initial_replay", ())
    )
    return PipelineConfig(**values)


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--shards-per-update", type=int, default=1)
    parser.add_argument("--replay-window", type=int, default=8)
    parser.add_argument("--maximum-prefetch-shards", type=int, default=1)
    parser.add_argument(
        "--concurrent-generators",
        type=int,
        default=1,
        help=(
            "generator processes running at once; a second one fills the "
            "parallelism gap while the first drains its in-flight games"
        ),
    )
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--policy-resolution", type=int, default=32)
    parser.add_argument("--radius", type=float, default=1.0 / 18.0)
    parser.add_argument("--generation-simulations", type=int, default=256)
    parser.add_argument("--maximum-plies", type=int, default=256)
    parser.add_argument("--coarse-pool", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--actors", type=int, default=64)
    parser.add_argument("--writer-queue-games", type=int, default=2)
    parser.add_argument("--leaf-batch", type=int, default=1)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--inference-delay-ms", type=int, default=1)
    parser.add_argument(
        "--inference-slots",
        type=int,
        default=2,
        help="execution slots behind self-play generation's shared batch queue",
    )
    parser.add_argument(
        "--provider", choices=("cpu", "cuda", "tensorrt"), default="tensorrt"
    )
    parser.add_argument("--inference-device-id", type=int, default=0)
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--warm-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "prebuild each TensorRT engine after export so model-load latency "
            "overlaps the current self-play tail"
        ),
    )
    parser.add_argument(
        "--architecture", choices=MODEL_ARCHITECTURES, default="ddrnet"
    )
    parser.add_argument(
        "--norm-groups",
        type=int,
        default=None,
        help=(
            "GroupNorm groups in every residual block (ddrnet only); "
            "supersedes --variance-scaled. Bounds activation magnitude "
            "structurally rather than by a constant fixed at initialization"
        ),
    )
    parser.add_argument(
        "--context-attention-blocks",
        type=int,
        default=0,
        help=(
            "trailing residual blocks in each ddrnet context stage to replace "
            "with transformer blocks; 0 is the plain convolutional net. "
            "Attention is the only part of this model that is not "
            "resolution-agnostic -- rotary tables are built per board size -- "
            "so a checkpoint carrying it is fixed to its raster"
        ),
    )
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument(
        "--muon-learning-rate",
        type=float,
        default=0.01,
        help=(
            "rate for the Muon group: the conv/linear trunk. Heads and norms "
            "stay on Adam at --learning-rate"
        ),
    )
    parser.add_argument(
        "--full-adam",
        action="store_true",
        help=(
            "put every parameter on Adam instead, which is what runs before "
            "Muon landed did. Kept because those runs' numbers are only "
            "comparable to another Adam run"
        ),
    )
    parser.add_argument(
        "--variance-scaled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "scale each residual branch by 1/sqrt(depth) and initialize convs "
            "at He scale; bounds trunk variance instead of letting training "
            "inflate weights (ddrnet only, cannot be warm-started onto an "
            "unscaled checkpoint)"
        ),
    )
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--training-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--warm-learning-rate", type=float, default=5e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument(
        "--drain-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="finish in-flight games when a shard fills rather than cancelling them",
    )
    parser.add_argument(
        "--recency-decay",
        type=float,
        default=1.0,
        help=(
            "per-shard sampling decay; 1.0 samples the window uniformly, 0.9 "
            "makes each older shard 10%% less likely than its successor"
        ),
    )
    parser.add_argument(
        "--ownership-weight",
        type=float,
        default=1.5,
        help=(
            "weight on the auxiliary ownership loss; 0 disables it and frees "
            "its targets from the replay window"
        ),
    )
    parser.add_argument("--training-threads", type=int, default=4)
    parser.add_argument("--training-device", default="cuda")
    parser.add_argument(
        "--training-precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--restore-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-epochs", type=float, default=1)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--overlap-actor-learner",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=0.0,
        help=(
            "concede a game once the side to move has been losing by this much "
            "for --resign-window consecutive plies; 0 disables resignation"
        ),
    )
    parser.add_argument(
        "--komi-low",
        type=float,
        default=0.0,
        help="lowest komi a game may draw; positive favours White",
    )
    parser.add_argument(
        "--komi-high",
        type=float,
        default=0.0,
        help=(
            "highest komi a game may draw. A range teaches the relationship "
            "between komi and the position; one value teaches one balance point"
        ),
    )
    parser.add_argument(
        "--dynamic-komi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "recenter the fixed-width komi distribution from exact outcomes in "
            "the trailing replay window"
        ),
    )
    parser.add_argument(
        "--komi-target-black-win-rate",
        type=float,
        default=0.5,
        help="Black win rate the dynamic komi controller targets",
    )
    parser.add_argument(
        "--komi-recenter-minimum-games",
        type=int,
        default=256,
        help="played games required before trusting a dynamic komi fit",
    )
    parser.add_argument(
        "--komi-recenter-maximum-step",
        type=float,
        default=0.025,
        help="largest change in the komi distribution's center per shard",
    )
    parser.add_argument(
        "--raster-kind",
        choices=("semantic", "compact", "compact-pass", "compact-dead-zone", "rgb"),
        default="semantic",
        help=(
            "channel layout. compact is four channels plus komi; compact-pass "
            "adds whether the previous move was a pass, without which a net "
            "cannot tell that passing now would end the game; compact-dead-zone "
            "is compact-pass with the official rules' capture predicate in "
            "place of settled, and the two differ in exactly one plane"
        ),
    )
    parser.add_argument("--resign-window", type=int, default=5)
    parser.add_argument(
        "--resign-minimum-ply",
        type=int,
        default=0,
        help=(
            "earliest ply a game may be conceded at. The window counts a "
            "seat's own turns and a seat moves every other ply, so window 5 "
            "concedes at ply 8 -- five stones each on a board holding 35"
        ),
    )
    parser.add_argument(
        "--resign-target-false-positive",
        type=float,
        default=0.0,
        help=(
            "choose the resign threshold per shard from trailing calibration "
            "instead of fixing it: the lowest threshold whose measured error "
            "stays under this rate. A fixed threshold cannot follow a model "
            "that is still learning -- 0.95 fired on 1%% of games early in "
            "ddrnet-wl and 26%% late. Zero keeps --resign-threshold fixed"
        ),
    )
    parser.add_argument(
        "--resign-soft-simulations",
        type=int,
        default=0,
        help=(
            "play a conceded game out at this many simulations instead of "
            "stopping, so a false positive corrects itself rather than writing "
            "the wrong label; measured at 10.7%% of concessions on a real shard"
        ),
    )
    parser.add_argument(
        "--resign-disable-fraction",
        type=float,
        default=0.1,
        help=(
            "fraction of games played to a real finish regardless of the "
            "threshold, which are what the shard's calibration is measured on"
        ),
    )
    parser.add_argument("--arena-pairs", type=int, default=16)
    parser.add_argument("--arena-simulations", type=int, default=256)
    parser.add_argument("--arena-actors", type=int, default=32)
    parser.add_argument("--telemetry-opponents", type=int, default=2)
    parser.add_argument("--telemetry-pairs", type=int, default=16)
    parser.add_argument("--arena-komi", type=float, default=0.104)
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=1,
        help=(
            "queue Elo work for every Nth checkpoint; the loop's rating curve is "
            "a trend and does not need every point"
        ),
    )
    parser.add_argument(
        "--retire-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compress replay shards once they leave the training window; runs "
            "off the critical path and never delays an update"
        ),
    )
    parser.add_argument("--seed", type=int, default=700_001)
    parser.add_argument("--arena-seed", type=int, default=900_001)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-onnx", type=Path)
    parser.add_argument("--initial-replay", type=Path, nargs="*", default=[])
