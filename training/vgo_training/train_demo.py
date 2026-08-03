from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import PreparedRasterDataset, RasterDataset
from .model import MODEL_ARCHITECTURES, build_model


LEGAL_CLEARANCE_CHANNEL = 7

# The eight symmetries of the square: (number of 90-degree rotations, flip?).
DIHEDRAL_TRANSFORMS = tuple((rotations, flip) for flip in (False, True) for rotations in range(4))


def apply_dihedral(
    states: torch.Tensor,
    policies: torch.Tensor,
    policy_masks: torch.Tensor,
    transform: int,
    height: int,
    width: int,
    policy_height: int | None = None,
    policy_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one of the eight square symmetries to a batch of states and targets.

    The board is square and every raster channel is a geometric field of the
    position (player-relative, so no colour swapping is involved), which makes
    the dihedral group an exact symmetry of the game: the transformed position is
    a real position whose transformed policy is the correct target for it.

    Policy vectors are `height * width` placement cells followed by one pass
    logit. Only the placement block is reindexed; pass is invariant under every
    symmetry, so it is carried through untouched.
    """
    if transform == 0:
        return states, policies, policy_masks
    if height != width:
        raise ValueError("dihedral augmentation requires a square raster")
    # The placement grid may be coarser than the raster; both are square, so the
    # same symmetry applies to each at its own scale.
    policy_height = height if policy_height is None else policy_height
    policy_width = width if policy_width is None else policy_width
    if policy_height != policy_width:
        raise ValueError("dihedral augmentation requires a square placement grid")
    rotations, flip = DIHEDRAL_TRANSFORMS[transform]

    def spatial(tensor: torch.Tensor) -> torch.Tensor:
        # [..., H, W] with the symmetry applied to the trailing two axes.
        if rotations:
            tensor = torch.rot90(tensor, rotations, dims=(-2, -1))
        if flip:
            tensor = torch.flip(tensor, dims=(-1,))
        return tensor.contiguous()

    cells = policy_height * policy_width

    def policy(tensor: torch.Tensor) -> torch.Tensor:
        placements, pass_logit = tensor[:, :cells], tensor[:, cells:]
        grid = placements.reshape(-1, policy_height, policy_width)
        return torch.cat((spatial(grid).reshape(placements.shape[0], -1), pass_logit), dim=1)

    return spatial(states), policy(policies), policy(policy_masks)


def importance_corrected_policy_targets(
    visits: torch.Tensor,
    betas: torch.Tensor,
    proposal_counts: torch.Tensor,
    explored_masks: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Build sparse importance-corrected policy targets.

    For replay-v3 rows, cumulative proposal counts from progressive widening
    provide the empirical sampling distribution ``beta_hat(a) = m(a) / K``.
    Sampled placements therefore receive unnormalized mass
    ``visits * m / (K * beta)``. Pass is enumerated deterministically and
    retains its raw visit mass. Rows without proposal counts (legacy, replay
    v1/v2, naive fallback, and pass-only states) use uncorrected normalized
    visits.

    The correction is computed in float64 log space so small positive proposal
    probabilities cannot overflow an inverse weight.
    """
    if visits.ndim != 2 or visits.shape != betas.shape:
        raise ValueError("visits and betas must be equally shaped rank-two tensors")
    if proposal_counts.shape != visits.shape:
        raise ValueError("proposal counts must match visits")
    if explored_masks.shape != visits.shape:
        raise ValueError("explored mask must match visits")
    if visits.shape[1] < 1:
        raise ValueError("policy tensors must contain the pass action")

    visits64 = visits.detach().to(dtype=torch.float64)
    betas64 = betas.detach().to(dtype=torch.float64)
    counts64 = proposal_counts.detach().to(dtype=torch.float64)
    explored = explored_masks.detach().bool()
    if validate:
        if not bool(torch.isfinite(visits64).all()) or bool((visits64 < 0.0).any()):
            raise ValueError("visit counts must be finite and nonnegative")
        if not bool(torch.isfinite(betas64).all()) or bool(
            ((betas64 < 0.0) | (betas64 > 1.0)).any()
        ):
            raise ValueError("sampling probabilities must be finite and in [0, 1]")
        if not bool(torch.isfinite(counts64).all()) or bool(
            ((counts64 < 0.0) | (counts64 != counts64.floor())).any()
        ):
            raise ValueError("proposal counts must be finite nonnegative integers")
        if bool(((visits64 > 0.0) & ~explored).any()):
            raise ValueError("positive visit counts must be explored")
        if bool(((betas64 > 0.0) & ~explored).any()):
            raise ValueError("positive sampling probabilities must be explored")
        if bool(((counts64 > 0.0) & ~explored).any()):
            raise ValueError("positive proposal counts must be explored")
        if bool((betas64[:, -1] != 0.0).any()):
            raise ValueError(
                "the deterministically enumerated pass action must have beta zero"
            )
        if bool((counts64[:, -1] != 0.0).any()):
            raise ValueError(
                "the deterministically enumerated pass action must have proposal count zero"
            )
        if bool((visits64.sum(dim=1) <= 0.0).any()):
            raise ValueError("every policy target must contain at least one visit")

    proposal_support = counts64[:, :-1] > 0.0
    beta_support = betas64[:, :-1] > 0.0
    proposal_totals = counts64[:, :-1].sum(dim=1)
    counted_rows = proposal_totals > 0.0
    if validate and bool(
        (
            counted_rows[:, None]
            & (proposal_support != beta_support)
        ).any()
    ):
        raise ValueError(
            "counted placement support must equal positive-beta placement support"
        )
    if validate and bool(
        (
            counted_rows[:, None]
            & explored[:, :-1]
            & ~proposal_support
        ).any()
    ):
        raise ValueError("counted replay rows must count every explored placement")

    # log(0) is -inf, which is exactly the desired zero target mass.
    log_weights = visits64.log()
    safe_betas = torch.where(
        proposal_support, betas64[:, :-1], torch.ones_like(betas64[:, :-1])
    )
    safe_counts = torch.where(
        proposal_support, counts64[:, :-1], torch.ones_like(counts64[:, :-1])
    )
    corrected = (
        visits64[:, :-1].log()
        + safe_counts.log()
        - proposal_totals.clamp_min(1.0).log()[:, None]
        - safe_betas.log()
    )
    log_weights[:, :-1] = torch.where(
        counted_rows[:, None] & proposal_support,
        corrected,
        log_weights[:, :-1],
    )
    # Counted rows have target support only on proposed placements and pass.
    # Pre-v3/all-zero-count rows retain their uncorrected explored visit target.
    log_weights[:, :-1] = torch.where(
        counted_rows[:, None] & ~proposal_support,
        torch.full_like(log_weights[:, :-1], -torch.inf),
        log_weights[:, :-1],
    )
    log_weights = log_weights.masked_fill(~explored, -torch.inf)
    targets = torch.softmax(log_weights, dim=1)
    if validate and not bool(torch.isfinite(targets).all()):
        raise ValueError("importance-corrected policy target is not finite")
    return targets.to(dtype=visits.dtype)


def full_legal_policy_masks(
    states: torch.Tensor, explored_masks: torch.Tensor
) -> torch.Tensor:
    """Return the full legality mask on the *policy* grid, preserving explored aliases.

    Legality is read from the raster's signed legal-clearance channel, which is at
    the render resolution. When the placement grid is coarser, each policy cell
    covers a block of raster pixels and is legal if *any* of them is legal --
    max-pooling, not averaging: a cell containing one playable point is a playable
    move, and averaging clearance would erase exactly the near-boundary moves.
    """
    if states.ndim != 4 or explored_masks.ndim != 2:
        raise ValueError("states must be rank four and policy masks rank two")
    if explored_masks.shape[0] != states.shape[0]:
        raise ValueError("policy mask batch does not match states")
    placement_cells = explored_masks.shape[1] - 1
    if states.shape[1] <= LEGAL_CLEARANCE_CHANNEL:
        # Only synthetic/legacy fixtures lack the semantic legal-clearance
        # channel. Their stored candidate mask is the best available contract.
        return explored_masks.bool()
    clearance = states[:, LEGAL_CLEARANCE_CHANNEL].unsqueeze(1)
    raster_cells = states.shape[2] * states.shape[3]
    if placement_cells != raster_cells:
        side = int(round(placement_cells**0.5))
        if side * side != placement_cells:
            raise ValueError("policy mask is not a square placement grid plus pass")
        # max_pool over clearance == "any legal pixel in this cell".
        clearance = nn.functional.adaptive_max_pool2d(clearance, (side, side))
    placements = clearance.reshape(states.shape[0], -1) >= 0.0
    if placements.shape[1] != placement_cells:
        raise ValueError("policy mask shape does not match the placement grid")
    passes = torch.ones(
        (states.shape[0], 1), dtype=torch.bool, device=states.device
    )
    # Exact continuous actions on the legal boundary can map to a cell whose
    # centre is just outside the legal set. Keep such explored aliases in the
    # denominator so every positive target remains representable.
    return torch.cat((placements, passes), dim=1) | explored_masks.bool()


def policy_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()


def sampled_policy_loss(
    logits: torch.Tensor,
    states: torch.Tensor,
    visits: torch.Tensor,
    betas: torch.Tensor,
    proposal_counts: torch.Tensor,
    explored_masks: torch.Tensor,
    *,
    validate_targets: bool = True,
) -> torch.Tensor:
    targets = importance_corrected_policy_targets(
        visits,
        betas,
        proposal_counts,
        explored_masks,
        validate=validate_targets,
    )
    legal_masks = full_legal_policy_masks(states, explored_masks)
    return policy_cross_entropy(logits, targets, legal_masks)


@torch.no_grad()
def prepare_policy_supervision(
    dataset: RasterDataset,
    batch_size: int,
    *,
    validate_targets: bool = True,
) -> PreparedRasterDataset:
    """Consume raw replay supervision into cached targets in bounded CPU batches.

    The normalized policy and explored-mask buffers are overwritten in place.
    The returned lightweight dataset deliberately omits visits, betas, proposal
    counts, and replay metadata so those tensors can be released before training.
    """
    if batch_size <= 0:
        raise ValueError("policy preparation batch size must be positive")
    tensors = (
        dataset.states,
        dataset.policies,
        dataset.policy_masks,
        dataset.visits,
        dataset.betas,
        dataset.proposal_counts,
        dataset.values,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("policy preparation requires a CPU dataset")
    for start in range(0, dataset.samples, batch_size):
        stop = min(start + batch_size, dataset.samples)
        explored_masks = dataset.policy_masks[start:stop]
        targets = importance_corrected_policy_targets(
            dataset.visits[start:stop],
            dataset.betas[start:stop],
            dataset.proposal_counts[start:stop],
            explored_masks,
            validate=validate_targets,
        )
        legal_masks = full_legal_policy_masks(
            dataset.states[start:stop],
            explored_masks,
        )
        dataset.policies[start:stop].copy_(targets)
        dataset.policy_masks[start:stop].copy_(legal_masks)
        del targets, legal_masks
    return PreparedRasterDataset(
        states=dataset.states,
        policies=dataset.policies,
        policy_masks=dataset.policy_masks,
        values=dataset.values,
        height=dataset.height,
        width=dataset.width,
        sources=dataset.sources,
        ownerships=getattr(dataset, "ownerships", None),
    )


@torch.no_grad()
def metrics(
    model: nn.Module,
    dataset: DatasetView | PreparedRasterDataset,
    device: torch.device,
    batch_size: int,
    *,
    value_weight: float,
) -> dict[str, float]:
    """Held-out metrics, accumulated one batch at a time.

    Every quantity here is a mean over samples, so a sum of per-batch means
    weighted by batch size and divided by the sample count reproduces the
    whole-set value exactly. Evaluating the split in a single forward pass
    instead made evaluation memory scale with the replay window rather than with
    the batch size, which capped how far the window could grow independently of
    the device.
    """
    model.eval()
    total = dataset.samples
    if total <= 0:
        raise ValueError("cannot compute metrics over an empty dataset")
    if batch_size <= 0:
        raise ValueError("metrics batch size must be positive")
    if not np.isfinite(value_weight) or value_weight < 0.0:
        raise ValueError("value weight must be finite and nonnegative")

    # Keep all reductions on the evaluation device. Converting each batch loss
    # to float forces a CUDA synchronization per metric per batch; accumulating
    # five scalar tensors and transferring once makes the pass one synchronization.
    accumulator = torch.zeros(5, dtype=torch.float64, device=device)

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        states, targets, masks, values = gather_batch(
            dataset, torch.arange(start, stop), device
        )
        logits, predictions = model(states)
        masked_logits = logits.masked_fill(
            ~masks, torch.finfo(logits.dtype).min
        )
        cross_entropy = -(
            targets * torch.log_softmax(masked_logits, dim=1)
        ).sum(dim=1)
        target_entropy = -(
            targets * targets.clamp_min(1e-12).log()
        ).sum(dim=1)
        accumulator += torch.stack(
            (
                cross_entropy.sum(),
                target_entropy.sum(),
                (predictions - values).square().sum(),
                (
                    masked_logits.argmax(dim=1) == targets.argmax(dim=1)
                ).float().sum(),
                (predictions - values).abs().sum(),
            )
        ).to(dtype=torch.float64)

    (
        cross_entropy_total,
        target_entropy_total,
        squared_error_total,
        top1_total,
        absolute_error_total,
    ) = accumulator.cpu().tolist()
    cross_entropy = cross_entropy_total / total
    target_entropy = target_entropy_total / total
    value_mse = squared_error_total / total
    return {
        "loss": cross_entropy + value_weight * value_mse,
        "policy_cross_entropy": cross_entropy,
        "policy_target_entropy": target_entropy,
        "policy_kl": cross_entropy - target_entropy,
        "policy_top1": top1_total / total,
        "value_mae": absolute_error_total / total,
    }


def gather_batch(
    dataset: "DatasetView | PreparedRasterDataset",
    selection: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather one batch onto `device` from either a split view or a whole dataset.

    `selection` indexes into whatever it is given: positions within the view for
    a `DatasetView`, rows of the dataset otherwise.
    """
    if isinstance(dataset, DatasetView):
        return dataset.batch(selection, device)
    return (
        dataset.states[selection].to(device),
        dataset.policies[selection].to(device),
        dataset.policy_masks[selection].to(device),
        dataset.values[selection].to(device),
    )


@dataclass(frozen=True)
class DatasetView:
    """A train/validation split held as an index rather than as gathered rows.

    Materializing the split with `dataset.states[indices]` allocates a full copy
    of every tensor, and because both halves are built while the original is
    still live the peak is roughly twice the replay window. At a coupled 128x128
    policy that was ~36 GB for a five-shard window. Carrying the indices instead
    makes a split free; rows are gathered per batch, which is the granularity the
    GPU consumes anyway.
    """

    base: PreparedRasterDataset
    indices: torch.Tensor

    @property
    def samples(self) -> int:
        return int(self.indices.shape[0])

    @property
    def height(self) -> int:
        return self.base.height

    @property
    def width(self) -> int:
        return self.base.width

    @property
    def channels(self) -> int:
        return self.base.channels

    @property
    def sources(self) -> tuple[str, ...]:
        return self.base.sources

    def batch(
        self, selection: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather one batch of rows onto `device`.

        `selection` indexes into this view, not into the underlying dataset.
        """
        rows = self.indices[selection]
        return (
            self.base.states[rows].to(device),
            self.base.policies[rows].to(device),
            self.base.policy_masks[rows].to(device),
            self.base.values[rows].to(device),
        )


def subset(dataset: PreparedRasterDataset, indices: torch.Tensor) -> DatasetView:
    return DatasetView(base=dataset, indices=indices)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def build_scheduler(
    optimizer: torch.optim.Optimizer, arguments: argparse.Namespace
) -> torch.optim.lr_scheduler.LRScheduler:
    """Learning-rate schedule over epochs.

    Cosine bakes the run length into the curve: `T_max` is the horizon, so
    training longer means re-tuning the shape, and a run that is still
    improving at the end has already annealed its learning rate away.

    WSD (warmup-stable-decay) holds a constant rate through the middle and
    only spends `--decay-fraction` of the run annealing to `eta_min`. The
    stable phase can be any length, so extending a run is just a larger
    `--epochs`. The short warmup covers the from-scratch case, where a full
    rate at step one is the likeliest source of a loss spike.
    """
    minimum = arguments.learning_rate * arguments.final_learning_rate_fraction
    if arguments.schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=arguments.epochs, eta_min=minimum
        )
    epochs = arguments.epochs
    warmup = min(arguments.warmup_epochs, max(epochs - 1, 0))
    decay = min(round(epochs * arguments.decay_fraction), max(epochs - warmup, 0))
    stable = max(epochs - warmup - decay, 0)
    floor = arguments.final_learning_rate_fraction

    def scale(epoch: int) -> float:
        # LambdaLR multiplies the base rate; `epoch` is zero-based here.
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        if epoch < warmup + stable:
            return 1.0
        if decay <= 0:
            return 1.0
        progress = (epoch - warmup - stable + 1) / decay
        return max(1.0 + (floor - 1.0) * min(progress, 1.0), floor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train(arguments: argparse.Namespace) -> dict[str, object]:
    # Keep the historical CLI as a one-update adapter. The implementation lives
    # in PersistentLearner so the RL loop can keep model weights, Adam moments,
    # compiled code, prepared shards, and staging buffers alive between updates.
    from .learner import LearnerConfig, LearnerUpdate, PersistentLearner

    config = LearnerConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        value_weight=arguments.value_weight,
        model_width=arguments.model_width,
        blocks=arguments.blocks,
        architecture=arguments.architecture,
        threads=arguments.threads,
        device=arguments.device,
        precision=arguments.precision,
        seed=arguments.seed,
        compile=arguments.compile,
        restore_optimizer=arguments.restore_optimizer,
        schedule=arguments.schedule,
        warmup_epochs=arguments.warmup_epochs,
        decay_fraction=arguments.decay_fraction,
        final_learning_rate_fraction=arguments.final_learning_rate_fraction,
        report_every=arguments.report_every,
        validation_fraction=arguments.validation_fraction,
        augment=arguments.augment,
    )
    learner = PersistentLearner(defaults=config)
    try:
        report = learner.update(
            LearnerUpdate(
                datasets=tuple(arguments.datasets),
                output=Path(arguments.output),
                initial_checkpoint=arguments.initial_checkpoint,
                config=config,
            )
        )
    finally:
        learner.close()
    print(json.dumps(report, indent=2))
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("../artifacts/raster-demo/model.pt"))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument(
        "--architecture", choices=MODEL_ARCHITECTURES, default="flat"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the training model and enable TF32 matmuls (CUDA only)",
    )
    parser.add_argument(
        "--restore-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restore Adam moments from --initial-checkpoint instead of starting "
        "them at zero. Matters most for short runs: beta2=0.999 needs ~2000 "
        "steps of history, and a 10-epoch iteration is only ~4300",
    )
    parser.add_argument(
        "--schedule",
        choices=("wsd", "cosine"),
        default="wsd",
        help="learning-rate schedule; wsd holds a constant rate and anneals "
        "only at the end, so a longer run needs no reshaping",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="wsd only: epochs ramping linearly to the full rate",
    )
    parser.add_argument(
        "--decay-fraction",
        type=float,
        default=0.2,
        help="wsd only: trailing fraction of the run spent annealing",
    )
    parser.add_argument(
        "--final-learning-rate-fraction",
        type=float,
        default=0.01,
        help="floor as a fraction of --learning-rate, for both schedules",
    )
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the eight dihedral symmetries of the square board to training batches",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_arguments())
