from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .dataset import PreparedRasterDataset, RasterDataset, load_datasets
from .model import build_model
from .serve import load_model


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
    )


@torch.no_grad()
def metrics(
    model: nn.Module,
    dataset: PreparedRasterDataset,
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

    cross_entropy_total = 0.0
    target_entropy_total = 0.0
    squared_error_total = 0.0
    top1_total = 0.0
    absolute_error_total = 0.0

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        count = stop - start
        states = dataset.states[start:stop].to(device)
        targets = dataset.policies[start:stop].to(device)
        masks = dataset.policy_masks[start:stop].to(device)
        values = dataset.values[start:stop].to(device)
        logits, predictions = model(states)
        # Reuses the training objective so the two can never drift apart.
        cross_entropy_total += float(policy_cross_entropy(logits, targets, masks)) * count
        target_entropy_total += float(
            -(targets * targets.clamp_min(1e-12).log()).sum(dim=1).mean()
        ) * count
        squared_error_total += float(nn.functional.mse_loss(predictions, values)) * count
        top1_total += float(
            (
                logits.masked_fill(~masks, torch.finfo(logits.dtype).min).argmax(dim=1)
                == targets.argmax(dim=1)
            )
            .float()
            .mean()
        ) * count
        absolute_error_total += float((predictions - values).abs().mean()) * count

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


def subset(
    dataset: PreparedRasterDataset, indices: torch.Tensor
) -> PreparedRasterDataset:
    return PreparedRasterDataset(
        states=dataset.states[indices],
        policies=dataset.policies[indices],
        policy_masks=dataset.policy_masks[indices],
        values=dataset.values[indices],
        height=dataset.height,
        width=dataset.width,
        sources=dataset.sources,
    )


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="ascii")
    temporary.replace(path)


def train(arguments: argparse.Namespace) -> dict[str, object]:
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    device = torch.device(arguments.device)
    raw_dataset = load_datasets(arguments.datasets)
    corrected_samples = int(
        torch.count_nonzero(raw_dataset.proposal_counts[:, :-1], dim=1)
        .gt(0)
        .sum()
        .item()
    )
    dataset = prepare_policy_supervision(
        raw_dataset,
        arguments.batch_size,
        validate_targets=False,
    )
    del raw_dataset
    policy_target_name = "progressive_empirical_importance_v1"
    dataset_samples = dataset.samples
    dataset_channels = dataset.channels
    dataset_height = dataset.height
    dataset_width = dataset.width
    # The replay policy vector is `policy_resolution^2 + 1`, which is how the
    # placement grid reaches training: it may be coarser than the raster the
    # states were rendered at. Equal sizes mean the grids are coupled.
    placement_cells = dataset.policies.shape[1] - 1
    policy_resolution = int(round(placement_cells**0.5))
    if policy_resolution * policy_resolution != placement_cells:
        raise ValueError(
            f"replay policy vector {dataset.policies.shape[1]} is not a square grid plus pass"
        )
    decoupled_policy = (
        policy_resolution if (policy_resolution, policy_resolution) != (dataset_height, dataset_width) else None
    )
    dataset_sources = dataset.sources
    dataset_shape = tuple(dataset.states.shape)
    generator = torch.Generator().manual_seed(arguments.seed)
    split = torch.randperm(dataset_samples, generator=generator)
    validation_samples = 0
    if dataset_samples > 1 and arguments.validation_fraction > 0.0:
        validation_samples = max(1, round(dataset_samples * arguments.validation_fraction))
        validation_samples = min(validation_samples, dataset_samples - 1)
    validation_indices = split[:validation_samples]
    training_indices = split[validation_samples:]
    training = subset(dataset, training_indices)
    validation = subset(dataset, validation_indices) if validation_samples else training
    del dataset, split, validation_indices, training_indices

    parent_checkpoint = None
    if arguments.initial_checkpoint is not None:
        model, parent = load_model(arguments.initial_checkpoint.resolve(strict=True))
        if (
            int(parent["channels"]),
            int(parent["height"]),
            int(parent["width"]),
        ) != (dataset_channels, dataset_height, dataset_width):
            raise ValueError("initial checkpoint does not match replay raster shape")
        parent_checkpoint = str(arguments.initial_checkpoint.resolve())
        model_width = int(parent["model_width"])
        blocks = int(parent["blocks"])
        architecture = str(parent.get("architecture", "flat"))
    else:
        model_width = arguments.model_width
        blocks = arguments.blocks
        architecture = arguments.architecture
        model = build_model(
            architecture=architecture,
            channels=dataset_channels,
            width=model_width,
            blocks=blocks,
            policy_resolution=decoupled_policy,
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=arguments.epochs,
        eta_min=arguments.learning_rate * 0.01,
    )
    initial_training = metrics(
        model,
        training,
        device,
        arguments.batch_size,
        value_weight=arguments.value_weight,
    )
    initial_validation = metrics(
        model,
        validation,
        device,
        arguments.batch_size,
        value_weight=arguments.value_weight,
    )
    best_epoch = 0
    best = initial_validation
    best_score = (
        initial_validation["policy_kl"]
        + arguments.value_weight * initial_validation["value_mae"]
    )
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    started = time.perf_counter()

    model.train()
    for epoch in range(1, arguments.epochs + 1):
        permutation = torch.randperm(training.samples, generator=generator)
        for start in range(0, training.samples, arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            states = training.states[indices].to(device)
            policy_targets = training.policies[indices].to(device)
            policy_masks = training.policy_masks[indices].to(device)
            value_targets = training.values[indices].to(device)
            if arguments.augment:
                # One symmetry per batch rather than per sample: the transform is
                # a cheap view-and-copy, and over many epochs each position is
                # still seen under all eight. Values are scalars and invariant.
                transform = int(
                    torch.randint(len(DIHEDRAL_TRANSFORMS), (1,), generator=generator).item()
                )
                states, policy_targets, policy_masks = apply_dihedral(
                    states,
                    policy_targets,
                    policy_masks,
                    transform,
                    training.height,
                    training.width,
                    policy_resolution,
                    policy_resolution,
                )
            logits, values = model(states)
            policy_loss = policy_cross_entropy(logits, policy_targets, policy_masks)
            value_loss = nn.functional.mse_loss(values, value_targets)
            loss = policy_loss + arguments.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % arguments.report_every == 0 or epoch == arguments.epochs:
            current = metrics(
                model,
                validation,
                device,
                arguments.batch_size,
                value_weight=arguments.value_weight,
            )
            score = current["policy_kl"] + arguments.value_weight * current["value_mae"]
            if score < best_score:
                best_epoch = epoch
                best = current
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            print(
                f"epoch={epoch:4d} policy_kl={current['policy_kl']:.5f} "
                f"top1={current['policy_top1']:.3f} value_mae={current['value_mae']:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )
            model.train()

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    final_training = metrics(
        model,
        training,
        device,
        arguments.batch_size,
        value_weight=arguments.value_weight,
    )
    final_validation = metrics(
        model,
        validation,
        device,
        arguments.batch_size,
        value_weight=arguments.value_weight,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "vgo.raster-policy-value.v1",
        "channels": dataset_channels,
        "height": dataset_height,
        "width": dataset_width,
        "policy_resolution": policy_resolution,
        "model_width": model_width,
        "blocks": blocks,
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "parent_checkpoint": parent_checkpoint,
        "replay_sources": list(dataset_sources),
        "policy_target": policy_target_name,
        "policy_denominator": "full_legal_raster_v1",
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        checkpoint,
        temporary,
    )
    temporary.replace(output)
    report = {
        "schema": "vgo.training-run.v2",
        "datasets": list(dataset_sources),
        "checkpoint": str(output),
        "parent_checkpoint": parent_checkpoint,
        "device": str(device),
        "samples": dataset_samples,
        "training_samples": training.samples,
        "validation_samples": validation_samples,
        "shape": list(dataset_shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "value_weight": arguments.value_weight,
        "policy_target": policy_target_name,
        "policy_denominator": "full_legal_raster_v1",
        "importance_corrected_samples": corrected_samples,
        "uncorrected_samples": dataset_samples - corrected_samples,
        "selection_metric": "policy_kl + value_weight * value_mae",
        "wall_seconds": elapsed,
        "best_epoch": best_epoch,
        "initial_training": initial_training,
        "initial_validation": initial_validation,
        "best_validation": best,
        "final_training": final_training,
        "final_validation": final_validation,
    }
    report_path = output.with_suffix(output.suffix + ".json")
    atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
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
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--architecture", choices=("flat", "unet"), default="flat")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
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
