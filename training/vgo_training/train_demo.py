from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .dataset import RasterDataset, load_datasets
from .model import build_model
from .serve import load_model


def policy_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def metrics(
    model: nn.Module, dataset: RasterDataset, device: torch.device, batch_size: int
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
    return {
        "loss": cross_entropy + squared_error_total / total,
        "policy_cross_entropy": cross_entropy,
        "policy_target_entropy": target_entropy,
        "policy_kl": cross_entropy - target_entropy,
        "policy_top1": top1_total / total,
        "value_mae": absolute_error_total / total,
    }


def subset(dataset: RasterDataset, indices: torch.Tensor) -> RasterDataset:
    return RasterDataset(
        states=dataset.states[indices],
        policies=dataset.policies[indices],
        policy_masks=dataset.policy_masks[indices],
        values=dataset.values[indices],
        selected_actions=dataset.selected_actions[indices],
        game_ids=dataset.game_ids[indices],
        plies=dataset.plies[indices],
        seeds=dataset.seeds[indices],
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
    dataset = load_datasets(arguments.datasets)
    generator = torch.Generator().manual_seed(arguments.seed)
    split = torch.randperm(dataset.samples, generator=generator)
    validation_samples = 0
    if dataset.samples > 1 and arguments.validation_fraction > 0.0:
        validation_samples = max(1, round(dataset.samples * arguments.validation_fraction))
        validation_samples = min(validation_samples, dataset.samples - 1)
    validation_indices = split[:validation_samples]
    training_indices = split[validation_samples:]
    training = subset(dataset, training_indices)
    validation = subset(dataset, validation_indices) if validation_samples else training

    parent_checkpoint = None
    if arguments.initial_checkpoint is not None:
        model, parent = load_model(arguments.initial_checkpoint.resolve(strict=True))
        if (
            int(parent["channels"]),
            int(parent["height"]),
            int(parent["width"]),
        ) != (dataset.channels, dataset.height, dataset.width):
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
            channels=dataset.channels,
            width=model_width,
            blocks=blocks,
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=arguments.epochs,
        eta_min=arguments.learning_rate * 0.01,
    )
    initial_training = metrics(model, training, device, arguments.batch_size)
    initial_validation = metrics(model, validation, device, arguments.batch_size)
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
            logits, values = model(states)
            policy_loss = policy_cross_entropy(logits, policy_targets, policy_masks)
            value_loss = nn.functional.mse_loss(values, value_targets)
            loss = policy_loss + arguments.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % arguments.report_every == 0 or epoch == arguments.epochs:
            current = metrics(model, validation, device, arguments.batch_size)
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
    final_training = metrics(model, training, device, arguments.batch_size)
    final_validation = metrics(model, validation, device, arguments.batch_size)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "vgo.raster-policy-value.v1",
        "channels": dataset.channels,
        "height": dataset.height,
        "width": dataset.width,
        "model_width": model_width,
        "blocks": blocks,
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "parent_checkpoint": parent_checkpoint,
        "replay_sources": list(dataset.sources),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        checkpoint,
        temporary,
    )
    temporary.replace(output)
    report = {
        "schema": "vgo.training-run.v2",
        "datasets": list(dataset.sources),
        "checkpoint": str(output),
        "parent_checkpoint": parent_checkpoint,
        "device": str(device),
        "samples": dataset.samples,
        "training_samples": training.samples,
        "validation_samples": validation_samples,
        "shape": list(dataset.states.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "value_weight": arguments.value_weight,
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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_arguments())
