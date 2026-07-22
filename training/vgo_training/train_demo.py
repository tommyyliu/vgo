from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .dataset import RasterDataset, load_dataset
from .model import RasterPolicyValueNet


def policy_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def metrics(
    model: nn.Module, dataset: RasterDataset, device: torch.device
) -> dict[str, float]:
    model.eval()
    states = dataset.states.to(device)
    targets = dataset.policies.to(device)
    masks = dataset.policy_masks.to(device)
    values = dataset.values.to(device)
    logits, predictions = model(states)
    cross_entropy = policy_cross_entropy(logits, targets, masks)
    target_entropy = -(targets * targets.clamp_min(1e-12).log()).sum(dim=1).mean()
    return {
        "loss": float(cross_entropy + nn.functional.mse_loss(predictions, values)),
        "policy_cross_entropy": float(cross_entropy),
        "policy_target_entropy": float(target_entropy),
        "policy_kl": float(cross_entropy - target_entropy),
        "policy_top1": float(
            (
                logits.masked_fill(~masks, torch.finfo(logits.dtype).min).argmax(dim=1)
                == targets.argmax(dim=1)
            )
            .float()
            .mean()
        ),
        "value_mae": float((predictions - values).abs().mean()),
    }


def train(arguments: argparse.Namespace) -> dict[str, object]:
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    device = torch.device(arguments.device)
    dataset = load_dataset(arguments.dataset)
    model = RasterPolicyValueNet(
        channels=dataset.channels,
        width=arguments.model_width,
        blocks=arguments.blocks,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=arguments.epochs,
        eta_min=arguments.learning_rate * 0.01,
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    initial = metrics(model, dataset, device)
    best_epoch = 0
    best = initial
    best_score = initial["policy_kl"] + initial["value_mae"]
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    started = time.perf_counter()

    model.train()
    for epoch in range(1, arguments.epochs + 1):
        permutation = torch.randperm(dataset.samples, generator=generator)
        for start in range(0, dataset.samples, arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            states = dataset.states[indices].to(device)
            policy_targets = dataset.policies[indices].to(device)
            policy_masks = dataset.policy_masks[indices].to(device)
            value_targets = dataset.values[indices].to(device)
            logits, values = model(states)
            policy_loss = policy_cross_entropy(logits, policy_targets, policy_masks)
            value_loss = nn.functional.mse_loss(values, value_targets)
            loss = policy_loss + arguments.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % arguments.report_every == 0 or epoch == arguments.epochs:
            current = metrics(model, dataset, device)
            score = current["policy_kl"] + current["value_mae"]
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
    final = metrics(model, dataset, device)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "vgo.raster-policy-value.v1",
            "channels": dataset.channels,
            "height": dataset.height,
            "width": dataset.width,
            "model_width": arguments.model_width,
            "blocks": arguments.blocks,
            "state_dict": model.state_dict(),
        },
        output,
    )
    report = {
        "schema": "vgo.training-canary.v1",
        "dataset": str(arguments.dataset),
        "checkpoint": str(output),
        "device": str(device),
        "samples": dataset.samples,
        "shape": list(dataset.states.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "wall_seconds": elapsed,
        "best_epoch": best_epoch,
        "initial": initial,
        "best": best,
        "final": final,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("../artifacts/raster-demo/model.pt"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_arguments())
