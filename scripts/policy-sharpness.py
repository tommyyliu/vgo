#!/usr/bin/env python3
"""Compare what two checkpoints predict on identical positions.

Self-play statistics conflate the model with the games it happened to play. To
ask whether one optimizer produces a *sharper* policy than another, the two
models have to see the same board states -- then any difference is theirs.

Reported per checkpoint, over the same sample:

*Policy entropy* in bits, over the move distribution. Lower means the model
concentrates its probability on fewer moves. This is the direct measure of
sharpness; a policy that has collapsed onto a narrow set of replies scores low
here regardless of whether it is any good.

*Top-1 mass* is the probability on the single best move -- the same thing seen
from the other end, and less sensitive to the long tail.

*Value spread* is the standard deviation of predicted outcomes. A model that
has learned to call games early pushes its values toward the extremes.

*Agreement* is how often two models pick the same top move, which says whether
sharper also means different.

    scripts/policy-sharpness.py shard-dir --checkpoint a.pt --checkpoint b.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.dataset import load_dataset  # noqa: E402
from vgo_training.serve import load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path,
                        help="shard directory or dataset.vgo")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    path = arguments.dataset
    if path.is_dir():
        path = path / "dataset.vgo"
    data = load_dataset(path)
    total = min(arguments.samples, data.samples)
    # A fixed stride rather than the first N: consecutive samples are plies of
    # the same game, so the head of a shard is a handful of openings.
    stride = max(data.samples // total, 1)
    rows = torch.arange(0, stride * total, stride)[:total]
    states = torch.as_tensor(data.states[rows.numpy()]).float()

    targets = torch.as_tensor(data.values[rows.numpy()]).float()
    # The search's own visit distribution is the policy target. Its argmax --
    # not `selected_actions`, which is temperature-sampled and so disagrees
    # with the target most of the time -- is what the learner scores top-1
    # against. Comparing both models to it on shared positions measures policy
    # quality on common ground, which each run's own validation curve cannot:
    # those are scored against data the run generated itself.
    policy_target = data.policies[rows]
    chosen = policy_target.argmax(dim=-1).long()

    labels = arguments.label or [c.stem for c in arguments.checkpoint]
    tops: list[torch.Tensor] = []
    print(f"positions : {len(states)} from {path}")
    print(f"targets   : mean |outcome| {targets.abs().mean():.3f}, "
          f"sd {targets.std():.3f}")
    print(f"\n{'checkpoint':<22}{'entropy':>9}{'top1 mass':>11}{'value sd':>10}"
          f"{'value MAE':>11}{'overconf':>10}{'pol top1':>10}{'pol KL':>9}")
    for checkpoint, label in zip(arguments.checkpoint, labels):
        model, _ = load_model(checkpoint.resolve(strict=True))
        model = model.to(arguments.device).eval()
        entropies, top_mass, values, argmax, kls = [], [], [], [], []
        with torch.no_grad():
            for start in range(0, len(states), arguments.batch):
                chunk = states[start:start + arguments.batch].to(arguments.device)
                logits, value = model(chunk)
                probabilities = torch.softmax(logits.float(), dim=-1)
                # Guard the log: a fully saturated softmax produces exact zeros.
                entropy = -(probabilities
                            * torch.log(probabilities.clamp_min(1e-12))).sum(-1)
                entropies.append((entropy / math.log(2)).cpu())
                best = probabilities.max(dim=-1)
                top_mass.append(best.values.cpu())
                argmax.append(best.indices.cpu())
                values.append(value.float().flatten().cpu())
                target = policy_target[start:start + arguments.batch].to(arguments.device)
                kls.append((target * (target.clamp_min(1e-12).log()
                            - probabilities.clamp_min(1e-12).log())).sum(-1).cpu())
        entropy = torch.cat(entropies).mean().item()
        mass = torch.cat(top_mass).mean().item()
        predicted = torch.cat(values)
        spread = predicted.std().item()
        mae = (predicted - targets).abs().mean().item()
        # Overconfidence: among positions the model calls decisively, how often
        # is it actually right? Extreme values are only a fault if unearned --
        # a model that says +0.9 and wins nine times in ten is simply correct.
        confident = predicted.abs() > 0.8
        if confident.any():
            correct = (torch.sign(predicted[confident])
                       == torch.sign(targets[confident])).float().mean().item()
            overconfidence = f"{correct:.3f}"
        else:
            overconfidence = "—"
        picks = torch.cat(argmax)
        tops.append(picks)
        top1 = (picks == chosen).float().mean().item()
        kl = torch.cat(kls).mean().item()
        print(f"{label:<22}{entropy:>9.3f}{mass:>11.4f}{spread:>10.4f}"
              f"{mae:>11.4f}{overconfidence:>10}{top1:>10.4f}{kl:>9.4f}")
        del model
        if arguments.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(tops) == 2:
        agree = (tops[0] == tops[1]).float().mean().item()
        print(f"\ntop-1 agreement between the two: {agree:.3f}")


if __name__ == "__main__":
    main()
