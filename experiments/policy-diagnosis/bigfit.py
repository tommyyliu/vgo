"""Fit the policy head on 2048 positions and report train vs held-out KL.

32 positions overfit to KL ~1e-4, and the RL loop plateaus at ~1.79 on ~12k.
This sits in between. Train KL low with held-out KL pinned near 1.79 means the
net memorises but cannot generalise; both high means it cannot even fit at this
size, which would point at capacity or optimisation instead.
"""

import sys
import time
from pathlib import Path

# Derived, not hardcoded: this file lives at <root>/experiments/policy-diagnosis/.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

import torch

from vgo_training.dataset import load_datasets
from vgo_training.model import build_model
from vgo_training.train_demo import (
    full_legal_policy_masks,
    importance_corrected_policy_targets,
    policy_cross_entropy,
)

TRAIN = 2048
HOLDOUT = 512
EPOCHS = 60
BATCH = 64


def prepare(dataset, start, stop, device):
    states = dataset.states[start:stop]
    targets = importance_corrected_policy_targets(
        dataset.visits[start:stop],
        dataset.betas[start:stop],
        dataset.proposal_counts[start:stop],
        dataset.policy_masks[start:stop],
    )
    masks = full_legal_policy_masks(states, dataset.policy_masks[start:stop])
    entropy = float(-(targets.clamp(min=1e-12).log() * targets).sum(1).mean())
    return states.to(device), targets.to(device), masks.to(device), entropy


@torch.no_grad()
def evaluate(model, states, targets, masks, entropy):
    model.eval()
    total = 0.0
    hits = 0
    for start in range(0, states.shape[0], BATCH):
        stop = min(start + BATCH, states.shape[0])
        logits, _ = model(states[start:stop])
        total += float(
            policy_cross_entropy(logits, targets[start:stop], masks[start:stop])
        ) * (stop - start)
        predicted = logits.masked_fill(~masks[start:stop], float("-inf")).argmax(1)
        hits += int((predicted == targets[start:stop].argmax(1)).sum())
    model.train()
    return total / states.shape[0] - entropy, hits / states.shape[0]


def main() -> None:
    shards = [
        ROOT / f"artifacts/decoupled-run2/iteration-00{i}/replay/dataset.vgo"
        for i in range(3)
    ]
    device = torch.device("cuda")
    dataset = load_datasets([s for s in shards if s.exists()])
    print(f"loaded {dataset.samples} positions")

    states, targets, masks, entropy = prepare(dataset, 0, TRAIN, device)
    validation = prepare(dataset, dataset.samples - HOLDOUT, dataset.samples, device)
    print(f"train {TRAIN}  holdout {HOLDOUT}  target_entropy {entropy:.4f}")

    cells = targets.shape[1] - 1
    side = int(round(cells**0.5))
    model = build_model(
        "unet", channels=states.shape[1], width=64, blocks=8, policy_resolution=side
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"{'epoch':>6} {'train_kl':>10} {'val_kl':>10} {'val_top1':>9} {'secs':>6}")
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        permutation = torch.randperm(TRAIN, device=device)
        for start in range(0, TRAIN, BATCH):
            index = permutation[start : start + BATCH]
            logits, _ = model(states[index])
            loss = policy_cross_entropy(logits, targets[index], masks[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            train_kl, _ = evaluate(model, states, targets, masks, entropy)
            val_kl, val_top1 = evaluate(model, *validation)
            print(
                f"{epoch:>6} {train_kl:>10.4f} {val_kl:>10.4f} {val_top1:>9.3f} "
                f"{time.perf_counter() - started:>6.0f}"
            )


if __name__ == "__main__":
    main()
