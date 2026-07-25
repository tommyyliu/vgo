"""Can the policy head fit a small fixed set of replay targets at all?

The RL loop shows policy_kl flat to three decimals over 80 epochs while the
value head, on the same batches and gradients, improves 35%. Two explanations
fit that: the target is hard to learn (needs more capacity/steps), or the target
is *inconsistent* -- the same board mapping to contradictory targets across
games, in which case no amount of training helps and the mean is the best any
model can do.

Overfitting separates them. Train on a handful of positions with no validation,
no augmentation, and no value loss, and see whether KL goes to zero. A network
with 2.8M parameters memorising 32 positions should drive KL to ~0 if each input
has one consistent target. If KL plateaus well above zero, the targets for
distinct positions are mutually contradictory and the ceiling is intrinsic.

The duplicate-board control makes that precise: it reports how much of the
plateau is explained by identical boards carrying different targets.
"""

import sys
from pathlib import Path

ROOT = Path("/home/tommy/PycharmProjects/vgo")
sys.path.insert(0, str(ROOT / "training"))

import torch
from torch import nn

from vgo_training.dataset import load_dataset
from vgo_training.model import build_model
from vgo_training.train_demo import (
    full_legal_policy_masks,
    importance_corrected_policy_targets,
    policy_cross_entropy,
)


def main() -> None:
    shard = ROOT / "artifacts/decoupled-run2/iteration-001/replay/dataset.vgo"
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    device = torch.device("cuda")

    dataset = load_dataset(shard)
    states = dataset.states[:samples].to(device)
    targets = importance_corrected_policy_targets(
        dataset.visits[:samples],
        dataset.betas[:samples],
        dataset.proposal_counts[:samples],
        dataset.policy_masks[:samples],
    ).to(device)
    masks = full_legal_policy_masks(
        dataset.states[:samples], dataset.policy_masks[:samples]
    ).to(device)

    # How much of any plateau is forced by identical inputs with different
    # targets? Hash each raster; boards colliding here cannot both be fit.
    keys = [hash(states[i].cpu().numpy().tobytes()) for i in range(samples)]
    duplicates = samples - len(set(keys))

    placement_cells = targets.shape[1] - 1
    side = int(round(placement_cells**0.5))
    model = build_model(
        "unet",
        channels=states.shape[1],
        width=64,
        blocks=8,
        policy_resolution=side if side * side == placement_cells else None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    entropy = float(-(targets.clamp(min=1e-12).log() * targets).sum(1).mean())
    print(f"samples={samples} duplicate_boards={duplicates} target_entropy={entropy:.4f}")
    print(f"policy grid {side}x{side}, legal cells/row {float(masks.sum(1).float().mean()):.1f}")
    print("A consistent target should drive KL toward 0; a plateau is the floor.")
    print()

    model.train()
    for step in range(1, steps + 1):
        logits, _ = model(states)
        loss = policy_cross_entropy(logits, targets, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0:
            with torch.no_grad():
                kl = float(loss) - entropy
                top1 = float(
                    (
                        logits.masked_fill(~masks, float("-inf")).argmax(1)
                        == targets.argmax(1)
                    )
                    .float()
                    .mean()
                )
            print(f"step={step:5d} cross_entropy={float(loss):.5f} kl={kl:.5f} top1={top1:.3f}")


if __name__ == "__main__":
    main()
