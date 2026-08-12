#!/usr/bin/env python3
"""Rate many checkpoints densely, via short random round-robins.

A complete round-robin does not scale: 80 checkpoints is 3160 pairings. But a
dense *curve* does not need every pair played -- it needs every checkpoint
connected to the field with enough games to place it, and shape read across
many points beats precision at a few. Forty points at +/-100 Elo says more
about whether improvement is decaying than six points at +/-60.

So the field is sampled instead. Each round loads a handful of checkpoints,
plays a complete round-robin among them, and appends its records; over many
rounds every checkpoint meets a different random subset, and one
Bradley-Terry fit ties them together.

Naive plays in a few rounds, not all of them. It is the one player whose
strength does not depend on any run, so anchoring on it puts ratings on an
absolute scale that survives across tournaments -- but it loses nearly every
game to a trained checkpoint, and a pairing decided 8-0 carries almost no
information. In every round it would take 22% of the games for that. It is
scheduled into the rounds holding the *earliest* checkpoints instead, which
are the ones it can still take games from and therefore the only ones it
measures anything against.

Connectivity does not need it: sampled round-robins over a shuffled pool come
out connected on their own (checked at 42 checkpoints, one component).

    scripts/dense-curve.py artifacts/ddrnet-fresh-attn artifacts/ddrnet-fresh-muon \\
        --stride 2 --rounds-per-checkpoint 5 --field 8 --output artifacts/dense-curve
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.pipeline import runtime_environment  # noqa: E402


def checkpoints(run: Path, stride: int) -> list[tuple[int, Path]]:
    """(update, onnx) for every `stride`-th checkpoint, endpoints included."""
    found = []
    for directory in sorted((run / "updates").glob("update-*")):
        onnx = directory / "candidate.onnx"
        if onnx.exists():
            found.append((int(directory.name.split("-")[1]), onnx))
    if not found:
        raise SystemExit(f"no checkpoints under {run}")
    picked = [c for c in found if c[0] % stride == 0]
    if found[-1] not in picked:
        picked.append(found[-1])
    return picked


def banded_rounds(pool, ratings, per_checkpoint, field, band, spanning_every, rng):
    """Rounds drawn from a rating neighbourhood, with periodic spanning rounds.

    Bradley-Terry information per game is proportional to p(1-p), so a game
    between evenly matched players is worth several times one decided 97-3.
    Measured on this field, random pairing averages 0.111 while banding to eight
    neighbours averages 0.241 -- 2.2x the information for the same games, with
    47% of random pairings currently differing by more than 400 Elo.

    Banding alone is not enough. Matching only near neighbours turns the
    comparison graph into a chain: every link is precise, but comparing the ends
    accumulates error through every link between, so long-range comparisons get
    worse. Every `spanning_every`-th round is therefore drawn stratified across
    the whole range -- poor information per game, but these are the long
    baselines that hold the scale together.

    Pairing on estimated rating does not bias the fit: the maximum-likelihood
    estimate stays unbiased as long as pairing depends on prior estimates rather
    than on outcomes. It changes variance, not location.
    """
    def rating(entry):
        return ratings.get(f"{entry[0]}/{entry[1]}", 0.0)

    order = sorted(pool, key=rating)
    total = max(1, -(-len(pool) * per_checkpoint // field))
    counts = {id(p): 0 for p in pool}
    out = []
    for index in range(total):
        if spanning_every and index % spanning_every == 0:
            # One from each stratum, so the round spans the whole rating range.
            step = max(1, len(order) // field)
            group = [rng.choice(order[i:i + step]) for i in range(0, len(order), step)][:field]
        else:
            # Seed on the least-played checkpoint so coverage stays even, then
            # fill from its neighbourhood.
            seed = min(pool, key=lambda p: (counts[id(p)], rng.random()))
            centre = order.index(seed)
            low = max(0, min(centre - band // 2, len(order) - band))
            window = order[low:low + band]
            group = rng.sample(window, min(field, len(window)))
            if seed not in group:
                group[0] = seed
        for entry in group:
            counts[id(entry)] += 1
        out.append(group)
    return out


def rounds(pool: list, per_checkpoint: int, field: int, rng: random.Random):
    """Chunk `per_checkpoint` shuffled copies of the pool into rounds.

    Repeating the pool and shuffling keeps every checkpoint in the same number
    of rounds, so no part of the curve is rated more precisely than another for
    no reason. Rounds are drawn without replacement so a checkpoint never plays
    itself.
    """
    slots = []
    for _ in range(per_checkpoint):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        slots.extend(shuffled)
    out, current = [], []
    for slot in slots:
        if slot in current:
            continue
        current.append(slot)
        if len(current) == field:
            out.append(current)
            current = []
    if len(current) >= 2:
        out.append(current)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--stride", type=int, default=2,
                        help="rate every Nth checkpoint")
    parser.add_argument("--rounds-per-checkpoint", type=int, default=5)
    parser.add_argument("--field", type=int, default=8,
                        help="checkpoints per round")
    parser.add_argument("--ratings", type=Path, default=None,
                        help="ratings JSON from build-dense-curve.py; enables "
                             "banded matchmaking")
    parser.add_argument("--band", type=int, default=12,
                        help="neighbours a banded round draws from")
    parser.add_argument("--spanning-every", type=int, default=4,
                        help="every Nth round spans the whole range instead")
    parser.add_argument("--naive-rounds", type=int, default=4,
                        help="uniform matchmaking only: rounds that also include "
                             "naive, chosen from those with the earliest "
                             "checkpoints. Under --ratings naive is banded as an "
                             "ordinary pool member and this is ignored.")
    parser.add_argument("--pairs", type=int, default=1,
                        help="colour-swapped pairs per pairing; each is 2 games")
    parser.add_argument("--simulations", type=int, default=1600)
    parser.add_argument("--concurrency", type=int, default=80)
    parser.add_argument("--maximum-plies", type=int, default=105)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pool = []
    for run in arguments.runs:
        for version, onnx in checkpoints(run, arguments.stride):
            pool.append((run.name, version, onnx))
    rng = random.Random(arguments.seed)
    if arguments.ratings and arguments.ratings.exists():
        ratings = json.loads(arguments.ratings.read_text(encoding="utf-8"))
        # Naive joins the pool rather than being bolted onto whole rounds. It
        # has a rating like anything else, so banding pairs it with the weakest
        # checkpoints -- the only ones it takes games from -- instead of letting
        # it play a full round of eight, most of them foregone. It also
        # self-regulates: as the field improves naive sinks in the ordering and
        # stops being drawn at all.
        naive_is_pooled = "naive/-1" in ratings
        if naive_is_pooled:
            pool.append(("naive", -1, None))
        schedule = banded_rounds(pool, ratings, arguments.rounds_per_checkpoint,
                                 arguments.field, arguments.band,
                                 arguments.spanning_every, rng)
        print(f"matchmaking: banded (window {arguments.band}, "
              f"spanning every {arguments.spanning_every}"
              + (", naive in the pool)" if "naive/-1" in ratings else ")"))
    else:
        naive_is_pooled = False
        schedule = rounds(pool, arguments.rounds_per_checkpoint, arguments.field, rng)
        print("matchmaking: uniform random")

    # Whole-round naive is the fallback for when there is no fit to band on.
    # The two mechanisms are exclusive: bolting naive onto a round *as well*
    # would hand it a second helping of games, most against opponents its own
    # band already ruled out.
    if naive_is_pooled:
        if arguments.naive_rounds:
            print(f"note     : ignoring --naive-rounds {arguments.naive_rounds}; "
                  "naive is banded as a pool member instead")
        with_naive = set()
    else:
        # Naive goes where it is still competitive -- the rounds whose
        # checkpoints are earliest. Against late checkpoints it scores zero,
        # and a pairing with no losses contributes nothing to anyone's rating.
        ranked = sorted(range(len(schedule)),
                        key=lambda i: sum(v for _, v, _ in schedule[i]) / len(schedule[i]))
        with_naive = set(ranked[:max(arguments.naive_rounds, 0)])

    def games_in(index: int) -> int:
        # A naive entry is already counted in the group when it is a pool
        # member; only the whole-round mode adds a player.
        extra = 1 if (index in with_naive
                      and not any(e[0] == "naive" for e in schedule[index])) else 0
        players = len(schedule[index]) + extra
        return players * (players - 1) // 2 * arguments.pairs * 2

    total = sum(games_in(i) for i in range(len(schedule)))
    print(f"pool     : {len(pool)} checkpoints over {len(arguments.runs)} runs "
          f"(stride {arguments.stride})")
    print(f"schedule : {len(schedule)} rounds, {len(with_naive)} with naive, "
          f"{total} games")
    # Measured on an idle box: 0.165 games/s at 800 simulations, 0.073 at 1600
    # while contending with a training run. Scale by the simulation count.
    rate = 0.165 * (800 / max(arguments.simulations, 1))
    print(f"estimate : {total / rate / 3600:.1f}h at ~{rate:.3f} games/s "
          f"({arguments.simulations} simulations)")
    if arguments.dry_run:
        for index, group in enumerate(schedule[:5]):
            print(f"  round {index}: "
                  + ", ".join(f"{r.split('-')[-1]}:v{v}" for r, v, _ in group)
                  + ("  + naive" if index in with_naive else ""))
        if len(schedule) > 5:
            print(f"  ... {len(schedule) - 5} more")
        print("  naive rounds: " + ", ".join(str(i) for i in sorted(with_naive)))
        return

    arguments.output.mkdir(parents=True, exist_ok=True)
    records = arguments.output / "records.jsonl"
    done_path = arguments.output / "rounds-done.json"
    # Resume by round index: the tournament appends, so re-running a completed
    # round would double-count its games rather than overwrite them.
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    if done:
        print(f"resuming : {len(done)} rounds already played")

    environment = runtime_environment()
    started = time.time()
    for index, group in enumerate(schedule):
        if index in done:
            continue
        command = [str(root / "target/release/vgo-tournament"),
                   "--pairs", str(arguments.pairs),
                   "--concurrency", str(arguments.concurrency),
                   "--simulations", str(arguments.simulations),
                   "--maximum-plies", str(arguments.maximum_plies),
                   "--coarse-pool", "16", "--leaf-batch", "4",
                   "--maximum-batch", "64", "--delay-ms", "1",
                   "--resolution", "128", "--policy-resolution", "128",
                   "--radius", "0.055714285714285716", "--komi", "0.034",
                   "--provider", "tensorrt",
                   "--cache-directory", str(root / "artifacts/onnx-cache"),
                   "--seed", str(arguments.seed + index)]
        for _, _, onnx in group:
            if onnx is not None:
                command += ["--model", str(onnx)]
        if any(entry[0] == "naive" for entry in group) or index in with_naive:
            command.append("--include-naive")
        names = ", ".join("naive" if r == "naive" else f"{r.split('-')[-1]}:v{v}"
                          for r, v, _ in group)
        if index in with_naive:
            names += " + naive"
        elapsed = time.time() - started
        print(f"\n[{index + 1}/{len(schedule)}] {names}"
              f"   ({elapsed / 60:.0f} min elapsed)", flush=True)
        with records.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(command, env=environment, stdout=stream)
        if completed.returncode != 0:
            print(f"  round failed ({completed.returncode}); "
                  f"its completed pairings are still on file")
            continue
        done.add(index)
        done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")

    print(f"\n-> {records}")
    print(f"   scripts/rate-tournament.py {records} --json {arguments.output}/elo.json")


if __name__ == "__main__":
    main()
