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

A run may carry its own stride as `run:stride`, so the arm under test can be
rated at every checkpoint while the ladder it is measured against stays sparse:

    scripts/dense-curve.py artifacts/ddrnet-fresh-attn:3 artifacts/shard-sweep-15000:1 \\
        --ratings ratings.json --output artifacts/dense-curve-6
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


def exported_maximum_batch(onnx: Path) -> int:
    """Read the dynamic batch ceiling recorded beside an exported model."""
    manifest = onnx.with_suffix(".onnx.json")
    try:
        report = json.loads(manifest.read_text(encoding="utf-8"))
        maximum_batch = int(report["input"]["maximum_batch"])
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(
            f"cannot determine ONNX maximum batch from {manifest}; "
            "re-export the checkpoint so it has a valid manifest"
        ) from error
    if maximum_batch < 1:
        raise SystemExit(
            f"invalid ONNX maximum batch {maximum_batch} in {manifest}"
        )
    return maximum_batch


def round_maximum_batch(group, exported_batches, requested: int) -> int:
    """Serving batch supported by every model participating in one round."""
    return min(
        [requested]
        + [
            exported_batches[onnx]
            for _, _, onnx in group
            if onnx is not None
        ]
    )


def seeded_ratings(pool, ratings):
    """Fill unrated checkpoints from their own run's rated neighbours.

    Banding looks up `ratings[run/version]`, and anything missing falls back to
    0.0 -- which is naive's rating, the bottom of the field. That is the worst
    available guess for the arm under test: an unrated checkpoint is unrated
    because it is *new*, and new checkpoints are a run's strongest. Seeding them
    at naive puts the whole point of the tournament into 8-0 pairings, which is
    the exact failure banding exists to prevent.

    Within one run adjacent checkpoints are close in strength, so a rated
    neighbour is a good prior: interpolate between the two nearest rated
    versions, hold flat beyond the last one. Extrapolating the trend instead
    would guess higher and is probably closer to the truth, but a prior that
    overshoots pairs a checkpoint above the field it can actually learn from,
    and flat is the conservative error.

    This only steers matchmaking. The fit afterwards reads game outcomes alone,
    so a wrong prior costs information, never accuracy.
    """
    seeded = {}
    by_run = {}
    for name, version, _ in pool:
        by_run.setdefault(name, []).append(version)
    for name, versions in by_run.items():
        known = sorted((v, ratings[f"{name}/{v}"])
                       for v in set(versions) if f"{name}/{v}" in ratings)
        if not known:
            continue
        for version in sorted(set(versions)):
            if f"{name}/{version}" in ratings:
                continue
            below = [k for k in known if k[0] <= version]
            above = [k for k in known if k[0] >= version]
            if below and above:
                (low, low_rating), (high, high_rating) = below[-1], above[0]
                span = high - low
                weight = 0.0 if span == 0 else (version - low) / span
                seeded[f"{name}/{version}"] = (
                    low_rating + weight * (high_rating - low_rating)
                )
            else:
                seeded[f"{name}/{version}"] = (below or above)[-1 if below else 0][1]
    return seeded


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


def parse_run(spec: str, default: int) -> tuple[Path, int]:
    """`path` or `path:stride`, defaulting to --stride.

    The run under test and the ladder it is measured against want different
    densities. A new arm needs every checkpoint -- its shape is the question --
    while the established run is only there to span the strength range and
    connect the graph, so rating it densely spends games re-deriving a curve
    that is already known. One global stride forces the two together and makes
    the field several times larger than the question needs.
    """
    head, separator, tail = spec.rpartition(":")
    if separator and head and tail.isdigit():
        return Path(head), int(tail)
    return Path(spec), default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+",
                        help="run directory, or run:stride to override "
                             "--stride for that run alone")
    parser.add_argument("--stride", type=int, default=2,
                        help="rate every Nth checkpoint, for runs that do not "
                             "carry their own")
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
    parser.add_argument("--maximum-batch", type=int, default=64,
                        help="desired inference batch ceiling; each round is "
                             "automatically clamped to the smallest maximum "
                             "supported by its ONNX models")
    parser.add_argument("--parallel-rounds", type=int, default=1,
                        help="rounds to play at once. One round cannot keep "
                             "the GPU busy: each model gets its own broker, "
                             "which blocks on an empty queue between batches, "
                             "so a round of 8 models and 112 games averages "
                             "~9 positions against a 64 batch. Concurrent "
                             "rounds give the card independent request "
                             "streams without changing the schedule.")
    parser.add_argument("--maximum-plies", type=int, default=105)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.maximum_batch < 1:
        parser.error("--maximum-batch must be positive")

    root = Path(__file__).resolve().parents[1]
    selected = [parse_run(spec, arguments.stride) for spec in arguments.runs]
    pool = []
    for run, stride in selected:
        for version, onnx in checkpoints(run, stride):
            pool.append((run.name, version, onnx))
    # Export maximums describe what a model *can* accept, not what TensorRT
    # will allocate or execute. Keep the requested serving batch independent,
    # then clamp mixed old/new rounds to their weakest artifact. This lets an
    # interrupted curve resume across batch-32 and batch-64 checkpoints.
    exported_batches = {
        onnx: exported_maximum_batch(onnx)
        for _, _, onnx in pool
        if onnx is not None
    }
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
        seeded = seeded_ratings(pool, ratings)
        if seeded:
            print(f"seeded   : {len(seeded)} unrated checkpoints placed from "
                  "their run's rated neighbours")
        ratings = {**seeded, **ratings}
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
    print(f"pool     : {len(pool)} checkpoints over {len(selected)} runs")
    for run, stride in selected:
        taken = sum(1 for name, _, _ in pool if name == run.name)
        print(f"           {run.name}: {taken} at stride {stride}")
    print(f"schedule : {len(schedule)} rounds, {len(with_naive)} with naive, "
          f"{total} games")
    export_ceiling = min(exported_batches.values(),
                         default=arguments.maximum_batch)
    if export_ceiling < arguments.maximum_batch:
        print(f"batch    : requested {arguments.maximum_batch}; rounds containing "
              f"older exports clamp as low as {export_ceiling}")
    else:
        print(f"batch    : {arguments.maximum_batch}")
    # Measured on an idle box: 0.165 games/s at 800 simulations, 0.073 at 1600
    # while contending with a training run. Scale by the simulation count.
    rate = 0.165 * (800 / max(arguments.simulations, 1))
    print(f"estimate : {total / rate / 3600:.1f}h at ~{rate:.3f} games/s "
          f"({arguments.simulations} simulations, one round at a time)")
    if arguments.parallel_rounds > 1:
        # Deliberately not divided by the lane count. The rate above is a
        # single round's, and the whole reason for running rounds concurrently
        # is that one round leaves the card idle -- so the speedup is whatever
        # of that idle time the extra rounds reclaim, which is measured, not
        # predicted. Quoting total/rate/lanes here would invent a number.
        print(f"           {arguments.parallel_rounds} rounds in flight; "
              "real rate to be measured against that baseline")
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

    def build(index: int) -> tuple[list[str], str, int]:
        group = schedule[index]
        maximum_batch = round_maximum_batch(
            group, exported_batches, arguments.maximum_batch
        )
        command = [str(root / "target/release/vgo-tournament"),
                   "--pairs", str(arguments.pairs),
                   "--concurrency", str(arguments.concurrency),
                   "--simulations", str(arguments.simulations),
                   "--maximum-plies", str(arguments.maximum_plies),
                   "--coarse-pool", "16", "--leaf-batch", "4",
                   "--maximum-batch", str(maximum_batch), "--delay-ms", "1",
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
        return command, names, maximum_batch

    # Rounds run concurrently but their records are appended by this process,
    # one finished round at a time. A record is a pretty-printed JSON object
    # spanning many lines, so two tournaments sharing the append handle would
    # interleave halves of two objects and corrupt the file for every reader.
    # Each round therefore writes to its own file and is spliced in on exit.
    running: dict[int, tuple[subprocess.Popen, Path, object]] = {}
    queued = [index for index in range(len(schedule)) if index not in done]
    lanes = max(1, arguments.parallel_rounds)

    def launch(index: int) -> None:
        command, names, maximum_batch = build(index)
        partial = arguments.output / f"round-{index:03d}.partial.jsonl"
        stream = partial.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=environment, stdout=stream)
        running[index] = (process, partial, stream)
        elapsed = (time.time() - started) / 60
        print(f"\n[{index + 1}/{len(schedule)}] {names}"
              f"   (batch {maximum_batch}, {elapsed:.0f} min elapsed, "
              f"{len(running)} rounds in flight)",
              flush=True)

    def harvest(index: int) -> None:
        process, partial, stream = running.pop(index)
        stream.close()
        if process.returncode != 0:
            # Do not splice a partial round in. Its pairings would be replayed
            # on the next resume and counted twice, which biases the fit toward
            # whatever the interrupted round happened to finish.
            kept = partial.with_name(f"round-{index:03d}.failed.jsonl")
            partial.replace(kept)
            print(f"  round {index + 1} failed ({process.returncode}); "
                  f"kept {kept.name}, will replay on resume", flush=True)
            return
        with records.open("a", encoding="utf-8") as sink:
            sink.write(partial.read_text(encoding="utf-8"))
        partial.unlink()
        done.add(index)
        done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
        print(f"  round {index + 1} complete ({len(done)}/{len(schedule)})",
              flush=True)

    while queued or running:
        while queued and len(running) < lanes:
            launch(queued.pop(0))
        finished = None
        while finished is None:
            for index, (process, _, _) in list(running.items()):
                if process.poll() is not None:
                    finished = index
                    break
            if finished is None:
                time.sleep(1.0)
        harvest(finished)

    print(f"\n-> {records}")
    print(f"   scripts/rate-tournament.py {records} --json {arguments.output}/elo.json")


if __name__ == "__main__":
    main()
