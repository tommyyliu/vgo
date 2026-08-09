#!/usr/bin/env python3
"""Rate two runs on one Elo scale, and print a curve for each.

Two runs rated separately are not comparable. Each fit is anchored inside its
own field, and the only shared opponent -- the naive evaluator -- stops
discriminating as soon as a run outgrows it: five checkpoints of the Adam run
scored a perfect 1.000 against naive, and an undefeated record has no finite
maximum-likelihood rating, so those numbers came from the prior rather than
from evidence.

Playing the runs against each other fixes that. Checkpoints from both go into
one field, `fit_ratings` sees a single connected graph, and the resulting
ratings sit on one scale by construction.

Version ids are offset per run (run 0 keeps its own numbers, run 1 is shifted
by 1000) because `fit_ratings` keys on integers and the two runs both start at
zero.

    scripts/joint-arena.py artifacts/ddrnet-fresh-attn artifacts/ddrnet-fresh-muon
    scripts/joint-arena.py A B --points 6 --pairs 8 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.bradley_terry import fit_ratings  # noqa: E402
from vgo_training.pipeline import runtime_environment  # noqa: E402

OFFSET = 1000


def checkpoints(run: Path) -> dict[int, Path]:
    found = {}
    for directory in sorted((run / "updates").glob("update-*")):
        onnx = directory / "candidate.onnx"
        if onnx.exists():
            found[int(directory.name.split("-")[1])] = onnx
    return found


def spread(versions: list[int], count: int) -> list[int]:
    """`count` versions across a run, endpoints always included."""
    if count >= len(versions):
        return versions
    if count <= 1:
        return versions[-1:]
    step = (len(versions) - 1) / (count - 1)
    picked = {versions[round(i * step)] for i in range(count)}
    picked.update({versions[0], versions[-1]})
    return sorted(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=2)
    parser.add_argument("--points", type=int, default=6,
                        help="checkpoints sampled per run")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("artifacts/joint-arena"))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (arguments.runs[0] / "pipeline-config.json").read_text(encoding="utf-8")
    )

    field: dict[int, tuple[Path, str]] = {}
    labels: dict[int, str] = {}
    for index, run in enumerate(arguments.runs):
        available = checkpoints(run)
        if not available:
            raise SystemExit(f"no checkpoints under {run}")
        for version in spread(sorted(available), arguments.points):
            key = version + index * OFFSET
            field[key] = (available[version], run.name)
            labels[key] = f"{run.name}:v{version}"

    keys = sorted(field)
    pairings = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    print(f"field    : {len(keys)} checkpoints over {len(arguments.runs)} runs")
    print(f"matches  : {len(pairings)} pairings x {arguments.pairs} pairs "
          f"= {len(pairings) * arguments.pairs * 2} games")

    # One process per candidate, opponents batched: a model load is ~21s
    # against ~1s per additional pair.
    grouped: dict[int, list[int]] = {}
    for a, b in pairings:
        grouped.setdefault(a, []).append(b)

    if arguments.dry_run:
        for candidate, opponents in sorted(grouped.items()):
            print(f"  {labels[candidate]} vs {', '.join(labels[o] for o in opponents)}")
        return

    arguments.output.mkdir(parents=True, exist_ok=True)
    matches_path = arguments.output / "matches.jsonl"
    matches: list[dict] = []
    if matches_path.exists():
        matches = [json.loads(line) for line in
                   matches_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"resuming : {len(matches)} existing records")
    already = {frozenset((m["a"], m["b"])) for m in matches}

    environment = runtime_environment()
    for candidate, opponents in sorted(grouped.items()):
        todo = [o for o in opponents if frozenset((candidate, o)) not in already]
        if not todo:
            continue
        command = [
            str(root / "target/release/vgo-arena"),
            "--candidate", str(field[candidate][0]),
            "--pairs", str(arguments.pairs),
            "--simulations", str(arguments.simulations),
            "--coarse-pool", str(int(config.get("coarse_pool", 16))),
            "--candidate-raster-kind", str(config.get("raster_kind", "compact")),
            "--max-plies", str(int(1.5 * int(config.get("maximum_plies", 70)))),
            "--threads", str(int(config.get("arena_actors", 64))),
            "--leaf-batch", str(int(config.get("leaf_batch", 4))),
            "--resolution", str(int(config.get("resolution", 128))),
            "--policy-resolution", str(int(config.get("policy_resolution", 128))),
            "--radius", str(config.get("radius", 0.055714285714285716)),
            "--komi", str((float(config.get("komi_low", 0.0))
                           + float(config.get("komi_high", 0.0))) / 2.0),
            "--seed", str(60000 + candidate),
            "--maximum-batch", "64",
            "--delay-ms", "1",
            "--provider", "tensorrt",
            "--fp16", "true",
            "--cache-directory", str(root / "artifacts/onnx-cache"),
        ]
        for opponent in todo:
            command += ["--opponent", str(field[opponent][0])]
        print(f"\n{labels[candidate]} vs {', '.join(labels[o] for o in todo)} ...",
              flush=True)
        completed = subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            print(f"  FAILED ({completed.returncode})")
            for line in completed.stderr.strip().splitlines()[-3:]:
                print(f"    {line}")
            continue
        records, depth, start = [], 0, None
        for position, character in enumerate(completed.stdout):
            if character == "{":
                if depth == 0:
                    start = position
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        value = json.loads(completed.stdout[start:position + 1])
                    except json.JSONDecodeError:
                        continue
                    if value.get("schema") == "vgo.arena.v1":
                        records.append(value)
        for opponent, record in zip(todo, records):
            if int(record["completed"]) == 0:
                print(f"  vs {labels[opponent]}: nothing decided; not recorded")
                continue
            match = {"a": candidate, "b": opponent,
                     "a_wins": int(record["candidate_wins"]),
                     "b_wins": int(record["candidate_losses"]),
                     "draws": int(record["draws"]),
                     "games": int(record["games"])}
            matches.append(match)
            with matches_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(match) + "\n")
            print(f"  vs {labels[opponent]}: {match['a_wins']}-{match['b_wins']}"
                  f"-{match['draws']}")

    if not matches:
        raise SystemExit("no matches played")
    ratings = fit_ratings(matches, anchor=min(labels), prior_games=0.25)
    (arguments.output / "ratings.json").write_text(
        json.dumps({labels.get(k, str(k)): v for k, v in sorted(ratings.items())},
                   indent=2) + "\n", encoding="utf-8")

    for run_index, run in enumerate(arguments.runs):
        print(f"\n{run.name}")
        print(f"{'update':>8}{'elo':>9}")
        for key in sorted(ratings):
            if key // OFFSET != run_index:
                continue
            print(f"{key - run_index * OFFSET:>8}{ratings[key]:>9.0f}")
    print(f"\n-> {arguments.output}/ratings.json")


if __name__ == "__main__":
    main()
