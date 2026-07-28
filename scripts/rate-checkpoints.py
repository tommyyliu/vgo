#!/usr/bin/env python3
"""Rate a subset of a run's checkpoints against a common anchor.

The pipeline queues Elo jobs but only drains them when asked
(`rl_loop.py --drain-telemetry` / `--telemetry-only`), and draining every
queued job costs an arena per update. What the RL loop's rating curve is
actually for is the trend: whether each generation beats the ones before it,
and by how much. A handful of points against a fixed anchor answers that.

This runs outside the pipeline entirely -- no state file, no queue, no resume
semantics. It reads the checkpoints a run has already written and plays them
directly, so it can be run against a finished run, an abandoned one, or a run
still in progress (though not while that run is using the GPU).

Usage:

    scripts/rate-checkpoints.py artifacts/ddrnet-pipe --count 6
    scripts/rate-checkpoints.py artifacts/ddrnet-pipe --versions 0,12,24,36,48
    scripts/rate-checkpoints.py artifacts/ddrnet-pipe --count 6 --dry-run

Writes `<run>/manual-elo/matches.jsonl` and `ratings.json`, and prints the
curve. Re-running extends the match history rather than replacing it, so a
later pass with more checkpoints refines the same fit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from vgo_training.bradley_terry import fit_ratings  # noqa: E402
from vgo_training.pipeline import runtime_environment  # noqa: E402


def discover_checkpoints(run: Path) -> dict[int, Path]:
    """Every published checkpoint, by version."""
    found: dict[int, Path] = {}
    for directory in sorted((run / "updates").glob("update-*")):
        onnx = directory / "candidate.onnx"
        if onnx.exists():
            found[int(directory.name.split("-")[1])] = onnx
    # The serial loop used a different layout; accept it so old runs can be
    # rated with the same tool.
    for directory in sorted(run.glob("iteration-*")):
        onnx = directory / "model" / "candidate.onnx"
        if onnx.exists():
            found.setdefault(int(directory.name.split("-")[1]), onnx)
    return found


def evenly_spaced(versions: list[int], count: int) -> list[int]:
    """`count` versions spread across the run, always including first and last.

    Endpoints matter more than even spacing here: the first is the anchor and
    the last is the result, so both are pinned before the interior is filled.
    """
    if count >= len(versions):
        return versions
    if count <= 1:
        return versions[-1:]
    step = (len(versions) - 1) / (count - 1)
    picked = {versions[round(index * step)] for index in range(count)}
    picked.update({versions[0], versions[-1]})
    return sorted(picked)


def run_config(run: Path) -> dict:
    """Search settings the checkpoints were produced under.

    The arena has to search the way the run did -- resolution, policy grid, and
    radius must match what the model was exported with, and simulations and
    coarse pool decide whether the comparison reflects the run's own play.
    """
    for name in ("pipeline-config.json", "run-config.json"):
        path = run / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(f"no pipeline-config.json or run-config.json under {run}")


def build_command(
    root: Path,
    config: dict,
    candidate: Path,
    opponents: list[Path],
    seed: int,
    pairs: int,
) -> list[str]:
    simulations = int(
        config.get("arena_simulations", config.get("generation_simulations", 512))
    )
    command = [
        str(root / "target/release/vgo-arena"),
        "--candidate", str(candidate),
        "--pairs", str(pairs),
        "--simulations", str(simulations),
        "--coarse-pool", str(int(config.get("coarse_pool", 16))),
        "--max-plies", str(int(config.get("maximum_plies", 256))),
        "--threads", str(int(config.get("arena_actors", 64))),
        "--leaf-batch", str(int(config.get("leaf_batch", 4))),
        "--resolution", str(int(config.get("resolution", 128))),
        "--policy-resolution", str(int(config.get("policy_resolution", 128))),
        "--radius", str(config.get("radius", 0.05555555555555555)),
        "--seed", str(seed),
        "--maximum-batch", str(int(config.get("maximum_batch", 64))),
        "--delay-ms", str(int(config.get("inference_delay_ms", config.get("delay_ms", 1)))),
        "--provider", str(config.get("provider", "tensorrt")),
        "--fp16", "true" if config.get("fp16", True) else "false",
        "--cache-directory", str(root / "artifacts/onnx-cache"),
    ]
    for opponent in opponents:
        command += ["--opponent", str(opponent)]
    return command


def parse_records(stdout: str) -> list[dict]:
    """Arena emits one JSON object per opponent, in the order given."""
    records, depth, start = [], 0, None
    for index, character in enumerate(stdout):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(stdout[start : index + 1])
                except json.JSONDecodeError:
                    continue
                if value.get("schema") == "vgo.arena.v1":
                    records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--count", type=int, default=6,
        help="how many checkpoints to rate, spread across the run",
    )
    parser.add_argument(
        "--versions",
        help="explicit comma-separated versions, overriding --count",
    )
    parser.add_argument(
        "--anchor", type=int,
        help="version every other plays against; defaults to the earliest rated",
    )
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5_000_001)
    parser.add_argument(
        "--prior-games", type=float, default=0.25,
        help="virtual even games regularizing a lopsided record",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    run = arguments.run.resolve()
    root = Path(__file__).resolve().parents[1]
    config = run_config(run)
    available = discover_checkpoints(run)
    if not available:
        raise SystemExit(f"no checkpoints under {run}")

    if arguments.versions:
        chosen = sorted({int(value) for value in arguments.versions.split(",")})
        missing = [value for value in chosen if value not in available]
        if missing:
            raise SystemExit(f"no checkpoint for version(s): {missing}")
    else:
        chosen = evenly_spaced(sorted(available), arguments.count)

    anchor = arguments.anchor if arguments.anchor is not None else chosen[0]
    if anchor not in available:
        raise SystemExit(f"no checkpoint for anchor version {anchor}")
    challengers = [version for version in chosen if version != anchor]
    if not challengers:
        raise SystemExit("need at least one checkpoint besides the anchor")

    print(f"run       : {run}")
    print(f"available : {len(available)} checkpoints (versions {min(available)}-{max(available)})")
    print(f"rating    : {challengers}")
    print(f"anchor    : {anchor}")
    print(f"matches   : {len(challengers)} x {arguments.pairs} pairs")

    # One process per candidate, with every opponent batched into it: the
    # arena's own help notes model load is ~21s against ~0.93s per extra pair,
    # so batching opponents is most of the cost saved.
    command = build_command(
        root, config, available[challengers[-1]], [available[anchor]],
        arguments.seed, arguments.pairs,
    )
    if arguments.dry_run:
        print("\nwould run, per challenger:")
        print("  " + " ".join(command))
        return

    output = run / "manual-elo"
    output.mkdir(parents=True, exist_ok=True)
    matches_path = output / "matches.jsonl"
    matches: list[dict] = []
    if matches_path.exists():
        matches = [
            json.loads(line)
            for line in matches_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"resuming  : {len(matches)} existing match record(s)")

    already = {(match["a"], match["b"]) for match in matches}
    environment = runtime_environment()
    started = time.perf_counter()

    for index, version in enumerate(challengers, start=1):
        if (version, anchor) in already:
            print(f"\n[{index}/{len(challengers)}] v{version} vs v{anchor}: already rated, skipping")
            continue
        command = build_command(
            root, config, available[version], [available[anchor]],
            arguments.seed + version * 1_000_003, arguments.pairs,
        )
        print(f"\n[{index}/{len(challengers)}] v{version} vs v{anchor} ...", flush=True)
        completed = subprocess.run(
            command, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        if completed.returncode != 0:
            tail = completed.stderr.strip().splitlines()[-3:]
            print(f"  FAILED (exit {completed.returncode})")
            for line in tail:
                print(f"    {line}")
            continue
        records = parse_records(completed.stdout)
        if not records:
            print("  FAILED: no arena record in output")
            continue
        record = records[0]
        match = {
            "a": version,
            "b": anchor,
            "a_wins": int(record["candidate_wins"]),
            "b_wins": int(record["candidate_losses"]),
            "draws": int(record["draws"]),
            "score": float(record["candidate_score"]),
            "games": int(record["games"]),
            "wall_seconds": float(record["wall_seconds"]),
        }
        matches.append(match)
        with matches_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(match) + "\n")
        print(
            f"  score {match['score']:.3f} "
            f"({match['a_wins']}-{match['b_wins']}-{match['draws']}) "
            f"in {match['wall_seconds']:.0f}s"
        )

    if not matches:
        raise SystemExit("no matches completed")

    ratings = fit_ratings(matches, anchor=anchor, prior_games=arguments.prior_games)
    (output / "ratings.json").write_text(
        json.dumps({str(k): v for k, v in sorted(ratings.items())}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'version':>9}{'elo':>9}{'score':>9}")
    scores = {match["a"]: match["score"] for match in matches}
    for version in sorted(ratings):
        score = scores.get(version)
        column = f"{score:.3f}" if score is not None else "anchor"
        print(f"{version:>9}{ratings[version]:>9.0f}{column:>9}")
    print(f"\n{time.perf_counter() - started:.0f}s total -> {output}/ratings.json")


if __name__ == "__main__":
    main()
