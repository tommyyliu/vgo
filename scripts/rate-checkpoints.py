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

_TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(_TRAINING))
from vgo_training.bradley_terry import fit_ratings  # noqa: E402
from vgo_training.pipeline import runtime_environment  # noqa: E402


def arena_environment() -> dict[str, str]:
    """Environment for vgo-arena, independent of the interpreter running us.

    `runtime_environment` derives the ONNX Runtime and TensorRT library paths
    from `sys.prefix`, so it only produces a usable environment when called from
    the training venv. Run under the system interpreter it silently yields no
    ORT_DYLIB_PATH, and the arena then hangs rather than failing: a failed dylib
    load inside `ort` builds its error through `ort::api()`, which waits on the
    same initialization lock the failing load still holds.

    Re-exec through the venv interpreter to collect the environment, so this
    script works however it was invoked.
    """
    if Path(sys.prefix).resolve() == _TRAINING / ".venv":
        return runtime_environment()
    interpreter = _TRAINING / ".venv/bin/python3"
    if not interpreter.exists():
        raise SystemExit(f"training venv not found at {interpreter}")
    collected = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import json,sys;sys.path.insert(0,%r);"
            "from vgo_training.pipeline import runtime_environment;"
            "print(json.dumps(runtime_environment()))" % str(_TRAINING),
        ],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    environment = json.loads(collected.stdout)
    if not environment.get("ORT_DYLIB_PATH"):
        raise SystemExit(
            "ORT_DYLIB_PATH is unset; the arena would hang instead of failing"
        )
    return environment


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
    # Midpoint of the run's komi range. Zero is outside what generation drew.
    komi = (
        float(config.get("komi_low", 0.0)) + float(config.get("komi_high", 0.0))
    ) / 2.0
    command = [
        str(root / "target/release/vgo-arena"),
        "--candidate", str(candidate),
        "--pairs", str(pairs),
        "--simulations", str(simulations),
        "--coarse-pool", str(int(config.get("coarse_pool", 16))),
        # A model exported under a non-default layout declares its own channel
        # count, and loading validates it against the raster the arena builds.
        "--candidate-raster-kind", str(config.get("raster_kind", "semantic")),
        # Above the run's own cap. Arena games have no resignation to shorten
        # them, so a cap tuned for self-play truncates more of them here.
        "--max-plies", str(int(config.get("arena_maximum_plies", 120))),
        "--threads", str(int(config.get("arena_actors", 64))),
        "--leaf-batch", str(int(config.get("leaf_batch", 4))),
        "--resolution", str(int(config.get("resolution", 128))),
        "--policy-resolution", str(int(config.get("policy_resolution", 128))),
        "--radius", str(config.get("radius", 0.05555555555555555)),
        # The midpoint of the run's komi range, not zero. Generation draws komi
        # per game, so a model rated at zero is judged on a game it never
        # trained for -- Black takes ~85% at the bottom of the range. Fixed
        # across the match so both halves of a colour-swapped pair face the
        # same game.
        "--komi", str(komi),
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
    parser.add_argument(
        "--round-robin", action="store_true",
        help=(
            "play every chosen checkpoint against every other, not just the "
            "anchor; keeps late ratings pinned by close games instead of by "
            "the prior once the field beats the anchor outright"
        ),
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

    if arguments.round_robin:
        # Every checkpoint against every other. Rating against one anchor stops
        # discriminating once the field beats it outright -- an undefeated record
        # has no finite Elo, so the fit falls back on the prior rather than on
        # evidence. Cross-play keeps every rating pinned by games that were
        # actually close.
        pairings = [
            (a, b)
            for index, a in enumerate(chosen)
            for b in chosen[index + 1 :]
        ]
    else:
        pairings = [(version, anchor) for version in chosen if version != anchor]
    if not pairings:
        raise SystemExit("need at least two checkpoints to play")

    print(f"run       : {run}")
    print(f"available : {len(available)} checkpoints (versions {min(available)}-{max(available)})")
    print(f"rating    : {chosen}")
    print(f"anchor    : {anchor} (ratings are relative to this)")
    print(f"matches   : {len(pairings)} pairing(s) x {arguments.pairs} pairs")

    if arguments.dry_run:
        preview: dict[int, list[int]] = {}
        for candidate, opponent in pairings:
            preview.setdefault(candidate, []).append(opponent)
        print(f"\nwould run {len(preview)} arena process(es):")
        for candidate, opponents in sorted(preview.items()):
            listed = ", ".join(f"v{version}" for version in opponents)
            print(f"  v{candidate} vs {listed}")
        sample = sorted(preview)[0]
        print("\nfirst command:")
        print("  " + " ".join(build_command(
            root, config, available[sample],
            [available[version] for version in preview[sample]],
            arguments.seed, arguments.pairs,
        )))
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

    # A pairing already played in either direction is not replayed: colours are
    # swapped within a match, so v10-vs-v0 and v0-vs-v10 measure the same thing.
    already = {frozenset((match["a"], match["b"])) for match in matches}
    todo = [
        pairing for pairing in pairings if frozenset(pairing) not in already
    ]
    skipped = len(pairings) - len(todo)
    if skipped:
        print(f"skipping  : {skipped} pairing(s) already in matches.jsonl")
    if not todo:
        print("nothing new to play")

    # One process per candidate with all of its opponents batched in. The
    # arena loads the candidate once and emits a record per opponent, and model
    # load is ~21s against ~0.93s per additional pair -- so grouping turns 10
    # processes into 4 here.
    grouped: dict[int, list[int]] = {}
    for candidate, opponent in todo:
        grouped.setdefault(candidate, []).append(opponent)

    environment = arena_environment()
    started = time.perf_counter()

    for index, (candidate, opponents) in enumerate(sorted(grouped.items()), start=1):
        command = build_command(
            root, config, available[candidate],
            [available[version] for version in opponents],
            arguments.seed + candidate * 1_000_003, arguments.pairs,
        )
        listed = ", ".join(f"v{version}" for version in opponents)
        print(f"\n[{index}/{len(grouped)}] v{candidate} vs {listed} ...", flush=True)
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
        if len(records) != len(opponents):
            print(
                f"  FAILED: expected {len(opponents)} arena record(s), "
                f"got {len(records)}"
            )
            continue
        # Records come back in the order the opponents were passed.
        for opponent, record in zip(opponents, records):
            # A pairing that decided nothing is no evidence. The arena still
            # reports candidate_score 0.0 there because the field has to stay a
            # JSON number, so `completed` is what distinguishes "lost every
            # game" from "played none to a result" -- recording the former fed
            # the fit a fabricated 16-game shutout.
            if int(record["completed"]) == 0:
                print(
                    f"  vs v{opponent}: no decided games in "
                    f"{int(record['games'])}; not recorded"
                )
                continue
            match = {
                "a": candidate,
                "b": opponent,
                "a_wins": int(record["candidate_wins"]),
                "b_wins": int(record["candidate_losses"]),
                "draws": int(record["draws"]),
                "score": float(record["candidate_score"]),
                "games": int(record["games"]),
                "completed": int(record["completed"]),
                "wall_seconds": float(record["wall_seconds"]),
            }
            matches.append(match)
            with matches_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(match) + "\n")
            print(
                f"  vs v{opponent}: score {match['score']:.3f} "
                f"({match['a_wins']}-{match['b_wins']}-{match['draws']}) "
                f"{match['completed']}/{match['games']} decided "
                f"in {match['wall_seconds']:.0f}s"
            )

    if not matches:
        raise SystemExit("no matches completed")

    ratings = fit_ratings(matches, anchor=anchor, prior_games=arguments.prior_games)
    (output / "ratings.json").write_text(
        json.dumps({str(k): v for k, v in sorted(ratings.items())}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Aggregate every game a version played, from either seat.
    wins: dict[int, float] = {}
    played: dict[int, int] = {}
    for match in matches:
        a, b = int(match["a"]), int(match["b"])
        half = 0.5 * float(match["draws"])
        wins[a] = wins.get(a, 0.0) + float(match["a_wins"]) + half
        wins[b] = wins.get(b, 0.0) + float(match["b_wins"]) + half
        played[a] = played.get(a, 0) + int(match["games"])
        played[b] = played.get(b, 0) + int(match["games"])

    print(f"\n{'version':>9}{'elo':>9}{'score':>9}{'games':>8}")
    for version in sorted(ratings):
        total = played.get(version, 0)
        score = wins.get(version, 0.0) / total if total else float("nan")
        flag = ""
        # An unbeaten or winless record has no finite maximum-likelihood rating,
        # so its Elo is set by --prior-games rather than by evidence. Say so
        # rather than letting the number be read as a measurement.
        if total and (score >= 1.0 or score <= 0.0):
            flag = "  <- no finite Elo; rating pinned by the prior"
        print(
            f"{version:>9}{ratings[version]:>9.0f}{score:>9.3f}{total:>8}{flag}"
        )
    print(f"\n{time.perf_counter() - started:.0f}s total -> {output}/ratings.json")


if __name__ == "__main__":
    main()
