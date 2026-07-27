from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import random
import time

from .bradley_terry import fit_ratings
from .model import MODEL_ARCHITECTURES


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    temporary.replace(path)


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def cargo_executable() -> str:
    discovered = shutil.which("cargo")
    if discovered:
        return discovered
    fallback = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("cargo executable was not found")


def runtime_environment() -> dict[str, str]:
    environment = os.environ.copy()
    candidates: list[Path] = []
    if os.name == "nt":
        packages = Path(sys.prefix) / "Lib" / "site-packages"
        candidates.extend([packages / "tensorrt_libs", packages / "torch" / "lib"])
        cuda = environment.get("CUDA_PATH")
        if cuda:
            candidates.append(Path(cuda) / "bin")
        existing = [str(path) for path in candidates if path.exists()]
        if existing:
            environment["PATH"] = os.pathsep.join(existing + [environment.get("PATH", "")])
        return environment

    # On Linux the ort crate is built with `load-dynamic`, so the Rust binaries
    # dlopen libonnxruntime.so at runtime from ORT_DYLIB_PATH. Blackwell (sm_120)
    # GPUs need a from-source build; see docs/NVRTX_HANDOFF.md. The CUDA runtime
    # libraries the provider needs live in the training venv, so they must be on
    # LD_LIBRARY_PATH for the child processes. Both are configurable via the
    # environment; we only fill in what the caller has not already set.
    packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    # Prefer the TensorRT-enabled onnxruntime build (superset: it also carries the
    # CUDA provider), falling back to the CUDA-only build if only that is present.
    onnxruntime_dir = packages / "onnxruntime_trt"
    if not onnxruntime_dir.exists():
        onnxruntime_dir = packages / "onnxruntime_blackwell"
    library_dirs = [
        onnxruntime_dir,
        packages / "tensorrt_libs",
        packages / "nvidia" / "cu13" / "lib",
        packages / "nvidia" / "cudnn" / "lib",
        packages / "torch" / "lib",
    ]
    existing = [str(path) for path in library_dirs if path.exists()]
    if existing:
        prior = environment.get("LD_LIBRARY_PATH", "")
        parts = existing + ([prior] if prior else [])
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    dylib = onnxruntime_dir / "libonnxruntime.so"
    if "ORT_DYLIB_PATH" not in environment and dylib.exists():
        environment["ORT_DYLIB_PATH"] = str(dylib)
    return environment


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    print(f"[{log_path.stem}] {' '.join(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}\n{result.stderr[-2000:]}"
        )
    print(f"[{log_path.stem}] complete in {time.perf_counter() - started:.1f}s", flush=True)
    return result


def run_json_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    result = run_logged(command, cwd=cwd, log_path=log_path, environment=environment)
    return json.loads(result.stdout)


def json_from_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    stdout = text.split("\nSTDOUT\n", 1)[1].split("\nSTDERR\n", 1)[0].strip()
    start = stdout.rfind("\n{")
    if start >= 0:
        stdout = stdout[start + 1 :]
    return json.loads(stdout)


def json_documents_from_log(path: Path) -> list[dict[str, object]]:
    """Every top-level JSON document in a log's stdout, in order.

    A batched arena emits one record per opponent, so the single-document
    reader above would silently keep only the last one.
    """
    text = path.read_text(encoding="utf-8")
    stdout = text.split("\nSTDOUT\n", 1)[1].split("\nSTDERR\n", 1)[0]
    documents: list[dict[str, object]] = []
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = stdout.find("{", index)
        if start < 0:
            return documents
        document, end = decoder.raw_decode(stdout, start)
        documents.append(document)
        index = end


def recover_progress(iteration_path: Path) -> dict[str, object]:
    progress_path = iteration_path / "progress.json"
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="ascii"))
    progress: dict[str, object] = {}
    for key, name in [
        ("generation", "generate.log"),
        ("training", "train.log"),
        ("export", "export.log"),
        ("baseline_arena", "arena-naive.log"),
        ("promotion_arena", "arena-incumbent.log"),
    ]:
        path = iteration_path / name
        if path.exists():
            try:
                progress[key] = json_from_log(path)
            except (IndexError, json.JSONDecodeError):
                continue
    if progress:
        atomic_json(progress_path, progress)
    return progress


def require_artifacts(
    progress: dict[str, object], key: str, *paths: Path
) -> bool:
    complete = key in progress and all(path.exists() for path in paths)
    if not complete:
        progress.pop(key, None)
    return complete


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = {
        "iterations": arguments.iterations,
        "samples": arguments.samples,
        "replay window": arguments.replay_window,
        "resolution": arguments.resolution,
        "generation simulations": arguments.generation_simulations,
        "maximum plies": arguments.maximum_plies,
        "epochs": arguments.epochs,
        "training batch": arguments.training_batch,
        "model width": arguments.model_width,
        "blocks": arguments.blocks,
        "training threads": arguments.training_threads,
        "arena pairs": arguments.arena_pairs,
        "arena simulations": arguments.arena_simulations,
        "actors": arguments.actors,
        "arena actors": arguments.arena_actors,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"counts must be positive: {', '.join(invalid)}")
    if arguments.maximum_batch < 2:
        raise ValueError("maximum batch must be at least two")
    if not 0.0 < arguments.radius < 0.5:
        raise ValueError("radius must be between zero and one half")
    if not 0.0 <= arguments.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if not 0.0 <= arguments.promotion_score <= 1.0:
        raise ValueError("promotion score must be in [0, 1]")
    if arguments.promotion_arena and arguments.promotion_score <= 0.0:
        raise ValueError(
            "--promotion-arena gates on a minimum score, so it needs a nonzero "
            "--promotion-score"
        )
    if not arguments.promotion_arena and arguments.promotion_score > 0.0:
        raise ValueError(
            "a nonzero --promotion-score has no effect without --promotion-arena"
        )
    if getattr(arguments, "elo_prior_games", 1.0) <= 0.0:
        raise ValueError("elo prior games must be positive")
    if not 0.0 <= arguments.maximum_truncation_rate <= 1.0:
        raise ValueError("maximum truncation rate must be in [0, 1]")
    if arguments.learning_rate <= 0.0 or arguments.warm_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if arguments.value_weight < 0.0:
        raise ValueError("value weight must be nonnegative")
    if arguments.coarse_pool < 0:
        raise ValueError("coarse pool must be nonnegative")
    if arguments.policy_resolution <= 0:
        raise ValueError("policy resolution must be positive")
    if arguments.coarse_pool > arguments.policy_resolution:
        raise ValueError("coarse pool must not exceed policy resolution")
    if arguments.temperature < 0.0:
        raise ValueError("temperature must be nonnegative")
    if arguments.temperature_plies < 0:
        raise ValueError("temperature plies must be nonnegative")


def promotion_decision(
    arena: dict[str, object], minimum_score: float, maximum_truncation_rate: float
) -> bool:
    games = int(arena["games"])
    completed = int(arena["completed"])
    if games <= 0 or not 0 <= completed <= games:
        raise ValueError("arena game counts are inconsistent")
    truncation_rate = (games - completed) / games
    return (
        completed > 0
        and truncation_rate <= maximum_truncation_rate
        and float(arena["candidate_score"]) >= minimum_score
    )


def rust_command(root: Path, binary: str) -> list[str]:
    return [
        cargo_executable(),
        "run",
        "--release",
        "-p",
        "vgo-selfplay",
        "--bin",
        binary,
        "--",
    ]


def generation_command(
    arguments: argparse.Namespace,
    root: Path,
    output: Path,
    seed: int,
    incumbent: Path | None,
) -> list[str]:
    command = rust_command(root, "vgo-generate-demo") + [
        "--samples",
        str(arguments.samples),
        "--resolution",
        str(arguments.resolution),
        "--policy-resolution",
        str(arguments.policy_resolution),
        "--simulations",
        str(arguments.generation_simulations),
        "--coarse-pool",
        str(arguments.coarse_pool),
        "--temperature",
        str(arguments.temperature),
        "--temperature-plies",
        str(arguments.temperature_plies),
        "--max-plies",
        str(arguments.maximum_plies),
        "--radius",
        str(arguments.radius),
        "--seed",
        str(seed),
        "--examples",
        str(arguments.examples),
        "--output",
        str(output),
        "--maximum-batch",
        str(arguments.maximum_batch),
        "--delay-ms",
        str(arguments.delay_ms),
        "--actors",
        str(arguments.actors),
        "--leaf-batch",
        str(arguments.leaf_batch),
        "--provider",
        arguments.provider,
        "--fp16",
        str(arguments.fp16).lower(),
        "--cache-directory",
        str((root / "artifacts" / "onnx-cache").resolve()),
    ]
    if incumbent is None:
        command.extend(["--runtime", "naive"])
    else:
        command.extend(["--runtime", "onnx", "--model", str(incumbent)])
    return command


def arena_command(
    arguments: argparse.Namespace,
    root: Path,
    candidate: Path,
    opponent: Path | None,
    seed: int,
    *,
    pairs: int | None = None,
) -> list[str]:
    command = rust_command(root, "vgo-arena") + [
        "--candidate",
        str(candidate),
        "--pairs",
        str(arguments.arena_pairs if pairs is None else pairs),
        "--simulations",
        str(arguments.arena_simulations),
        "--coarse-pool",
        str(arguments.coarse_pool),
        "--max-plies",
        str(arguments.maximum_plies),
        "--threads",
        str(arguments.arena_actors),
        "--leaf-batch",
        str(arguments.leaf_batch),
        "--resolution",
        str(arguments.resolution),
        "--policy-resolution",
        str(arguments.policy_resolution),
        "--radius",
        str(arguments.radius),
        "--seed",
        str(seed),
        "--maximum-batch",
        str(arguments.maximum_batch),
        "--delay-ms",
        str(arguments.delay_ms),
        "--provider",
        arguments.provider,
        "--fp16",
        str(arguments.fp16).lower(),
        "--cache-directory",
        str((root / "artifacts" / "onnx-cache").resolve()),
    ]
    # `opponent` may be one path or several; vgo-arena repeats --opponent and
    # emits one JSON record per opponent, reusing the loaded candidate.
    if opponent is not None:
        for path in [opponent] if isinstance(opponent, (str, Path)) else opponent:
            command.extend(["--opponent", str(path)])
    return command


def update_elo_pool(
    arguments: argparse.Namespace,
    root: Path,
    output: Path,
    candidate_onnx: Path,
    iteration: int,
    past_generations: list[tuple[int, Path]],
    environment: dict[str, str],
) -> dict[str, float]:
    """Play the new generation against a light random sample of past generations,
    append the results to a persistent match history, and re-fit Bradley-Terry
    ratings for every generation. Pure telemetry — never gates promotion. Returns
    {generation_index: elo} (empty until there is at least one past generation).

    Resilient by design: a failed sample match is skipped, not fatal, so a flaky
    arena never crashes the training loop over telemetry.
    """
    if arguments.elo_pool_samples <= 0 or not past_generations:
        return {}
    elo_dir = output / "elo"
    elo_dir.mkdir(parents=True, exist_ok=True)
    matches_path = elo_dir / "matches.jsonl"

    sample = random.Random(arguments.arena_seed + iteration * 977)
    opponents = sample.sample(
        past_generations, min(arguments.elo_pool_samples, len(past_generations))
    )
    playable = [
        (index, path) for index, path in opponents if Path(path).exists()
    ]
    new_records: list[dict[str, object]] = []
    if playable:
        # One process for every opponent: vgo-arena loads the candidate once and
        # emits a record per opponent, in the order given.
        log_path = elo_dir / f"iter{iteration:03d}-pool.log"
        command = arena_command(
            arguments,
            root,
            candidate_onnx,
            [path for _, path in playable],
            arguments.arena_seed + iteration * 10_000 + 7_000,
            pairs=arguments.elo_pool_pairs,
        )
        try:
            run_logged(command, cwd=root, log_path=log_path, environment=environment)
            results = json_documents_from_log(log_path)
        except Exception as error:  # noqa: BLE001 — telemetry must not kill the loop
            print(f"[elo] skipped pool matches: {error}", flush=True)
            results = []
        if results and len(results) != len(playable):
            print(
                f"[elo] expected {len(playable)} records, got {len(results)}; "
                "discarding to avoid mis-attributing matches",
                flush=True,
            )
            results = []
        for (opponent_iteration, _), result in zip(playable, results):
            # Only count decisive completed games; a truncated game is not a result.
            new_records.append(
                {
                    "a": iteration,
                    "b": opponent_iteration,
                    "a_wins": int(result.get("candidate_wins", 0)),
                    "b_wins": int(result.get("candidate_losses", 0)),
                    "draws": int(result.get("draws", 0)),
                }
            )

    if new_records:
        with matches_path.open("a", encoding="ascii") as handle:
            for record in new_records:
                handle.write(json.dumps(record) + "\n")

    matches: list[dict] = []
    if matches_path.exists():
        for line in matches_path.read_text(encoding="ascii").splitlines():
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    if not matches:
        return {}
    ratings = fit_ratings(matches, anchor=0, prior_games=arguments.elo_prior_games)
    ranked = dict(sorted(ratings.items()))
    atomic_json(elo_dir / "ratings.json", {str(k): round(v, 1) for k, v in ranked.items()})
    current = ratings.get(iteration)
    if current is not None:
        print(f"[elo] gen {iteration} rating = {current:+.0f}", flush=True)
    return {str(k): round(v, 1) for k, v in ranked.items()}


def run(arguments: argparse.Namespace) -> dict[str, object]:
    validate_arguments(arguments)
    training = Path(__file__).resolve().parents[1]
    root = training.parent
    output = arguments.output.resolve()
    if (output / "run.json").exists():
        raise FileExistsError(f"completed RL run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment()
    replay_paths = [path.resolve(strict=True) for path in arguments.initial_replay]
    incumbent_checkpoint = (
        arguments.initial_checkpoint.resolve(strict=True)
        if arguments.initial_checkpoint is not None
        else None
    )
    incumbent_onnx = (
        arguments.initial_onnx.resolve(strict=True)
        if arguments.initial_onnx is not None
        else None
    )
    if (incumbent_checkpoint is None) != (incumbent_onnx is None):
        raise ValueError("initial checkpoint and ONNX model must be supplied together")
    run_config = jsonable(
        vars(arguments)
        | {
            "output": str(output),
            "initial_checkpoint": (
                str(incumbent_checkpoint) if incumbent_checkpoint is not None else None
            ),
            "initial_onnx": str(incumbent_onnx) if incumbent_onnx is not None else None,
            "initial_replay": [str(path) for path in replay_paths],
        }
    )
    config_path = output / "run-config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="ascii"))
        if previous_config != run_config:
            raise ValueError("RL run configuration differs from the interrupted run")
    else:
        atomic_json(config_path, run_config)
    iterations: list[dict[str, object]] = []
    # (iteration_index, onnx_path) for every completed generation — the Elo pool.
    past_generations: list[tuple[int, Path]] = []
    started = time.perf_counter()

    for iteration in range(arguments.iterations):
        iteration_path = output / f"iteration-{iteration:03d}"
        report_path = iteration_path / "iteration.json"
        if not report_path.exists():
            break
        report = json.loads(report_path.read_text(encoding="ascii"))
        iterations.append(report)
        replay_paths.append(iteration_path / "replay" / "dataset.vgo")
        past_generations.append((iteration, iteration_path / "model" / "candidate.onnx"))
        if report.get("accepted"):
            incumbent_checkpoint = Path(str(report["incumbent_checkpoint"]))
            incumbent_onnx = Path(str(report["incumbent_onnx"]))
    start_iteration = len(iterations)

    for iteration in range(start_iteration, arguments.iterations):
        iteration_path = output / f"iteration-{iteration:03d}"
        replay_path = iteration_path / "replay"
        replay_staging = iteration_path / "replay.staging"
        model_path = iteration_path / "model"
        iteration_path.mkdir(parents=True, exist_ok=True)
        model_path.mkdir(exist_ok=True)
        progress = recover_progress(iteration_path)
        progress_path = iteration_path / "progress.json"
        generation_seed = arguments.seed + iteration * 100_000
        generate_command = generation_command(
            arguments,
            root,
            replay_staging,
            generation_seed,
            incumbent_onnx,
        )
        generation_complete = require_artifacts(
            progress,
            "generation",
            replay_path / "dataset.vgo",
            replay_path / "manifest.json",
        )
        if not generation_complete:
            for key in ("training", "export", "baseline_arena", "promotion_arena"):
                progress.pop(key, None)
            if replay_staging.exists():
                shutil.rmtree(replay_staging)
            progress["generation"] = run_json_command(
                generate_command,
                cwd=root,
                log_path=iteration_path / "generate.log",
                environment=environment,
            )
            if replay_path.exists():
                raise FileExistsError(f"incomplete replay output exists: {replay_path}")
            replay_staging.replace(replay_path)
            atomic_json(progress_path, progress)
        generation = progress["generation"]
        if isinstance(generation, dict):
            generation["dataset"] = str(replay_path / "dataset.vgo")
            if "diagnostics" not in generation:
                # Cheap health check on the shard we just wrote. A ply-0 Jaccard
                # near zero means the policy target is still sampler noise, which
                # no amount of training will fix; see docs/RL_LOOP.md.
                from .dataset import load_dataset, replay_diagnostics

                try:
                    generation["diagnostics"] = replay_diagnostics(
                        load_dataset(replay_path / "dataset.vgo")
                    )
                except Exception as error:  # diagnostics must never fail a run
                    generation["diagnostics"] = {"error": str(error)}
                atomic_json(progress_path, progress)
        replay_paths.append(replay_path / "dataset.vgo")
        active_replay = replay_paths[-arguments.replay_window :]

        checkpoint = model_path / "candidate.pt"
        training_command = [
            sys.executable,
            "-m",
            "vgo_training.train_demo",
            *[str(path) for path in active_replay],
            "--output",
            str(checkpoint),
            "--epochs",
            str(arguments.epochs),
            "--batch-size",
            str(arguments.training_batch),
            "--learning-rate",
            str(
                arguments.learning_rate
                if incumbent_checkpoint is None
                else arguments.warm_learning_rate
            ),
            "--value-weight",
            str(arguments.value_weight),
            "--model-width",
            str(arguments.model_width),
            "--blocks",
            str(arguments.blocks),
            "--architecture",
            arguments.architecture,
            "--threads",
            str(arguments.training_threads),
            "--device",
            arguments.device,
            "--schedule",
            arguments.schedule,
            "--compile" if arguments.compile else "--no-compile",
            "--restore-optimizer"
            if arguments.restore_optimizer
            else "--no-restore-optimizer",
            "--seed",
            str(arguments.seed + iteration),
            "--report-every",
            str(arguments.report_every),
            "--validation-fraction",
            str(arguments.validation_fraction),
        ]
        if incumbent_checkpoint is not None:
            training_command.extend(["--initial-checkpoint", str(incumbent_checkpoint)])
        training_report_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        training_complete = require_artifacts(
            progress, "training", checkpoint, training_report_path
        )
        if not training_complete:
            for key in ("export", "baseline_arena", "promotion_arena"):
                progress.pop(key, None)
            run_logged(
                training_command,
                cwd=training,
                log_path=iteration_path / "train.log",
                environment=environment,
            )
            progress["training"] = json.loads(
                training_report_path.read_text(encoding="ascii")
            )
            atomic_json(progress_path, progress)
        training_report = progress["training"]

        onnx = model_path / "candidate.onnx"
        export_command = [
            sys.executable,
            "-m",
            "vgo_training.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(onnx),
            "--maximum-batch",
            str(arguments.maximum_batch),
        ]
        export_report_path = onnx.with_suffix(onnx.suffix + ".json")
        export_complete = require_artifacts(progress, "export", onnx, export_report_path)
        if not export_complete:
            for key in ("baseline_arena", "promotion_arena"):
                progress.pop(key, None)
            run_logged(
                export_command,
                cwd=training,
                log_path=iteration_path / "export.log",
                environment=environment,
            )
            progress["export"] = json.loads(export_report_path.read_text(encoding="ascii"))
            atomic_json(progress_path, progress)
        export_report = progress["export"]

        # The baseline (vs-naive) arena drives promotion only when there is no
        # incumbent yet (iteration 0). Once an incumbent exists it is pure
        # telemetry, so --skip-baseline-arena drops it to save serial arena time.
        run_baseline = incumbent_onnx is None or not arguments.skip_baseline_arena
        if run_baseline and "baseline_arena" not in progress:
            progress["baseline_arena"] = run_json_command(
                arena_command(
                    arguments,
                    root,
                    onnx,
                    None,
                    arguments.arena_seed + iteration * 10_000,
                ),
                cwd=root,
                log_path=iteration_path / "arena-naive.log",
                environment=environment,
            )
            atomic_json(progress_path, progress)
        baseline_arena = progress.get("baseline_arena")
        if incumbent_onnx is None:
            promotion_arena = baseline_arena
        elif not arguments.promotion_arena:
            # AlphaZero-style: no gate, so the 60-pair vs-incumbent arena is pure
            # telemetry -- and the least informative kind, since consecutive
            # generations are nearly identical. The Elo pool measures the same
            # progress against many opponents for a fraction of the games.
            promotion_arena = None
        else:
            if "promotion_arena" not in progress:
                progress["promotion_arena"] = run_json_command(
                    arena_command(
                        arguments,
                        root,
                        onnx,
                        incumbent_onnx,
                        arguments.arena_seed + iteration * 10_000 + 5_000,
                    ),
                    cwd=root,
                    log_path=iteration_path / "arena-incumbent.log",
                    environment=environment,
                )
                atomic_json(progress_path, progress)
            promotion_arena = progress["promotion_arena"]
        accepted = (
            True
            if promotion_arena is None
            else promotion_decision(
                promotion_arena,
                arguments.promotion_score,
                arguments.maximum_truncation_rate,
            )
        )
        if accepted:
            incumbent_checkpoint = checkpoint
            incumbent_onnx = onnx

        elo_ratings = update_elo_pool(
            arguments, root, output, onnx, iteration, past_generations, environment
        )
        past_generations.append((iteration, onnx))

        iteration_report = {
            "schema": "vgo.rl-iteration.v1",
            "iteration": iteration,
            "generation": generation,
            "active_replay": [str(path) for path in active_replay],
            "training": training_report,
            "export": export_report,
            "baseline_arena": baseline_arena,
            "promotion_arena": promotion_arena,
            "promotion_criteria": {
                "minimum_score": arguments.promotion_score,
                "maximum_truncation_rate": arguments.maximum_truncation_rate,
            },
            "accepted": accepted,
            "incumbent_checkpoint": str(incumbent_checkpoint) if incumbent_checkpoint else None,
            "incumbent_onnx": str(incumbent_onnx) if incumbent_onnx else None,
            "elo": elo_ratings,
        }
        atomic_json(iteration_path / "iteration.json", iteration_report)
        iterations.append(iteration_report)
        baseline_text = (
            f"{baseline_arena['candidate_score']:.3f}" if baseline_arena else "skipped"
        )
        promotion_text = (
            f"{promotion_arena['candidate_score']:.3f}" if promotion_arena else "skipped"
        )
        print(
            f"iteration={iteration} baseline_score={baseline_text} "
            f"promotion_score={promotion_text} accepted={accepted}",
            flush=True,
        )

    report = {
        "schema": "vgo.rl-run.v1",
        "wall_seconds": time.perf_counter() - started,
        "config": run_config,
        "iterations": iterations,
        "final_checkpoint": str(incumbent_checkpoint) if incumbent_checkpoint else None,
        "final_onnx": str(incumbent_onnx) if incumbent_onnx else None,
    }
    atomic_json(output / "run.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete VGO reinforcement-learning loop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-onnx", type=Path)
    parser.add_argument("--initial-replay", type=Path, nargs="*", default=[])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--replay-window", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--radius", type=float, default=1.0 / 6.0)
    parser.add_argument("--generation-simulations", type=int, default=64)
    parser.add_argument(
        "--coarse-pool",
        type=int,
        default=0,
        help="fine cells per coarse sampling region; 0 uses legacy candidates",
    )
    parser.add_argument(
        "--policy-resolution",
        type=int,
        default=32,
        help=(
            "placement grid the policy head emits, independent of --resolution. "
            "Coarser placement concentrates the fixed coarse->fine proposal budget "
            "over far fewer cells; rendering keeps --resolution detail."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "softmax temperature on root visit counts for the opening plies of "
            "self-play generation; 0 is deterministic argmax. Arenas are always "
            "deterministic regardless of this value."
        ),
    )
    parser.add_argument(
        "--temperature-plies",
        type=int,
        default=30,
        help="plies over which --temperature applies before reverting to argmax",
    )
    parser.add_argument("--maximum-plies", type=int, default=48)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--training-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--warm-learning-rate", type=float, default=5e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument(
        "--architecture", choices=MODEL_ARCHITECTURES, default="flat"
    )
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--training-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--schedule",
        choices=("wsd", "cosine"),
        default="wsd",
        help="learning-rate schedule forwarded to training",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the training model and enable TF32 (forwarded to training)",
    )
    parser.add_argument(
        "--restore-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="carry Adam moments across iterations instead of restarting them "
        "cold each time (forwarded to training)",
    )
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--skip-baseline-arena",
        action="store_true",
        help="skip the vs-naive arena once an incumbent exists (it is telemetry, "
        "not a promotion gate, from iteration 1 on)",
    )
    parser.add_argument(
        "--promotion-arena",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="gate promotion on a vs-incumbent arena. Off by default: it only "
        "answers 'did N beat N-1', where the two nets are nearly identical and "
        "the result is mostly sampling noise. Prefer an Elo pool, which "
        "measures the same progress against many opponents for fewer games. "
        "Enabling it requires a nonzero --promotion-score",
    )
    parser.add_argument(
        "--elo-pool-samples",
        type=int,
        default=2,
        help="past generations to play the new net against each iteration for Elo "
        "tracking (0 disables); pure telemetry, does not gate promotion. With no "
        "promotion arena this is the only progress signal, so it defaults on",
    )
    parser.add_argument(
        "--elo-pool-pairs",
        type=int,
        default=16,
        help="color-swapped pairs per sampled opponent. Concurrency is capped by "
        "game count, so a small match leaves most arena threads idle and the "
        "inference broker never fills a batch: measured cost per game is 3.08s "
        "at 4 pairs, 1.13s at 16, and 0.92s at 60. Prefer few large matches -- "
        "2x16 is 64 games in ~72s, where 4x4 is 32 games in ~99s",
    )
    parser.add_argument(
        "--elo-prior-games",
        type=float,
        default=0.25,
        help="virtual even games regularizing each generation's rating. Keeps an "
        "undefeated net from diverging, but shrinks every rating toward the "
        "anchor: at the old default of 2.0 a simulated 40-generation ladder "
        "read 468 Elo against a true 585, and 0.25 recovers it to 623",
    )
    parser.add_argument("--arena-pairs", type=int, default=16)
    parser.add_argument("--arena-simulations", type=int, default=16)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument(
        "--leaf-batch",
        type=int,
        default=1,
        help="leaves evaluated together per simulation round, raising in-flight "
        "evaluations per game from one to this many. Above 1 the search explores "
        "different nodes, so it is not a free throughput knob: 1 is the "
        "sequential path pinned by test, and both arena seats must agree",
    )
    parser.add_argument("--arena-actors", type=int, default=1)
    parser.add_argument("--maximum-batch", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=1)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt"), default="tensorrt")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--promotion-score",
        type=float,
        default=0.0,
        help="minimum vs-incumbent arena score to promote; requires "
        "--promotion-arena. Zero (the default) accepts every candidate",
    )
    parser.add_argument("--maximum-truncation-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=700_001)
    parser.add_argument("--arena-seed", type=int, default=900_001)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
