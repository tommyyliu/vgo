from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


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
    if not 0.0 <= arguments.maximum_truncation_rate <= 1.0:
        raise ValueError("maximum truncation rate must be in [0, 1]")
    if arguments.learning_rate <= 0.0 or arguments.warm_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if arguments.value_weight < 0.0:
        raise ValueError("value weight must be nonnegative")


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


def arena_command(
    arguments: argparse.Namespace,
    root: Path,
    candidate: Path,
    opponent: Path | None,
    seed: int,
) -> list[str]:
    command = rust_command(root, "vgo-arena") + [
        "--candidate",
        str(candidate),
        "--pairs",
        str(arguments.arena_pairs),
        "--simulations",
        str(arguments.arena_simulations),
        "--max-plies",
        str(arguments.maximum_plies),
        "--threads",
        str(arguments.arena_actors),
        "--resolution",
        str(arguments.resolution),
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
    if opponent is not None:
        command.extend(["--opponent", str(opponent)])
    return command


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
    started = time.perf_counter()

    for iteration in range(arguments.iterations):
        iteration_path = output / f"iteration-{iteration:03d}"
        report_path = iteration_path / "iteration.json"
        if not report_path.exists():
            break
        report = json.loads(report_path.read_text(encoding="ascii"))
        iterations.append(report)
        replay_paths.append(iteration_path / "replay" / "dataset.vgo")
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
        generation_command = rust_command(root, "vgo-generate-demo") + [
            "--samples",
            str(arguments.samples),
            "--resolution",
            str(arguments.resolution),
            "--simulations",
            str(arguments.generation_simulations),
            "--max-plies",
            str(arguments.maximum_plies),
            "--radius",
            str(arguments.radius),
            "--seed",
            str(generation_seed),
            "--examples",
            str(arguments.examples),
            "--output",
            str(replay_staging),
            "--maximum-batch",
            str(arguments.maximum_batch),
            "--delay-ms",
            str(arguments.delay_ms),
            "--actors",
            str(arguments.actors),
            "--provider",
            arguments.provider,
            "--fp16",
            str(arguments.fp16).lower(),
            "--cache-directory",
            str((root / "artifacts" / "onnx-cache").resolve()),
        ]
        if incumbent_onnx is None:
            generation_command.extend(["--runtime", "naive"])
        else:
            generation_command.extend(
                ["--runtime", "onnx", "--model", str(incumbent_onnx)]
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
                generation_command,
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
            "--threads",
            str(arguments.training_threads),
            "--device",
            arguments.device,
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

        if "baseline_arena" not in progress:
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
        baseline_arena = progress["baseline_arena"]
        if incumbent_onnx is None:
            promotion_arena = baseline_arena
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
        accepted = promotion_decision(
            promotion_arena,
            arguments.promotion_score,
            arguments.maximum_truncation_rate,
        )
        if accepted:
            incumbent_checkpoint = checkpoint
            incumbent_onnx = onnx
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
        }
        atomic_json(iteration_path / "iteration.json", iteration_report)
        iterations.append(iteration_report)
        print(
            f"iteration={iteration} baseline_score={baseline_arena['candidate_score']:.3f} "
            f"promotion_score={promotion_arena['candidate_score']:.3f} accepted={accepted}",
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete VGO reinforcement-learning loop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-onnx", type=Path)
    parser.add_argument("--initial-replay", type=Path, nargs="*", default=[])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--replay-window", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1.0 / 6.0)
    parser.add_argument("--generation-simulations", type=int, default=64)
    parser.add_argument("--maximum-plies", type=int, default=48)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--training-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--warm-learning-rate", type=float, default=5e-4)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--training-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--arena-pairs", type=int, default=16)
    parser.add_argument("--arena-simulations", type=int, default=16)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--arena-actors", type=int, default=1)
    parser.add_argument("--maximum-batch", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=1)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt"), default="tensorrt")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--promotion-score", type=float, default=0.52)
    parser.add_argument("--maximum-truncation-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=700_001)
    parser.add_argument("--arena-seed", type=int, default=900_001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
