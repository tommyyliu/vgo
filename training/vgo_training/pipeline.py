from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, BinaryIO

from .bradley_terry import fit_ratings
from .model import MODEL_ARCHITECTURES


PIPELINE_SCHEMA = "vgo.pipeline-state.v1"
CONFIG_SCHEMA = "vgo.pipeline-config.v1"
UPDATE_SCHEMA = "vgo.pipeline-update.v1"
TELEMETRY_SCHEMA = "vgo.telemetry-job.v1"
LEARNER_PROTOCOL_SCHEMA = "vgo.learner.protocol.v1"
OPERATIONAL_CONFIG_FIELDS = {
    "output",
    "updates",
    "maximum_prefetch_shards",
    "actors",
    "writer_queue_games",
    "inference_delay_ms",
    "inference_slots",
    "inference_device_id",
    "warm_inference",
    "training_threads",
    "training_device",
    "compile",
    "overlap_actor_learner",
    "arena_actors",
    "telemetry_opponents",
    "telemetry_pairs",
    # Neither changes what is learned: one picks which checkpoints get rated,
    # the other is disk housekeeping. Both must be adjustable on a resume.
    "telemetry_every",
    "retire_shards",
    # Inference precision is an execution choice, not a learning one. It does
    # perturb the numbers a generator sees, so this is a judgement rather than
    # an identity: measured on ddrnet-vs update 14 over 256 real positions,
    # fp16 agrees with fp32 on 100% of policy argmaxes and differs on value by
    # 0.004, which is far below the noise between two searches of the same
    # position. What it does change is speed -- 41.4 ms/batch against 17.5 --
    # and a run that cannot adopt that without starting over pays for the
    # decision twice.
    #
    # The failure this guards against is real -- fp16 overflows when
    # activations approach 65504, which is what ended ddrnet-short -- but a
    # config digest is the wrong place to catch it. Overflow is a property of
    # the checkpoint, which drifts every update, not of the flag, which only
    # changes when someone edits it. What actually catches it is generation
    # failing loudly on a non-finite evaluation, plus checking peak activation
    # against the limit before enabling this (diagnostics in the run's
    # launch.sh). Freezing the flag would only mean a run cannot adopt fp16
    # without starting over.
    "fp16",
}


def _compress_shard(path: Path) -> tuple[Path, int, int]:
    """Compress one shard in place, leaving `<name>.zst` and removing the source.

    Verifies the archive round-trips to the exact original bytes before deleting
    anything: the shard is the only copy of that generation's self-play.
    """
    archive = path.with_suffix(path.suffix + ".zst")
    before = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    expected = digest.hexdigest()

    temporary = archive.with_suffix(archive.suffix + ".tmp")
    subprocess.run(
        ["zstd", "-3", "-T2", "-q", "-f", str(path), "-o", str(temporary)],
        check=True,
    )
    verify = subprocess.run(
        ["zstd", "-dc", str(temporary)], stdout=subprocess.PIPE, check=True
    )
    if hashlib.sha256(verify.stdout).hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"compressed shard does not round-trip: {path}")
    os.replace(temporary, archive)
    after = archive.stat().st_size
    path.unlink()
    return path, before, after


def _log_completed_retirement(path: Path, before: int, after: int) -> None:
    print(
        f"[retire] {path.parent.name}: "
        f"{before / 1e9:.2f} GB -> {after / 1e9:.3f} GB "
        f"({before / max(after, 1):.1f}x)",
        flush=True,
    )


def _log_retirement(task: "asyncio.Task[tuple[Path, int, int]]") -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        print(f"[retire] failed: {error}", flush=True)
        return
    _log_completed_retirement(*task.result())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Make a preceding atomic rename durable where directory fsync exists."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def identity_config(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in OPERATIONAL_CONFIG_FIELDS
    }


def cargo_executable() -> str:
    discovered = shutil.which("cargo")
    if discovered:
        return discovered
    fallback = Path.home() / ".cargo" / "bin" / (
        "cargo.exe" if os.name == "nt" else "cargo"
    )
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("cargo executable was not found")


def runtime_environment() -> dict[str, str]:
    """Environment needed by Rust ONNX Runtime and the Python learner."""

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
            environment["PATH"] = os.pathsep.join(
                existing + [environment.get("PATH", "")]
            )
        return environment

    packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
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
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            existing + ([prior] if prior else [])
        )
    dylib = onnxruntime_dir / "libonnxruntime.so"
    if "ORT_DYLIB_PATH" not in environment and dylib.exists():
        environment["ORT_DYLIB_PATH"] = str(dylib)
    return environment


@dataclass(frozen=True)
class ModelArtifact:
    version: int
    checkpoint: str
    onnx: str
    checkpoint_sha256: str
    onnx_sha256: str
    parent_version: int | None
    accepted: bool = True

    @classmethod
    def from_paths(
        cls,
        *,
        version: int,
        checkpoint: Path,
        onnx: Path,
        parent_version: int | None,
        accepted: bool = True,
    ) -> "ModelArtifact":
        return cls(
            version=version,
            checkpoint=str(checkpoint.resolve()),
            onnx=str(onnx.resolve()),
            checkpoint_sha256=file_sha256(checkpoint),
            onnx_sha256=file_sha256(onnx),
            parent_version=parent_version,
            accepted=accepted,
        )


@dataclass(frozen=True)
class ReplayArtifact:
    sequence: int
    path: str
    manifest: str
    samples: int
    behavior_model_sha256: str | None
    dataset_sha256: str
    seed: int


@dataclass(frozen=True)
class PipelineConfig:
    output: str
    updates: int = 10
    samples_per_shard: int = 1024
    shards_per_update: int = 1
    replay_window: int = 8
    maximum_prefetch_shards: int = 1
    resolution: int = 96
    policy_resolution: int = 32
    radius: float = 1.0 / 18.0
    generation_simulations: int = 256
    maximum_plies: int = 256
    coarse_pool: int = 4
    temperature: float = 1.0
    temperature_plies: int = 30
    actors: int = 64
    writer_queue_games: int = 2
    leaf_batch: int = 1
    inference_batch: int = 16
    inference_delay_ms: int = 1
    inference_slots: int = 2
    provider: str = "tensorrt"
    inference_device_id: int = 0
    fp16: bool = True
    warm_inference: bool = True
    architecture: str = "ddrnet"
    variance_scaled: bool = False
    norm_groups: int | None = None
    model_width: int = 64
    blocks: int = 8
    training_epochs: int = 10
    training_batch: int = 64
    learning_rate: float = 2e-3
    warm_learning_rate: float = 5e-4
    value_weight: float = 1.0
    training_threads: int = 4
    training_device: str = "cuda"
    training_precision: str = "bfloat16"
    schedule: str = "wsd"
    compile: bool = True
    restore_optimizer: bool = True
    warmup_epochs: int = 1
    report_every: int = 5
    validation_fraction: float = 0.1
    overlap_actor_learner: bool = True
    promotion_arena: bool = False
    promotion_score: float = 0.0
    maximum_truncation_rate: float = 0.02
    # Concede once the side to move has been losing for resign_window
    # consecutive plies. Zero disables it. This belongs to run identity: it
    # changes which positions reach the shard.
    resign_threshold: float = 0.0
    resign_window: int = 5
    resign_minimum_ply: int = 0
    # Uniform per game, in [komi_low, komi_high]. Positive favours White:
    # scoring is `black - white - komi > 0`.
    komi_low: float = 0.0
    komi_high: float = 0.0
    raster_kind: str = "semantic"
    resign_disable_fraction: float = 0.1
    arena_pairs: int = 16
    arena_simulations: int = 256
    arena_actors: int = 32
    telemetry_opponents: int = 2
    telemetry_pairs: int = 16
    # Rate an every-Nth checkpoint instead of all of them. The loop's Elo curve
    # is a trend, and a rating every update costs an arena per update while
    # telling you little the neighbouring points did not. 1 rates everything.
    telemetry_every: int = 1
    # Compress shards once they leave the replay window. Housekeeping only: it
    # never blocks an update, and a missed shard is retried after the next one.
    retire_shards: bool = True
    seed: int = 700_001
    arena_seed: int = 900_001
    initial_checkpoint: str | None = None
    initial_onnx: str | None = None
    initial_replay: tuple[str, ...] = ()

    @property
    def output_path(self) -> Path:
        return Path(self.output).resolve()

    def validate(self) -> None:
        positive = {
            "updates": self.updates,
            "samples per shard": self.samples_per_shard,
            "shards per update": self.shards_per_update,
            "replay window": self.replay_window,
            "resolution": self.resolution,
            "policy resolution": self.policy_resolution,
            "generation simulations": self.generation_simulations,
            "maximum plies": self.maximum_plies,
            "actors": self.actors,
            "writer queue games": self.writer_queue_games,
            "leaf batch": self.leaf_batch,
            "inference batch": self.inference_batch,
            "inference slots": self.inference_slots,
            "training epochs": self.training_epochs,
            "training batch": self.training_batch,
            "training threads": self.training_threads,
            "model width": self.model_width,
            "blocks": self.blocks,
            "arena pairs": self.arena_pairs,
            "arena simulations": self.arena_simulations,
            "arena actors": self.arena_actors,
            "telemetry pairs": self.telemetry_pairs,
        }
        invalid = [name for name, count in positive.items() if count <= 0]
        if invalid:
            raise ValueError(f"counts must be positive: {', '.join(invalid)}")
        if self.inference_batch < 2:
            raise ValueError("inference batch must be at least two")
        if self.leaf_batch > self.inference_batch:
            raise ValueError(
                "leaf batch must not exceed the inference batch capacity"
            )
        if self.maximum_prefetch_shards < 0:
            raise ValueError("maximum prefetch shards must be nonnegative")
        if self.inference_delay_ms < 0:
            raise ValueError("inference delay must be nonnegative")
        if self.temperature_plies < 0:
            raise ValueError("temperature plies must be nonnegative")
        if self.telemetry_opponents < 0:
            raise ValueError("telemetry opponents must be nonnegative")
        if self.seed < 0 or self.arena_seed < 0:
            raise ValueError("random seeds must be nonnegative")
        if self.report_every <= 0:
            raise ValueError("report interval must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup epochs must be nonnegative")
        if self.replay_window < self.shards_per_update:
            raise ValueError("replay window must hold at least one update quantum")
        if self.coarse_pool < 0 or self.coarse_pool > self.policy_resolution:
            raise ValueError("coarse pool must fit within the policy resolution")
        if self.policy_resolution > self.resolution:
            raise ValueError("policy resolution must not exceed raster resolution")
        if not 0.0 < self.radius < 0.5:
            raise ValueError("radius must be between zero and one half")
        if self.temperature < 0.0:
            raise ValueError("temperature must be nonnegative")
        if self.learning_rate <= 0.0 or self.warm_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.value_weight < 0.0:
            raise ValueError("value weight must be nonnegative")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation fraction must be in [0, 1)")
        if not 0.0 <= self.maximum_truncation_rate <= 1.0:
            raise ValueError("maximum truncation rate must be in [0, 1]")
        if not 0.0 <= self.promotion_score <= 1.0:
            raise ValueError("promotion score must be in [0, 1]")
        if self.provider not in {"cpu", "cuda", "tensorrt"}:
            raise ValueError(f"unsupported inference provider: {self.provider}")
        if self.inference_device_id < 0:
            raise ValueError("inference device id must be nonnegative")
        if self.architecture not in MODEL_ARCHITECTURES:
            raise ValueError(f"unsupported model architecture: {self.architecture}")
        if self.schedule not in {"wsd", "cosine"}:
            raise ValueError(f"unsupported learning-rate schedule: {self.schedule}")
        if self.training_precision not in {"float32", "bfloat16"}:
            raise ValueError(
                f"unsupported training precision: {self.training_precision}"
            )
        if (self.initial_checkpoint is None) != (self.initial_onnx is None):
            raise ValueError(
                "initial checkpoint and initial ONNX model must be supplied together"
            )
        if self.promotion_arena and self.promotion_score <= 0.0:
            raise ValueError("a promotion arena requires a nonzero promotion score")
        if not self.promotion_arena and self.promotion_score != 0.0:
            raise ValueError("promotion score requires the promotion arena")


@dataclass
class PipelineState:
    config_digest: str
    next_shard: int = 0
    updates_completed: int = 0
    consumed_through_shard: int = -1
    replay: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    rejected_models: list[dict[str, Any]] = field(default_factory=list)
    telemetry_pending: list[dict[str, Any]] = field(default_factory=list)
    telemetry_completed: list[str] = field(default_factory=list)
    started_unix_seconds: float = field(default_factory=time.time)
    active_wall_seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"schema": PIPELINE_SCHEMA, **asdict(self)}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "PipelineState":
        if value.get("schema") != PIPELINE_SCHEMA:
            raise ValueError(f"unsupported pipeline state: {value.get('schema')!r}")
        return cls(
            config_digest=str(value["config_digest"]),
            next_shard=int(value.get("next_shard", 0)),
            updates_completed=int(value.get("updates_completed", 0)),
            consumed_through_shard=int(value.get("consumed_through_shard", -1)),
            replay=list(value.get("replay", [])),
            models=list(value.get("models", [])),
            rejected_models=list(value.get("rejected_models", [])),
            telemetry_pending=list(value.get("telemetry_pending", [])),
            telemetry_completed=list(value.get("telemetry_completed", [])),
            started_unix_seconds=float(value.get("started_unix_seconds", time.time())),
            active_wall_seconds=float(value.get("active_wall_seconds", 0.0)),
        )


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    wall_seconds: float
    stdout_path: Path
    stderr_path: Path

    def json_documents(self) -> list[dict[str, Any]]:
        text = self.stdout_path.read_text(encoding="utf-8")
        documents: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        index = 0
        while True:
            start = text.find("{", index)
            if start < 0:
                return documents
            document, end = decoder.raw_decode(text, start)
            if isinstance(document, dict):
                documents.append(document)
            index = end

    def final_json(self) -> dict[str, Any]:
        documents = self.json_documents()
        if not documents:
            raise ValueError(f"command emitted no JSON document: {self.stdout_path}")
        return documents[-1]


class CommandRunner:
    def __init__(self, environment: dict[str, str]) -> None:
        self.environment = environment

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_prefix: Path,
    ) -> CommandResult:
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        command_path = log_prefix.with_suffix(".command.json")
        stdout_path = log_prefix.with_suffix(".stdout.log")
        stderr_path = log_prefix.with_suffix(".stderr.log")
        atomic_json(
            command_path,
            {
                "command": list(command),
                "cwd": str(cwd),
                "started_unix_seconds": time.time(),
            },
        )
        started = time.perf_counter()
        print(f"[{log_prefix.name}] {' '.join(command)}", flush=True)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=self.environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name != "nt",
            )
            try:
                returncode = await process.wait()
            except asyncio.CancelledError:
                self._terminate(process)
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except TimeoutError:
                    self._kill(process)
                    await process.wait()
                raise
        elapsed = time.perf_counter() - started
        result = CommandResult(
            command=tuple(command),
            returncode=returncode,
            wall_seconds=elapsed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        finished = json.loads(command_path.read_text(encoding="utf-8"))
        finished |= {
            "finished_unix_seconds": time.time(),
            "wall_seconds": elapsed,
            "returncode": returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        atomic_json(command_path, finished)
        if returncode != 0:
            tail = stderr_path.read_text(
                encoding="utf-8", errors="backslashreplace"
            )[-4000:]
            raise RuntimeError(
                f"command failed with exit code {returncode}; see {stderr_path}\n{tail}"
            )
        print(f"[{log_prefix.name}] complete in {elapsed:.1f}s", flush=True)
        return result

    @staticmethod
    def _terminate(process: asyncio.subprocess.Process) -> None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class LearnerService:
    """Supervise the long-lived JSON-lines learner process."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        stderr_task: asyncio.Task[None],
    ) -> None:
        self.process = process
        self.stderr_task = stderr_task
        self._lock = asyncio.Lock()
        self._next_request_id = 1

    @classmethod
    async def start(
        cls,
        *,
        cwd: Path,
        environment: dict[str, str],
        stderr_path: Path,
    ) -> "LearnerService":
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "vgo_training.learner",
            "--serve",
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stderr is not None

        async def drain_stderr() -> None:
            with stderr_path.open("ab") as destination:
                while block := await process.stderr.read(64 * 1024):
                    destination.write(block)
                    destination.flush()

        stderr_task = asyncio.create_task(drain_stderr())
        service = cls(process, stderr_task)
        try:
            ready = await asyncio.wait_for(service._read_response(), timeout=60.0)
        except BaseException:
            await service.close(force=True)
            raise
        if (
            ready.get("schema") != LEARNER_PROTOCOL_SCHEMA
            or ready.get("event") != "ready"
        ):
            await service.close(force=True)
            raise RuntimeError(f"learner did not become ready: {ready}")
        return service

    async def _read_response(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = await self.process.stdout.readline()
        if not line:
            returncode = await self.process.wait()
            raise RuntimeError(
                f"learner stopped before responding (exit {returncode})"
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError(f"learner response is not an object: {response!r}")
        if response.get("schema") != LEARNER_PROTOCOL_SCHEMA:
            raise RuntimeError(f"unsupported learner response: {response!r}")
        if response.get("ok") is False:
            kind = response.get("error_type", "LearnerError")
            detail = response.get("error", "learner update failed")
            raise RuntimeError(f"{kind}: {detail}")
        return response

    async def request(self, value: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self.process.returncode is not None:
                raise RuntimeError(f"learner has stopped with {self.process.returncode}")
            request = dict(value)
            request_id = self._next_request_id
            self._next_request_id += 1
            request["request_id"] = request_id
            assert self.process.stdin is not None
            self.process.stdin.write(
                (json.dumps(request, separators=(",", ":")) + "\n").encode()
            )
            await self.process.stdin.drain()
            response = await self._read_response()
            if response.get("request_id") != request_id:
                raise RuntimeError(
                    "learner response id does not match the request: "
                    f"{response.get('request_id')!r} != {request_id!r}"
                )
            return response

    async def close(self, *, force: bool = False) -> None:
        if self.process.returncode is None:
            if force:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
            else:
                try:
                    await self.request({"command": "shutdown"})
                except (BrokenPipeError, ConnectionError, RuntimeError):
                    try:
                        self.process.terminate()
                    except ProcessLookupError:
                        pass
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=15.0)
                except TimeoutError:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                    await self.process.wait()
        await self.stderr_task


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config
        self.output = config.output_path
        self.training = Path(__file__).resolve().parents[1]
        self.root = self.training.parent
        self.environment = runtime_environment()
        self.runner = CommandRunner(self.environment)
        self.state_path = self.output / "pipeline-state.json"
        # Strong references to in-flight shard compressions. asyncio only holds
        # weak references to tasks, so without this a retirement can be garbage
        # collected mid-run; each task discards itself on completion.
        self._retirements: set[asyncio.Task[Any]] = set()
        # Normalize tuples and Path-like values to their on-disk JSON forms so
        # a freshly constructed config compares equal to the same resumed run.
        config_value = json.loads(
            json.dumps({"schema": CONFIG_SCHEMA, **asdict(config)})
        )
        config_value["output"] = str(self.output)
        self._config_value = config_value
        self.config_digest = canonical_digest(identity_config(config_value))
        self.learner: LearnerService | None = None
        self._gpu_lock = asyncio.Lock()
        self._lease: BinaryIO | None = None
        self.output.mkdir(parents=True, exist_ok=True)
        # Creation itself publishes config/state, so it needs the same lease as
        # execution. The lease is reacquired and state is reloaded at run time;
        # a preconstructed object can therefore never overwrite newer progress.
        with self._run_lease():
            self.state = self._load_or_create_state(self._config_value)

    @classmethod
    def resume(cls, output: Path) -> "Pipeline":
        config_path = output.resolve() / "pipeline-config.json"
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if value.pop("schema", None) != CONFIG_SCHEMA:
            raise ValueError(f"unsupported pipeline configuration: {config_path}")
        value["initial_replay"] = tuple(value.get("initial_replay", ()))
        return cls(PipelineConfig(**value))

    def _load_or_create_state(self, config_value: dict[str, Any]) -> PipelineState:
        self.output.mkdir(parents=True, exist_ok=True)
        config_path = self.output / "pipeline-config.json"
        prior: dict[str, Any] | None = None
        if config_path.exists():
            prior = json.loads(config_path.read_text(encoding="utf-8"))
            if identity_config(prior) != identity_config(config_value):
                raise ValueError(
                    "pipeline's learning configuration differs from the existing run"
                )
        if self.state_path.exists():
            state = PipelineState.from_json(
                json.loads(self.state_path.read_text(encoding="utf-8"))
            )
            if state.config_digest != self.config_digest:
                raise ValueError("pipeline state belongs to a different configuration")
        else:
            state = PipelineState(config_digest=self.config_digest)
            for index, replay in enumerate(self.config.initial_replay):
                path = Path(replay).resolve(strict=True)
                if path.is_dir():
                    path = (path / "dataset.vgo").resolve(strict=True)
                state.replay.append(
                    asdict(
                        self._replay_artifact(
                            path.parent,
                            -len(self.config.initial_replay) + index,
                            verify_dataset_digest=True,
                            expected_samples=None,
                        )
                    )
                )
            if self.config.initial_checkpoint is not None:
                checkpoint = Path(self.config.initial_checkpoint).resolve(strict=True)
                onnx = Path(str(self.config.initial_onnx)).resolve(strict=True)
                state.models.append(
                    asdict(
                        ModelArtifact.from_paths(
                            version=-1,
                            checkpoint=checkpoint,
                            onnx=onnx,
                            parent_version=None,
                        )
                    )
                )
            atomic_json(self.state_path, state.to_json())

        if self.config.updates < state.updates_completed:
            raise ValueError(
                "updates cannot be reduced below already completed work"
            )
        if prior != config_value:
            history_path = self.output / "pipeline-config-history.json"
            history = (
                json.loads(history_path.read_text(encoding="utf-8"))
                if history_path.exists()
                else []
            )
            history.append(
                {
                    "effective_unix_seconds": time.time(),
                    "updates_completed": state.updates_completed,
                    "next_shard": state.next_shard,
                    "config": config_value,
                }
            )
            atomic_json(history_path, history)
            atomic_json(config_path, config_value)
        return state

    def _save_state(self) -> None:
        atomic_json(self.state_path, self.state.to_json())

    @contextmanager
    def _run_lease(self) -> Iterator[None]:
        """Exclude a second coordinator from mutating the same run."""

        if self._lease is not None:
            yield
            return
        path = self.output / ".pipeline.lock"
        lease = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if lease.seek(0, os.SEEK_END) == 0:
                    lease.write(b"\0")
                    lease.flush()
                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            lease.close()
            raise RuntimeError(
                f"another pipeline owns the run at {self.output}"
            ) from error
        self._lease = lease
        try:
            yield
        finally:
            self._lease = None
            if os.name == "nt":
                import msvcrt

                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()

    @property
    def incumbent(self) -> ModelArtifact | None:
        if not self.state.models:
            return None
        return ModelArtifact(**self.state.models[-1])

    def _rust_command(self, binary: str) -> list[str]:
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
        self,
        *,
        output: Path,
        sequence: int,
        model: ModelArtifact | None,
    ) -> list[str]:
        config = self.config
        command = self._rust_command("vgo-generate-demo") + [
            "--samples",
            str(config.samples_per_shard),
            "--resolution",
            str(config.resolution),
            "--policy-resolution",
            str(config.policy_resolution),
            "--simulations",
            str(config.generation_simulations),
            "--coarse-pool",
            str(config.coarse_pool),
            "--temperature",
            str(config.temperature),
            "--temperature-plies",
            str(config.temperature_plies),
            "--max-plies",
            str(config.maximum_plies),
            "--resign-threshold",
            str(config.resign_threshold),
            "--resign-window",
            str(config.resign_window),
            "--resign-minimum-ply",
            str(config.resign_minimum_ply),
            # `=` form: a negative komi otherwise parses as a flag, since clap
            # cannot tell `-0.1` from a short option.
            f"--komi-low={config.komi_low}",
            f"--komi-high={config.komi_high}",
            "--raster-kind",
            str(config.raster_kind),
            "--model-raster-kind",
            str(config.raster_kind),
            "--resign-disable-fraction",
            str(config.resign_disable_fraction),
            "--radius",
            str(config.radius),
            "--seed",
            str(config.seed + sequence * 1_000_003),
            "--examples",
            "0",
            "--output",
            str(output),
            "--maximum-batch",
            str(config.inference_batch),
            "--delay-ms",
            str(config.inference_delay_ms),
            "--inference-slots",
            str(config.inference_slots),
            "--actors",
            str(config.actors),
            "--writer-queue-games",
            str(config.writer_queue_games),
            "--leaf-batch",
            str(config.leaf_batch),
            "--provider",
            config.provider,
            "--device-id",
            str(config.inference_device_id),
            "--fp16",
            str(config.fp16).lower(),
            "--cache-directory",
            str((self.root / "artifacts" / "onnx-cache").resolve()),
        ]
        if model is None:
            command.extend(["--runtime", "naive"])
        else:
            command.extend(["--runtime", "onnx", "--model", model.onnx])
        return command

    def arena_command(
        self,
        *,
        candidate: Path,
        opponents: Sequence[Path],
        seed: int,
        pairs: int,
    ) -> list[str]:
        config = self.config
        command = self._rust_command("vgo-arena") + [
            "--candidate",
            str(candidate),
            # Both seats are this run's own models, so both read its layout.
            "--candidate-raster-kind",
            str(self.config.raster_kind),
            "--pairs",
            str(pairs),
            "--simulations",
            str(config.arena_simulations),
            "--coarse-pool",
            str(config.coarse_pool),
            "--max-plies",
            str(config.maximum_plies),
            "--threads",
            str(config.arena_actors),
            "--leaf-batch",
            str(config.leaf_batch),
            "--resolution",
            str(config.resolution),
            "--policy-resolution",
            str(config.policy_resolution),
            "--radius",
            str(config.radius),
            "--seed",
            str(seed),
            "--maximum-batch",
            str(config.inference_batch),
            "--delay-ms",
            str(config.inference_delay_ms),
            "--provider",
            config.provider,
            "--device-id",
            str(config.inference_device_id),
            "--fp16",
            str(config.fp16).lower(),
            "--cache-directory",
            str((self.root / "artifacts" / "onnx-cache").resolve()),
        ]
        for opponent in opponents:
            command.extend(["--opponent", str(opponent)])
        return command

    @asynccontextmanager
    async def _gpu_lease(self) -> AsyncIterator[None]:
        if self.config.overlap_actor_learner:
            yield
            return
        async with self._gpu_lock:
            yield

    async def _generate_shard(
        self, model: ModelArtifact | None
    ) -> ReplayArtifact:
        sequence = self.state.next_shard
        replay_root = self.output / "replay"
        final = replay_root / f"shard-{sequence:06d}"
        staging = replay_root / f"shard-{sequence:06d}.staging"
        if final.exists():
            return self._replay_artifact(
                final, sequence, verify_dataset_digest=True
            )
        if staging.exists():
            shutil.rmtree(staging)
        staging.parent.mkdir(parents=True, exist_ok=True)
        command = self.generation_command(
            output=staging, sequence=sequence, model=model
        )
        async with self._gpu_lease():
            result = await self.runner.run(
                command,
                cwd=self.root,
                log_prefix=self.output / "logs" / f"generate-{sequence:06d}",
            )
        report = result.final_json()
        dataset = staging / "dataset.vgo"
        manifest_path = staging / "manifest.json"
        if not dataset.exists() or not manifest_path.exists():
            raise RuntimeError("generator returned success without publishing replay")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("dataset_sha256") != manifest.get("dataset_sha256"):
            raise RuntimeError("generator report and manifest digests disagree")
        behavior_digest = manifest.get(
            "behavior_model_sha256", manifest.get("model_sha256")
        )
        expected_digest = None if model is None else model.onnx_sha256
        if behavior_digest != expected_digest:
            raise RuntimeError(
                "replay behavior-model digest does not match the pinned actor model"
            )
        staging.replace(final)
        sync_directory(final.parent)
        return self._replay_artifact(
            final, sequence, verify_dataset_digest=False
        )

    def _replay_artifact(
        self,
        directory: Path,
        sequence: int,
        *,
        verify_dataset_digest: bool,
        expected_samples: int | None = None,
    ) -> ReplayArtifact:
        dataset = directory / "dataset.vgo"
        manifest_path = directory / "manifest.json"
        if not dataset.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"incomplete replay shard: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "vgo.replay-shard.v1":
            raise RuntimeError(f"unsupported replay manifest: {manifest_path}")
        samples = int(manifest.get("samples", -1))
        expected_samples = (
            self.config.samples_per_shard
            if expected_samples is None and sequence >= 0
            else expected_samples
        )
        if expected_samples is not None and samples != expected_samples:
            raise RuntimeError(
                f"replay shard has {samples} samples, expected "
                f"{expected_samples}"
            )
        digest = manifest.get("dataset_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("replay manifest has no valid dataset digest")
        if manifest.get("shard_id") not in {None, f"sha256:{digest}"}:
            raise RuntimeError("replay shard identity does not match its digest")
        expected_bytes = manifest.get("dataset_bytes")
        if expected_bytes is not None and dataset.stat().st_size != int(expected_bytes):
            raise RuntimeError("replay dataset size does not match its manifest")
        if verify_dataset_digest and file_sha256(dataset) != digest:
            raise RuntimeError("replay dataset checksum does not match its manifest")
        behavior_digest = manifest.get(
            "behavior_model_sha256", manifest.get("model_sha256")
        )
        if behavior_digest is not None and (
            not isinstance(behavior_digest, str) or len(behavior_digest) != 64
        ):
            raise RuntimeError("replay manifest has an invalid behavior model digest")
        return ReplayArtifact(
            sequence=sequence,
            path=str(dataset.resolve()),
            manifest=str(manifest_path.resolve()),
            samples=samples,
            behavior_model_sha256=behavior_digest,
            dataset_sha256=digest,
            seed=int(manifest.get("seed", self.config.seed)),
        )

    def _commit_replay(self, replay: ReplayArtifact) -> None:
        existing = next(
            (
                value
                for value in self.state.replay
                if int(value["sequence"]) == replay.sequence
            ),
            None,
        )
        if existing is not None:
            if existing["dataset_sha256"] != replay.dataset_sha256:
                raise RuntimeError(
                    f"replay sequence {replay.sequence} changed identity"
                )
            self.state.next_shard = max(
                self.state.next_shard, replay.sequence + 1
            )
            self._save_state()
            return
        self.state.replay.append(asdict(replay))
        self.state.replay.sort(key=lambda value: int(value["sequence"]))
        self.state.next_shard = max(self.state.next_shard, replay.sequence + 1)
        self._save_state()

    def _pending_replay(self) -> list[ReplayArtifact]:
        return [
            ReplayArtifact(**value)
            for value in self.state.replay
            if int(value["sequence"]) > self.state.consumed_through_shard
        ]

    def _active_replay(self, through_sequence: int) -> list[ReplayArtifact]:
        eligible = [
            ReplayArtifact(**value)
            for value in self.state.replay
            if int(value["sequence"]) <= through_sequence
        ]
        return eligible[-self.config.replay_window :]

    def _update_spec(self, update: int) -> tuple[dict[str, Any], Path]:
        update_path = self.output / "updates" / f"update-{update:06d}"
        spec_path = update_path / "update-spec.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            self._validate_update_spec(spec, update, spec_path)
            return spec, update_path
        pending = self._pending_replay()
        if len(pending) < self.config.shards_per_update:
            raise RuntimeError("learner update requested without enough new replay")
        through = pending[self.config.shards_per_update - 1].sequence
        active = self._active_replay(through)
        parent = self.incumbent
        spec = {
            "schema": UPDATE_SCHEMA,
            "update": update,
            "through_shard": through,
            "active_replay": [asdict(replay) for replay in active],
            "parent_model": asdict(parent) if parent is not None else None,
            "created_unix_seconds": time.time(),
        }
        atomic_json(spec_path, spec)
        return spec, update_path

    def _validate_update_spec(
        self, spec: dict[str, Any], update: int, path: Path
    ) -> None:
        """Reconcile durable update intent with the authoritative run state."""

        if spec.get("schema") != UPDATE_SCHEMA:
            raise RuntimeError(f"unsupported update spec: {path}")
        if int(spec.get("update", -1)) != update:
            raise RuntimeError(f"update spec has the wrong sequence: {path}")
        pending = self._pending_replay()
        if len(pending) < self.config.shards_per_update:
            raise RuntimeError(
                "durable update spec has no matching unconsumed replay boundary"
            )
        expected_through = pending[self.config.shards_per_update - 1].sequence
        if int(spec.get("through_shard", -1)) != expected_through:
            raise RuntimeError(
                "durable update spec consumes a different replay boundary"
            )
        expected_replay = [
            asdict(replay) for replay in self._active_replay(expected_through)
        ]
        if spec.get("active_replay") != expected_replay:
            raise RuntimeError(
                "durable update spec names a different replay snapshot"
            )
        parent = self.incumbent
        expected_parent = asdict(parent) if parent is not None else None
        if spec.get("parent_model") != expected_parent:
            raise RuntimeError(
                "durable update spec names a different parent model"
            )

    async def _train(self, spec: dict[str, Any], update_path: Path) -> dict[str, Any]:
        if self.learner is None:
            raise RuntimeError("learner service is not running")
        checkpoint = update_path / "candidate.pt"
        report_path = checkpoint.with_suffix(".pt.json")
        if checkpoint.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self._validate_training_artifact(
                spec, checkpoint, report, verify_checkpoint_digest=True
            )
            atomic_json(report_path, report)
            return report
        parent = spec.get("parent_model")
        request: dict[str, Any] = {
            "command": "update",
            "datasets": [
                replay["path"] for replay in spec["active_replay"]
            ],
            "output": str(checkpoint),
            "epochs": self.config.training_epochs,
            "batch_size": self.config.training_batch,
            "learning_rate": (
                self.config.learning_rate
                if parent is None
                else self.config.warm_learning_rate
            ),
            "value_weight": self.config.value_weight,
            "model_width": self.config.model_width,
            "blocks": self.config.blocks,
            "architecture": self.config.architecture,
            "variance_scaled": self.config.variance_scaled,
            "norm_groups": self.config.norm_groups,
            "threads": self.config.training_threads,
            "device": self.config.training_device,
            "precision": self.config.training_precision,
            "schedule": self.config.schedule,
            "compile": self.config.compile,
            "restore_optimizer": self.config.restore_optimizer,
            "warmup_epochs": self.config.warmup_epochs,
            "seed": self.config.seed + int(spec["update"]),
            "report_every": self.config.report_every,
            "validation_fraction": self.config.validation_fraction,
            "initial_checkpoint": parent["checkpoint"] if parent else None,
        }
        response = await self.learner.request(request)
        report = response.get("result", response.get("report", response))
        if not isinstance(report, dict):
            raise RuntimeError(f"learner returned an invalid report: {response}")
        if not checkpoint.exists():
            raise RuntimeError("learner returned success without a checkpoint")
        self._validate_training_artifact(
            spec, checkpoint, report, verify_checkpoint_digest=False
        )
        # The response is authoritative even if a stale sidecar survived an
        # interrupted prior attempt. Re-publish it durably after validating the
        # new checkpoint rather than leaving a mismatched recovery fast path.
        atomic_json(report_path, report)
        return report

    def _validate_training_artifact(
        self,
        spec: dict[str, Any],
        checkpoint: Path,
        report: dict[str, Any],
        *,
        verify_checkpoint_digest: bool,
    ) -> None:
        if report.get("schema") != "vgo.training-run.v3":
            raise RuntimeError("unsupported learner training report")
        parent = spec.get("parent_model")
        expected_parent_digest = (
            None if parent is None else parent["checkpoint_sha256"]
        )
        if report.get("parent_checkpoint_sha256") != expected_parent_digest:
            raise RuntimeError(
                "learner did not train from the update's authoritative parent"
            )
        expected_datasets = [
            str(Path(replay["path"]).resolve())
            for replay in spec["active_replay"]
        ]
        if report.get("datasets") != expected_datasets:
            raise RuntimeError("training report names a different replay snapshot")
        expected_digests = [
            replay["dataset_sha256"] for replay in spec["active_replay"]
        ]
        if report.get("dataset_digests") != expected_digests:
            raise RuntimeError("training report replay digests do not match the update")
        digest = report.get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("training report has no valid checkpoint digest")
        if str(checkpoint.resolve()) != report.get("checkpoint"):
            raise RuntimeError("training report checkpoint path does not match the update")
        if verify_checkpoint_digest and file_sha256(checkpoint) != digest:
            raise RuntimeError("checkpoint checksum does not match its training report")

    async def _export(self, update_path: Path) -> dict[str, Any]:
        checkpoint = update_path / "candidate.pt"
        onnx = update_path / "candidate.onnx"
        report_path = onnx.with_suffix(".onnx.json")
        if onnx.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self._validate_export_artifact(checkpoint, onnx, report)
            atomic_json(report_path, report)
            return report
        command = [
            sys.executable,
            "-m",
            "vgo_training.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(onnx),
            "--maximum-batch",
            str(self.config.inference_batch),
        ]
        result = await self.runner.run(
            command,
            cwd=self.training,
            log_prefix=self.output
            / "logs"
            / f"export-{update_path.name.removeprefix('update-')}",
        )
        report = result.final_json()
        if not onnx.exists():
            raise RuntimeError("export returned success without an ONNX model")
        self._validate_export_artifact(checkpoint, onnx, report)
        atomic_json(report_path, report)
        return report

    async def _warm_inference(
        self, update: int, update_path: Path
    ) -> dict[str, Any] | None:
        """Build the candidate's TensorRT engine during the current actor tail."""

        if not self.config.warm_inference or self.config.provider != "tensorrt":
            return None
        command = [
            cargo_executable(),
            "run",
            "--release",
            "-p",
            "vgo-inference",
            "--bin",
            "vgo-onnx-bench",
            "--",
            "--model",
            str(update_path / "candidate.onnx"),
            "--provider",
            self.config.provider,
            "--device-id",
            str(self.config.inference_device_id),
            "--cache-directory",
            str((self.root / "artifacts" / "onnx-cache").resolve()),
            "--resolution",
            str(self.config.resolution),
            "--policy-resolution",
            str(self.config.policy_resolution),
            "--raster-kind",
            str(self.config.raster_kind),
            "--batch",
            str(self.config.inference_batch),
            "--fp16",
            str(self.config.fp16).lower(),
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--compare-python",
            "false",
        ]
        result = await self.runner.run(
            command,
            cwd=self.root,
            log_prefix=self.output / "logs" / f"warmup-{update:06d}",
        )
        report = result.final_json()
        expected = {
            "provider": self.config.provider,
            "resolution": self.config.resolution,
            "policy_resolution": self.config.policy_resolution,
            "batch": self.config.inference_batch,
            "fp16": self.config.fp16,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "inference warmup did not use the exported model's runtime contract"
            )
        return report

    def _validate_export_artifact(
        self,
        checkpoint: Path,
        onnx: Path,
        report: dict[str, Any],
    ) -> None:
        if report.get("schema") != "vgo.onnx-manifest.v1":
            raise RuntimeError("unsupported ONNX export report")
        if report.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise RuntimeError("ONNX export names a different checkpoint")
        if report.get("checkpoint") != str(checkpoint.resolve()):
            raise RuntimeError("ONNX export checkpoint path does not match the update")
        if report.get("onnx") != str(onnx.resolve()):
            raise RuntimeError("ONNX export path does not match the update")
        digest = report.get("onnx_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("ONNX export report has no valid model digest")
        maximum_batch = report.get("input", {}).get("maximum_batch")
        if int(maximum_batch) != self.config.inference_batch:
            raise RuntimeError("ONNX export batch contract does not match the pipeline")
        if file_sha256(onnx) != digest:
            raise RuntimeError("ONNX checksum does not match its export report")

    @staticmethod
    def _promotion_decision(
        arena: dict[str, Any],
        minimum_score: float,
        maximum_truncation_rate: float,
    ) -> bool:
        games = int(arena["games"])
        completed = int(arena["completed"])
        if games <= 0 or not 0 <= completed <= games:
            raise ValueError("arena game counts are inconsistent")
        return (
            completed > 0
            and (games - completed) / games <= maximum_truncation_rate
            and float(arena["candidate_score"]) >= minimum_score
        )

    async def _gate_candidate(
        self, update: int, update_path: Path
    ) -> tuple[bool, dict[str, Any] | None]:
        incumbent = self.incumbent
        if not self.config.promotion_arena or incumbent is None:
            return True, None
        command = self.arena_command(
            candidate=update_path / "candidate.onnx",
            opponents=[Path(incumbent.onnx)],
            seed=self.config.arena_seed + update * 10_003,
            pairs=self.config.arena_pairs,
        )
        result = await self.runner.run(
            command,
            cwd=self.root,
            log_prefix=self.output / "logs" / f"promotion-{update:06d}",
        )
        arena = result.final_json()
        return (
            self._promotion_decision(
                arena,
                self.config.promotion_score,
                self.config.maximum_truncation_rate,
            ),
            arena,
        )

    def _queue_telemetry(self, model: ModelArtifact) -> None:
        if self.config.telemetry_opponents <= 0:
            return
        # Rating every checkpoint costs an arena per update to resolve a curve
        # that is legible from a fraction of the points. Version 0 is always
        # rated so the trend has an anchor at the origin.
        every = max(1, self.config.telemetry_every)
        if model.version % every != 0:
            return
        opponents = [
            ModelArtifact(**value)
            for value in self.state.models
            if int(value["version"]) < model.version
        ]
        if not opponents:
            return
        # A counter-derived permutation gives a deterministic sample without
        # depending on process-global random state.
        opponents.sort(
            key=lambda value: hashlib.sha256(
                f"{self.config.arena_seed}:{model.version}:{value.version}".encode()
            ).digest()
        )
        selected = opponents[: self.config.telemetry_opponents]
        group_seed = (
            self.config.arena_seed + model.version * 1_000_003
        )
        for order, opponent in enumerate(selected):
            job_id = f"v{model.version:06d}-vs-v{opponent.version:06d}"
            if job_id in self.state.telemetry_completed or any(
                pending["id"] == job_id for pending in self.state.telemetry_pending
            ):
                continue
            self.state.telemetry_pending.append(
                {
                    "schema": TELEMETRY_SCHEMA,
                    "id": job_id,
                    "candidate_version": model.version,
                    "candidate": model.onnx,
                    "opponent_version": opponent.version,
                    "opponent": opponent.onnx,
                    "pairs": self.config.telemetry_pairs,
                    "group_seed": group_seed,
                    "group_index": order,
                    "seed": group_seed + order * 1_000_003,
                }
            )
        self._save_state()

    def _commit_publication(
        self,
        report: dict[str, Any],
        *,
        verify_model_files: bool = False,
    ) -> ModelArtifact | None:
        if report.get("schema") != "vgo.pipeline-publication.v1":
            raise RuntimeError("unsupported model publication")
        update = int(report["update"])
        model = ModelArtifact(**report["model"])
        accepted = bool(report["accepted"])
        if model.version != update:
            raise RuntimeError("publication model version does not match its update")
        if model.accepted != accepted:
            raise RuntimeError("publication decision and model metadata disagree")
        if verify_model_files:
            if file_sha256(Path(model.checkpoint)) != model.checkpoint_sha256:
                raise RuntimeError("published checkpoint checksum mismatch")
            if file_sha256(Path(model.onnx)) != model.onnx_sha256:
                raise RuntimeError("published ONNX checksum mismatch")
        accepted_existing = next(
            (
                value
                for value in self.state.models
                if int(value["version"]) == model.version
            ),
            None,
        )
        rejected_existing = next(
            (
                value
                for value in self.state.rejected_models
                if int(value["version"]) == model.version
            ),
            None,
        )
        existing = accepted_existing or rejected_existing
        if existing is not None:
            if existing != asdict(model) or bool(accepted_existing) != accepted:
                raise RuntimeError(
                    f"model version {model.version} changed publication identity"
                )
        else:
            if accepted:
                self.state.models.append(asdict(model))
                self._queue_telemetry(model)
            else:
                self.state.rejected_models.append(asdict(model))
        self.state.updates_completed = max(
            self.state.updates_completed, update + 1
        )
        self.state.consumed_through_shard = max(
            self.state.consumed_through_shard,
            int(report["through_shard"]),
        )
        self._save_state()
        return model if accepted else None

    async def _learn_and_publish(
        self,
        update: int,
        spec: dict[str, Any] | None = None,
        update_path: Path | None = None,
    ) -> ModelArtifact | None:
        if spec is None or update_path is None:
            spec, update_path = self._update_spec(update)
        publication_path = update_path / "publication.json"
        if publication_path.exists():
            report = json.loads(publication_path.read_text(encoding="utf-8"))
            if int(report.get("update", -1)) != update:
                raise RuntimeError("publication belongs to a different update")
            if int(report.get("through_shard", -1)) != int(spec["through_shard"]):
                raise RuntimeError("publication consumed a different replay boundary")
            parent = spec.get("parent_model")
            expected_parent = None if parent is None else int(parent["version"])
            if report.get("model", {}).get("parent_version") != expected_parent:
                raise RuntimeError("publication names a different parent model")
            return self._commit_publication(
                report, verify_model_files=True
            )
        started = time.perf_counter()
        # In serialized mode one lease spans the complete publication path, so
        # a waiting actor cannot slip between training and promotion and delay
        # the new incumbent by an entire shard.
        async with self._gpu_lease():
            training_report = await self._train(spec, update_path)
            export_report = await self._export(update_path)
            if (
                training_report["checkpoint_sha256"]
                != export_report["checkpoint_sha256"]
            ):
                raise RuntimeError(
                    "training and export disagree on the candidate checkpoint"
                )
            warmup_report = await self._warm_inference(update, update_path)
            accepted, arena = await self._gate_candidate(update, update_path)
        parent = self.incumbent
        model = ModelArtifact(
            version=update,
            checkpoint=str((update_path / "candidate.pt").resolve()),
            onnx=str((update_path / "candidate.onnx").resolve()),
            checkpoint_sha256=str(export_report["checkpoint_sha256"]),
            onnx_sha256=str(export_report["onnx_sha256"]),
            parent_version=parent.version if parent else None,
            accepted=accepted,
        )
        report = {
            "schema": "vgo.pipeline-publication.v1",
            "update": update,
            "through_shard": int(spec["through_shard"]),
            "training": training_report,
            "export": export_report,
            "inference_warmup": warmup_report,
            "promotion_arena": arena,
            "accepted": accepted,
            "model": asdict(model),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(publication_path, report)
        published = self._commit_publication(report)
        self._retire_aged_shards(int(spec["through_shard"]))
        return published

    def _retire_aged_shards(self, through_sequence: int) -> None:
        """Compress shards the learner has finished with.

        An update trains on `_active_replay(through_sequence)`, the last
        `replay_window` shards, so anything older is never read again by this
        run. This is the moment that is certainly true: the update has published
        and its replay boundary is durably recorded.

        The shards are still worth keeping -- each one records the model that
        generated it, so the set is a reproducible history -- but a 6.04 GB
        dense shard compresses 30-70x here, because five policy-shaped arrays
        store 16385 float32 slots per sample to hold ~24 nonzero values.

        Compression runs in a thread and is never awaited: it is housekeeping,
        and a slow or failed compression must not delay the next update. A shard
        that is missed simply gets picked up after the following update.
        """
        if not self.config.retire_shards:
            return
        cutoff = through_sequence - self.config.replay_window + 1
        if cutoff <= 0:
            return
        stale = [
            Path(value["path"])
            for value in self.state.replay
            if int(value["sequence"]) < cutoff
        ]
        pending = [path for path in stale if path.exists()]
        if not pending:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Retirement is housekeeping scheduled from the running pipeline. If
            # there is no loop -- a synchronous caller, or a test -- do it inline
            # rather than failing the update that just published.
            for path in pending:
                try:
                    _log_completed_retirement(*_compress_shard(path))
                except Exception as error:  # never fail an update over cleanup
                    print(f"[retire] failed: {error}", flush=True)
            return
        for path in pending:
            # Fire and forget. asyncio.to_thread returns a task we deliberately
            # do not await; the callback exists only so a failure is visible in
            # the log rather than swallowed as a never-retrieved exception.
            task = asyncio.ensure_future(asyncio.to_thread(_compress_shard, path))
            task.add_done_callback(_log_retirement)
            self._retirements.add(task)
            task.add_done_callback(self._retirements.discard)

    def _should_start_generation(
        self,
        pending: Sequence[ReplayArtifact],
        *,
        learning_through: int | None,
        generation_active: bool,
    ) -> bool:
        if generation_active:
            return False
        remaining_updates = self.config.updates - self.state.updates_completed
        remaining_shards = remaining_updates * self.config.shards_per_update
        if len(pending) >= remaining_shards:
            return False
        if learning_through is None:
            return len(pending) < self.config.shards_per_update
        prefetched = sum(
            replay.sequence > learning_through for replay in pending
        )
        return prefetched < self.config.maximum_prefetch_shards

    def report(self) -> dict[str, Any]:
        return {
            "schema": "vgo.pipeline-run.v1",
            "config_digest": self.config_digest,
            "wall_seconds": self.state.active_wall_seconds,
            "calendar_seconds": time.time() - self.state.started_unix_seconds,
            "updates_completed": self.state.updates_completed,
            "replay_shards": len(self.state.replay),
            "models": self.state.models,
            "rejected_models": self.state.rejected_models,
            "telemetry_pending": len(self.state.telemetry_pending),
            "telemetry_completed": len(self.state.telemetry_completed),
            "final_model": self.state.models[-1] if self.state.models else None,
            "utilization": self._utilization_report(),
        }

    def _utilization_report(self) -> dict[str, Any]:
        """Aggregate the counters that explain where pipeline capacity went."""

        generation_wall = 0.0
        summed_game = 0.0
        writer_backpressure = 0.0
        inference_seconds = 0.0
        inference_positions = 0
        inference_batches = 0
        generated_shards = 0
        for replay in self.state.replay:
            if int(replay.get("sequence", -1)) < 0:
                continue
            try:
                manifest = json.loads(
                    Path(str(replay["manifest"])).read_text(encoding="utf-8")
                )
                generation = manifest["generation_metrics"]
                broker = manifest["broker_metrics"]
                generation_wall += float(generation["wall_seconds"])
                summed_game += float(generation["summed_game_seconds"])
                writer_backpressure += float(
                    generation["writer_backpressure_seconds"]
                )
                inference_seconds += float(broker["inference_seconds"])
                inference_positions += int(broker["positions"])
                inference_batches += int(broker["batches"])
                generated_shards += 1
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        learning_wall = 0.0
        learner_wall = 0.0
        optimization_wall = 0.0
        replay_cache_hits = 0
        replay_cache_misses = 0
        measured_updates = 0
        for update in range(self.state.updates_completed):
            publication_path = (
                self.output
                / "updates"
                / f"update-{update:06d}"
                / "publication.json"
            )
            try:
                publication = json.loads(
                    publication_path.read_text(encoding="utf-8")
                )
                training = publication["training"]
                learning_wall += float(publication["wall_seconds"])
                learner_wall += float(training["wall_seconds"])
                optimization_wall += float(training["optimization_seconds"])
                replay_cache_hits += int(training.get("replay_cache_hits", 0))
                replay_cache_misses += int(
                    training.get("replay_cache_misses", 0)
                )
                measured_updates += 1
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        batch_capacity = inference_batches * self.config.inference_batch
        cache_accesses = replay_cache_hits + replay_cache_misses
        stage_work = generation_wall + learning_wall
        active_wall = self.state.active_wall_seconds
        return {
            "measured_generation_shards": generated_shards,
            "measured_updates": measured_updates,
            "generation_wall_seconds": generation_wall,
            "learning_wall_seconds": learning_wall,
            "learner_wall_seconds": learner_wall,
            "learner_optimization_seconds": optimization_wall,
            "stage_work_seconds": stage_work,
            "pipeline_overlap_factor": (
                stage_work / active_wall if active_wall > 0.0 else None
            ),
            "average_active_games": (
                summed_game / generation_wall
                if generation_wall > 0.0
                else None
            ),
            "inference_batch_fill": (
                inference_positions / batch_capacity
                if batch_capacity > 0
                else None
            ),
            "inference_positions": inference_positions,
            "inference_batches": inference_batches,
            "inference_seconds": inference_seconds,
            "writer_backpressure_seconds": writer_backpressure,
            "learner_optimization_fraction": (
                optimization_wall / learner_wall
                if learner_wall > 0.0
                else None
            ),
            "replay_cache_hit_ratio": (
                replay_cache_hits / cache_accesses
                if cache_accesses > 0
                else None
            ),
            "replay_cache_hits": replay_cache_hits,
            "replay_cache_misses": replay_cache_misses,
        }

    async def run(self) -> dict[str, Any]:
        with self._run_lease():
            return await self._run()

    async def _run(self) -> dict[str, Any]:
        self.state = self._load_or_create_state(self._config_value)
        if self.state.updates_completed >= self.config.updates:
            final_path = self.output / "run.json"
            report = self.report()
            atomic_json(final_path, report)
            return report
        active_started = time.perf_counter()
        try:
            self.learner = await LearnerService.start(
                cwd=self.training,
                environment=self.environment,
                stderr_path=self.output / "logs" / "learner.stderr.log",
            )
        except BaseException:
            self.state.active_wall_seconds += (
                time.perf_counter() - active_started
            )
            self._save_state()
            raise
        generation: asyncio.Task[ReplayArtifact] | None = None
        learning: asyncio.Task[ModelArtifact | None] | None = None
        learning_through: int | None = None
        failed = True
        try:
            while self.state.updates_completed < self.config.updates:
                pending = self._pending_replay()

                if (
                    learning is None
                    and len(pending) >= self.config.shards_per_update
                ):
                    update = self.state.updates_completed
                    spec, update_path = self._update_spec(update)
                    learning_through = int(spec["through_shard"])
                    learning = asyncio.create_task(
                        self._learn_and_publish(update, spec, update_path)
                    )

                if self._should_start_generation(
                    pending,
                    learning_through=learning_through,
                    generation_active=generation is not None,
                ):
                    # The incumbent is captured once per shard. A publication
                    # can finish while this task runs, but the shard remains a
                    # valid, explicitly stamped sample from the older policy.
                    generation = asyncio.create_task(
                        self._generate_shard(self.incumbent)
                    )

                tasks = {
                    task
                    for task in (generation, learning)
                    if task is not None
                }
                if not tasks:
                    raise RuntimeError("pipeline has no runnable work")
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                if generation in done:
                    replay = generation.result()
                    self._commit_replay(replay)
                    generation = None
                if learning in done:
                    learning.result()
                    learning = None
                    learning_through = None
            failed = False
        finally:
            active = [
                task
                for task in (generation, learning)
                if task is not None and not task.done()
            ]
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            try:
                await self.learner.close(force=failed)
            finally:
                self.learner = None
                self.state.active_wall_seconds += (
                    time.perf_counter() - active_started
                )
                self._save_state()

        report = self.report()
        atomic_json(self.output / "run.json", report)
        return report

    async def drain_telemetry(self) -> None:
        with self._run_lease():
            await self._drain_telemetry()

    async def _drain_telemetry(self) -> None:
        self.state = self._load_or_create_state(self._config_value)
        while self.state.telemetry_pending:
            first = self.state.telemetry_pending[0]
            candidate_version = int(first["candidate_version"])
            candidate = str(first["candidate"])
            jobs = [
                job
                for job in self.state.telemetry_pending
                if int(job["candidate_version"]) == candidate_version
                and str(job["candidate"]) == candidate
                and job.get("group_seed") == first.get("group_seed")
            ]
            jobs.sort(key=lambda job: int(job.get("group_index", 0)))
            missing = [
                job
                for job in jobs
                if not (
                    self.output / "telemetry" / f"{job['id']}.json"
                ).exists()
            ]
            if missing:
                pairs = {int(job["pairs"]) for job in missing}
                if len(pairs) != 1:
                    raise RuntimeError("batched telemetry jobs disagree on pair count")
                group_seed = missing[0].get("group_seed")
                first_index = int(missing[0].get("group_index", 0))
                if group_seed is None or any(
                    int(job["seed"])
                    != int(group_seed)
                    + (first_index + index) * 1_000_003
                    for index, job in enumerate(missing)
                ):
                    # Legacy pending jobs have independent seeds. Preserve their
                    # exact experiments rather than silently rebasing them.
                    missing = missing[:1]
                    pairs = {int(missing[0]["pairs"])}
                    group_seed = int(missing[0]["seed"])
                    first_index = 0
                command = self.arena_command(
                    candidate=Path(candidate),
                    opponents=[
                        Path(str(job["opponent"])) for job in missing
                    ],
                    # Arena derives a deterministic per-opponent offset from
                    # this group seed. Job identity remains stable even though
                    # execution is amortized across one model load.
                    seed=int(group_seed) + first_index * 1_000_003,
                    pairs=pairs.pop(),
                )
                result = await self.runner.run(
                    command,
                    cwd=self.root,
                    log_prefix=self.output
                    / "logs"
                    / f"telemetry-v{candidate_version:06d}",
                )
                arenas = result.json_documents()
                if len(arenas) != len(missing):
                    raise RuntimeError(
                        "batched arena emitted an unexpected number of results"
                    )
                for job, arena in zip(missing, arenas, strict=True):
                    opponent = Path(str(job["opponent"])).resolve()
                    reported = arena.get("opponent_model")
                    if reported and Path(str(reported)).resolve() != opponent:
                        raise RuntimeError(
                            "batched arena result order does not match its jobs"
                        )
                    atomic_json(
                        self.output / "telemetry" / f"{job['id']}.json",
                        {
                            "schema": "vgo.telemetry-result.v1",
                            "id": str(job["id"]),
                            "candidate_version": candidate_version,
                            "opponent_version": int(job["opponent_version"]),
                            "arena": arena,
                        },
                    )
            completed_ids = {
                str(job["id"])
                for job in jobs
                if (
                    self.output / "telemetry" / f"{job['id']}.json"
                ).exists()
            }
            self.state.telemetry_pending = [
                job
                for job in self.state.telemetry_pending
                if str(job["id"]) not in completed_ids
            ]
            for job_id in sorted(completed_ids):
                if job_id not in self.state.telemetry_completed:
                    self.state.telemetry_completed.append(job_id)
            self._save_state()
        self._refresh_ratings()
        if self.state.updates_completed >= self.config.updates:
            atomic_json(self.output / "run.json", self.report())

    def _refresh_ratings(self) -> None:
        matches: list[dict[str, Any]] = []
        for path in sorted((self.output / "telemetry").glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("schema") != "vgo.telemetry-result.v1":
                continue
            arena = result["arena"]
            matches.append(
                {
                    "a": int(result["candidate_version"]),
                    "b": int(result["opponent_version"]),
                    "a_wins": int(arena["candidate_wins"]),
                    "b_wins": int(arena["candidate_losses"]),
                    "draws": int(arena.get("draws", 0)),
                }
            )
        if not matches:
            return
        anchor = min(
            min(int(match["a"]), int(match["b"])) for match in matches
        )
        ratings = fit_ratings(matches, anchor=anchor, prior_games=0.25)
        atomic_json(
            self.output / "telemetry" / "ratings.json",
            {str(version): round(rating, 1) for version, rating in ratings.items()},
        )


def config_from_arguments(arguments: argparse.Namespace) -> PipelineConfig:
    values = vars(arguments).copy()
    values.pop("drain_telemetry", None)
    values.pop("telemetry_only", None)
    values["output"] = str(Path(values["output"]).resolve())
    values["initial_checkpoint"] = (
        str(Path(values["initial_checkpoint"]).resolve())
        if values.get("initial_checkpoint") is not None
        else None
    )
    values["initial_onnx"] = (
        str(Path(values["initial_onnx"]).resolve())
        if values.get("initial_onnx") is not None
        else None
    )
    values["initial_replay"] = tuple(
        str(Path(path).resolve()) for path in values.get("initial_replay", ())
    )
    return PipelineConfig(**values)


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--shards-per-update", type=int, default=1)
    parser.add_argument("--replay-window", type=int, default=8)
    parser.add_argument("--maximum-prefetch-shards", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--policy-resolution", type=int, default=32)
    parser.add_argument("--radius", type=float, default=1.0 / 18.0)
    parser.add_argument("--generation-simulations", type=int, default=256)
    parser.add_argument("--maximum-plies", type=int, default=256)
    parser.add_argument("--coarse-pool", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--actors", type=int, default=64)
    parser.add_argument("--writer-queue-games", type=int, default=2)
    parser.add_argument("--leaf-batch", type=int, default=1)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--inference-delay-ms", type=int, default=1)
    parser.add_argument(
        "--inference-slots",
        type=int,
        default=2,
        help="independent inference lanes used by self-play generation",
    )
    parser.add_argument(
        "--provider", choices=("cpu", "cuda", "tensorrt"), default="tensorrt"
    )
    parser.add_argument("--inference-device-id", type=int, default=0)
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--warm-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "prebuild each TensorRT engine after export so model-load latency "
            "overlaps the current self-play tail"
        ),
    )
    parser.add_argument(
        "--architecture", choices=MODEL_ARCHITECTURES, default="ddrnet"
    )
    parser.add_argument(
        "--norm-groups",
        type=int,
        default=None,
        help=(
            "GroupNorm groups in every residual block (ddrnet only); "
            "supersedes --variance-scaled. Bounds activation magnitude "
            "structurally rather than by a constant fixed at initialization"
        ),
    )
    parser.add_argument(
        "--variance-scaled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "scale each residual branch by 1/sqrt(depth) and initialize convs "
            "at He scale; bounds trunk variance instead of letting training "
            "inflate weights (ddrnet only, cannot be warm-started onto an "
            "unscaled checkpoint)"
        ),
    )
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--training-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--warm-learning-rate", type=float, default=5e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--training-threads", type=int, default=4)
    parser.add_argument("--training-device", default="cuda")
    parser.add_argument(
        "--training-precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--restore-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--overlap-actor-learner",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--promotion-arena",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--promotion-score", type=float, default=0.0)
    parser.add_argument("--maximum-truncation-rate", type=float, default=0.02)
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=0.0,
        help=(
            "concede a game once the side to move has been losing by this much "
            "for --resign-window consecutive plies; 0 disables resignation"
        ),
    )
    parser.add_argument(
        "--komi-low",
        type=float,
        default=0.0,
        help="lowest komi a game may draw; positive favours White",
    )
    parser.add_argument(
        "--komi-high",
        type=float,
        default=0.0,
        help=(
            "highest komi a game may draw. A range teaches the relationship "
            "between komi and the position; one value teaches one balance point"
        ),
    )
    parser.add_argument(
        "--raster-kind",
        choices=("semantic", "compact", "rgb"),
        default="semantic",
        help="channel layout; compact is four channels plus komi",
    )
    parser.add_argument("--resign-window", type=int, default=5)
    parser.add_argument(
        "--resign-minimum-ply",
        type=int,
        default=0,
        help=(
            "earliest ply a game may be conceded at. The window counts a "
            "seat's own turns and a seat moves every other ply, so window 5 "
            "concedes at ply 8 -- five stones each on a board holding 35"
        ),
    )
    parser.add_argument(
        "--resign-disable-fraction",
        type=float,
        default=0.1,
        help=(
            "fraction of games played to a real finish regardless of the "
            "threshold, which are what the shard's calibration is measured on"
        ),
    )
    parser.add_argument("--arena-pairs", type=int, default=16)
    parser.add_argument("--arena-simulations", type=int, default=256)
    parser.add_argument("--arena-actors", type=int, default=32)
    parser.add_argument("--telemetry-opponents", type=int, default=2)
    parser.add_argument("--telemetry-pairs", type=int, default=16)
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=1,
        help=(
            "queue Elo work for every Nth checkpoint; the loop's rating curve is "
            "a trend and does not need every point"
        ),
    )
    parser.add_argument(
        "--retire-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compress replay shards once they leave the training window; runs "
            "off the critical path and never delays an update"
        ),
    )
    parser.add_argument("--seed", type=int, default=700_001)
    parser.add_argument("--arena-seed", type=int, default=900_001)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-onnx", type=Path)
    parser.add_argument("--initial-replay", type=Path, nargs="*", default=[])
