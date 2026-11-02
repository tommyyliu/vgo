#!/usr/bin/env python3
"""Sustained positions/s of an exported model on the generator's inference path.

Configures ONNX Runtime the way `crates/vgo-inference/src/onnx.rs` does: TensorRT
at fp16, a batch 1 to N profile, one session per lane. This is the card's ceiling.
Its session time matched a real generator's within 1%, but a lane there also spends
about 0.3 ms per batch on the CPU, and that decides the rate on a fast net: 39k here
was 31k in the generator. Use this to rank models, then confirm with a short run.

    source scripts/env/ort.sh
    training/.venv/bin/python performance/model_timing.py model.onnx --raster 128

Packed models only. Weights don't affect timing: a trained net and a random one
agreed within 1%, so an untrained export is enough. Differences under 3% are noise,
so repeat a baseline between runs. Run nothing else alongside it.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

BINARY, CONTINUOUS, SCALAR = 3, 1, 3  # the 7-channel compact-radius layout


def profile(batch: int, raster: int) -> str:
    pixels = raster * raster
    return (
        f"bits:{batch}x{BINARY}x{-(-pixels // 8)},"
        f"dense:{batch}x{CONTINUOUS}x{raster}x{raster},"
        f"scalars:{batch}x{SCALAR}"
    )


def open_session(
    model: Path, raster: int, batch: int, minimum: int, cache: Path
) -> ort.InferenceSession:
    engine, timing = cache / "engine" / model.stem, cache / f"timing-{raster}-b{batch}"
    engine.mkdir(parents=True, exist_ok=True)
    timing.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    tensorrt = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(timing),
        "trt_profile_min_shapes": profile(minimum, raster),
        "trt_profile_opt_shapes": profile(batch, raster),
        "trt_profile_max_shapes": profile(batch, raster),
    }
    session = ort.InferenceSession(
        str(model), options, providers=[("TensorrtExecutionProvider", tensorrt)]
    )
    if session.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TensorRT did not load: {session.get_providers()}")
    return session


def example(batch: int, raster: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "bits": rng.integers(0, 256, (batch, BINARY, -(-(raster**2) // 8)), dtype=np.uint8),
        "dense": rng.random((batch, CONTINUOUS, raster, raster), dtype=np.float32).astype(np.float16),
        "scalars": rng.random((batch, SCALAR), dtype=np.float32).astype(np.float16),
    }


def sustained(sessions, feed, batch: int, seconds: float, warmup: float) -> float:
    start, stop = threading.Event(), threading.Event()
    counts = [0] * len(sessions)

    def lane(index: int, session) -> None:
        while not start.is_set():
            session.run(None, feed)
        while not stop.is_set():
            session.run(None, feed)
            counts[index] += 1

    threads = [threading.Thread(target=lane, args=pair) for pair in enumerate(sessions)]
    for thread in threads:
        thread.start()
    time.sleep(warmup)
    began = time.perf_counter()
    start.set()
    time.sleep(seconds)
    stop.set()
    for thread in threads:
        thread.join()
    return sum(counts) * batch / (time.perf_counter() - began)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--raster", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--minimum-batch", type=int, default=1, help="equal to --batch for a fixed-batch engine"
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--cache", type=Path, default=None, help="reuse engines between runs")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        cache = arguments.cache or Path(scratch)
        lanes = [
            open_session(
                arguments.model, arguments.raster, arguments.batch, arguments.minimum_batch, cache
            )
            for _ in range(2)
        ]
        feed = example(arguments.batch, arguments.raster)
        two = sustained(lanes, feed, arguments.batch, arguments.seconds, 3.0)
        one = sustained(lanes[:1], feed, arguments.batch, arguments.seconds / 2, 1.0)
    print(json.dumps({"two_lanes": round(two), "one_lane": round(one)}))


if __name__ == "__main__":
    main()
