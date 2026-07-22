from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn

from .serve import load_model, resolve_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    model: nn.Module,
    states: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(states)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(states)
        synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "milliseconds_per_batch": seconds * 1000.0 / iterations,
        "positions_per_second": states.shape[0] * iterations / seconds,
    }


def measure_slots(
    eager: nn.Module,
    shape: tuple[int, int, int, int],
    device: torch.device,
    slot_counts: list[int],
    warmup: int,
    iterations: int,
) -> list[dict[str, float | int]]:
    if device.type != "cuda":
        raise ValueError("concurrent model slots currently require CUDA")
    maximum_slots = max(slot_counts)
    models = [
        torch.compile(
            copy.deepcopy(eager), mode="reduce-overhead", fullgraph=True
        )
        for _ in range(maximum_slots)
    ]
    streams = [torch.cuda.Stream(device=device) for _ in range(maximum_slots)]
    states = [torch.zeros(shape, device=device) for _ in range(maximum_slots)]
    for model, stream, inputs in zip(models, streams, states, strict=True):
        with torch.cuda.stream(stream), torch.inference_mode():
            model(inputs)
            model(inputs)
    synchronize(device)

    results = []
    for slots in slot_counts:
        with torch.inference_mode():
            for _ in range(warmup):
                for index in range(slots):
                    with torch.cuda.stream(streams[index]):
                        models[index](states[index])
            synchronize(device)
            started = time.perf_counter()
            for _ in range(iterations):
                for index in range(slots):
                    with torch.cuda.stream(streams[index]):
                        models[index](states[index])
            synchronize(device)
        seconds = time.perf_counter() - started
        positions = shape[0] * slots * iterations
        results.append(
            {
                "slots": slots,
                "batch_per_slot": shape[0],
                "seconds": seconds,
                "milliseconds_per_round": seconds * 1000.0 / iterations,
                "positions_per_second": positions / seconds,
            }
        )
    return results


def benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.iterations <= 0 or arguments.warmup <= 0:
        raise ValueError("iteration counts must be positive")
    batches = [int(value) for value in arguments.batches.split(",")]
    slots = [int(value) for value in arguments.slots.split(",")]
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("batch sizes must be positive")
    if arguments.slot_batch <= 0 or not slots or any(slot <= 0 for slot in slots):
        raise ValueError("slot counts must be positive")

    device = resolve_device(arguments.device)
    torch.set_num_threads(arguments.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    eager, metadata = load_model(arguments.checkpoint)
    eager = eager.to(device).eval()
    compiled = (
        torch.compile(eager, mode="reduce-overhead", fullgraph=True)
        if arguments.compile
        else None
    )

    results = []
    for batch in batches:
        states = torch.zeros(
            (
                batch,
                int(metadata["channels"]),
                int(metadata["height"]),
                int(metadata["width"]),
            ),
            dtype=torch.float32,
            device=device,
        )
        result: dict[str, object] = {
            "batch": batch,
            "eager": measure(
                eager, states, device, arguments.warmup, arguments.iterations
            ),
        }
        if compiled is not None:
            result["compiled"] = measure(
                compiled, states, device, arguments.warmup, arguments.iterations
            )
        results.append(result)

    concurrent_slots = None
    if arguments.compile:
        concurrent_slots = measure_slots(
            eager,
            (
                arguments.slot_batch,
                int(metadata["channels"]),
                int(metadata["height"]),
                int(metadata["width"]),
            ),
            device,
            slots,
            arguments.warmup,
            arguments.iterations,
        )

    report = {
        "schema": "vgo.model-throughput.v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "compile": arguments.compile,
        "iterations": arguments.iterations,
        "warmup": arguments.warmup,
        "results": results,
        "concurrent_slots": concurrent_slots,
    }
    print(json.dumps(report, indent=2))
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batches", default="1,8,16,32,64")
    parser.add_argument("--slot-batch", type=int, default=16)
    parser.add_argument("--slots", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_arguments())
