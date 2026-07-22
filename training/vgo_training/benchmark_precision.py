from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .dataset import RasterDataset, load_dataset
from .serve import load_model, resolve_device


Quantizer = Callable[[torch.Tensor], torch.Tensor]


def roundtrip(dtype: torch.dtype) -> Quantizer:
    return lambda states: states.to(dtype).to(torch.float32)


def fixed_symmetric(states: torch.Tensor, levels: int) -> torch.Tensor:
    return (states * levels).round().clamp(-levels, levels) / levels


def channel_fixed8(states: torch.Tensor) -> torch.Tensor:
    quantized = (states * 255.0).round().clamp(0.0, 255.0) / 255.0
    quantized[:, 7] = fixed_symmetric(states[:, 7], 127)
    return quantized


def centered_snorm8(states: torch.Tensor) -> torch.Tensor:
    quantized = (fixed_symmetric(states * 2.0 - 1.0, 127) + 1.0) * 0.5
    quantized[:, 7] = fixed_symmetric(states[:, 7], 127)
    return quantized


def infer(
    model: nn.Module,
    states: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    policies = []
    values = []
    with torch.inference_mode():
        for start in range(0, states.shape[0], batch_size):
            policy, value = model(states[start : start + batch_size].to(device))
            policies.append(policy.cpu())
            values.append(value.cpu())
    return torch.cat(policies), torch.cat(values)


def precision_result(
    name: str,
    bytes_per_value: float,
    quantize: Quantizer,
    dataset: RasterDataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    baseline_policy: torch.Tensor,
    baseline_values: torch.Tensor,
) -> dict[str, object]:
    states = quantize(dataset.states)
    errors = (states - dataset.states).abs()
    policy, values = infer(model, states, device, batch_size)
    policy_errors = (policy - baseline_policy).abs()
    value_errors = (values - baseline_values).abs()
    full_top1 = policy.argmax(dim=1)
    baseline_full_top1 = baseline_policy.argmax(dim=1)
    masked_policy = policy.masked_fill(~dataset.policy_masks, -torch.inf)
    baseline_masked_policy = baseline_policy.masked_fill(
        ~dataset.policy_masks, -torch.inf
    )
    return {
        "name": name,
        "bytes_per_value": bytes_per_value,
        "bytes_per_position": int(
            dataset.channels * dataset.height * dataset.width * bytes_per_value
        ),
        "state_mean_absolute_error": float(errors.mean()),
        "state_max_absolute_error": float(errors.max()),
        "channel_mean_absolute_error": [
            float(errors[:, channel].mean()) for channel in range(dataset.channels)
        ],
        "channel_max_absolute_error": [
            float(errors[:, channel].max()) for channel in range(dataset.channels)
        ],
        "policy_mean_absolute_difference": float(policy_errors.mean()),
        "policy_max_absolute_difference": float(policy_errors.max()),
        "policy_full_top1_agreement": float(
            (full_top1 == baseline_full_top1).float().mean()
        ),
        "policy_sampled_top1_agreement": float(
            (
                masked_policy.argmax(dim=1)
                == baseline_masked_policy.argmax(dim=1)
            )
            .float()
            .mean()
        ),
        "value_mean_absolute_difference": float(value_errors.mean()),
        "value_max_absolute_difference": float(value_errors.max()),
    }


def benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = resolve_device(arguments.device)
    dataset = load_dataset(arguments.dataset)
    model, metadata = load_model(arguments.checkpoint)
    expected_shape = (
        int(metadata["channels"]),
        int(metadata["height"]),
        int(metadata["width"]),
    )
    if tuple(dataset.states.shape[1:]) != expected_shape:
        raise ValueError("dataset and checkpoint tensor shapes differ")
    model = model.to(device).eval()
    baseline_policy, baseline_values = infer(
        model, dataset.states, device, arguments.batch_size
    )
    formats: list[tuple[str, float, Quantizer]] = [
        ("fp16", 2.0, roundtrip(torch.float16)),
        ("bfloat16", 2.0, roundtrip(torch.bfloat16)),
        ("fp8_e4m3fn", 1.0, roundtrip(torch.float8_e4m3fn)),
        ("fp8_e5m2", 1.0, roundtrip(torch.float8_e5m2)),
        ("symmetric_int8", 1.0, lambda states: fixed_symmetric(states, 127)),
        ("channel_fixed8", 1.0, channel_fixed8),
        ("centered_snorm8", 1.0, centered_snorm8),
        ("symmetric_int4", 0.5, lambda states: fixed_symmetric(states, 7)),
    ]
    report = {
        "schema": "vgo.raster-precision.v1",
        "device": str(device),
        "samples": dataset.samples,
        "shape": list(dataset.states.shape),
        "baseline_bytes_per_position": (
            dataset.channels * dataset.height * dataset.width * 4
        ),
        "formats": [
            precision_result(
                name,
                bytes_per_value,
                quantize,
                dataset,
                model,
                device,
                arguments.batch_size,
                baseline_policy,
                baseline_values,
            )
            for name, bytes_per_value, quantize in formats
        ],
    }
    print(json.dumps(report, indent=2))
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_arguments())
