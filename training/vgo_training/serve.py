from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from torch import nn

from .model import RasterPolicyValueNet, build_model


REQUEST_MAGIC = b"VGOIFR01"
RESPONSE_MAGIC = b"VGOOFR01"
VERSION = 1
REQUEST_HEADER = struct.Struct("<8s5I")
RESPONSE_HEADER = struct.Struct("<8s3I")
IDENTIFIER = struct.Struct("<Q")
VALUE = struct.Struct("<f")


def read_exact(stream, size: int, *, allow_eof: bool = False) -> bytes | None:
    data = stream.read(size)
    if allow_eof and not data:
        return None
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, received {len(data)}")
    return data


def load_model(checkpoint_path: Path) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        architecture=str(checkpoint.get("architecture", "flat")),
        channels=int(checkpoint["channels"]),
        width=int(checkpoint["model_width"]),
        blocks=int(checkpoint["blocks"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def prepare_model(
    model: nn.Module,
    device: torch.device,
    compile_model: bool,
    batch_shape: tuple[int, int, int, int],
) -> nn.Module:
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    warmup = torch.zeros(batch_shape, dtype=torch.float32, device=device)
    with torch.inference_mode():
        model(warmup)
        model(warmup)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model


def serve(
    checkpoint_path: Path,
    threads: int,
    device_name: str,
    compile_model: bool,
    maximum_batch: int,
) -> None:
    if threads <= 0 or maximum_batch <= 0:
        raise ValueError("thread and batch counts must be positive")
    torch.set_num_threads(threads)
    model, metadata = load_model(checkpoint_path)
    expected_channels = int(metadata["channels"])
    expected_height = int(metadata["height"])
    expected_width = int(metadata["width"])
    device = resolve_device(device_name)
    model = prepare_model(
        model,
        device,
        compile_model,
        (maximum_batch, expected_channels, expected_height, expected_width),
    )
    print(
        f"vgo inference ready: device={device} compile={compile_model} "
        f"maximum_batch={maximum_batch}",
        file=sys.stderr,
        flush=True,
    )
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer

    while True:
        header_bytes = read_exact(input_stream, REQUEST_HEADER.size, allow_eof=True)
        if header_bytes is None:
            return
        magic, version, batch, channels, height, width = REQUEST_HEADER.unpack(header_bytes)
        if magic != REQUEST_MAGIC or version != VERSION:
            raise ValueError("unsupported inference request")
        if batch == 0 or batch > maximum_batch:
            raise ValueError(
                f"request batch {batch} is outside supported range 1..{maximum_batch}"
            )
        if (channels, height, width) != (
            expected_channels,
            expected_height,
            expected_width,
        ):
            raise ValueError(
                f"tensor shape {(channels, height, width)} does not match checkpoint "
                f"{(expected_channels, expected_height, expected_width)}"
            )
        identifiers = []
        inference_batch = maximum_batch if compile_model else batch
        states = np.zeros((inference_batch, channels, height, width), dtype=np.float32)
        tensor_bytes = channels * height * width * np.dtype("<f4").itemsize
        for index in range(batch):
            identifiers.append(
                IDENTIFIER.unpack(read_exact(input_stream, IDENTIFIER.size))[0]
            )
            payload = read_exact(input_stream, tensor_bytes)
            states[index] = np.frombuffer(payload, dtype="<f4").reshape(
                channels, height, width
            )

        with torch.inference_mode():
            policy, values = model(torch.from_numpy(states).to(device))
        policy = policy[:batch].detach().cpu().numpy().astype("<f4", copy=False)
        values = values[:batch].detach().cpu().numpy().astype("<f4", copy=False)
        output_stream.write(
            RESPONSE_HEADER.pack(RESPONSE_MAGIC, VERSION, batch, policy.shape[1])
        )
        for identifier, value, logits in zip(identifiers, values, policy, strict=True):
            output_stream.write(IDENTIFIER.pack(identifier))
            output_stream.write(VALUE.pack(float(value)))
            output_stream.write(logits.tobytes())
        output_stream.flush()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--maximum-batch", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    serve(
        arguments.checkpoint,
        arguments.threads,
        arguments.device,
        arguments.compile,
        arguments.maximum_batch,
    )
