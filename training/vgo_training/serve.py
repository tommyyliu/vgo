from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

import numpy as np
import torch

from .model import RasterPolicyValueNet


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


def load_model(checkpoint_path: Path) -> tuple[RasterPolicyValueNet, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RasterPolicyValueNet(
        channels=int(checkpoint["channels"]),
        width=int(checkpoint["model_width"]),
        blocks=int(checkpoint["blocks"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def serve(checkpoint_path: Path, threads: int) -> None:
    torch.set_num_threads(threads)
    model, metadata = load_model(checkpoint_path)
    expected_channels = int(metadata["channels"])
    expected_height = int(metadata["height"])
    expected_width = int(metadata["width"])
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer

    while True:
        header_bytes = read_exact(input_stream, REQUEST_HEADER.size, allow_eof=True)
        if header_bytes is None:
            return
        magic, version, batch, channels, height, width = REQUEST_HEADER.unpack(header_bytes)
        if magic != REQUEST_MAGIC or version != VERSION:
            raise ValueError("unsupported inference request")
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
        states = np.empty((batch, channels, height, width), dtype=np.float32)
        tensor_bytes = channels * height * width * np.dtype("<f4").itemsize
        for index in range(batch):
            identifiers.append(IDENTIFIER.unpack(read_exact(input_stream, IDENTIFIER.size))[0])
            payload = read_exact(input_stream, tensor_bytes)
            states[index] = np.frombuffer(payload, dtype="<f4").reshape(
                channels, height, width
            )

        with torch.inference_mode():
            policy, values = model(torch.from_numpy(states))
        policy = policy.detach().cpu().numpy().astype("<f4", copy=False)
        values = values.detach().cpu().numpy().astype("<f4", copy=False)
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
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    serve(arguments.checkpoint, arguments.threads)
