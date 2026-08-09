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
    # Checkpoints written before the placement grid was decoupled have no
    # policy_resolution and are raster-coupled.
    stored_policy = checkpoint.get("policy_resolution")
    policy_resolution = (
        int(stored_policy)
        if stored_policy is not None and int(stored_policy) != int(checkpoint["height"])
        else None
    )
    model = build_model(
        architecture=str(checkpoint.get("architecture", "flat")),
        channels=int(checkpoint["channels"]),
        width=int(checkpoint["model_width"]),
        blocks=int(checkpoint["blocks"]),
        policy_resolution=policy_resolution,
        variance_scaled=bool(checkpoint.get("variance_scaled", False)),
        norm_groups=checkpoint.get("norm_groups"),
        # Absent from every checkpoint written before context attention
        # existed, where the default 0 rebuilds exactly what was saved. A
        # checkpoint that does carry attention has to rebuild with it or the
        # state dict will not fit -- and the rotary tables are sized from the
        # raster, so that has to come back too.
        context_attention_blocks=int(checkpoint.get("context_attention_blocks", 0)),
        attention_heads=int(checkpoint.get("attention_heads", 8)),
        raster_resolution=int(checkpoint["height"]),
    )
    # The batch-normalized twin heads exist only while training; inference runs
    # the unnormalized heads, so their weights are absent from an exported model
    # and present-but-unused in a training checkpoint. Either way they must not
    # fail the load -- but everything inference *does* read still has to be
    # there, so a genuinely truncated checkpoint is not quietly accepted.
    # A checkpoint from before the value head became categorical has a
    # single-output final projection where this model has two. PyTorch raises on
    # the shape mismatch rather than loading it quietly, which is the right
    # default -- the two heads mean different things. Drop those tensors so the
    # trunk and policy still transfer and the value head starts fresh, which is
    # what a migration wants anyway: the old weights encode pre-tanh margins
    # reaching 17, all of which a logit head would have to unlearn.
    state = dict(checkpoint["state_dict"])
    reinitialized_value_head = False
    for name, parameter in list(state.items()):
        current = model.state_dict().get(name)
        if current is not None and current.shape != parameter.shape:
            if "value_head" not in name:
                raise RuntimeError(
                    f"checkpoint tensor {name} has shape {tuple(parameter.shape)}, "
                    f"model expects {tuple(current.shape)}"
                )
            del state[name]
            reinitialized_value_head = True
    if reinitialized_value_head:
        print(
            "value head reinitialized: checkpoint predates the win/loss head",
            flush=True,
        )
    missing, _ = model.load_state_dict(state, strict=False)
    required = [
        name
        for name in missing
        if not (
            name.endswith("_normed")
            or "_normed." in name
            or name.endswith("_norm.num_batches_tracked")
            or "_norm." in name
            # Ownership is an auxiliary training target and never enters the
            # exported graph, so a checkpoint predating the head is complete
            # for inference and must still load.
            or name.startswith("ownership_")
            # Dropped just above when the checkpoint predates the win/loss head.
            or (reinitialized_value_head and "value_head" in name)
        )
    ]
    if required:
        raise RuntimeError(
            f"checkpoint is missing {len(required)} inference weight(s), "
            f"starting with {required[0]}"
        )
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
