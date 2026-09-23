from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import onnx
import torch

from .packed_input import PackedInputModel
from .packed_states import _LAYOUTS
from .model import load_model
from .supervision import atomic_write_text


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.maximum_batch < 2:
        raise ValueError("maximum batch must be at least two for dynamic export")
    checkpoint_path = arguments.checkpoint.resolve(strict=True)
    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    model, checkpoint = load_model(checkpoint_path)
    channels = int(checkpoint["channels"])
    height = int(checkpoint["height"])
    width = int(checkpoint["width"])
    model_width = int(checkpoint["model_width"])
    blocks = int(checkpoint["blocks"])
    checkpoint_digest = file_sha256(checkpoint_path)
    policy_resolution = int(checkpoint.get("policy_resolution", height))
    policy_size = policy_resolution * policy_resolution + 1
    batch = torch.export.Dim("batch", min=1, max=arguments.maximum_batch)

    if arguments.packed_input:
        # Three compressed inputs instead of one dense raster; the graph
        # expands them. See vgo_training/packed_input.py for why.
        layout = _LAYOUTS.get(channels)
        if layout is None:
            raise ValueError(
                f"--packed-input needs a known layout for {channels} channels; "
                f"packed_states._LAYOUTS has {sorted(_LAYOUTS)}"
            )
        pixels = height * width
        wrapped = PackedInputModel(
            model,
            binary=layout.binary,
            continuous=layout.continuous,
            scalar=layout.scalar,
            height=height,
            width=width,
        ).eval()
        example = (
            torch.zeros((2, len(layout.binary), -(-pixels // 8)), dtype=torch.uint8),
            torch.zeros((2, len(layout.continuous), height, width), dtype=torch.float16),
            torch.zeros((2, len(layout.scalar)), dtype=torch.float16),
        )
        input_names = ["bits", "dense", "scalars"]
        dynamic_shapes = ({0: batch}, {0: batch}, {0: batch})
        export_model = wrapped
    else:
        example = (torch.zeros((2, channels, height, width), dtype=torch.float32),)
        input_names = ["states"]
        dynamic_shapes = ({0: batch},)
        export_model = model

    program = torch.onnx.export(
        export_model,
        example,
        input_names=input_names,
        output_names=["policy_logits", "values"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        optimize=True,
        opset_version=20,
        external_data=False,
    )
    program.save(temporary_path, external_data=False)
    model_proto = onnx.load(temporary_path, load_external_data=False)
    properties = {
        "vgo.schema": "vgo.raster-policy-value.onnx.v1",
        "vgo.checkpoint_sha256": checkpoint_digest,
        "vgo.channels": str(channels),
        "vgo.height": str(height),
        "vgo.width": str(width),
        "vgo.policy_size": str(policy_size),
        "vgo.maximum_batch": str(arguments.maximum_batch),
        "vgo.input_precision": "float16" if arguments.packed_input else "float32",
        "vgo.input_layout": "packed" if arguments.packed_input else "dense",
    }
    onnx.helper.set_model_props(model_proto, properties)
    onnx.checker.check_model(model_proto, full_check=True)
    onnx.save_model(model_proto, temporary_path, save_as_external_data=False)
    with temporary_path.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_path, output_path)
    if os.name != "nt":
        descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    manifest = {
        "schema": "vgo.onnx-manifest.v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "onnx": str(output_path),
        "onnx_sha256": file_sha256(output_path),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "opset": model_proto.opset_import[0].version,
        "input": {
            "name": "states",
            "dtype": "float32",
            "shape": ["batch", channels, height, width],
            "minimum_batch": 1,
            "maximum_batch": arguments.maximum_batch,
        },
        "outputs": [
            {
                "name": "policy_logits",
                "dtype": "float32",
                "shape": ["batch", policy_size],
            },
            {"name": "values", "dtype": "float32", "shape": ["batch"]},
        ],
        "model": {
            "architecture": "ddrnet",
            # Which planes this network expects, carried so a serving path can
            # rasterize the way it was trained. `channels` above says how many;
            # it does not say what they mean. Feeding the wrong layout is a
            # silent failure at inference, unlike in training, where the shape
            # check catches most of it.
            #
            # Null for a checkpoint trained before the field existed.
            "raster_kind": checkpoint.get("raster_kind"),
            "width": model_width,
            "blocks": blocks,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-batch", type=int, default=32)
    parser.add_argument(
        "--packed-input",
        action="store_true",
        help=(
            "Take the compressed input triple (bits, dense, scalars) instead of a "
            "dense float32 raster, and expand it inside the graph. 11.8x fewer "
            "bytes staged per position at 256x256, bit-identical arithmetic."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_arguments())
