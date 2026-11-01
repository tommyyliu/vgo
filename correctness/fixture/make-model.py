#!/usr/bin/env python3
"""Rebuild the untrained model behind `correctness/game-digest.sh`.

The ONNX file is committed so a torch or exporter upgrade cannot move the digest.
Rerun this only when the model contract changes, then refresh the golden file.

    cd training && PYTHONPATH=. .venv/bin/python ../correctness/fixture/make-model.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from vgo_training.model import build_model

OUTPUT = Path(__file__).resolve().parent / "model.onnx"
RESOLUTION = 64
WIDTH = 16
NORM_GROUPS = 4

torch.manual_seed(20261101)
model = build_model(
    channels=7, width=WIDTH, blocks=1, norm_groups=NORM_GROUPS, raster_resolution=RESOLUTION
)
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "model.pt"
    torch.save(
        {
            "architecture": "ddrnet",
            "state_dict": model.state_dict(),
            "channels": 7,
            "height": RESOLUTION,
            "width": RESOLUTION,
            "policy_resolution": RESOLUTION,
            "model_width": WIDTH,
            "blocks": 1,
            "norm_groups": NORM_GROUPS,
            "raster_kind": "compact-radius",
        },
        checkpoint,
    )
    subprocess.run(
        [sys.executable, "-m", "vgo_training.export_onnx", "--checkpoint", checkpoint,
         "--output", OUTPUT, "--maximum-batch", "32", "--packed-input"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
OUTPUT.with_suffix(".onnx.json").unlink()
