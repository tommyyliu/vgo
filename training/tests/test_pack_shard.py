"""`pack_shard` must agree with `pack()` on the dense render, byte for byte.

Training renders straight into the packed planes: the dense float32 raster is
never built, because it existed only to be packed and dropped. That makes the
Rust packer and `vgo_training.packed_states.pack` two implementations of one
bit layout, and nothing else compares them. A divergence would not crash --
it would feed the network subtly wrong planes while every metric kept reporting
normally, which is the train/serve skew the position format exists to prevent.

Both binaries are built from the same rasteriser, so this pins the *packing*,
not the geometry:

    cargo build --release -p vgo-raster
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import vgo_training.dataset as dataset_module
from vgo_training.packed_states import pack

# The module, not the class: importing a TestCase by name re-exports it, and
# discovery would then run test_dataset's cases a second time from here.
from tests import test_dataset as _dataset_tests

_ROOT = Path(__file__).resolve().parents[2]
_RENDERER = _ROOT / "target/release/vgo-render-shard"
_PACKER = _ROOT / "target/release/vgo-pack-shard"
# Small enough to keep the test quick; a multiple of eight so the bit planes end
# on a byte boundary, which is the interesting case for `np.packbits`.
_RESOLUTION = 16
_KIND = "compact-radius"
_CHANNELS = 7


@unittest.skipUnless(
    _RENDERER.exists() and _PACKER.exists(),
    "needs `cargo build --release -p vgo-raster`",
)
class PackShardMatchesThePythonPacker(unittest.TestCase):
    def _shard(self, directory: Path) -> Path:
        # Reused rather than rewritten: a second hand-rolled v4 writer would
        # drift from the format the moment a field moved.
        helper = _dataset_tests.V4PositionShardTests(
            "test_positions_render_the_channels_they_describe"
        )
        return helper._shard(directory, samples=5)

    def test_the_two_packers_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = self._shard(root)

            dense_path = root / "dense.bin"
            subprocess.run(
                [str(_RENDERER), str(shard), str(dense_path), str(_RESOLUTION), _KIND],
                check=True, capture_output=True, text=True,
            )
            dense = np.fromfile(dense_path, dtype="<f4").reshape(
                -1, _CHANNELS, _RESOLUTION, _RESOLUTION
            )
            expected = pack(torch.from_numpy(dense))

            # Through the production reader, so the parser is under test too.
            actual = dataset_module._packed_states(shard, _RESOLUTION, _KIND)
            self.assertIsNotNone(actual, "the packer refused a layout it should pack")

            self.assertEqual(actual.samples, dense.shape[0])
            for field in ("bits", "continuous", "scalars"):
                want, got = getattr(expected, field), getattr(actual, field)
                self.assertEqual(want.shape, got.shape, field)
                self.assertEqual(want.dtype, got.dtype, field)
                self.assertTrue(torch.equal(want, got), f"{field} diverged")

    def test_expanding_the_packed_planes_reproduces_the_render(self) -> None:
        """The round trip, not just the storage: what the net reads must match.

        `expand` is what the batch stager calls, so agreeing on the packed bytes
        is only half the contract -- the planes it reconstructs are what the
        model actually sees.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = self._shard(root)
            dense_path = root / "dense.bin"
            subprocess.run(
                [str(_RENDERER), str(shard), str(dense_path), str(_RESOLUTION), _KIND],
                check=True, capture_output=True, text=True,
            )
            dense = torch.from_numpy(
                np.fromfile(dense_path, dtype="<f4").reshape(
                    -1, _CHANNELS, _RESOLUTION, _RESOLUTION
                )
            )
            packed = dataset_module._packed_states(shard, _RESOLUTION, _KIND)
            self.assertIsNotNone(packed)
            # fp16 is the storage precision, so compare against the same
            # rounding rather than against the f32 the renderer emitted.
            self.assertTrue(torch.equal(packed.expand(), dense.half()))

    def test_a_missing_binary_falls_back_rather_than_failing(self) -> None:
        """The dense path is the fallback, and it has to stay reachable."""
        with tempfile.TemporaryDirectory() as directory:
            shard = self._shard(Path(directory))
            saved = dataset_module._RUST_PACKER
            dataset_module._RUST_PACKER = Path("/nonexistent/pack_shard")
            try:
                self.assertIsNone(
                    dataset_module._packed_states(shard, _RESOLUTION, _KIND)
                )
            finally:
                dataset_module._RUST_PACKER = saved


if __name__ == "__main__":
    unittest.main()
