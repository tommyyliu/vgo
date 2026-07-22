from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import torch


MAGIC = b"VGODATA1"
VERSION = 2
HEADER = struct.Struct("<8s6I")


@dataclass(frozen=True)
class RasterDataset:
    states: torch.Tensor
    policies: torch.Tensor
    policy_masks: torch.Tensor
    values: torch.Tensor
    height: int
    width: int

    @property
    def samples(self) -> int:
        return self.states.shape[0]

    @property
    def channels(self) -> int:
        return self.states.shape[1]


def load_dataset(path: str | Path) -> RasterDataset:
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError("dataset header is truncated")
    magic, version, samples, channels, height, width, policy_size = HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError(f"unexpected dataset magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported dataset version: {version}")
    if samples == 0 or channels == 0 or height == 0 or width == 0:
        raise ValueError("dataset dimensions must be positive")
    if policy_size != height * width + 1:
        raise ValueError("policy size must equal raster pixels plus pass")

    state_size = channels * height * width
    record_size = state_size + 2 * policy_size + 1
    expected_bytes = HEADER.size + samples * record_size * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"dataset size mismatch: expected {expected_bytes}, got {path.stat().st_size}"
        )

    records = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=HEADER.size,
        shape=(samples, record_size),
    )
    states = np.array(records[:, :state_size], copy=True).reshape(
        samples, channels, height, width
    )
    policies = np.array(records[:, state_size : state_size + policy_size], copy=True)
    policy_masks = np.array(
        records[:, state_size + policy_size : state_size + 2 * policy_size], copy=True
    )
    values = np.array(records[:, -1], copy=True)

    if not np.isfinite(states).all() or not np.isfinite(policies).all():
        raise ValueError("dataset contains non-finite tensors")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0):
        raise ValueError("value targets must be finite and in [-1, 1]")
    if np.any(policies < 0.0) or not np.allclose(policies.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("policy targets must be probability distributions")
    if np.any((policy_masks != 0.0) & (policy_masks != 1.0)):
        raise ValueError("policy masks must be binary")
    if np.any(policy_masks.sum(axis=1) == 0.0):
        raise ValueError("every sample must expose at least one policy action")
    if np.any((policies > 0.0) & (policy_masks == 0.0)):
        raise ValueError("positive policy targets must be included in the mask")

    return RasterDataset(
        states=torch.from_numpy(states),
        policies=torch.from_numpy(policies),
        policy_masks=torch.from_numpy(policy_masks).bool(),
        values=torch.from_numpy(values),
        height=height,
        width=width,
    )
