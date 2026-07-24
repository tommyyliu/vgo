from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import struct

import numpy as np
import torch


MAGIC = b"VGODATA1"
VERSION = 2
REPLAY_MAGIC = b"VGORPLY1"
# v1: state, policy, mask, value, ... ; v2 inserts per-cell visits + beta after
# the mask (for off-policy reweighting). Both are accepted; v1 records synthesize
# visits from the policy and leave beta at zero. REPLAY_VERSION is the version the
# current generator writes.
REPLAY_VERSION = 2
REPLAY_VERSIONS = (1, 2)
HEADER = struct.Struct("<8s6I")


@dataclass(frozen=True)
class RasterDataset:
    states: torch.Tensor
    policies: torch.Tensor
    policy_masks: torch.Tensor
    # Raw MCTS visit counts and coarse->fine sampling probabilities (beta) per
    # cell. For legacy/v1 shards visits proxy the policy and betas are zero.
    visits: torch.Tensor
    betas: torch.Tensor
    values: torch.Tensor
    selected_actions: torch.Tensor
    game_ids: torch.Tensor
    plies: torch.Tensor
    seeds: torch.Tensor
    height: int
    width: int
    sources: tuple[str, ...]

    @property
    def samples(self) -> int:
        return self.states.shape[0]

    @property
    def channels(self) -> int:
        return self.states.shape[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(path: Path) -> tuple[bytes, int, int, int, int, int, int]:
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError("dataset header is truncated")
    magic, version, samples, channels, height, width, policy_size = HEADER.unpack(header)
    if magic not in (MAGIC, REPLAY_MAGIC):
        raise ValueError(f"unexpected dataset magic: {magic!r}")
    if magic == MAGIC:
        if version != VERSION:
            raise ValueError(f"unsupported dataset version: {version}")
    elif version not in REPLAY_VERSIONS:
        raise ValueError(f"unsupported replay version: {version}")
    if samples == 0 or channels == 0 or height == 0 or width == 0:
        raise ValueError("dataset dimensions must be positive")
    if policy_size != height * width + 1:
        raise ValueError("policy size must equal raster pixels plus pass")
    return magic, version, samples, channels, height, width, policy_size


def _validate_manifest(path: Path) -> None:
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        raise ValueError("replay shard is incomplete: manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "vgo.replay-shard.v1":
        raise ValueError(f"unsupported replay manifest: {manifest.get('schema')!r}")
    expected = manifest.get("dataset_sha256")
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(f"replay checksum mismatch: expected {expected}, got {actual}")


def _load_legacy(
    path: Path,
    samples: int,
    channels: int,
    height: int,
    width: int,
    policy_size: int,
) -> tuple[np.ndarray, ...]:
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
    states = np.array(records[:, :state_size], copy=True)
    policies = np.array(records[:, state_size : state_size + policy_size], copy=True)
    masks = np.array(
        records[:, state_size + policy_size : state_size + 2 * policy_size], copy=True
    )
    values = np.array(records[:, -1], copy=True)
    selected = np.full(samples, -1, dtype=np.int64)
    zeros = np.zeros(samples, dtype=np.int64)
    # legacy datasets carry no raw visits/beta: proxy visits with the policy.
    visits = policies.copy()
    beta = np.zeros_like(policies)
    return (
        states,
        policies,
        masks,
        visits,
        beta,
        values,
        selected,
        zeros,
        zeros.copy(),
        zeros.copy(),
    )


def _load_replay(
    path: Path,
    version: int,
    samples: int,
    channels: int,
    height: int,
    width: int,
    policy_size: int,
) -> tuple[np.ndarray, ...]:
    state_size = channels * height * width
    fields = [
        ("state", "<f4", (state_size,)),
        ("policy", "<f4", (policy_size,)),
        ("mask", "<f4", (policy_size,)),
    ]
    if version >= 2:
        fields += [
            ("visits", "<f4", (policy_size,)),
            ("beta", "<f4", (policy_size,)),
        ]
    fields += [
        ("value", "<f4"),
        ("selected_action", "<u4"),
        ("game", "<u8"),
        ("ply", "<u4"),
        ("seed", "<u8"),
    ]
    record_dtype = np.dtype(fields, align=False)
    expected_bytes = HEADER.size + samples * record_dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"replay size mismatch: expected {expected_bytes}, got {path.stat().st_size}"
        )
    records = np.memmap(
        path,
        dtype=record_dtype,
        mode="r",
        offset=HEADER.size,
        shape=(samples,),
    )
    policies = np.array(records["policy"], copy=True)
    masks = np.array(records["mask"], copy=True)
    if version >= 2:
        visits = np.array(records["visits"], copy=True)
        beta = np.array(records["beta"], copy=True)
    else:
        # v1 has no raw visits/beta: the normalized policy is the best proxy for
        # visit share, and beta is zero (no factored sampling was recorded).
        visits = policies.copy()
        beta = np.zeros_like(policies)
    return (
        np.array(records["state"], copy=True),
        policies,
        masks,
        visits,
        beta,
        np.array(records["value"], copy=True),
        np.array(records["selected_action"], copy=True).astype(np.int64),
        np.array(records["game"], copy=True).astype(np.int64),
        np.array(records["ply"], copy=True).astype(np.int64),
        np.array(records["seed"], copy=True).astype(np.int64),
    )


def load_dataset(path: str | Path) -> RasterDataset:
    path = Path(path).resolve(strict=True)
    magic, version, samples, channels, height, width, policy_size = _read_header(path)
    if magic == REPLAY_MAGIC:
        _validate_manifest(path)
        arrays = _load_replay(path, version, samples, channels, height, width, policy_size)
    else:
        arrays = _load_legacy(path, samples, channels, height, width, policy_size)
    (
        states,
        policies,
        policy_masks,
        visits,
        beta,
        values,
        selected,
        games,
        plies,
        seeds,
    ) = arrays
    states = states.reshape(samples, channels, height, width)

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
    replay_actions = selected >= 0
    if np.any(selected[replay_actions] >= policy_size):
        raise ValueError("selected replay action is outside the policy tensor")
    if np.any(policy_masks[np.arange(samples)[replay_actions], selected[replay_actions]] == 0.0):
        raise ValueError("selected replay action is absent from the policy mask")

    return RasterDataset(
        states=torch.from_numpy(states),
        policies=torch.from_numpy(policies),
        policy_masks=torch.from_numpy(policy_masks).bool(),
        visits=torch.from_numpy(visits),
        betas=torch.from_numpy(beta),
        values=torch.from_numpy(values),
        selected_actions=torch.from_numpy(selected),
        game_ids=torch.from_numpy(games),
        plies=torch.from_numpy(plies),
        seeds=torch.from_numpy(seeds),
        height=height,
        width=width,
        sources=(str(path),),
    )


def load_datasets(paths: Iterable[str | Path]) -> RasterDataset:
    datasets = [load_dataset(path) for path in paths]
    if not datasets:
        raise ValueError("at least one replay dataset is required")
    first = datasets[0]
    for dataset in datasets[1:]:
        if (dataset.channels, dataset.height, dataset.width) != (
            first.channels,
            first.height,
            first.width,
        ):
            raise ValueError("all replay datasets must have the same raster shape")
    return RasterDataset(
        states=torch.cat([dataset.states for dataset in datasets]),
        policies=torch.cat([dataset.policies for dataset in datasets]),
        policy_masks=torch.cat([dataset.policy_masks for dataset in datasets]),
        visits=torch.cat([dataset.visits for dataset in datasets]),
        betas=torch.cat([dataset.betas for dataset in datasets]),
        values=torch.cat([dataset.values for dataset in datasets]),
        selected_actions=torch.cat([dataset.selected_actions for dataset in datasets]),
        game_ids=torch.cat([dataset.game_ids for dataset in datasets]),
        plies=torch.cat([dataset.plies for dataset in datasets]),
        seeds=torch.cat([dataset.seeds for dataset in datasets]),
        height=first.height,
        width=first.width,
        sources=tuple(source for dataset in datasets for source in dataset.sources),
    )
