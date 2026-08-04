from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
import hashlib
import json
import struct

import subprocess

if TYPE_CHECKING:
    from .packed_states import PackedStates
import tempfile

import numpy as np
import torch


MAGIC = b"VGODATA1"
VERSION = 2
REPLAY_MAGIC = b"VGORPLY1"
# v1: state, policy, mask, value, ...
# v2: inserts per-cell visits + beta after the mask
# v3: inserts per-cell u32 proposal multiplicities immediately after beta
#
# Older records synthesize unavailable fields so replay windows can span schema
# versions. REPLAY_VERSION is the version the current generator writes.
REPLAY_VERSION = 5
REPLAY_VERSIONS = (1, 2, 3, 4, 5)

# v4 stores the position and a sparse policy instead of a rendered raster.
# These capacities are the writer's, in crates/vgo-selfplay/src/replay_stream.rs,
# and a mismatch silently misparses every record -- the size check in
# _load_replay_v4 is what catches it.
V4_STONE_CAPACITY = 128
V4_POLICY_CAPACITY = 64
CHANNEL_COUNT = 10

# Built by `cargo build --release -p vgo-raster --example render_shard`.
# Channel count -> the raster kind that produces it. The shard header records
# the count, so the kind never has to be configured separately.
_RASTER_KINDS = {3: "rgb", 5: "compact", 12: "semantic"}

_RUST_RENDERER = (
    Path(__file__).resolve().parents[2] / "target/release/examples/render_shard"
)
HEADER = struct.Struct("<8s6I")


@dataclass(frozen=True)
class SparsePolicy:
    """Touched cells only, in the shard's own layout.

    A record stores at most `V4_POLICY_CAPACITY` (64) cells, but the dense
    tensor is `policy_size` wide -- 16385 at policy resolution 128. Expanding at
    load time costs 24 GB of a 39 GB replay window, all of it zeros, which is
    what caps the window and therefore how many independent game outcomes
    training can see at once. Held sparse here and expanded per batch instead.
    """

    indices: torch.Tensor      # (samples, capacity) int64, valid where < counts
    policies: torch.Tensor     # (samples, capacity) float32
    visits: torch.Tensor
    betas: torch.Tensor
    proposal_counts: torch.Tensor
    counts: torch.Tensor       # (samples,) int64: live cells per sample
    policy_size: int

    def expand(self, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Dense policy, mask, visits, beta and proposals for `rows`."""
        picked = self.counts[rows]
        slot = torch.arange(self.indices.shape[1]).unsqueeze(0)
        live = slot < picked.unsqueeze(1)
        size = (rows.numel(), self.policy_size)
        columns = self.indices[rows]
        # Park dead slots on a scratch column so one scatter handles the whole
        # batch; the column is dropped before returning.
        scratch = torch.where(live, columns, torch.full_like(columns, self.policy_size))
        dense = []
        for source in (self.policies, self.visits, self.betas):
            out = torch.zeros(size[0], size[1] + 1, dtype=torch.float32)
            out.scatter_(1, scratch, torch.where(live, source[rows], torch.zeros(())))
            dense.append(out[:, : self.policy_size])
        mask = torch.zeros(size[0], size[1] + 1, dtype=torch.float32)
        mask.scatter_(1, scratch, live.float())
        counts_dense = torch.zeros(size[0], size[1] + 1, dtype=torch.float32)
        counts_dense.scatter_(
            1, scratch, torch.where(live, self.proposal_counts[rows], torch.zeros(()))
        )
        return (
            dense[0],
            mask[:, : self.policy_size],
            dense[1],
            dense[2],
            counts_dense[:, : self.policy_size],
        )


@dataclass(frozen=True)
class RasterDataset:
    states: torch.Tensor
    policies: torch.Tensor
    policy_masks: torch.Tensor
    # Raw MCTS visit counts, coarse->fine sampling probabilities (beta), and raw
    # cumulative proposal multiplicities per cell. Legacy/v1 shards proxy visits
    # with the policy; pre-v3 shards synthesize zero proposal counts.
    visits: torch.Tensor
    betas: torch.Tensor
    proposal_counts: torch.Tensor
    values: torch.Tensor
    selected_actions: torch.Tensor
    game_ids: torch.Tensor
    plies: torch.Tensor
    seeds: torch.Tensor
    height: int
    width: int
    sources: tuple[str, ...]
    # The same policy targets in the shard's own sparse layout. The dense
    # tensors above stay for callers that want a whole shard at once; the
    # replay window holds only this, which is 254x smaller.
    sparse: "SparsePolicy | None" = None
    # Final-board ownership per sample, mover-relative, flattened to
    # policy_resolution**2. None for shards written before `final_stones`
    # existed, which the loss masks rather than treating as "nobody owns it".
    ownerships: "torch.Tensor | None" = None

    @property
    def samples(self) -> int:
        return self.states.shape[0]

    @property
    def channels(self) -> int:
        return self.states.shape[1]


@dataclass(frozen=True)
class PreparedRasterDataset:
    """Training tensors after raw replay policy supervision has been consumed."""

    states: torch.Tensor
    policies: torch.Tensor
    policy_masks: torch.Tensor
    values: torch.Tensor
    height: int
    width: int
    sources: tuple[str, ...]
    # Carried from the raw dataset. None when no shard in the window had a
    # `final_stones` sidecar, which the ownership loss masks rather than
    # training on as "nobody owns anything".
    ownerships: "torch.Tensor | None" = None
    # The same states in packed form, set by the replay cache when the layout
    # allows it. A cached shard stays resident for every update its window
    # spans, so packing here is what bounds the window's memory. `states` then
    # holds an empty view kept for its dtype and channel count, and callers
    # wanting rows expand this instead. See vgo_training/packed_states.py.
    packed_states: "PackedStates | None" = None

    @property
    def samples(self) -> int:
        if self.packed_states is not None:
            return self.packed_states.samples
        return self.states.shape[0]

    @property
    def channels(self) -> int:
        if self.packed_states is not None:
            return self.packed_states.channels
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
    # The placement grid may be coarser than the render raster, so policy_size
    # need not equal raster pixels. When it does not, the placement grid must be
    # square (the decoupled grid is always `policy_resolution^2 + 1`) and no
    # finer than the raster it is pooled from.
    placement_cells = policy_size - 1
    if placement_cells <= 0:
        raise ValueError("policy size must include at least one placement cell plus pass")
    if placement_cells != height * width:
        side = round(placement_cells**0.5)
        if side * side != placement_cells:
            raise ValueError(
                "decoupled policy size must be a square placement grid plus pass"
            )
        if side > min(height, width):
            raise ValueError("placement grid must not exceed the raster resolution")
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
    proposal_counts = np.zeros_like(policies, dtype=np.uint32)
    return (
        states,
        # Legacy shards have no sparse cell list; the window falls back to the
        # dense tensors for them.
        None,
        None,
        policies,
        masks,
        visits,
        beta,
        proposal_counts,
        values,
        selected,
        zeros,
        zeros.copy(),
        zeros.copy(),
    )


def _render_states(
    path: Path,
    records: np.ndarray,
    channels: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Renders a shard's positions, preferring the Rust rasterizer.

    `rasterize_records` below is a hand-kept NumPy port of `rasterize_into`.
    That was affordable while every channel was a closed-form function of
    stone distances. It stopped being so once `settled` arrived: it is a
    radial contour solve over the legal set, and a second implementation of
    that geometry would have to agree exactly, which is the drift the position
    format exists to remove.

    So when the Rust renderer is available and the caller wants channels the
    port does not implement, shell out to it. The port still serves the ten
    channels it was written for, which keeps older runs loading unchanged.
    """
    if channels == CHANNEL_COUNT:
        return rasterize_records(records, channels, height, width)
    binary = _RUST_RENDERER
    if not binary.exists():
        raise ValueError(
            f"{channels} channels needs the Rust rasterizer at {binary}; "
            "build it with "
            "`cargo build --release -p vgo-raster --example render_shard`"
        )
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "rasters.bin"
        completed = subprocess.run(
            [
                str(binary),
                str(path),
                str(destination),
                str(width),
                _RASTER_KINDS[channels],
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        produced = completed.stdout.split()
        if len(produced) != 3 or int(produced[1]) != channels:
            raise ValueError(
                f"renderer produced {completed.stdout.strip()!r}, expected "
                f"{len(records)} {channels} {width}"
            )
        return np.fromfile(destination, dtype="<f4").reshape(
            len(records), channels, height, width
        )


def rasterize_records(
    records: np.ndarray, channels: int, height: int, width: int
) -> np.ndarray:
    """Renders v4 positions into semantic rasters.

    A port of `rasterize_into` in `crates/vgo-raster/src/lib.rs`, vectorized over
    samples and pixels. The two must agree exactly: generation feeds the model
    from the Rust path and training feeds it from here, so any divergence is a
    train/serve skew that no test downstream of this would catch. See
    `tests/test_dataset.py::rasterization_matches_the_rust_reference`.

    Distances are accumulated squared, as in the Rust, so the two take square
    roots at the same points and round identically.
    """
    if channels != CHANNEL_COUNT:
        raise ValueError(
            f"v4 shards render {CHANNEL_COUNT} semantic channels, header says {channels}"
        )
    samples = records.shape[0]
    # Pixel centres, matching the Rust's (index + 0.5) / extent.
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    radius = np.asarray(records["radius"], dtype=np.float64)
    to_move = np.asarray(records["to_move"], dtype=np.uint8)
    passes = np.asarray(records["consecutive_passes"], dtype=np.uint32)
    counts = np.asarray(records["stone_count"], dtype=np.int64)
    stone_x = np.asarray(records["stones"]["x"], dtype=np.float64)
    stone_y = np.asarray(records["stones"]["y"], dtype=np.float64)
    stone_c = np.asarray(records["stones"]["color"], dtype=np.uint8)

    out = np.zeros((samples, CHANNEL_COUNT, height * width), dtype=np.float32)
    infinity = np.float64(np.inf)
    empty = np.full(grid_x.shape, infinity)
    for index in range(samples):
        # Slice to the live stones before computing anything. Records pad to
        # V4_STONE_CAPACITY, and a typical position fills ~30 of 128 slots, so
        # working at capacity spends most of the time on padding -- measured at
        # 47.9 ms against 2.9 ms for the same block over live stones only.
        #
        # One sample at a time rather than one array over all of them: stones x
        # pixels is already several megabytes, and samples x stones x pixels at
        # 128x128 would be tens of gigabytes.
        count = int(counts[index])
        if count:
            sx = stone_x[index, :count]
            sy = stone_y[index, :count]
            dx = grid_x[None, :] - sx[:, None]
            dy = grid_y[None, :] - sy[:, None]
            square = dx * dx + dy * dy

            current_mask = stone_c[index, :count] == to_move[index]
            current_square = (
                square[current_mask].min(axis=0) if current_mask.any() else empty
            )
            opponent_square = (
                square[~current_mask].min(axis=0) if (~current_mask).any() else empty
            )
            if count >= 2:
                # Only the two smallest are needed, so partition rather than sort.
                partitioned = np.partition(square, 1, axis=0)
                nearest_square = partitioned[0]
                second_square = partitioned[1]
            else:
                nearest_square = square[0]
                second_square = empty
        else:
            current_square = empty
            opponent_square = empty
            nearest_square = empty
            second_square = empty

        r = radius[index]
        current_distance = np.sqrt(current_square)
        opponent_distance = np.sqrt(opponent_square)
        nearest = np.sqrt(nearest_square)
        second = np.sqrt(second_square)
        scale = max(4.0 * r, np.finfo(np.float64).eps)

        out[index, 0] = (current_distance <= r).astype(np.float32)
        out[index, 1] = (opponent_distance <= r).astype(np.float32)
        # ownership(): nearer side takes the cell, a tie splits it, and neither
        # owns a cell when the board is empty.
        both_infinite = ~np.isfinite(current_distance) & ~np.isfinite(opponent_distance)
        current_area = np.where(current_distance < opponent_distance, 1.0, 0.0)
        opponent_area = np.where(opponent_distance < current_distance, 1.0, 0.0)
        tie = (current_distance == opponent_distance) & ~both_infinite
        current_area = np.where(tie, 0.5, current_area)
        opponent_area = np.where(tie, 0.5, opponent_area)
        out[index, 2] = current_area
        out[index, 3] = opponent_area
        out[index, 4] = np.where(
            np.isfinite(current_distance), np.clip(current_distance / scale, 0.0, 1.0), 1.0
        )
        out[index, 5] = np.where(
            np.isfinite(opponent_distance), np.clip(opponent_distance / scale, 0.0, 1.0), 1.0
        )
        # second - nearest is inf - inf on an empty board; isfinite discards it,
        # but compute it under errstate so the warning does not train readers to
        # ignore warnings.
        with np.errstate(invalid="ignore"):
            ridge = np.clip(1.0 - (second - nearest) / r, 0.0, 1.0)
        out[index, 6] = np.where(np.isfinite(second), ridge, 0.0)
        board_clearance = np.minimum(
            np.minimum(grid_x - r, 1.0 - r - grid_x),
            np.minimum(grid_y - r, 1.0 - r - grid_y),
        )
        stone_clearance = np.where(np.isfinite(nearest), nearest - 2.0 * r, infinity)
        out[index, 7] = np.clip(
            np.minimum(board_clearance, stone_clearance) / r, -1.0, 1.0
        )
        out[index, 8] = 2.0 * r
        out[index, 9] = 1.0 if passes[index] > 0 else 0.0

    return out.reshape(samples, CHANNEL_COUNT * height * width)


def _v4_record_dtype(policy_size: int, version: int = REPLAY_VERSION) -> np.dtype:
    """Fixed-size v4 record: position, sparse policy, scalars.

    Both variable-length parts pad to a capacity so records stay memory-mappable;
    the live counts precede them. Capacities are the writer's, in
    `crates/vgo-selfplay/src/replay_stream.rs`, and must match exactly.
    """
    return np.dtype(
        [
            ("radius", "<f8"),
            *(
                # v5 onward. Part of the position: the same stones under
                # different komi have different winners, so a v4 shard read
                # with this field would misparse every record after it.
                [("komi", "<f8")] if version >= 5 else []
            ),
            ("to_move", "u1"),
            ("consecutive_passes", "<u4"),
            ("phase", "u1"),
            ("stone_count", "<u4"),
            (
                "stones",
                np.dtype([("x", "<f8"), ("y", "<f8"), ("color", "u1")]),
                (V4_STONE_CAPACITY,),
            ),
            ("touched", "<u4"),
            (
                "cells",
                np.dtype(
                    [
                        ("index", "<u4"),
                        ("policy", "<f4"),
                        ("visits", "<f4"),
                        ("beta", "<f4"),
                        ("proposal_counts", "<u4"),
                    ]
                ),
                (V4_POLICY_CAPACITY,),
            ),
            ("value", "<f4"),
            ("selected_action", "<u4"),
            ("game", "<u8"),
            ("ply", "<u4"),
            ("seed", "<u8"),
        ],
        align=False,
    )


def _expand_sparse_policy(
    records: np.ndarray, samples: int, policy_size: int
) -> tuple[np.ndarray, ...]:
    """Scatter the touched cells back into dense per-sample arrays.

    The mask is not stored: presence in the sparse list is the mask, so a cell
    absent from a record's list has zero policy, visits, beta, and proposals.
    """
    policies = np.zeros((samples, policy_size), dtype=np.float32)
    masks = np.zeros((samples, policy_size), dtype=np.float32)
    visits = np.zeros((samples, policy_size), dtype=np.float32)
    beta = np.zeros((samples, policy_size), dtype=np.float32)
    proposal_counts = np.zeros((samples, policy_size), dtype=np.uint32)

    counts = np.asarray(records["touched"], dtype=np.int64)
    if int(counts.max(initial=0)) > V4_POLICY_CAPACITY:
        raise ValueError("replay record claims more touched cells than the capacity")
    cells = records["cells"]
    # One flat scatter rather than a per-sample loop: build the row index for
    # every live cell, then index once.
    slot = np.arange(V4_POLICY_CAPACITY, dtype=np.int64)[None, :]
    live = slot < counts[:, None]
    rows = np.broadcast_to(
        np.arange(samples, dtype=np.int64)[:, None], live.shape
    )[live]
    columns = np.asarray(cells["index"], dtype=np.int64)[live]
    if columns.size and int(columns.max()) >= policy_size:
        raise ValueError("replay cell index is outside the policy tensor")
    policies[rows, columns] = np.asarray(cells["policy"])[live]
    masks[rows, columns] = 1.0
    visits[rows, columns] = np.asarray(cells["visits"])[live]
    beta[rows, columns] = np.asarray(cells["beta"])[live]
    proposal_counts[rows, columns] = np.asarray(cells["proposal_counts"])[live]
    return policies, masks, visits, beta, proposal_counts


def _load_replay_v4(
    path: Path,
    samples: int,
    channels: int,
    height: int,
    width: int,
    policy_size: int,
    version: int = REPLAY_VERSION,
) -> tuple[np.ndarray, ...]:
    """Loads a position shard, rendering each state at load time.

    v4 stores the position rather than a picture of it, so the raster is
    produced here and the layout is a training-time choice rather than a
    property of the data -- see docs/POSITION_SHARDS.md.
    """
    record_dtype = _v4_record_dtype(policy_size, version)
    expected_bytes = HEADER.size + samples * record_dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"replay size mismatch: expected {expected_bytes}, got {path.stat().st_size}"
        )
    records = np.memmap(
        path, dtype=record_dtype, mode="r", offset=HEADER.size, shape=(samples,)
    )
    if int(np.asarray(records["stone_count"]).max(initial=0)) > V4_STONE_CAPACITY:
        raise ValueError("replay record claims more stones than the capacity")

    states = _render_states(path, records, channels, height, width)
    policies, masks, visits, beta, proposal_counts = _expand_sparse_policy(
        records, samples, policy_size
    )
    # The sparse cells as stored, copied off the memmap so the dataset does not
    # pin the file. This is what the replay window keeps.
    sparse_cells = np.array(records["cells"], copy=True)
    sparse_counts = np.array(records["touched"], copy=True)
    return (
        states,
        sparse_cells,
        sparse_counts,
        policies,
        masks,
        visits,
        beta,
        proposal_counts,
        np.array(records["value"], copy=True),
        np.array(records["selected_action"], copy=True).astype(np.int64),
        np.array(records["game"], copy=True).astype(np.int64),
        np.array(records["ply"], copy=True).astype(np.int64),
        np.array(records["seed"], copy=True).astype(np.int64),
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
    if version >= 3:
        fields.append(("proposal_counts", "<u4", (policy_size,)))
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
    if version >= 3:
        proposal_counts = np.array(records["proposal_counts"], copy=True)
    else:
        proposal_counts = np.zeros_like(policies, dtype=np.uint32)
    return (
        np.array(records["state"], copy=True),
        None,
        None,
        policies,
        masks,
        visits,
        beta,
        proposal_counts,
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
        if version >= 4:
            arrays = _load_replay_v4(
                path, samples, channels, height, width, policy_size, version
            )
        else:
            arrays = _load_replay(
                path, version, samples, channels, height, width, policy_size
            )
    else:
        arrays = _load_legacy(path, samples, channels, height, width, policy_size)
    (
        states,
        sparse_cells,
        sparse_counts,
        policies,
        policy_masks,
        visits,
        beta,
        proposal_counts,
        values,
        selected,
        games,
        plies,
        seeds,
    ) = arrays
    states = states.reshape(samples, channels, height, width)

    if not np.isfinite(states).all() or not np.isfinite(policies).all():
        raise ValueError("dataset contains non-finite tensors")
    if not np.isfinite(visits).all() or np.any(visits < 0.0):
        raise ValueError("visit counts must be finite and nonnegative")
    if not np.isfinite(beta).all() or np.any((beta < 0.0) | (beta > 1.0)):
        raise ValueError("sampling probabilities must be finite and in [0, 1]")
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
    if np.any((visits > 0.0) & (policy_masks == 0.0)):
        raise ValueError("positive visit counts must be included in the mask")
    if np.any((beta > 0.0) & (policy_masks == 0.0)):
        raise ValueError("positive sampling probabilities must be included in the mask")
    if np.any((proposal_counts > 0) & (policy_masks == 0.0)):
        raise ValueError("positive proposal counts must be included in the mask")
    visit_totals = visits.sum(axis=1, keepdims=True)
    if np.any(visit_totals <= 0.0):
        raise ValueError("every sample must contain at least one visit")
    if not np.allclose(visits / visit_totals, policies, atol=1e-5):
        raise ValueError("policy targets must equal normalized visit counts")
    if np.any(beta[:, -1] != 0.0):
        raise ValueError("the deterministically enumerated pass action must have beta zero")
    if np.any(proposal_counts[:, -1] != 0):
        raise ValueError(
            "the deterministically enumerated pass action must have proposal count zero"
        )
    counted_rows = np.any(proposal_counts[:, :-1] > 0, axis=1)
    proposal_support = proposal_counts[:, :-1] > 0
    beta_support = beta[:, :-1] > 0.0
    if np.any(counted_rows[:, None] & (proposal_support != beta_support)):
        raise ValueError(
            "counted placement support must equal positive-beta placement support"
        )
    coarse_rows = np.any(beta[:, :-1] > 0.0, axis=1)
    missing_beta = (
        coarse_rows[:, None]
        & (policy_masks[:, :-1] != 0.0)
        & (beta[:, :-1] == 0.0)
    )
    if np.any(missing_beta):
        raise ValueError("coarse-sampled placement candidates must have positive beta")
    replay_actions = selected >= 0
    if np.any(selected[replay_actions] >= policy_size):
        raise ValueError("selected replay action is outside the policy tensor")
    if np.any(policy_masks[np.arange(samples)[replay_actions], selected[replay_actions]] == 0.0):
        raise ValueError("selected replay action is absent from the policy mask")

    return RasterDataset(
        # Half precision. A raster is 0/1 planes and one ramp; fp16 is exact for
        # the first and keeps about three decimals of the second, which an
        # ablation could not distinguish from fp32 (2.8211 CE against 2.8441 for
        # a 256-level quantization -- but that arm lost, so the margin is where
        # fp16 sits, not below it). Halving the replay window's footprint is
        # what lets it hold sixteen shards rather than five.
        states=torch.from_numpy(states).half(),
        policies=torch.from_numpy(policies),
        policy_masks=torch.from_numpy(policy_masks).bool(),
        visits=torch.from_numpy(visits),
        betas=torch.from_numpy(beta),
        proposal_counts=torch.from_numpy(proposal_counts),
        values=torch.from_numpy(values),
        selected_actions=torch.from_numpy(selected),
        game_ids=torch.from_numpy(games),
        plies=torch.from_numpy(plies),
        seeds=torch.from_numpy(seeds),
        height=height,
        width=width,
        sources=(str(path),),
        ownerships=(
            None
            if (own := shard_ownership(path, samples, int(round((policy_size - 1) ** 0.5))))
            is None
            else torch.from_numpy(own)
        ),
        sparse=None if sparse_cells is None else SparsePolicy(
            indices=torch.from_numpy(sparse_cells["index"].astype(np.int64)),
            policies=torch.from_numpy(sparse_cells["policy"].astype(np.float32)),
            visits=torch.from_numpy(sparse_cells["visits"].astype(np.float32)),
            betas=torch.from_numpy(sparse_cells["beta"].astype(np.float32)),
            proposal_counts=torch.from_numpy(
                sparse_cells["proposal_counts"].astype(np.float32)
            ),
            counts=torch.from_numpy(sparse_counts.astype(np.int64)),
            policy_size=policy_size,
        ),
    )


_CONCATENATED_FIELDS = (
    "states",
    "policies",
    "policy_masks",
    "visits",
    "betas",
    "proposal_counts",
    "values",
    "selected_actions",
    "game_ids",
    "plies",
    "seeds",
)


def load_datasets(paths: Iterable[str | Path]) -> RasterDataset:
    """Concatenate replay shards without ever holding two full copies.

    `torch.cat` over a list of loaded shards allocates the whole output while
    every input is still referenced, so peak memory is twice the window. At a
    coupled 128x128 policy a shard is ~6 GB, and a five-shard window peaked at
    60 GB against 60 GB of RAM -- see the ddrnet-wide OOM. Here each shard is
    copied into a preallocated output and then dropped, so the peak is the
    window plus one shard rather than the window twice.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("at least one replay dataset is required")

    # Headers are cheap to read and give the total sample count up front, which
    # is what lets the output be allocated once.
    headers = [_read_header(Path(path).resolve(strict=True)) for path in paths]
    first_shape = headers[0][3:6]
    for header in headers[1:]:
        if header[3:6] != first_shape:
            raise ValueError("all replay datasets must have the same raster shape")
    total = sum(header[2] for header in headers)

    output: dict[str, torch.Tensor] = {}
    sources: list[str] = []
    # The sparse cells concatenate too, at 1/254 the size of the dense form.
    sparse_parts: list[SparsePolicy] = []
    ownership_parts: list[torch.Tensor] = []
    offset = 0
    for path in paths:
        shard = load_dataset(path)
        if not output:
            for field in _CONCATENATED_FIELDS:
                template = getattr(shard, field)
                output[field] = torch.empty(
                    (total, *template.shape[1:]), dtype=template.dtype
                )
        stop = offset + shard.samples
        for field in _CONCATENATED_FIELDS:
            output[field][offset:stop].copy_(getattr(shard, field))
        sources.extend(shard.sources)
        if shard.sparse is not None:
            sparse_parts.append(shard.sparse)
        if shard.ownerships is not None:
            ownership_parts.append(shard.ownerships)
        offset = stop
        # Drop the shard before loading the next one; without this the peak is
        # the window plus every shard already consumed.
        del shard

    if offset != total:
        raise ValueError(f"expected {total} concatenated samples, wrote {offset}")

    sparse = None
    if len(sparse_parts) == len(paths) and sparse_parts:
        sparse = SparsePolicy(
            indices=torch.cat([part.indices for part in sparse_parts]),
            policies=torch.cat([part.policies for part in sparse_parts]),
            visits=torch.cat([part.visits for part in sparse_parts]),
            betas=torch.cat([part.betas for part in sparse_parts]),
            proposal_counts=torch.cat(
                [part.proposal_counts for part in sparse_parts]
            ),
            counts=torch.cat([part.counts for part in sparse_parts]),
            policy_size=sparse_parts[0].policy_size,
        )

    # Only when every shard has it: a partial stack would silently misalign
    # rows against the other tensors.
    ownerships = (
        torch.cat(ownership_parts)
        if len(ownership_parts) == len(paths) and ownership_parts
        else None
    )
    return RasterDataset(
        **output,
        height=first_shape[1],
        width=first_shape[2],
        sources=tuple(sources),
        sparse=sparse,
        ownerships=ownerships,
    )


def shard_ownership(path: Path, samples: int, resolution: int) -> "np.ndarray | None":
    """Per-sample ownership targets for a shard, or None if it has no sidecar.

    One map is rendered per *game* and broadcast across that game's sample
    range: the target is the final board, which every position in a game shares.
    Rendering per position would be ~55x the work for identical rows.

    `first_sample` and `sample_count` in games.jsonl are the join, which is what
    they were written for. Games missing `final_stones` -- shards from before
    that field existed -- leave their rows zero, and the caller masks them.
    """
    sidecar = path.parent / "games.jsonl"
    if not sidecar.exists():
        return None
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    if not rows or "final_stones" not in rows[0]:
        return None

    cells = resolution * resolution
    out = np.zeros((samples, cells), dtype=np.float32)
    for row in rows:
        stones = row.get("final_stones") or []
        start, count = int(row["first_sample"]), int(row["sample_count"])
        if not stones or count <= 0 or start + count > samples:
            continue
        # Mover-relative, so the sign follows whose turn it is. Ply parity gives
        # the mover: even plies are Black's.
        plies = np.arange(count)
        mover_is_black = (plies % 2) == 0
        out[start : start + count] = ownership_targets(
            [tuple(stone) for stone in stones], mover_is_black, resolution
        )
    return out


def ownership_targets(
    final_stones: list[tuple[float, float, int]],
    mover_is_black: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Per-cell ownership of the final board, from each sample's mover's view.

    Ownership at a point is the colour of the nearest stone -- the same rule the
    rasteriser uses for the voronoi channels, so the target cannot drift from
    the inputs the net reads. Rendered from the stored final position rather
    than stored as a map: 26x smaller, and it stays correct if the resolution
    changes.

    Returns +1 where the sample's mover owns the cell and -1 where the opponent
    does, so the target is mover-relative exactly as the value target is.
    """
    cells = resolution * resolution
    if not final_stones:
        return np.zeros((mover_is_black.shape[0], cells), dtype=np.float32)

    step = 1.0 / resolution
    axis = (np.arange(resolution, dtype=np.float64) + 0.5) * step
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    stones = np.asarray(final_stones, dtype=np.float64)
    deltas = points[:, None, :] - stones[None, :, :2]
    nearest = np.argmin(np.einsum("ijk,ijk->ij", deltas, deltas), axis=1)
    # Stored colour is 0 for Black and 1 for White.
    black_owns = stones[nearest, 2] == 0.0

    # Mover-relative: flip the sign for samples where White is to move.
    view = np.where(mover_is_black[:, None], 1.0, -1.0)
    return (np.where(black_owns, 1.0, -1.0)[None, :] * view).astype(np.float32)


def replay_diagnostics(dataset: RasterDataset, *, maximum_pairs: int = 400) -> dict[str, object]:
    """Health metrics for a replay shard, keyed on whether the policy target is
    learnable at all.

    `ply0_candidate_jaccard` is the mean pairwise overlap of candidate sets across
    games at ply zero, where every game sees the identical empty board. If the
    coarse-to-fine sampler is drawing from a board-dependent map, those sets
    should substantially agree. Near zero means the target's support relocates
    every game, which is the unlearnable-target failure the redesign exists to
    fix -- see docs/POLICY_REDESIGN.md.

    `top1_visit_share` is the mean fraction of root visits landing on the
    most-visited child. Very high with a small candidate count suggests search is
    committing before progressive widening has introduced later candidates.

    `distinct_opening_moves` counts how many different first moves the shard
    contains. Under deterministic (argmax) selection with a stable candidate
    sampler this collapses toward one.
    """
    import itertools

    plies = dataset.plies
    opening = (plies == 0).nonzero().flatten()
    visits = dataset.visits.float()
    totals = visits.sum(dim=1).clamp(min=1.0)
    top1 = float((visits.max(dim=1).values / totals).mean()) if dataset.samples else 0.0

    jaccard = float("nan")
    distinct_openings = 0
    if len(opening) >= 1:
        distinct_openings = int(torch.unique(dataset.selected_actions[opening]).numel())
    if len(opening) >= 2:
        # uint32 proposal counts do not support comparison on all torch builds.
        support = dataset.proposal_counts[opening].long() > 0
        overlaps = []
        for left, right in itertools.islice(
            itertools.combinations(range(len(opening)), 2), maximum_pairs
        ):
            first, second = support[left], support[right]
            union = (first | second).sum().clamp(min=1)
            overlaps.append(float((first & second).sum() / union))
        if overlaps:
            jaccard = sum(overlaps) / len(overlaps)

    explored = (visits > 0).sum(dim=1).float()
    return {
        "ply0_games": int(len(opening)),
        "ply0_candidate_jaccard": jaccard,
        "distinct_opening_moves": distinct_openings,
        "top1_visit_share": top1,
        "explored_candidates_per_position": float(explored.mean()) if dataset.samples else 0.0,
    }
