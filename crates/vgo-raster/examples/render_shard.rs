//! Render a v4/v5 shard's positions to raw f32 rasters, for offline experiments.
//!
//! Training renders shards through `rasterize_records` in Python, a hand-kept
//! port. Settled is a contour solve; porting it would be a second geometry
//! implementation that has to agree exactly, which is the drift the v4 format
//! exists to remove.
//!
//!     render_shard <shard.vgo> <out.bin> [resolution] [kind]

use std::fs;

use vgo_core::{Color, Position, Stone};
use vgo_raster::{RasterConfig, RasterKind, rasterize_any_into};

// v7 appended the policy capacity, so its header is one u32 longer.
const HEADER_V6: usize = 32;
const HEADER_V7: usize = 36;
// v8 appends the stone capacity, so boards larger than the mini one fit.
const HEADER_V8: usize = 40;

const fn header_size(version: u32) -> usize {
    if version >= 8 {
        HEADER_V8
    } else if version >= 7 {
        HEADER_V7
    } else {
        HEADER_V6
    }
}
const STONE: usize = 8 + 8 + 1;
const STONE_CAPACITY: usize = 128;
// Slots per record. v4 held 64 and v6 held 128, both fixed constants three
// files had to agree on. v7 writes the capacity into its header instead, so
// this covers only the older shards.
const POLICY_CAPACITY_V4: usize = 64;
const POLICY_CAPACITY_V6: usize = 128;

const fn policy_capacity(version: u32) -> usize {
    if version >= 6 {
        POLICY_CAPACITY_V6
    } else {
        POLICY_CAPACITY_V4
    }
}

/// Bytes per stored policy cell. v7 repacked it from 20 to 12 by dropping the
/// policy -- which is the normalised visit counts -- and narrowing the index
/// and proposal counter.
const fn cell_bytes(version: u32) -> usize {
    if version >= 7 { 12 } else { 20 }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn read_f64(bytes: &[u8], at: usize) -> f64 {
    f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let source = args.next().expect("shard path");
    let destination = args.next().expect("output path");
    let resolution: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(128);
    // The shard header records how many channels it was written with, and the
    // kind follows from that -- a caller should not have to repeat it.
    let kind: RasterKind = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(RasterKind::Semantic);

    let blob = fs::read(&source).expect("read shard");
    let version = read_u32(&blob, 8);
    assert!(
        (4..=8).contains(&version),
        "expected replay version 4 through 7, found {version}"
    );
    let samples = read_u32(&blob, 12) as usize;

    // v5 inserts komi after radius, shifting everything below it.
    let komi_bytes = if version >= 5 { 8 } else { 0 };
    let stones_at = 8 + komi_bytes + 1 + 4 + 1 + 4;
    let count_at = 8 + komi_bytes + 1 + 4 + 1;

    let header = header_size(version);
    let capacity = if version >= 7 {
        read_u32(&blob, 32) as usize
    } else {
        policy_capacity(version)
    };
    // v8 sizes the stone slots per shard, because a board mix spans 74 stones
    // at 18 units to ~540 at 50, and one constant would pad every small board.
    let stone_capacity = if version >= 8 {
        read_u32(&blob, 36) as usize
    } else {
        STONE_CAPACITY
    };
    let stride = (blob.len() - header) / samples;
    let expected = stones_at
        + stone_capacity * STONE
        + 4
        + capacity * cell_bytes(version)
        + 4 + 4 + 8 + 4 + 8;
    assert_eq!(
        stride, expected,
        "record stride {stride} does not match the expected {expected} for \
         version {version}; the layout changed and this example needs updating"
    );

    let config = RasterConfig::square_of(resolution, kind);
    let channels = config.channels();
    let pixels = config.pixels();
    let mut out = Vec::with_capacity(samples * channels * pixels * 4);
    let mut data = vec![0.0_f32; channels * pixels];

    for index in 0..samples {
        let base = header + index * stride;
        let radius = read_f64(&blob, base);
        let komi = if version >= 5 { read_f64(&blob, base + 8) } else { 0.0 };
        let to_move = if blob[base + 8 + komi_bytes] == 0 {
            Color::Black
        } else {
            Color::White
        };
        // Immediately after to_move, and dropped here until the raster grew a
        // plane that needed it. A record stores it, so a reader that rebuilds
        // the position without it renders `previous_pass` as zero for every
        // sample -- which trains a channel that is always off and then meets a
        // live one at inference.
        let passes = read_u32(&blob, base + 8 + komi_bytes + 1);
        let count = read_u32(&blob, base + count_at) as usize;
        let mut stones = Vec::with_capacity(count);
        for stone in 0..count {
            let at = base + stones_at + stone * STONE;
            let colour = if blob[at + 16] == 0 { Color::Black } else { Color::White };
            stones.push(Stone::new(read_f64(&blob, at), read_f64(&blob, at + 8), colour));
        }
        let position = Position::new(radius, stones, to_move)
            .with_komi(komi)
            .with_passes(passes);
        rasterize_any_into(&position, config, &mut data);
        for value in &data {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    fs::write(&destination, &out).expect("write rasters");
    println!("{samples} {channels} {resolution}");
}
