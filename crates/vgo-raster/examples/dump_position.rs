//! Dump one shard position's planes for both six-channel layouts.
//!
//!     dump_position <shard.vgo> <sample-index> <out-prefix> [resolution]
//!
//! Writes `<prefix>.bin` -- the union of the planes both layouts need, as u8,
//! in the order stated by `<prefix>.json` -- plus the stones and scalars needed
//! to draw the board over them. `compact-pass` and `compact-dead-zone` share
//! every plane but slot 3, so this emits five spatial planes rather than twelve.
//!
//! Bytes rather than floats because four of the five are strictly binary and the
//! fifth, the Voronoi ridge, is a display quantity here: 256 levels is finer
//! than a screen shows.
use std::fs;

use vgo_core::{Color, Position, Ruleset, Stone};
use vgo_raster::{RasterConfig, RasterKind, rasterize_any_into};

const HEADER: usize = 32;
const STONE: usize = 8 + 8 + 1;
const STONE_CAPACITY: usize = 128;

const fn policy_capacity(version: u32) -> usize {
    if version >= 6 { 128 } else { 64 }
}
fn read_u32(b: &[u8], at: usize) -> u32 { u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) }
fn read_f64(b: &[u8], at: usize) -> f64 { f64::from_le_bytes(b[at..at + 8].try_into().unwrap()) }

fn main() {
    let mut args = std::env::args().skip(1);
    let source = args.next().expect("shard path");
    let index: usize = args.next().expect("sample index").parse().expect("index");
    let prefix = args.next().expect("output prefix");
    let resolution: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(128);

    let blob = fs::read(&source).expect("read shard");
    let version = read_u32(&blob, 8);
    let samples = read_u32(&blob, 12) as usize;
    let komi_bytes = if version >= 5 { 8 } else { 0 };
    let stones_at = 8 + komi_bytes + 1 + 4 + 1 + 4;
    let count_at = 8 + komi_bytes + 1 + 4 + 1;
    let stride = (blob.len() - HEADER) / samples;

    let base = HEADER + index * stride;
    let radius = read_f64(&blob, base);
    let komi = if version >= 5 { read_f64(&blob, base + 8) } else { 0.0 };
    let to_move = if blob[base + 8 + komi_bytes] == 0 { Color::Black } else { Color::White };
    let passes = read_u32(&blob, base + 8 + komi_bytes + 1);
    let count = read_u32(&blob, base + count_at) as usize;
    let mut stones = Vec::with_capacity(count);
    for stone in 0..count {
        let at = base + stones_at + stone * STONE;
        let colour = if blob[at + 16] == 0 { Color::Black } else { Color::White };
        stones.push(Stone::new(read_f64(&blob, at), read_f64(&blob, at + 8), colour));
    }
    assert!(count <= STONE_CAPACITY);
    let _ = policy_capacity(version);

    let position = Position::new(radius, stones.clone(), to_move)
        .with_komi(komi)
        .with_passes(passes)
        .with_ruleset(Ruleset::Official);

    let pixels = resolution * resolution;
    let mut out: Vec<u8> = Vec::with_capacity(5 * pixels);
    let mut names: Vec<&str> = Vec::new();

    // compact-pass gives slots 0,1,2 (shared), 3 = settled, 4 komi, 5 pass.
    let mut ours = vec![0.0_f32; 6 * pixels];
    rasterize_any_into(
        &position,
        RasterConfig::square_of(resolution, RasterKind::CompactPass),
        &mut ours,
    );
    let mut theirs = vec![0.0_f32; 6 * pixels];
    rasterize_any_into(
        &position,
        RasterConfig::square_of(resolution, RasterKind::CompactDeadZone),
        &mut theirs,
    );

    for (slot, name) in [(0usize, "current_stones"), (1, "opponent_stones"), (2, "voronoi_ridge")] {
        names.push(name);
        out.extend(ours[slot * pixels..(slot + 1) * pixels]
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
    }
    names.push("settled");
    out.extend(ours[3 * pixels..4 * pixels].iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
    names.push("dead_zone");
    out.extend(theirs[3 * pixels..4 * pixels].iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));

    // The connection planes come from the nine-channel layout, whose slots 5
    // and 6 are the mover's connections and the opponent's.
    let mut connected = vec![0.0_f32; 9 * pixels];
    rasterize_any_into(
        &position,
        RasterConfig::square_of(resolution, RasterKind::CompactConnected),
        &mut connected,
    );
    for (slot, name) in [(5usize, "current_connections"), (6, "opponent_connections")] {
        names.push(name);
        out.extend(connected[slot * pixels..(slot + 1) * pixels]
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
    }

    fs::write(format!("{prefix}.bin"), &out).expect("write planes");

    let stone_json: Vec<String> = stones
        .iter()
        .map(|s| format!(
            "{{\"x\":{:.6},\"y\":{:.6},\"c\":\"{}\"}}",
            s.x, s.y,
            if s.color == Color::Black { "B" } else { "W" }
        ))
        .collect();
    let meta = format!(
        "{{\"resolution\":{resolution},\"planes\":[{}],\"radius\":{radius},\"komi\":{komi},\
\"passes\":{passes},\"to_move\":\"{}\",\"stones\":[{}],\
\"settled_fraction\":{:.4},\"dead_zone_fraction\":{:.4}}}",
        names.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(","),
        if to_move == Color::Black { "B" } else { "W" },
        stone_json.join(","),
        ours[3 * pixels..4 * pixels].iter().sum::<f32>() / pixels as f32,
        theirs[3 * pixels..4 * pixels].iter().sum::<f32>() / pixels as f32,
    );
    fs::write(format!("{prefix}.json"), meta).expect("write metadata");
    println!("{} planes at {resolution}^2, {count} stones, komi {komi:.4}, passes {passes}", names.len());
}
