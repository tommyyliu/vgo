//! Render a shard's positions straight into the packed planes training holds.
//!
//!     pack_shard <shard.vgo> <out.bin> [resolution] [kind]
//!
//! `render_shard` emits dense f32, which training then packs and throws away:
//! at 256x256x7 that is 1.75 MB per sample crossing the process boundary to
//! produce 152 KB of resident window, an 11.8x amplification. A 285,718-sample
//! window pushed ~500 GB through a tmpfs temp file to build 43 GB of tensors.
//!
//! Packing here instead writes what the window actually keeps. The layout is
//! `training/vgo_training/packed_states.py`, and it must match byte for byte --
//! `tests/test_pack_shard.py` compares this against `pack()` on the dense
//! render, which is the only thing that keeps the two in step.
//!
//! Output is three contiguous sections rather than per-sample interleaving, so
//! the reader can map each one at its final shape without a de-interleave copy:
//!
//!     [header][bits: samples x binary x ceil(pixels/8) u8]
//!             [continuous: samples x continuous x pixels f16]
//!             [scalars: samples x scalar f16]

use std::fs;
use std::io::Write;

use half::f16;
use vgo_core::{Color, Position, Stone};
use vgo_raster::{RasterConfig, RasterKind, rasterize_any_into};

const HEADER_V6: usize = 32;
const HEADER_V7: usize = 36;
const HEADER_V8: usize = 40;
/// Bytes of our own header: magic, samples, channels, height, width, and the
/// three class counts. A reader that mismatches any of these is reading a
/// layout this did not write.
const OUT_HEADER: usize = 8 + 4 * 7;
const OUT_MAGIC: &[u8; 8] = b"VGOPACK1";

const fn header_size(version: u32) -> usize {
    if version >= 8 { HEADER_V8 } else if version >= 7 { HEADER_V7 } else { HEADER_V6 }
}
const STONE: usize = 8 + 8 + 1;
const STONE_CAPACITY: usize = 128;
const POLICY_CAPACITY_V4: usize = 64;
const POLICY_CAPACITY_V6: usize = 128;
const fn policy_capacity(version: u32) -> usize {
    if version >= 6 { POLICY_CAPACITY_V6 } else { POLICY_CAPACITY_V4 }
}
const fn cell_bytes(version: u32) -> usize {
    if version >= 7 { 12 } else { 20 }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn read_f64(bytes: &[u8], at: usize) -> f64 {
    f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// How one raster layout's planes divide into storage classes.
///
/// Mirrors `_LAYOUTS` in packed_states.py, keyed the same way -- by channel
/// count -- so a layout this does not know falls through to an error rather
/// than being packed under the wrong classification.
struct Layout {
    binary: &'static [usize],
    scalar: &'static [usize],
    continuous: &'static [usize],
}

fn layout_for(channels: usize) -> Option<Layout> {
    match channels {
        5 => Some(Layout { binary: &[0, 1, 3], scalar: &[4], continuous: &[2] }),
        6 => Some(Layout { binary: &[0, 1, 3], scalar: &[4, 5], continuous: &[2] }),
        7 => Some(Layout { binary: &[0, 1, 3], scalar: &[4, 5, 6], continuous: &[2] }),
        9 => Some(Layout {
            binary: &[0, 1, 3, 4, 5, 6],
            scalar: &[7, 8],
            continuous: &[2],
        }),
        _ => None,
    }
}

/// Where each field sits in a record, computed once from the header.
struct Records {
    header: usize,
    stride: usize,
    version: u32,
    komi_bytes: usize,
    stones_at: usize,
    count_at: usize,
}

/// Rebuilds one stored position. Reads only `blob`, so several threads may call
/// it at once.
fn position_at(blob: &[u8], records: &Records, index: usize) -> Position {
    let base = records.header + index * records.stride;
    let radius = read_f64(blob, base);
    let komi = if records.version >= 5 { read_f64(blob, base + 8) } else { 0.0 };
    let to_move = if blob[base + 8 + records.komi_bytes] == 0 {
        Color::Black
    } else {
        Color::White
    };
    let passes = read_u32(blob, base + 8 + records.komi_bytes + 1);
    let count = read_u32(blob, base + records.count_at) as usize;
    let mut stones = Vec::with_capacity(count);
    for stone in 0..count {
        let at = base + records.stones_at + stone * STONE;
        let colour = if blob[at + 16] == 0 { Color::Black } else { Color::White };
        stones.push(Stone::new(read_f64(blob, at), read_f64(blob, at + 8), colour));
    }
    Position::new(radius, stones, to_move)
        .with_komi(komi)
        .with_passes(passes)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let source = args.next().expect("shard path");
    let destination = args.next().expect("output path");
    let resolution: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(128);
    let kind: RasterKind = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(RasterKind::Semantic);

    let blob = fs::read(&source).expect("read shard");
    let version = read_u32(&blob, 8);
    assert!(
        (4..=8).contains(&version),
        "expected replay version 4 through 8, found {version}"
    );
    let samples = read_u32(&blob, 12) as usize;
    let komi_bytes = if version >= 5 { 8 } else { 0 };
    let stones_at = 8 + komi_bytes + 1 + 4 + 1 + 4;
    let count_at = 8 + komi_bytes + 1 + 4 + 1;
    let header = header_size(version);
    let capacity = if version >= 7 {
        read_u32(&blob, 32) as usize
    } else {
        policy_capacity(version)
    };
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
    let layout = layout_for(channels)
        .unwrap_or_else(|| panic!("no packing layout for {channels} channels"));
    let bit_bytes = pixels.div_ceil(8);

    let bits_per_sample = layout.binary.len() * bit_bytes;
    let cont_per_sample = layout.continuous.len() * pixels * 2;
    let scalars_per_sample = layout.scalar.len() * 2;

    let mut bits = vec![0_u8; samples * bits_per_sample];
    let mut continuous = vec![0_u8; samples * cont_per_sample];
    let mut scalars = vec![0_u8; samples * scalars_per_sample];

    let records = Records { header, stride, version, komi_bytes, stones_at, count_at };
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(samples.max(1));
    let per_worker = samples.div_ceil(workers);

    std::thread::scope(|scope| {
        let chunks = bits
            .chunks_mut(per_worker * bits_per_sample)
            .zip(continuous.chunks_mut(per_worker * cont_per_sample))
            .zip(scalars.chunks_mut(per_worker * scalars_per_sample));
        for (index, ((bit_chunk, cont_chunk), scalar_chunk)) in chunks.enumerate() {
            let blob = &blob;
            let records = &records;
            let layout = &layout;
            scope.spawn(move || {
                let mut data = vec![0.0_f32; channels * pixels];
                let first = index * per_worker;
                for local in 0..bit_chunk.len() / bits_per_sample {
                    let position = position_at(blob, records, first + local);
                    rasterize_any_into(&position, config, &mut data);

                    // Bits, least-significant first within each byte, matching
                    // `np.packbits(..., bitorder="little")`.
                    for (slot, &channel) in layout.binary.iter().enumerate() {
                        let plane = &data[channel * pixels..(channel + 1) * pixels];
                        let at = local * bits_per_sample + slot * bit_bytes;
                        for (pixel, &value) in plane.iter().enumerate() {
                            assert!(
                                value == 0.0 || value == 1.0,
                                "channel {channel} is not binary ({value}) and \
                                 cannot be packed to bits"
                            );
                            if value == 1.0 {
                                bit_chunk[at + pixel / 8] |= 1 << (pixel % 8);
                            }
                        }
                    }
                    for (slot, &channel) in layout.continuous.iter().enumerate() {
                        let plane = &data[channel * pixels..(channel + 1) * pixels];
                        let at = local * cont_per_sample + slot * pixels * 2;
                        for (pixel, &value) in plane.iter().enumerate() {
                            let half = f16::from_f32(value).to_le_bytes();
                            cont_chunk[at + pixel * 2..at + pixel * 2 + 2]
                                .copy_from_slice(&half);
                        }
                    }
                    // One value per plane. The packer takes pixel zero and
                    // requires the plane be constant, so check rather than
                    // trust: a plane that stopped being constant would be
                    // silently truncated to its first pixel.
                    for (slot, &channel) in layout.scalar.iter().enumerate() {
                        let plane = &data[channel * pixels..(channel + 1) * pixels];
                        let first_value = plane[0];
                        assert!(
                            plane.iter().all(|&v| v == first_value),
                            "channel {channel} is not constant across its plane"
                        );
                        let at = local * scalars_per_sample + slot * 2;
                        scalar_chunk[at..at + 2]
                            .copy_from_slice(&f16::from_f32(first_value).to_le_bytes());
                    }
                }
            });
        }
    });

    let mut out = Vec::with_capacity(
        OUT_HEADER + bits.len() + continuous.len() + scalars.len(),
    );
    out.extend_from_slice(OUT_MAGIC);
    for value in [
        samples as u32,
        channels as u32,
        resolution as u32,
        resolution as u32,
        layout.binary.len() as u32,
        layout.continuous.len() as u32,
        layout.scalar.len() as u32,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&bits);
    out.extend_from_slice(&continuous);
    out.extend_from_slice(&scalars);
    let mut file = fs::File::create(&destination).expect("create output");
    file.write_all(&out).expect("write packed planes");

    println!("{samples} {channels} {resolution}");
}
