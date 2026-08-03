//! Does a 128x128 raster resolve what the model needs to see?
//!
//! Two questions, both about whether the picture the net reads is faithful to
//! the position the engine computes:
//!
//!   * **Settled.** The channel is a region boundary solved in closed form.
//!     Rasterising it samples that region at cell centres, so a cell whose
//!     centre falls on the wrong side of the boundary is a misclassified pixel.
//!     Comparing the rendered channel against `SettledRegion::contains` at a
//!     much finer grid measures how much of the board that costs.
//!
//!   * **Off-grid stones.** Snapping pushes a placement off its binding
//!     constraint by a margin, so legal moves land at irrational coordinates by
//!     design -- the engine's own moves do it too. The rasteriser takes exact
//!     float coordinates and never snaps, but that is worth demonstrating
//!     rather than asserting.
//!
//!     raster_fidelity <sgf-with-moves>

use std::fs;

use vgo_core::{Color, Point, Position, SettledRegion, Stone, legal_set_vertices, place};
use vgo_raster::{COMPACT_CHANNELS, RasterConfig, RasterKind, rasterize_any_into};

/// Stones parsed out of an SGF's move list, applied in order.
fn parse_moves(text: &str) -> Vec<Stone> {
    let mut stones = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(';') {
        rest = &rest[start + 1..];
        let colour = match rest.as_bytes().first() {
            Some(b'B') => Color::Black,
            Some(b'W') => Color::White,
            _ => continue,
        };
        let Some(open) = rest.find('[') else { continue };
        let Some(close) = rest.find(']') else { continue };
        let body = &rest[open + 1..close];
        let Some((x, y)) = body.split_once(',') else { continue };
        let (Ok(x), Ok(y)) = (x.trim().parse::<f64>(), y.trim().parse::<f64>()) else {
            continue;
        };
        stones.push(Stone::new(x, y, colour));
    }
    stones
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("sgf path");
    let text = fs::read_to_string(&path).expect("read sgf");
    let radius = text
        .find("RA[")
        .and_then(|at| {
            let rest = &text[at + 3..];
            rest.find(']').and_then(|end| rest[..end].parse::<f64>().ok())
        })
        .expect("RA[] radius");

    let stones = parse_moves(&text);
    println!("{} stones, radius {radius}", stones.len());

    // How far each stone sits from the nearest 128-grid cell centre.
    let resolution: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let cell = 1.0 / resolution as f64;
    let mut off_grid = 0usize;
    let mut worst: f64 = 0.0;
    for stone in &stones {
        let dx = (stone.x / cell - 0.5 - (stone.x / cell - 0.5).round()).abs() * cell;
        let dy = (stone.y / cell - 0.5 - (stone.y / cell - 0.5).round()).abs() * cell;
        let offset = dx.max(dy);
        if offset > 1.0e-5 {
            off_grid += 1;
        }
        worst = worst.max(offset);
    }
    println!(
        "off-grid stones: {off_grid} of {}, worst offset {worst:.6} ({:.2} of a cell)",
        stones.len(),
        worst / cell
    );

    // Replay through the engine rather than dropping every stone on at once:
    // captures remove stones, so the raw move list is not a legal position.
    let mut position = Position::new(radius, Vec::new(), Color::Black);
    let mut applied = 0usize;
    for stone in &stones {
        match place(&position, stone.x, stone.y) {
            Ok(result) => {
                position = result.position;
                applied += 1;
            }
            Err(error) => {
                println!("  move {} rejected by the engine: {error:?}", applied + 1);
                break;
            }
        }
    }
    println!("replayed {applied} of {} moves; {} stones on the final board",
             stones.len(), position.stones().len());

    // Settled fidelity: render the channel, then ask the closed-form region
    // about the same points and count disagreements.
    let config = RasterConfig::square_of(resolution, RasterKind::Compact);
    let mut data = vec![0.0_f32; config.pixels() * COMPACT_CHANNELS.len()];
    rasterize_any_into(&position, config, &mut data);
    // Compact plane 3 is `settled`.
    let plane = &data[3 * config.pixels()..4 * config.pixels()];

    let vertices = legal_set_vertices(&position);
    let regions: Vec<SettledRegion> = (0..position.stones().len())
        .map(|index| SettledRegion::new(&position, index, &vertices))
        .collect();

    let mut disagreements = 0usize;
    let mut settled_pixels = 0usize;
    for row in 0..resolution {
        for column in 0..resolution {
            let point = Point::new(
                (column as f64 + 0.5) * cell,
                (row as f64 + 0.5) * cell,
            );
            let truth = regions.iter().any(|region| region.contains(point));
            let drawn = plane[row * resolution + column] > 0.5;
            if truth {
                settled_pixels += 1;
            }
            if truth != drawn {
                disagreements += 1;
            }
        }
    }
    // Dump the planes as raw f32 so they can be viewed, not just counted.
    if let Some(dump) = std::env::var_os("RASTER_DUMP") {
        fs::write(&dump, unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&data[..]),
            )
        })
        .expect("write raster dump");
        println!("wrote {} planes to {}", COMPACT_CHANNELS.len(), dump.to_string_lossy());
    }

    let total = resolution * resolution;
    println!(
        "settled: {settled_pixels} of {total} pixels ({:.1}%), \
         disagreements with the closed form: {disagreements} ({:.3}%)",
        100.0 * settled_pixels as f64 / total as f64,
        100.0 * disagreements as f64 / total as f64,
    );
}
