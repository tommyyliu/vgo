//! Render a position as SVG, matching what the web client draws.
//!
//! The training rasterizer answers "what does the model see": ten semantic
//! channels at 128x128, each derived per pixel from nearest-stone distance.
//! That is the wrong tool for reading a game by eye. It cannot show a cell
//! boundary as a line, cannot distinguish a boundary between two of one
//! player's groups from the boundary between opposing groups, and quantizes
//! every edge to the pixel grid.
//!
//! `renderBoard()` in `reference/js-reference/voronoi_go.html` draws exact
//! polygons with stroked edges, and the distinction it draws that matters most
//! is soft-versus-hard boundaries: grey between cells of the same group, bright
//! white between different groups. That is connectivity, which is what life and
//! death turn on, and it is invisible in every raster channel.
//!
//! Everything needed is already computed. `Analysis` carries the cell polygons,
//! the group partition, and which groups are alive or settled, so this is a
//! serializer rather than a second geometry implementation -- and it emits
//! vectors, so the output is exact rather than sampled.
//!
//! Colours are the client's and are absolute: Black is always blue, White
//! always orange. The rasterizer's are relative to the side to move, which is
//! right for a model reading one position and wrong for a human reading a game,
//! where a stone changing colour between plies is only confusing.

use std::fmt::Write as _;

use vgo_core::{Analysis, Color, Point, Position};

const BLACK_STONE: &str = "#5aa2ec";
const BLACK_REGION: &str = "#22405c";
const WHITE_STONE: &str = "#f0975a";
const WHITE_REGION: &str = "#68391a";
const BACKGROUND: &str = "#0e1116";
/// `freeMaskSVG`: legal cells tinted at alpha 42/255, illegal left bare.
const LEGAL_TINT: &str = "rgb(205,214,232)";
const LEGAL_TINT_ALPHA: f64 = 42.0 / 255.0;

/// What to draw. The client exposes these as checkboxes; the defaults here are
/// its defaults.
#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    pub regions: bool,
    pub boundaries: bool,
    pub legal_mask: bool,
    pub settled: bool,
    pub stone_ids: bool,
    /// Board edge in pixels; the viewBox stays in board units either way.
    pub size: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            regions: true,
            boundaries: true,
            legal_mask: true,
            settled: false,
            stone_ids: false,
            size: 640,
        }
    }
}

fn coordinate(value: f64) -> String {
    // Five decimals is the client's display precision (`f`), which is finer
    // than any renderer resolves and keeps the document small.
    let text = format!("{value:.5}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn polygon_path(points: &[Point]) -> String {
    let mut path = String::new();
    for (index, point) in points.iter().enumerate() {
        let command = if index == 0 { 'M' } else { 'L' };
        let _ = write!(
            path,
            "{command}{} {}",
            coordinate(point.x),
            coordinate(point.y)
        );
    }
    path.push('Z');
    path
}

/// The legal-placement overlay, sampled on a grid.
///
/// The client calls this "the one intentionally raster overlay": the legal set
/// is an intersection of disc complements and half-planes, so its boundary is
/// made of circular arcs that SVG could express but that are not worth the
/// construction. A grid of rects at the same 256 resolution the client uses
/// reads identically once drawn.
fn legal_mask(position: &Position, samples: u32) -> String {
    let mut runs = String::new();
    let step = 1.0 / f64::from(samples);
    for row in 0..samples {
        let y = (f64::from(row) + 0.5) * step;
        let mut run_start: Option<u32> = None;
        for column in 0..=samples {
            let inside = column < samples && {
                let x = (f64::from(column) + 0.5) * step;
                vgo_core::is_legal_placement(position, x, y)
            };
            match (inside, run_start) {
                (true, None) => run_start = Some(column),
                (false, Some(start)) => {
                    // One subpath per run, all in a single <path>. Separate
                    // translucent rects composite against each other wherever
                    // their edges meet, and the seam reads as banding across
                    // the mask; subpaths of one path are filled once.
                    let (x0, y0) = (f64::from(start) * step, f64::from(row) * step);
                    let (x1, y1) = (f64::from(column) * step, y0 + step);
                    let _ = write!(
                        runs,
                        "M{} {}L{} {}L{} {}L{} {}Z",
                        coordinate(x0),
                        coordinate(y0),
                        coordinate(x1),
                        coordinate(y0),
                        coordinate(x1),
                        coordinate(y1),
                        coordinate(x0),
                        coordinate(y1),
                    );
                    run_start = None;
                }
                _ => {}
            }
        }
    }
    if runs.is_empty() {
        return String::new();
    }
    format!(
        r#"<path d="{runs}" fill="{LEGAL_TINT}" fill-opacity="{LEGAL_TINT_ALPHA}" stroke="none"/>"#
    )
}

/// One position as a standalone SVG document.
#[must_use]
pub fn render(position: &Position, options: RenderOptions) -> String {
    let analysis = Analysis::new(position);
    let geometry = &analysis.geometry;
    let stones = position.stones();
    let radius = position.radius();
    let size = options.size;

    let mut body = String::new();
    let _ = write!(
        body,
        r#"<rect x="0" y="0" width="1" height="1" fill="{BACKGROUND}"/>"#
    );

    // 1. filled cells, exact polygons
    if options.regions {
        for (index, cell) in geometry.cells.iter().enumerate() {
            if cell.polygon.len() < 3 {
                continue;
            }
            let fill = match stones[index].color {
                Color::Black => BLACK_REGION,
                Color::White => WHITE_REGION,
            };
            let _ = write!(
                body,
                r#"<path d="{}" fill="{fill}"/>"#,
                polygon_path(&cell.polygon)
            );
        }
    }

    // 2. overlays, under the edges as in the client
    if options.legal_mask {
        body.push_str(&legal_mask(position, 256));
    }
    if options.settled {
        let mut settled = String::new();
        for (index, cell) in geometry.cells.iter().enumerate() {
            if cell.polygon.len() >= 3
                && analysis.settled_groups.contains(&geometry.groups[index])
            {
                settled.push_str(&polygon_path(&cell.polygon));
            }
        }
        if !settled.is_empty() {
            let _ = write!(
                body,
                r#"<path d="{settled}" fill="rgba(4,6,10,0.40)" stroke="none"/>"#
            );
        }
    }

    // 3. cell boundaries: soft within a group, hard between groups. This is the
    //    connectivity the raster channels cannot express.
    if options.boundaries {
        let (mut soft, mut hard) = (String::new(), String::new());
        for (index, cell) in geometry.cells.iter().enumerate() {
            for edge in &cell.edges {
                let Some(other) = edge.other_stone() else {
                    continue; // board edge
                };
                if other < index {
                    continue; // each shared edge once
                }
                let segment = format!(
                    "M{} {}L{} {}",
                    coordinate(edge.start.x),
                    coordinate(edge.start.y),
                    coordinate(edge.end.x),
                    coordinate(edge.end.y),
                );
                if geometry.groups[index] == geometry.groups[other] {
                    soft.push_str(&segment);
                } else {
                    hard.push_str(&segment);
                }
            }
        }
        if !soft.is_empty() {
            let _ = write!(
                body,
                r#"<path d="{soft}" fill="none" stroke="rgba(150,160,180,.35)" stroke-width="1" vector-effect="non-scaling-stroke"/>"#
            );
        }
        if !hard.is_empty() {
            let _ = write!(
                body,
                r#"<path d="{hard}" fill="none" stroke="rgba(236,239,246,.92)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>"#
            );
        }
    }

    // 4. stones, lighter than their territory so they read against it
    for (index, stone) in stones.iter().enumerate() {
        let fill = match stone.color {
            Color::Black => BLACK_STONE,
            Color::White => WHITE_STONE,
        };
        let _ = write!(
            body,
            r#"<circle cx="{}" cy="{}" r="{}" fill="{fill}" stroke="rgba(0,0,0,.6)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>"#,
            coordinate(stone.x),
            coordinate(stone.y),
            coordinate(radius),
        );
        if options.stone_ids {
            let _ = write!(
                body,
                r#"<text x="{}" y="{}" font-size="0.02" font-family="monospace" text-anchor="middle" dominant-baseline="central" fill="rgba(0,0,0,.8)">{index}</text>"#,
                coordinate(stone.x),
                coordinate(stone.y),
            );
        }
    }

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="{size}" height="{size}">{body}</svg>"#
    )
}
