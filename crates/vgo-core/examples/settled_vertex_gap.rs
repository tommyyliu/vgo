//! Does testing only cell vertices ever get settlement wrong?
//!
//!     cargo run --release -p vgo-core --example settled_vertex_gap
//!
//! `alive_groups_of` decides a group by walking its cells' polygon **vertices**
//! and asking whether a legal centre can get strictly closer to that vertex than
//! the owning stone. The rule it is implementing is about the whole region, not
//! its corners, and those are not the same question: the settled region is
//! star-shaped about its stone with radial boundary `T(u)`, the cell boundary is
//! `R(u)`, the cell is alive where `R(u) > T(u)`, and the vertices only sample
//! `u` at the corner directions.
//!
//! So a cell whose every corner is settled can still have an unsettled point in
//! the middle of an edge. This looks for one, by sampling each cell's boundary
//! densely and comparing the verdict with the vertices-only verdict.
//!
//! The escape test needs no private API: `exists p in L with |x-p| < |x-s|` is
//! exactly `dist(x, L) < |x - s|`, since `dist` is that minimum.
use vgo_core::{
    Analysis, Color, Point, Position, Stone, distance_to_legal_set, legal_set_vertices,
    planar_length,
};

/// Can some legal centre get strictly closer to `point` than `stone` is?
fn escapable(position: &Position, point: Point, stone: Point, vertices: &[Point]) -> bool {
    let owner = planar_length(point.x - stone.x, point.y - stone.y);
    distance_to_legal_set(position, point, Some(vertices)) < owner
}

fn fixture(count: usize, radius: f64, seed: u64) -> Position {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut stones: Vec<Stone> = Vec::new();
    let mut attempts = 0;
    while stones.len() < count && attempts < count * 400 {
        attempts += 1;
        let x = radius + next() * (1.0 - 2.0 * radius);
        let y = radius + next() * (1.0 - 2.0 * radius);
        if stones
            .iter()
            .any(|s| planar_length(s.x - x, s.y - y) < 2.0 * radius)
        {
            continue;
        }
        let colour = if stones.len() % 2 == 0 { Color::Black } else { Color::White };
        stones.push(Stone::new(x, y, colour));
    }
    Position::new(radius, stones, Color::Black).with_komi(0.104)
}

fn main() {
    let radius = 1.0 / 18.0;
    let samples_per_edge = 256;

    let (mut cells, mut disagreed, mut positions_with_gap) = (0usize, 0usize, 0usize);
    let mut worst: Option<(usize, usize, f64)> = None;

    for seed in 1..=4000u64 {
        // Sweep density: sparse boards have big cells and plenty of legal set;
        // near-full boards have small isolated legal pockets, which is the shape
        // the failure needs.
        let count = 3 + (seed as usize % 70);
        let position = fixture(count, radius, seed);
        if !position.validate().is_playable() || position.stones().len() < 2 {
            continue;
        }
        let analysis = Analysis::new(&position);
        let vertices = legal_set_vertices(&position);
        let mut gap_here = false;

        for (index, cell) in analysis.geometry.cells.iter().enumerate() {
            if cell.polygon.len() < 3 {
                continue;
            }
            cells += 1;
            let s = position.stones()[index];
            let stone = Point::new(s.x, s.y);

            let by_vertex = cell
                .polygon
                .iter()
                .any(|&v| escapable(&position, v, stone, &vertices));

            // The boundary, sampled. An unsettled point of the cell can be
            // anywhere in it, but the settled region is star-shaped about the
            // stone, so if any point along a direction is unsettled then the
            // cell's own exit point along it is too -- and that lies on the
            // boundary. Sampling edges therefore finds every direction that
            // matters.
            let mut by_edge = false;
            let mut margin = 0.0_f64;
            'edges: for (a, b) in cell
                .polygon
                .iter()
                .zip(cell.polygon.iter().cycle().skip(1))
                .take(cell.polygon.len())
            {
                for step in 1..samples_per_edge {
                    let t = step as f64 / samples_per_edge as f64;
                    let point = Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
                    let owner = planar_length(point.x - stone.x, point.y - stone.y);
                    let legal = distance_to_legal_set(&position, point, Some(&vertices));
                    if legal < owner {
                        by_edge = true;
                        margin = margin.max(owner - legal);
                        if by_vertex {
                            break 'edges;
                        }
                    }
                }
            }

            if by_edge && !by_vertex {
                disagreed += 1;
                gap_here = true;
                if worst.is_none_or(|(_, _, m)| margin > m) {
                    worst = Some((seed as usize, index, margin));
                }
            }
        }
        if gap_here {
            positions_with_gap += 1;
        }
    }

    println!("cells examined                {cells}");
    println!("vertices said settled, an edge point said alive: {disagreed}");
    println!("positions containing one      {positions_with_gap} of 4000");
    match worst {
        Some((seed, index, margin)) => println!(
            "widest margin  seed {seed}, stone {index}, {margin:.3e} \
             ({:.2}% of a radius)",
            margin / radius * 100.0
        ),
        None => println!("no disagreement found"),
    }
}
