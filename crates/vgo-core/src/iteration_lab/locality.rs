//! Differential probes for the exact-arithmetic locality argument.
//! These do not introduce a production cutoff or prove floating-point parity.
use crate::{Color, Point, Position, Stone, legal_set, voronoi};

fn cell_alive(
    position: &Position,
    owner: usize,
    geometry: &voronoi::Geometry,
    vertices: &[Point],
) -> bool {
    let s = position.stones()[owner];
    geometry.cells[owner].polygon.iter().any(|&v| {
        legal_set::escape_witness(position, v, Point::new(s.x, s.y), Some(vertices)).is_some()
    })
}

#[test]
fn a_cell_can_lose_its_liberty_without_any_capture() {
    let r = 1.0 / 38.0;
    let side = 7;
    let missing = 3 * side + 3;
    let all: Vec<_> = (0..side * side)
        .map(|i| {
            Stone::new(
                0.25 + (i % side) as f64 * 2.1 * r,
                0.25 + (i / side) as f64 * 2.1 * r,
                Color::Black,
            )
        })
        .collect();
    let owner = all[missing + 1];
    let before = Position::new(
        r,
        all.iter()
            .enumerate()
            .filter_map(|(i, &s)| (i != missing).then_some(s))
            .collect(),
        Color::Black,
    );
    let index = before.stones().iter().position(|&s| s == owner).unwrap();
    assert!(cell_alive(
        &before,
        index,
        &voronoi::compute(&before),
        &legal_set::vertices(&before)
    ));
    let result = crate::place(&before, all[missing].x, all[missing].y).unwrap();
    assert_eq!(result.captured, 0);
    assert_eq!(result.position.stones().len(), all.len());
    assert!(result.events.is_empty());
    let index = result
        .position
        .stones()
        .iter()
        .position(|&s| s == owner)
        .unwrap();
    assert!(!cell_alive(
        &result.position,
        index,
        &result.analysis.geometry,
        &result.analysis.legal_vertices
    ));
    assert!(
        result
            .analysis
            .alive_groups
            .contains(&result.analysis.geometry.groups[index])
    );
}

#[test]
fn a_stone_at_7_4918_r_can_change_individual_cell_status() {
    // Exact decimal construction and independent rational certificate:
    // docs/research/LOCAL_CONTESTABILITY_FLOOR.md, diagnostics/check_locality_floor.py.
    let r = 1.0 / 12.0;
    let coordinates = [
        (1.0, 1.0),
        (1.0, 3.0),
        (3.2371, 2.8783),
        (5.2070, 2.5322),
        (8.4918, 1.0),
    ];
    let stones: Vec<_> = coordinates
        .iter()
        .enumerate()
        .map(|(index, &(x, y))| {
            Stone::new(
                x * r,
                y * r,
                if index == 0 { Color::Black } else { Color::White },
            )
        })
        .collect();
    let full = Position::new(r, stones.clone(), Color::White);
    let reduced = Position::new(r, stones[..4].to_vec(), Color::White);
    for (position, expected) in [(&full, false), (&reduced, true)] {
        assert!(position.validate().is_playable());
        assert_eq!(
            cell_alive(
                position,
                0,
                &voronoi::compute(position),
                &legal_set::vertices(position),
            ),
            expected,
        );
    }
    let witness = Point::new(6.4928 * r, r);
    assert!(!legal_set::contains(&full, witness.x, witness.y));
    assert!(legal_set::contains(&reduced, witness.x, witness.y));
    assert!(crate::Analysis::new(&reduced).settled_groups.is_empty());
    let result = crate::place(&reduced, stones[4].x, stones[4].y).unwrap();
    assert_eq!(result.captured, 1);
    assert!(!result.position.stones().contains(&stones[0]));
    assert_eq!(result.position.stones().len(), 4);
}

#[test]
fn removing_distant_stones_preserves_individual_cell_status() {
    // Compare the original, adaptive, and sharper D = (2+4sqrt(2))r bounds.
    // Include a small outward slack in this probe;
    // a production numerical contract still requires separate analysis.
    let mut checks = 0;
    let mut live = 0;
    let mut dead = 0;
    let mut reduced = 0;
    let mut adaptive_classes = [0usize; 3];
    for (r, side) in [(1.0 / 18.0, 8), (1.0 / 38.0, 18), (1.0 / 80.0, 24)] {
        for pattern in 0..3 {
            let spacing = (1.0 - 2.0 * r) / (side - 1) as f64;
            let stones: Vec<_> = (0..side * side)
                .filter(|&i| {
                    pattern == 0
                        || (i * 37 + i / side * 11) % (if pattern == 1 { 7 } else { 3 }) != 0
                })
                .map(|i| {
                    Stone::new(
                        r + (i % side) as f64 * spacing,
                        r + (i / side) as f64 * spacing,
                        if i % 2 == 0 {
                            Color::Black
                        } else {
                            Color::White
                        },
                    )
                })
                .collect();
            let position = Position::new(r, stones, Color::Black);
            assert!(position.validate().is_playable());
            let geometry = voronoi::compute(&position);
            let vertices = legal_set::vertices(&position);
            for sample in 0..24 {
                let owner = sample * (position.stones().len() - 1) / 23;
                let stone = position.stones()[owner];
                let expected = cell_alive(&position, owner, &geometry, &vertices);
                let edge_distances = [stone.x, 1.0 - stone.x, stone.y, 1.0 - stone.y];
                let class = if edge_distances.iter().all(|&d| d >= 3.0 * r + 1e-10) {
                    0
                } else if edge_distances
                    .iter()
                    .filter(|&&d| d < 4.0 * r + 1e-10)
                    .count()
                    <= 1
                {
                    1
                } else {
                    2
                };
                adaptive_classes[class] += 1;
                let general = 6.0 + 2.0 * 2.0_f64.sqrt();
                let sharp = 2.0 + 4.0 * 2.0_f64.sqrt();
                for multiplier in [general, [6.0, 8.0, general][class], sharp] {
                    let reach = multiplier * r + 1e-10;
                    let local_stones: Vec<_> = position
                        .stones()
                        .iter()
                        .copied()
                        .filter(|s| (s.x - stone.x).hypot(s.y - stone.y) <= reach)
                        .collect();
                    let local_owner = local_stones.iter().position(|&s| s == stone).unwrap();
                    let local = Position::new(r, local_stones, Color::Black);
                    let actual = cell_alive(
                        &local,
                        local_owner,
                        &voronoi::compute(&local),
                        &legal_set::vertices(&local),
                    );
                    assert_eq!(expected, actual, "r={r} pattern={pattern} owner={owner}");
                    checks += 1;
                    live += usize::from(expected);
                    dead += usize::from(!expected);
                    reduced += usize::from(local.stones().len() < position.stones().len());
                }
            }
        }
    }
    assert!(live > 0 && dead > 0 && reduced > 0);
    assert!(adaptive_classes.iter().all(|&n| n > 0));
    eprintln!(
        "locality checks={checks}, live={live}, dead={dead}, reduced-neighborhood checks={reduced}"
    );
    eprintln!("adaptive classes (interior, edge, general)={adaptive_classes:?}");
}
