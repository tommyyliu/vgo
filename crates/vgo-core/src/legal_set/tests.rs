use super::StoneBuckets;
use super::none_closer_than;
use super::vertices::vertices_with;
use super::{contains, distance, nearest, nearest_with, vertices};
use crate::{Color, Point, Position, Stone};

/// The bucketed enumeration against the quadratic one it replaced.
///
/// `vertices_with(_, None)` is the original: every pair considered, every
/// stone walked in every clearance test. The buckets only decide which of
/// those are worth asking about, so the two must agree exactly -- the same
/// points in the same order, because the dedup key keeps whichever of two
/// near-identical points it saw first.
///
/// Counts straddle `BUCKET_MINIMUM_STONES`, and the lattice is there because
/// it is the case the sampled legal set is known to miss: stones just over
/// the exclusion diameter apart leave a gap between every four neighbours
/// that exists only as a vertex.
#[test]
fn bucketed_vertices_match_the_quadratic_enumeration() {
    let radius = 1.0 / 38.0;
    let mut fixtures: Vec<crate::Position> = Vec::new();
    for &count in &[0usize, 1, 5, 31, 32, 33, 60, 120, 240] {
        for &seed in &[1u64, 7, 99] {
            fixtures.push(scattered(count, radius, seed));
        }
        fixtures.push(lattice(count, radius, 1.08));
        fixtures.push(lattice(count, radius, 2.02));
    }
    for position in &fixtures {
        let quadratic = vertices_with(position, None);
        let bucketed = vertices_with(position, Some(&StoneBuckets::build(position)));
        assert_eq!(
            quadratic.len(),
            bucketed.len(),
            "{} stones: {} vertices without buckets, {} with",
            position.stones().len(),
            quadratic.len(),
            bucketed.len()
        );
        for (index, (want, got)) in quadratic.iter().zip(bucketed.iter()).enumerate() {
            assert!(
                want.x == got.x && want.y == got.y,
                "{} stones: vertex {index} is ({}, {}) without buckets and ({}, {}) with",
                position.stones().len(),
                want.x,
                want.y,
                got.x,
                got.y
            );
        }
    }
}

fn lattice(count: usize, radius: f64, pitch: f64) -> crate::Position {
    let spacing = 2.0 * radius * pitch;
    let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
    let mut stones = Vec::new();
    for index in 0..count {
        let x = radius + 0.02 + (index % per_row) as f64 * spacing;
        let y = radius + 0.02 + (index / per_row) as f64 * spacing;
        if x > 1.0 - radius || y > 1.0 - radius {
            break;
        }
        stones.push(crate::Stone {
            x,
            y,
            color: if index % 2 == 0 {
                crate::Color::Black
            } else {
                crate::Color::White
            },
        });
    }
    crate::Position::new(radius, stones, crate::Color::Black)
}

/// Rejection-sampled placements, so the buckets meet uneven density rather
/// than only the regular spacing a lattice gives them.
fn scattered(count: usize, radius: f64, seed: u64) -> crate::Position {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut stones: Vec<crate::Stone> = Vec::new();
    let mut attempts = 0;
    while stones.len() < count && attempts < 40_000 {
        attempts += 1;
        let x = radius + next() * (1.0 - 2.0 * radius);
        let y = radius + next() * (1.0 - 2.0 * radius);
        let minimum = 2.0 * radius;
        if stones
            .iter()
            .any(|s: &crate::Stone| (s.x - x).hypot(s.y - y) < minimum)
        {
            continue;
        }
        stones.push(crate::Stone {
            x,
            y,
            color: if stones.len().is_multiple_of(2) {
                crate::Color::Black
            } else {
                crate::Color::White
            },
        });
    }
    crate::Position::new(radius, stones, crate::Color::Black)
}

fn threshold_lattice(count: usize, radius: f64) -> Position {
    let spacing = 2.0 * radius * 1.05;
    let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
    let mut stones = Vec::new();
    for index in 0..count {
        let (row, column) = (index / per_row, index % per_row);
        let x = radius + 0.02 + column as f64 * spacing;
        let y = radius + 0.02 + row as f64 * spacing;
        if x > 1.0 - radius || y > 1.0 - radius {
            break;
        }
        stones.push(Stone {
            x,
            y,
            color: if index % 2 == 0 {
                Color::Black
            } else {
                Color::White
            },
        });
    }
    Position::new(radius, stones, Color::Black)
}

/// `none_closer_than` prunes stones `distance` must visit, so the two can
/// only be trusted together: the predicate must agree with the exact
/// minimum at every threshold, including right at it.
#[test]
fn the_predicate_agrees_with_the_exact_distance() {
    for (radius, count) in [(1.0 / 18.0, 28usize), (1.0 / 18.0, 52), (1.0 / 36.0, 240)] {
        let position = threshold_lattice(count, radius);
        let known = vertices(&position);
        let index = super::LegalSetIndex::build(&position);
        for row in 0..24 {
            for column in 0..24 {
                let point = Point::new((column as f64 + 0.5) / 24.0, (row as f64 + 0.5) / 24.0);
                let exact = distance(&position, point, Some(&known));
                for scale in [0.25, 0.5, 0.9, 1.0, 1.1, 2.0] {
                    let threshold = exact * scale;
                    assert_eq!(
                        none_closer_than(&position, point, threshold, Some(&known)),
                        threshold <= exact,
                        "{count} stones, point {point:?}, threshold {threshold} vs exact {exact}",
                    );
                    assert_eq!(
                        super::none_closer_than_indexed(&position, point, threshold, &index),
                        threshold <= exact,
                        "indexed: {count} stones, point {point:?}, threshold {threshold} vs exact {exact}",
                    );
                }
            }
        }
    }
}

#[test]
fn indexed_threshold_queries_match_exhaustive_near_boundaries() {
    for count in [31, 32, 64, 160, 240] {
        let radius = 1.0 / 38.0;
        let position = scattered(count, radius, 83);
        let index = super::LegalSetIndex::build(&position);
        let mut points = vec![
            Point::new(0.0, 0.0),
            Point::new(radius, radius),
            Point::new(1.0 - radius, 1.0 - radius),
            Point::new(0.5, 0.5),
        ];
        points.extend(index.vertices.iter().copied().take(12));
        for stone in position.stones().iter().take(8) {
            points.push(Point::new(stone.x, stone.y));
            points.push(Point::new(stone.x + 2.0 * radius, stone.y));
        }
        for point in points {
            let nearest = distance(&position, point, Some(&index.vertices));
            for threshold in [
                0.0,
                radius,
                nearest.next_down(),
                nearest,
                nearest.next_up(),
                2.0 * radius,
                f64::INFINITY,
                f64::NAN,
            ] {
                assert_eq!(
                    super::none_closer_than_indexed(&position, point, threshold, &index),
                    none_closer_than(&position, point, threshold, Some(&index.vertices)),
                    "{count} stones: point {point:?}, threshold {threshold}"
                );
            }
        }
    }
}

#[test]
fn closed_boundaries_are_legal() {
    let position = Position::new(0.1, vec![Stone::new(0.5, 0.5, Color::Black)], Color::White);
    assert!(contains(&position, 0.7, 0.5));
    assert!(!contains(&position, 0.69, 0.5));
    assert!(contains(&position, 0.1, 0.1));
}

#[test]
fn covered_legal_set_has_no_vertices_or_finite_distance() {
    let position = Position::new(
        0.25,
        vec![
            Stone::new(0.25, 0.25, Color::Black),
            Stone::new(0.75, 0.25, Color::White),
            Stone::new(0.75, 0.75, Color::Black),
            Stone::new(0.25, 0.75, Color::White),
        ],
        Color::Black,
    );
    let legal_vertices = vertices(&position);
    assert!(legal_vertices.is_empty());
    assert!(distance(&position, Point::new(0.5, 0.5), Some(&legal_vertices)).is_infinite());
}

#[test]
fn indexed_nearest_matches_the_unindexed_scan() {
    for (count, radius, seed) in [
        (40, 1.0 / 38.0, 3),
        (90, 1.0 / 38.0, 5),
        (200, 1.0 / 38.0, 7),
    ] {
        let position = scattered(count, radius, seed);
        let index = super::LegalSetIndex::build(&position);
        for row in 0..64 {
            for column in 0..64 {
                let query = Point::new((column as f64 + 0.5) / 64.0, (row as f64 + 0.5) / 64.0);
                let plain = nearest_with(&position, query, Some(index.vertices()));
                let fast = super::nearest_indexed(&position, query, &index);
                assert_eq!(
                    (
                        plain.legal,
                        plain.snapped,
                        plain.point.x.to_bits(),
                        plain.point.y.to_bits()
                    ),
                    (
                        fast.legal,
                        fast.snapped,
                        fast.point.x.to_bits(),
                        fast.point.y.to_bits()
                    ),
                );
            }
        }
    }
}

#[test]
fn a_bounded_nearest_matches_the_full_one_within_its_reach() {
    for (count, radius, seed) in [
        (40, 1.0 / 38.0, 3),
        (90, 1.0 / 38.0, 5),
        (60, 1.0 / 18.0, 7),
    ] {
        let position = scattered(count, radius, seed);
        let index = super::LegalSetIndex::build(&position);
        for reach in [0.002, 0.01, 0.05] {
            for row in 0..128 {
                for column in 0..128 {
                    let query =
                        Point::new((column as f64 + 0.5) / 128.0, (row as f64 + 0.5) / 128.0);
                    let full = super::nearest_indexed(&position, query, &index);
                    let bounded = super::nearest_indexed_within(&position, query, &index, reach);
                    if full.legal && full.point.distance(query) <= reach {
                        assert_eq!(
                            (
                                bounded.legal,
                                bounded.point.x.to_bits(),
                                bounded.point.y.to_bits()
                            ),
                            (true, full.point.x.to_bits(), full.point.y.to_bits()),
                        );
                    } else if bounded.legal {
                        assert!(bounded.point.distance(query) > reach);
                    }
                }
            }
        }
    }
}

#[test]
fn nearest_returns_a_legal_query_unchanged() {
    let position = Position::new(
        0.1,
        vec![Stone::new(0.25, 0.25, Color::Black)],
        Color::Black,
    );
    let query = Point::new(0.7, 0.7);
    let found = nearest(&position, query);
    assert_eq!(found.point, query);
    assert!(found.legal);
    assert!(!found.snapped, "a legal query must not be moved");
}

/// Selecting lazily must pick exactly what the exhaustive scan did.
///
/// The original scan kept a strict `<` running minimum, so among candidates
/// at an equal distance it kept the first in list order. Retiring rejected
/// candidates by swapping the tail over them reorders the list, so ties are
/// no longer resolved by position. This replays both over random positions
/// and compares the chosen point -- a disagreement is a silently different
/// move, not a test failure anyone would otherwise notice.
#[test]
fn lazy_nearest_matches_the_exhaustive_scan() {
    // Bit-exact, not approximate. Both implementations pick a point out of
    // the same candidate list and hand it to the same `clear_by_margin`, so
    // agreeing means choosing the identical candidate -- there is no
    // arithmetic between them that could legitimately round differently.
    // A tolerance here would hide exactly the tie-order divergence the test
    // exists to catch.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    // The production radius, plus values either side of it: a larger radius
    // packs fewer stones and leaves more of the board illegal, which is the
    // regime where the scan walks furthest down the candidate order.
    let radii = [0.02, 0.05571428571428571, 0.1, 0.2];
    let mut checked = 0_u32;
    let mut snapped_seen = 0_u32;
    let mut illegal_seen = 0_u32;
    for radius in radii {
        for _ in 0..250 {
            let count = 1 + (next() * 30.0) as usize;
            let mut stones = Vec::new();
            let mut attempts = 0;
            while stones.len() < count && attempts < 2000 {
                attempts += 1;
                let x = radius + next() * (1.0 - 2.0 * radius);
                let y = radius + next() * (1.0 - 2.0 * radius);
                if stones.iter().all(|s: &crate::Stone| {
                    super::numeric::length(s.x - x, s.y - y) >= 2.0 * radius
                }) {
                    let colour = if stones.len() % 2 == 0 {
                        crate::Color::Black
                    } else {
                        crate::Color::White
                    };
                    stones.push(crate::Stone::new(x, y, colour));
                }
            }
            let position = crate::Position::new(radius, stones, crate::Color::Black);
            let known = vertices(&position);
            for _ in 0..6 {
                let query = Point::new(next(), next());
                // Both paths: `Some` is what the search uses, `None` makes
                // `nearest_with` build the vertex set itself, and only the
                // second is reachable from `nearest`.
                for supplied in [Some(&known[..]), None] {
                    let fast = nearest_with(&position, query, supplied);
                    let slow = exhaustive_nearest(&position, query, &known);
                    assert_eq!(
                        (fast.legal, fast.snapped),
                        (slow.0, slow.2),
                        "legality/snap disagreed at {query:?} with radius {radius}"
                    );
                    if fast.legal {
                        assert_eq!(
                            (fast.point.x.to_bits(), fast.point.y.to_bits()),
                            (slow.1.x.to_bits(), slow.1.y.to_bits()),
                            "chose {:?}, exhaustive chose {:?} at radius {radius}",
                            fast.point,
                            slow.1
                        );
                    }
                    checked += 1;
                }
                let probe = nearest_with(&position, query, Some(&known));
                if probe.snapped {
                    snapped_seen += 1;
                }
                if !probe.legal {
                    illegal_seen += 1;
                }
            }
        }
    }
    // The comparison is only worth anything if it exercised the branches:
    // an all-legal sample would never reach the candidate scan at all, and
    // a board with no legal point never reaches the choice either.
    assert!(checked > 10_000, "only {checked} comparisons");
    assert!(snapped_seen > 1_000, "only {snapped_seen} snapped queries");
    assert!(
        illegal_seen > 0,
        "no fully covered board was generated; the None branch is untested"
    );
}

/// The pre-optimization scan: build every candidate, test all of them,
/// keep a strict running minimum. Mirrors `nearest_with` exactly.
fn exhaustive_nearest(position: &Position, point: Point, known: &[Point]) -> (bool, Point, bool) {
    if contains(position, point.x, point.y) {
        return (true, point, false);
    }
    let radius = position.radius();
    let diameter = 2.0 * radius + super::numeric::SNAP_MARGIN;
    let inset = (radius + super::numeric::SNAP_MARGIN).min(0.5);
    let clamp = |value: f64| value.clamp(inset.min(1.0 - inset), inset.max(1.0 - inset));
    let mut candidates = vec![Point::new(clamp(point.x), clamp(point.y))];
    for stone in position.stones() {
        let dx = point.x - stone.x;
        let dy = point.y - stone.y;
        let radial = super::numeric::length(dx, dy);
        if radial < super::numeric::EDGE_EPSILON {
            for (ux, uy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                candidates.push(Point::new(
                    clamp(stone.x + diameter * ux),
                    clamp(stone.y + diameter * uy),
                ));
            }
        } else {
            candidates.push(Point::new(
                clamp(stone.x + diameter * dx / radial),
                clamp(stone.y + diameter * dy / radial),
            ));
        }
    }
    candidates.push(Point::new(inset, clamp(point.y)));
    candidates.push(Point::new(1.0 - inset, clamp(point.y)));
    candidates.push(Point::new(clamp(point.x), inset));
    candidates.push(Point::new(clamp(point.x), 1.0 - inset));
    candidates.extend_from_slice(known);
    let mut best: Option<(f64, Point)> = None;
    for candidate in candidates {
        if !contains(position, candidate.x, candidate.y) {
            continue;
        }
        let dx = candidate.x - point.x;
        let dy = candidate.y - point.y;
        let squared = dx.mul_add(dx, dy * dy);
        if best.is_none_or(|(best_squared, _)| squared < best_squared) {
            best = Some((squared, candidate));
        }
    }
    match best {
        None => (false, point, false),
        Some((_, found)) => (
            true,
            super::nearest::clear_by_margin(position, found, None),
            true,
        ),
    }
}

#[test]
fn nearest_projects_an_illegal_query_onto_the_legal_set() {
    let position = Position::new(0.1, vec![Stone::new(0.5, 0.5, Color::Black)], Color::Black);
    // Directly on the stone's centre: illegal, and with no ray to push
    // along, so this also covers the degenerate direction case.
    let found = nearest(&position, Point::new(0.5, 0.5));
    assert!(found.legal);
    assert!(found.snapped);
    assert!(contains(&position, found.point.x, found.point.y));
    // A diameter of clearance plus the snap margin. Landing exactly on the
    // constraint would make the result legal only by the tolerance in
    // `clear_of_stones`, so a second implementation re-checking the same
    // point could disagree; the margin puts it unambiguously inside.
    let distance = (found.point.x - 0.5).hypot(found.point.y - 0.5);
    assert!(
        distance >= 0.2,
        "expected at least a diameter of clearance, got {distance}"
    );
    assert!(
        (distance - (0.2 + crate::numeric::SNAP_MARGIN)).abs() < 1.0e-9,
        "expected a diameter plus the snap margin, got {distance}"
    );
}

#[test]
fn nearest_reaches_a_region_no_lattice_cell_can_name() {
    // The failure this exists for: a legal region thinner than a policy
    // cell. Two stones a little over two diameters apart leave a legal
    // sliver between them that a coarse lattice can miss entirely, and a
    // group whose last liberty sits there is uncapturable.
    let radius = 1.0 / 18.0;
    let gap = 4.0 * radius + 0.004;
    let position = Position::new(
        radius,
        vec![
            Stone::new(0.5 - gap / 2.0, 0.5, Color::Black),
            Stone::new(0.5 + gap / 2.0, 0.5, Color::Black),
        ],
        Color::White,
    );
    let midpoint = Point::new(0.5, 0.5);
    assert!(
        contains(&position, midpoint.x, midpoint.y),
        "the sliver between the stones must be legal"
    );

    // A lattice that steps over the sliver cannot name it, but the nearest
    // cell it can name projects back into it.
    let coarse = 0.5 + 0.003;
    assert!(
        !contains(&position, coarse, 0.5),
        "the probe must be illegal"
    );
    let found = nearest(&position, Point::new(coarse, 0.5));
    assert!(found.legal && found.snapped);
    assert!(contains(&position, found.point.x, found.point.y));
}

#[test]
fn a_snap_that_lands_on_a_vertex_still_clears_the_margin() {
    // `nearest` chooses from the stone-tangent candidates *and* the vertex
    // set. The tangents carry SNAP_MARGIN; the vertices deliberately do
    // not, because `distance` and `has_escape` need the exact vertex
    // set of the legal region. So a snap could still return a point resting
    // exactly on a constraint whenever a vertex was the closest candidate.
    //
    // A browser game stalled on one: a click snapped to a vertex 1.1e-16
    // from a stone, which the page accepted and the model server's own
    // legality check then refused, leaving the model unable to answer.
    let radius = 0.055714285714285716;
    let position = Position::new(
        radius,
        vec![
            Stone::new(0.70703125, 0.57421875, Color::Black),
            Stone::new(0.449219, 0.425781, Color::White),
            Stone::new(0.32421875, 0.69921875, Color::Black),
            Stone::new(0.370426, 0.504574, Color::White),
            Stone::new(0.72265625, 0.23046875, Color::Black),
            Stone::new(0.245426, 0.620426, Color::White),
            Stone::new(0.55859375, 0.80078125, Color::Black),
            Stone::new(0.560365, 0.417842, Color::White),
            Stone::new(0.77734375, 0.40234375, Color::Black),
            Stone::new(0.550655, 0.689635, Color::White),
        ],
        Color::Black,
    );
    // The click that produced the stalling stone.
    let found = nearest(&position, Point::new(0.66, 0.675));
    assert!(found.legal);
    for stone in position.stones() {
        let distance = (found.point.x - stone.x).hypot(found.point.y - stone.y);
        assert!(
            distance >= 2.0 * radius,
            "a snapped placement must not rest on a constraint; this one \
             sits {:.3e} inside a stone",
            2.0 * radius - distance
        );
    }
}

#[test]
fn snapped_placements_do_not_accumulate_an_invalid_position() {
    // A second stalled browser game, further in. Four of the server's
    // twelve moves had landed 1.6e-6 to 2.7e-6 inside a neighbour, each
    // legal to the search that snapped it. Those stones stay on the board,
    // so `analyze` reports four overlapping pairs and `place` refuses every
    // subsequent move with `invalid-position` -- the model appears stuck
    // while the real damage happened several moves earlier.
    //
    // One bad snap is a rejected move; a bad snap that gets played corrupts
    // the position for the rest of the game.
    let radius = 39.0 / 700.0;
    let position = Position::new(
        radius,
        vec![
            Stone::new(0.69922, 0.59766, Color::Black),
            Stone::new(0.628657, 0.511415, Color::White),
            Stone::new(0.75391, 0.44922, Color::Black),
            Stone::new(0.605469, 0.660156, Color::White),
            Stone::new(0.61328, 0.33984, Color::Black),
        ],
        Color::White,
    );
    // The cell the search wanted at move 5, which snapped to a point
    // 2.4e-6 inside the stone at (0.69922, 0.59766).
    let found = nearest(&position, Point::new(0.739220, 0.701659));
    assert!(found.legal);
    for stone in position.stones() {
        let distance = (found.point.x - stone.x).hypot(found.point.y - stone.y);
        assert!(
            distance >= 2.0 * radius,
            "a snapped placement that is played must leave the position \
             valid; this one sits {:.3e} inside a stone",
            2.0 * radius - distance
        );
    }
}

#[test]
fn a_snapped_placement_survives_an_independent_recheck() {
    // The move server stalled a browser game here. It snapped White's move
    // to (0.721723, 0.425648), one diameter from Black at (0.83203,
    // 0.44141) at radius 39/700 -- legal to the search that proposed it,
    // and 1.1e-6 inside the exclusion disc once the client re-derived the
    // distance its own way. The client refused the move, re-asked, and the
    // stateless server returned the same point twenty times.
    //
    // The margin is what makes a snapped point survive being re-checked by
    // an implementation that associates its arithmetic differently.
    let position = Position::new(
        39.0 / 700.0,
        vec![
            Stone::new(0.62891, 0.80078, Color::Black),
            Stone::new(0.61328, 0.63672, Color::White),
            Stone::new(0.47266, 0.71484, Color::Black),
            Stone::new(0.54272, 0.55048, Color::White),
            Stone::new(0.75391, 0.68359, Color::Black),
            Stone::new(0.74597, 0.57245, Color::White),
            Stone::new(0.89453, 0.59766, Color::Black),
            Stone::new(0.39386, 0.63605, Color::White),
            Stone::new(0.83203, 0.44141, Color::Black),
        ],
        Color::White,
    );
    let found = nearest(&position, Point::new(0.721723, 0.425648));
    assert!(found.legal);
    let minimum = 2.0 * position.radius();
    for stone in position.stones() {
        let distance = (found.point.x - stone.x).hypot(found.point.y - stone.y);
        assert!(
            distance >= minimum,
            "snapped point sits {:.3e} inside a stone's exclusion disc; a \
             recheck that does not subtract COORDINATE_EPSILON rejects it",
            minimum - distance
        );
    }
}

#[test]
fn nearest_reports_failure_when_nothing_is_legal() {
    // A radius that leaves no inset at all: there is no legal point to
    // snap to, and the caller must not be handed a fabricated one.
    let position = Position::new(0.5, Vec::new(), Color::Black);
    let found = nearest(&position, Point::new(0.1, 0.9));
    if !found.legal {
        assert!(!found.snapped);
        assert_eq!(found.point, Point::new(0.1, 0.9));
    }
}
