//! The settled mask as a comparison of two distance fields.
//!
//! `vgo-core`'s `settled.rs` solves a per-stone radial equation, extracts a
//! contour and scanline-fills it. That is `O(n²)` in the stone count and 92-96%
//! of rasterization cost. This is a different formulation of the same set:
//!
//! ```text
//!     settled(x)  <=>  D_S(x) <= D_L(x)
//! ```
//!
//! where `D_S` is the distance to the nearest stone and `D_L` the distance to
//! the legal set. Both are distance transforms, so the whole thing is two
//! sweeps over the grid instead of a geometric solve per stone — `O(pixels)`
//! rather than `O(pixels · n²)`, which means it stays flat as stones accumulate
//! instead of tripling between 28 and 52.
//!
//! `D_L` uses the exact separable Euclidean transform (Felzenszwalb and
//! Huttenlocher, *Distance Transforms of Sampled Functions*): a 1-D lower
//! envelope of parabolas per column, then per row, each linear in the row
//! length. `D_S` is taken from the continuous stone coordinates rather than the
//! grid, because it is cheap and exact there.
//!
//! **This is an approximation, unlike the other two implementations.** `D_L` is
//! the distance to the nearest pixel *centre* lying in the legal set, not to the
//! continuous set, so it is an overestimate by up to half a pixel diagonal. An
//! overestimated `D_L` makes the comparison easier to satisfy, so this errs
//! toward reporting too much settled. `resolution` oversamples the mask to
//! shrink that; `examples/settled_edt.rs` measures what it costs.

use vgo_core::{COORDINATE_EPSILON, Point, Position, distance_to_legal_set, legal_set_vertices};

use crate::RasterConfig;

/// Stands in for "no source here".
///
/// A literal infinity breaks the parabola intersection below: two infinite
/// parabolas give inf - inf, and the resulting NaN or -inf walks `k` off the
/// bottom of the hull. A finite sentinel keeps every intersection finite and
/// well-ordered, and is far beyond any real squared distance on a unit board
/// (the largest is 2, in pixel units at most `2 * size²`).
const ABSENT: f64 = 1.0e20;

/// Squared exact Euclidean distance transform of `f`, in place into `d`.
///
/// `f` holds 0 where the set is present and [`ABSENT`] elsewhere. Scratch
/// buffers are passed in so the 2-D driver can reuse them across every row and
/// column.
fn transform_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = intersection(f, q, v[k]);
        // `k > 0` is a guard, not an optimisation: z[0] is -inf and a finite
        // sentinel keeps s finite, so s <= z[0] is false in the well-behaved
        // case -- but a degenerate row should not be able to underflow.
        while k > 0 && s <= z[k] {
            k -= 1;
            s = intersection(f, q, v[k]);
        }
        if k == 0 && s <= z[0] {
            v[0] = q;
            z[1] = f64::INFINITY;
            continue;
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let offset = q as f64 - v[k] as f64;
        d[q] = offset * offset + f[v[k]];
    }
}

fn intersection(f: &[f64], q: usize, vk: usize) -> f64 {
    let (fq, fv) = (f[q], f[vk]);
    let (qf, vf) = (q as f64, vk as f64);
    ((fq + qf * qf) - (fv + vf * vf)) / (2.0 * qf - 2.0 * vf)
}

/// Squared distance, in pixel units, from each cell to the nearest `true` cell.
fn squared_distance_transform(mask: &[bool], width: usize, height: usize) -> Vec<f64> {
    let mut field: Vec<f64> = mask
        .iter()
        .map(|inside| if *inside { 0.0 } else { ABSENT })
        .collect();

    let longest = width.max(height);
    let mut source = vec![0.0_f64; longest];
    let mut result = vec![0.0_f64; longest];
    let mut vertices = vec![0usize; longest];
    let mut boundaries = vec![0.0_f64; longest + 1];

    for column in 0..width {
        for row in 0..height {
            source[row] = field[row * width + column];
        }
        transform_1d(&source[..height], &mut result[..height], &mut vertices, &mut boundaries);
        for row in 0..height {
            field[row * width + column] = result[row];
        }
    }
    for row in 0..height {
        let base = row * width;
        source[..width].copy_from_slice(&field[base..base + width]);
        transform_1d(&source[..width], &mut result[..width], &mut vertices, &mut boundaries);
        field[base..base + width].copy_from_slice(&result[..width]);
    }
    field
}


/// The legal set sampled onto a grid, built by stamping exclusion discs.
fn sampled_legal_set(position: &Position, fine_width: usize, fine_height: usize) -> Vec<bool> {
    let stones = position.stones();
    //
    // Testing every pixel against every stone is O(pixels · n) and dominates at
    // any useful oversample: 512² × 28 is 7.3M predicate calls. But a stone
    // only forbids a disc of radius 2r around itself, so scattering that disc
    // touches O(n · r²·pixels) cells instead -- about 370k for the same case,
    // and independent of how many stones are far away. Start from the inset
    // rectangle and clear each stone's exclusion disc.
    let radius = position.radius();
    let exclusion = 2.0 * radius - COORDINATE_EPSILON;
    let exclusion_squared = exclusion * exclusion;
    let mut legal = vec![false; fine_width * fine_height];
    let inset_low = radius - COORDINATE_EPSILON;
    let inset_high = 1.0 - radius + COORDINATE_EPSILON;
    for row in 0..fine_height {
        let y = (row as f64 + 0.5) / fine_height as f64;
        if y < inset_low || y > inset_high {
            continue;
        }
        for column in 0..fine_width {
            let x = (column as f64 + 0.5) / fine_width as f64;
            legal[row * fine_width + column] = x >= inset_low && x <= inset_high;
        }
    }
    for stone in stones {
        // Bounding box of the exclusion disc, clipped to the grid.
        let low_row = (((stone.y - exclusion) * fine_height as f64 - 0.5).floor()).max(0.0) as usize;
        let high_row = ((((stone.y + exclusion) * fine_height as f64 - 0.5).ceil()) as usize)
            .min(fine_height - 1);
        let low_column =
            (((stone.x - exclusion) * fine_width as f64 - 0.5).floor()).max(0.0) as usize;
        let high_column = ((((stone.x + exclusion) * fine_width as f64 - 0.5).ceil()) as usize)
            .min(fine_width - 1);
        for row in low_row..=high_row {
            let y = (row as f64 + 0.5) / fine_height as f64;
            let dy = y - stone.y;
            let dy_squared = dy * dy;
            if dy_squared > exclusion_squared {
                continue;
            }
            let base = row * fine_width;
            for column in low_column..=high_column {
                let x = (column as f64 + 0.5) / fine_width as f64;
                let dx = x - stone.x;
                if dx.mul_add(dx, dy_squared) < exclusion_squared {
                    legal[base + column] = false;
                }
            }
        }
    }
    legal
}

/// The settled mask, via distance transforms.
///
/// `oversample` multiplies the grid the legal-set distance is measured on; 1
/// samples it at the output resolution, 4 at four times that in each axis. The
/// output is always `config.pixels()` long.
#[must_use]
pub fn settled_mask_by_distance(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
) -> Vec<bool> {
    let pixels = config.pixels();
    let stones = position.stones();
    if stones.is_empty() {
        // No stone can own anything, and the comparison would be inf <= inf.
        return vec![false; pixels];
    }
    let scale = oversample.max(1);
    let (fine_width, fine_height) = (config.width * scale, config.height * scale);

    let legal = sampled_legal_set(position, fine_width, fine_height);
    let squared = squared_distance_transform(&legal, fine_width, fine_height);

    let mut mask = vec![false; pixels];
    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        // Centre of the output cell in fine-grid coordinates.
        let fine_row = (row * scale + scale / 2).min(fine_height - 1);
        for column in 0..config.width {
            let x = (column as f64 + 0.5) / config.width as f64;
            let fine_column = (column * scale + scale / 2).min(fine_width - 1);

            let mut nearest = f64::INFINITY;
            for stone in stones {
                let (dx, dy) = (x - stone.x, y - stone.y);
                let distance = dx.mul_add(dx, dy * dy);
                if distance < nearest {
                    nearest = distance;
                }
            }
            let legal_squared = squared[fine_row * fine_width + fine_column];
            // Both sides are squared distances; the pixel-unit one is scaled
            // into normalised coordinates by the fine grid's spacing.
            let spacing = 1.0 / fine_width as f64;
            let legal_normalised = legal_squared * spacing * spacing;
            mask[row * config.width + column] = nearest <= legal_normalised;
        }
    }
    mask
}

/// The settled mask, using distance-transform bounds to skip most of the work.
///
/// The sampled `D_L` is not merely approximate, it is approximate in a *known
/// direction*: it measures the distance to the nearest legal pixel centre, and
/// every such centre is a point of the legal set, so
///
/// ```text
///     D_L_true  <=  D_L_grid  <=  D_L_true + e
/// ```
///
/// where `e` is half a grid diagonal. That two-sided bound decides almost every
/// pixel outright:
///
///   * `D_S <= D_L_grid - e`  implies `D_S <= D_L_true`  — settled
///   * `D_S >  D_L_grid`      implies `D_S >  D_L_true`  — not settled
///
/// Only the band between them is genuinely undecided, and those pixels get the
/// exact continuous test — 55 of 16384 at oversample 1, so the cost is
/// dominated by the cheap path, which is the point: the exact test is what made
/// the direct formulation 42x too slow to use everywhere.
///
/// **The upper bound is not guaranteed.** It assumes every point of the legal
/// set has a sampled cell centre within half a diagonal, which fails where the
/// legal set is a sliver thinner than the grid — then `D_L_grid` overshoots by
/// more than the slack and a pixel can be called settled when it is not. The
/// lower direction (`D_S > D_L_grid` implies not settled) is always sound,
/// since every legal cell centre really is a point of the set.
///
/// In practice that costs at most one pixel of 16384 at the densest fixture
/// tested, against two for the shipping implementation, which walks a contour
/// at 1/128 tolerance and is not exact either. Oversampling to 3 removed it
/// entirely on every fixture.
///
/// Returns the mask and how many pixels needed the exact test, so callers can
/// see whether the band is staying small.
#[must_use]
pub fn settled_mask_by_bounded_distance(
    position: &Position,
    config: RasterConfig,
    oversample: usize,
) -> (Vec<bool>, usize) {
    let pixels = config.pixels();
    let stones = position.stones();
    if stones.is_empty() {
        return (vec![false; pixels], 0);
    }
    let scale = oversample.max(1) | 1;
    let (fine_width, fine_height) = (config.width * scale, config.height * scale);
    let legal = sampled_legal_set(position, fine_width, fine_height);
    let squared = squared_distance_transform(&legal, fine_width, fine_height);
    let spacing = 1.0 / fine_width as f64;
    // How far the sampled distance can overstate the true one.
    //
    // Half a cell diagonal is the tempting answer and it is wrong: it assumes
    // every point of the legal set has a *cell centre* within that distance,
    // which fails all along the set's boundary, where the cell containing a
    // legal point can easily have its centre outside. Measured, that cost two
    // wrong pixels at eight stones, where slivers cannot be the explanation.
    // A full diagonal covers the boundary case; nothing covers a sliver
    // narrower than a cell, which is why this function is not exact.
    let slack = spacing * std::f64::consts::SQRT_2;

    let mut vertices: Option<Vec<Point>> = None;
    let mut mask = vec![false; pixels];
    let mut exact_tests = 0usize;
    for row in 0..config.height {
        let y = (row as f64 + 0.5) / config.height as f64;
        let fine_row = (row * scale + scale / 2).min(fine_height - 1);
        for column in 0..config.width {
            let x = (column as f64 + 0.5) / config.width as f64;
            let fine_column = (column * scale + scale / 2).min(fine_width - 1);

            let mut nearest_squared = f64::INFINITY;
            for stone in stones {
                let (dx, dy) = (x - stone.x, y - stone.y);
                let distance = dx.mul_add(dx, dy * dy);
                if distance < nearest_squared {
                    nearest_squared = distance;
                }
            }
            let nearest = nearest_squared.sqrt();
            let sampled = squared[fine_row * fine_width + fine_column].sqrt() * spacing;

            mask[row * config.width + column] = if nearest <= sampled - slack {
                true
            } else if nearest > sampled {
                false
            } else {
                exact_tests += 1;
                let known = vertices.get_or_insert_with(|| legal_set_vertices(position));
                nearest <= distance_to_legal_set(position, Point::new(x, y), Some(known))
            };
        }
    }
    (mask, exact_tests)
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Stone, distance_to_legal_set, legal_set_vertices};

    use super::*;
    use crate::{RasterKind, settled_mask};

    fn fixture(count: usize, radius: f64) -> Position {
        let spacing = 2.0 * radius * 1.08;
        let per_row = ((0.86_f64 / spacing).floor() as usize).max(1);
        let mut stones = Vec::new();
        for index in 0..count {
            let (row, column) = (index / per_row, index % per_row);
            let x = 0.07 + (column as f64 + 0.5) * spacing;
            let y = 0.07 + (row as f64 + 0.5) * spacing;
            if x > 0.96 || y > 0.96 {
                break;
            }
            let colour = if index % 2 == 0 { Color::Black } else { Color::White };
            stones.push(Stone::new(x, y, colour));
        }
        Position::new(radius, stones, Color::Black).with_komi(0.104)
    }

    /// The bounded form must agree with the definition, not merely with the
    /// other implementation -- which walks a contour at 1/128 and is itself
    /// wrong on a pixel or two at high stone counts.
    #[test]
    fn bounded_distance_agrees_with_the_definition() {
        let radius = 0.055_714_285_714_285_716;
        let config = RasterConfig::square_of(128, RasterKind::Compact);
        for count in [8usize, 28, 52] {
            let position = fixture(count, radius);
            if !position.validate().is_playable() || position.stones().is_empty() {
                continue;
            }
            let vertices = legal_set_vertices(&position);
            // Oversample 1 is what the raster uses: it is the fastest setting
            // and, with the slack corrected to a full cell diagonal, was exact on
            // every fixture here. The function is still not exact in general --
            // see its doc comment on slivers.
            let (mask, exact_tests) = settled_mask_by_bounded_distance(&position, config, 1);

            let mut wrong = 0usize;
            for pixel in 0..config.pixels() {
                let x = ((pixel % config.width) as f64 + 0.5) / config.width as f64;
                let y = ((pixel / config.width) as f64 + 0.5) / config.height as f64;
                let nearest = position
                    .stones()
                    .iter()
                    .map(|s| ((s.x - x).powi(2) + (s.y - y).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let truth =
                    nearest <= distance_to_legal_set(&position, Point::new(x, y), Some(&vertices));
                if truth != mask[pixel] {
                    wrong += 1;
                }
            }
            assert_eq!(wrong, 0, "{count} stones: {wrong} pixels disagree with the definition");
            // The bound has to be doing the work; if the fallback ran
            // everywhere this would pass while being slower than the exact path.
            assert!(
                exact_tests * 20 < config.pixels(),
                "{count} stones: {exact_tests} exact tests is too many to be a fallback"
            );
        }
    }

    #[test]
    fn an_empty_board_settles_nothing() {
        let config = RasterConfig::square_of(32, RasterKind::Compact);
        let position = Position::new(0.05, Vec::new(), Color::Black);
        let (mask, _) = settled_mask_by_bounded_distance(&position, config, 1);
        assert!(mask.iter().all(|settled| !settled));
        assert_eq!(settled_mask(&position, config), mask);
    }
}
