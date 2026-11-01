//! Exact Euclidean distance transform of a binary mask.

/// The whole field when the mask has no `true` cell, far beyond any real squared
/// distance on a unit board, so every pixel reads as settled by a wide margin.
const ABSENT: f64 = 1.0e20;

/// Buffers retained by a caller rendering many positions at one size.
#[derive(Debug, Default)]
pub(crate) struct EdtScratch {
    field: Vec<f64>,
    gaps: Vec<u32>,
    parabolas: Vec<u32>,
    starts: Vec<u32>,
}

impl EdtScratch {
    /// The squared field from the last transform.
    pub(crate) fn field(&self) -> &[f64] {
        &self.field
    }
}

/// Squared Euclidean distance, in pixels, from each cell to the nearest `true`
/// cell, or [`ABSENT`] everywhere if no cell is `true`.
///
/// Each result is an exact integer, so any correct method gives identical bits.
/// Proof: [E3].
/// This one suits a binary mask (Meijster, Roerdink and Hesselink's transform):
///
///   1. Each cell's vertical distance to a `true` cell comes from one forward
///      and one backward sweep. Both run across whole rows, so they vectorize
///      and never stride through memory a column at a time.
///   2. Each row then takes a lower envelope of parabolas in integers. The
///      overlap test evaluates two parabolas instead of dividing.
///
/// `none` exceeds every real vertical distance by enough that a column with no
/// `true` cell can never win: its parabola is above `width^2 + height^2`
/// everywhere, and the mask is known to hold a `true` cell.
pub(crate) fn squared_distance_transform_into(
    mask: &[bool],
    width: usize,
    height: usize,
    scratch: &mut EdtScratch,
) {
    scratch.field.resize(mask.len(), 0.0);
    if !mask.iter().any(|inside| *inside) {
        scratch.field.fill(ABSENT);
        return;
    }
    let none = (width + height + 1) as u32;
    assert!(
        none < 1 << 14,
        "the envelope works in i32: see `envelope_row`"
    );
    scratch.gaps.resize(mask.len(), 0);
    let gaps = scratch.gaps.as_mut_slice();
    let mut above = vec![none; width];
    for (cells, inside) in gaps.chunks_exact_mut(width).zip(mask.chunks_exact(width)) {
        for ((cell, previous), &inside) in cells.iter_mut().zip(above.iter_mut()).zip(inside) {
            let next = (*previous + 1).min(none);
            *previous = if inside { 0 } else { next };
            *cell = *previous;
        }
    }
    above.fill(none);
    for cells in gaps.chunks_exact_mut(width).rev() {
        for (cell, previous) in cells.iter_mut().zip(above.iter_mut()) {
            let next = (*previous + 1).min(none);
            *previous = next.min(*cell);
            *cell = (*previous) * (*previous);
        }
    }

    scratch.parabolas.resize(width, 0);
    scratch.starts.resize(width, 0);
    for (row, out) in gaps
        .chunks_exact(width)
        .zip(scratch.field.chunks_exact_mut(width))
    {
        envelope_row(row, &mut scratch.parabolas, &mut scratch.starts, out);
    }
}

/// `min over c of (x - c)^2 + row[c]` at every `x`.
///
/// In `i32`, which [E3] shows holds every value here. The narrower divide is the
/// point: 64-bit division costs several times the 32-bit one on many server CPUs.
fn envelope_row(row: &[u32], parabolas: &mut [u32], starts: &mut [u32], out: &mut [f64]) {
    let parabolas = &mut parabolas[..row.len()];
    let starts = &mut starts[..row.len()];
    let height_at = |at: i32, of: u32| {
        let offset = at - of as i32;
        offset * offset + row[of as usize] as i32
    };
    let width = row.len() as i32;
    let mut top = 0usize;
    parabolas[0] = 0;
    starts[0] = 0;
    for column in 1..row.len() as u32 {
        let beaten = loop {
            let start = starts[top] as i32;
            if height_at(start, parabolas[top]) <= height_at(start, column) {
                break false;
            }
            if top == 0 {
                break true;
            }
            top -= 1;
        };
        if beaten {
            parabolas[0] = column;
            continue;
        }
        // First pixel where `column` is strictly below the parabola on top.
        let left = parabolas[top] as i32;
        let right = column as i32;
        let numerator =
            right * right - left * left + row[right as usize] as i32 - row[left as usize] as i32;
        let crossing = numerator.div_euclid(2 * (right - left)) + 1;
        if crossing < width {
            top += 1;
            parabolas[top] = column;
            starts[top] = crossing as u32;
        }
    }
    for at in (0..row.len()).rev() {
        out[at] = f64::from(height_at(at as i32, parabolas[top]));
        if at as u32 == starts[top] && top > 0 {
            top -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_distance_transform(mask: &[bool], width: usize, height: usize) -> Vec<f64> {
        let mut scratch = EdtScratch::default();
        squared_distance_transform_into(mask, width, height, &mut scratch);
        scratch.field
    }

    #[test]
    fn the_transform_matches_brute_force() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (width, height, per_thousand) in [
            (1, 1, 500),
            (1, 9, 300),
            (9, 1, 300),
            (17, 5, 2),
            (5, 17, 2),
            (32, 32, 1),
            (32, 32, 20),
            (32, 32, 300),
            (31, 47, 950),
            (64, 64, 3),
        ] {
            for _ in 0..40 {
                let mask: Vec<bool> = (0..width * height)
                    .map(|_| next() % 1000 < per_thousand)
                    .collect();
                let got = squared_distance_transform(&mask, width, height);
                let sources: Vec<(i64, i64)> = (0..width * height)
                    .filter(|&cell| mask[cell])
                    .map(|cell| ((cell % width) as i64, (cell / width) as i64))
                    .collect();
                for (cell, value) in got.iter().enumerate() {
                    let (x, y) = ((cell % width) as i64, (cell / width) as i64);
                    let expected = sources
                        .iter()
                        .map(|(a, b)| (a - x).pow(2) + (b - y).pow(2))
                        .min()
                        .map_or(ABSENT, |square| square as f64);
                    assert_eq!(*value, expected, "{width}x{height}, cell {cell}");
                }
            }
        }
    }
}
