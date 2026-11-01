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
    start_heights: Vec<i32>,
}

impl EdtScratch {
    /// The squared field from the last transform.
    pub(crate) fn field(&self) -> &[f64] {
        &self.field
    }

    /// The squared vertical distances each row of the last transform started from,
    /// empty if its mask held no `true` cell.
    pub(crate) fn heights(&self) -> &[u32] {
        &self.gaps
    }
}

/// An earlier transform on the same grid, which a row with the same input can copy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EarlierTransform<'a> {
    pub(crate) heights: &'a [u32],
    pub(crate) field: &'a [f64],
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
    earlier: Option<EarlierTransform<'_>>,
) {
    scratch.field.resize(mask.len(), 0.0);
    if !mask.iter().any(|inside| *inside) {
        scratch.field.fill(ABSENT);
        scratch.gaps.clear();
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
    scratch.start_heights.resize(width, 0);
    let earlier = earlier.filter(|earlier| earlier.heights.len() == mask.len());
    for (index, (row, out)) in gaps
        .chunks_exact(width)
        .zip(scratch.field.chunks_exact_mut(width))
        .enumerate()
    {
        // A row's envelope reads nothing but that row, so one that matches needs no
        // envelope. A new stone leaves most rows alone. Proof: [E17].
        if let Some(EarlierTransform { heights, field }) = earlier
            && heights[index * width..][..width] == *row
        {
            out.copy_from_slice(&field[index * width..][..width]);
            continue;
        }
        envelope_row(
            row,
            none * none,
            &mut scratch.parabolas,
            &mut scratch.starts,
            &mut scratch.start_heights,
            out,
        );
    }
}

/// `min over c of (x - c)^2 + row[c]` at every `x`.
///
/// In `i32`, which [E3] shows holds every value here. The narrower divide is the
/// point: 64-bit division costs several times the 32-bit one on many server CPUs.
fn envelope_row(
    row: &[u32],
    absent: u32,
    parabolas: &mut [u32],
    starts: &mut [u32],
    start_heights: &mut [i32],
    out: &mut [f64],
) {
    let size = row.len();
    let parabolas = &mut parabolas[..size];
    let starts = &mut starts[..size];
    let start_heights = &mut start_heights[..size];
    let out = &mut out[..size];
    let width = size as i32;
    let square = |value: i32| value * value;
    let mut top = 0usize;
    parabolas[0] = 0;
    starts[0] = 0;
    start_heights[0] = row[0] as i32;
    for (column, &height) in row.iter().enumerate().skip(1) {
        // A column holding no legal cell cannot win while any column holds one,
        // which [E3] makes the caller's condition.
        if height >= absent {
            continue;
        }
        let (right, height) = (column as i32, height as i32);
        let at_start = loop {
            let start = starts[top] as i32;
            let candidate = square(start - right) + height;
            if start_heights[top] <= candidate {
                break None;
            }
            if top == 0 {
                break Some(candidate);
            }
            top -= 1;
        };
        if let Some(candidate) = at_start {
            parabolas[0] = column as u32;
            start_heights[0] = candidate;
            continue;
        }
        // First pixel where `column` is strictly below the parabola on top.
        let left = parabolas[top] as i32;
        let numerator = right * right - left * left + height - row[left as usize] as i32;
        let crossing = numerator.div_euclid(2 * (right - left)) + 1;
        if crossing < width {
            top += 1;
            parabolas[top] = column as u32;
            starts[top] = crossing as u32;
            start_heights[top] = square(crossing - right) + height;
        }
    }
    // One run per parabola, so the arithmetic is a flat loop over contiguous pixels.
    let mut end = size;
    for index in (0..=top).rev() {
        let (centre, start) = (parabolas[index] as i32, starts[index] as usize);
        let height = row[parabolas[index] as usize] as i32;
        for (at, value) in out[start..end].iter_mut().enumerate() {
            let offset = (start + at) as i32 - centre;
            *value = f64::from(offset * offset + height);
        }
        end = start;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_distance_transform(mask: &[bool], width: usize, height: usize) -> Vec<f64> {
        let mut scratch = EdtScratch::default();
        squared_distance_transform_into(mask, width, height, &mut scratch, None);
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

    /// A mask with no true cell has no heights to offer. Keeping the last mask's
    /// would pair them with a field of nothing but `ABSENT`.
    #[test]
    fn an_empty_mask_leaves_no_rows_to_copy() {
        let (width, height) = (8, 8);
        let mut mask = vec![false; width * height];
        mask[27] = true;
        let mut earlier = EdtScratch::default();
        squared_distance_transform_into(&mask, width, height, &mut earlier, None);
        let empty = vec![false; width * height];
        squared_distance_transform_into(&empty, width, height, &mut earlier, None);

        let mut copied = EdtScratch::default();
        let rows = EarlierTransform {
            heights: earlier.heights(),
            field: earlier.field(),
        };
        squared_distance_transform_into(&mask, width, height, &mut copied, Some(rows));
        assert_eq!(
            copied.field,
            squared_distance_transform(&mask, width, height)
        );
    }

    /// Copying rows from another mask's transform has to change nothing, whether
    /// the masks differ by a cell, by a lot, or in having any true cell at all.
    #[test]
    fn rows_copied_from_an_earlier_transform_match_a_fresh_one() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (width, height) = (32, 32);
        for per_thousand in [0, 3, 20, 300, 1000] {
            for flips in [0, 1, 5, 400] {
                let before: Vec<bool> = (0..width * height)
                    .map(|_| next() % 1000 < per_thousand)
                    .collect();
                let mut after = before.clone();
                for _ in 0..flips {
                    let cell = next() as usize % after.len();
                    after[cell] = !after[cell];
                }
                let mut first = EdtScratch::default();
                squared_distance_transform_into(&before, width, height, &mut first, None);
                let mut copied = EdtScratch::default();
                let earlier = EarlierTransform {
                    heights: first.heights(),
                    field: first.field(),
                };
                squared_distance_transform_into(&after, width, height, &mut copied, Some(earlier));
                assert_eq!(
                    copied.field,
                    squared_distance_transform(&after, width, height),
                    "{per_thousand} per thousand, {flips} flips"
                );
            }
        }
    }
}
