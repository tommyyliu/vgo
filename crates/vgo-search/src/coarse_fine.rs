//! Coarse->fine factored candidate sampling from a dense policy map.
//!
//! The net emits a fine grid of placement logits. Candidates are drawn from the
//! net's own distribution (not an external quasi-random sequence) so the same
//! board yields a stable candidate distribution and the policy target becomes a
//! learnable function of the board. See docs/history/POLICY_REDESIGN.md.
//!
//! Each candidate is sampled in two stages, with replacement:
//!   1. a coarse cell `C` ~ softmax over the coarse map (coarse cell value = MAX
//!      of the fine logits it covers),
//!   2. a fine cell within `C` ~ softmax over the fine logits restricted to `C`.
//!
//! The exact sampling probability is `beta = P_coarse(C) * P_fine(a | C)`, which
//! we return so the training side can importance-correct the target.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};
use vgo_core::{
    COORDINATE_EPSILON, Point, Position, StoneFit, StoneRun, nearest_legal_placement_indexed_within,
};

/// A sampled placement candidate and the probability it was drawn with.
#[derive(Clone, Copy, Debug)]
pub struct CandidateSample {
    pub point: Point,
    /// beta(a) = P_coarse(C(a)) * P_fine(a | C(a)); exact for the two-stage
    /// with-replacement sampler. Used for the Sampled-AlphaZero importance
    /// correction on the training target.
    pub beta: f64,
}

/// How much farther than `2r` plus half a cell a stone must be to leave a cell's
/// snap alone. Past the most a snap can move, and far below a cell. See [E18].
const SNAP_REACH_MARGIN: f64 = 1.0e-4;

/// What snapping found for a cell, on the stones of a [`SnapAnchor`].
#[derive(Clone, Copy, Debug, Default)]
enum SavedSnap {
    #[default]
    Unknown,
    Rejected,
    Kept(Point),
}

/// Snaps a search's grids can share.
///
/// Every node's grid snaps the cells around every stone, and a node differs from the
/// root by a few stones, so nearly all of that work was the same each time. This
/// keeps the root's answers. A cell far enough from the stones a node differs by
/// reads the saved one. Nothing here is needed for correctness: with no root
/// announced, every grid snaps from nothing.
#[derive(Debug, Default)]
struct SnapAnchor {
    width: usize,
    height: usize,
    radius_bits: u64,
    stones: StoneRun,
    answers: Vec<SavedSnap>,
    /// Set by [`next_grid_is_search_root`], and spent by the next grid.
    next_is_root: bool,
}

impl SnapAnchor {
    /// Compares a grid with the anchor, moving the anchor to it if a root was
    /// announced. Returns whether the grid can use the saved answers.
    fn prepare(&mut self, position: &Position, width: usize, height: usize) -> bool {
        let same_board = self.width == width
            && self.height == height
            && self.radius_bits == position.radius().to_bits();
        let fit = if same_board {
            self.stones.fit(position.stones())
        } else {
            self.stones.clear_differing();
            StoneFit::Unrelated
        };
        if !std::mem::take(&mut self.next_is_root)
            || (fit == StoneFit::Extends && self.stones.differing().is_empty())
        {
            return fit != StoneFit::Unrelated;
        }
        self.width = width;
        self.height = height;
        self.radius_bits = position.radius().to_bits();
        self.stones.set(position.stones());
        self.answers.clear();
        self.answers.resize(width * height, SavedSnap::Unknown);
        true
    }
}

thread_local! {
    /// One grid per tree node, each needing this only while it is built.
    static NEAREST_SQUARES: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SNAP_ANCHOR: RefCell<SnapAnchor> = RefCell::default();
}

/// Says the next grid on this thread is the root of a search, whose nodes' grids are
/// the ones after it.
pub(crate) fn next_grid_is_search_root() {
    SNAP_ANCHOR.with_borrow_mut(|anchor| anchor.next_is_root = true);
}

/// A fine placement grid of logits over `[0,1]^2`, plus the legal mask, ready for
/// coarse->fine sampling. Illegal cells are held out of every softmax.
pub struct FineGrid {
    width: usize,
    height: usize,
    coarse: usize, // coarse-cell size in fine cells (pool factor), both axes
    logits: Arc<Vec<f32>>,
    // Rust Vec<bool> uses a byte per cell. Pack exact legality into bits.
    legal: Vec<u64>,
    /// Cells whose illegal centre was projected onto legal area inside the
    /// cell, keyed by row-major cell index. Ordinary legal cells place at their
    /// centre and need no stored point. Build order keeps this sorted for lookup.
    placement_overrides: Vec<(usize, Point)>,
    /// The coarse distribution, kept because every widening of a node reads the
    /// same immutable logits and would otherwise rescan the whole grid.
    coarse_distribution: OnceLock<CoarseDistribution>,
}

/// Legal coarse cells with their max logit, and the softmax over them.
struct CoarseDistribution {
    cells: Vec<(usize, usize, f32)>,
    probabilities: Vec<f64>,
}

impl FineGrid {
    fn coarse_distribution(&self) -> &CoarseDistribution {
        self.coarse_distribution.get_or_init(|| {
            let (width, height) = self.coarse_dims();
            let mut cells = Vec::new();
            for crow in 0..height {
                for ccol in 0..width {
                    if let Some(maximum) = self.coarse_max(crow, ccol) {
                        cells.push((crow, ccol, maximum));
                    }
                }
            }
            let probabilities = softmax(cells.iter().map(|&(_, _, maximum)| maximum));
            CoarseDistribution {
                cells,
                probabilities,
            }
        })
    }

    fn legal_at(&self, index: usize) -> bool {
        self.legal[index / 64] & (1u64 << (index % 64)) != 0
    }

    #[cfg(test)]
    fn unpacked_legal(&self) -> Vec<bool> {
        (0..self.width * self.height)
            .map(|i| self.legal_at(i))
            .collect()
    }
    pub(crate) fn account_memory(&self, counter: &mut crate::memory::MemoryCounter) {
        for allocation in [
            crate::HeapAllocation::vector(&self.logits),
            crate::HeapAllocation::vector(&self.legal),
            crate::HeapAllocation::vector(&self.placement_overrides),
        ] {
            counter.allocation(allocation, false);
        }
        if let Some(distribution) = self.coarse_distribution.get() {
            counter.allocation(crate::HeapAllocation::vector(&distribution.cells), false);
            counter.allocation(
                crate::HeapAllocation::vector(&distribution.probabilities),
                false,
            );
        }
    }
    // The production-like 128x128/35-stone fixture has 219 snapped cells. A
    // 256-entry slab avoids growth while occupying 6 KiB, versus the former
    // dense placement table's 256 KiB.
    const PLACEMENT_OVERRIDE_CAPACITY: usize = 256;

    /// Build from a fine logit map. `logit_at(row, col)` returns the net's placement
    /// logit for that fine cell; legality is decided by placing a stone at the cell
    /// centre. `coarse` is the pool factor (e.g. 8 turns a 128 grid into 16 coarse
    /// cells per axis); it is clamped so at least one coarse cell exists.
    pub fn build(
        position: &Position,
        width: usize,
        height: usize,
        coarse: usize,
        mut logit_at: impl FnMut(usize, usize) -> f32,
    ) -> Self {
        Self::build_storage(position, width, height, coarse, None, &mut logit_at)
    }

    /// Reuse immutable policy logits. Illegal entries are ignored by the legal
    /// mask, not overwritten. A trailing pass logit is permitted.
    pub fn from_shared_logits(
        position: &Position,
        width: usize,
        height: usize,
        coarse: usize,
        logits: Arc<Vec<f32>>,
    ) -> Self {
        assert!(width > 0 && height > 0);
        assert!(logits.len() == width * height || logits.len() == width * height + 1);
        Self::build_storage(
            position,
            width,
            height,
            coarse,
            Some(logits),
            &mut |_, _| unreachable!(),
        )
    }

    fn build_storage(
        position: &Position,
        width: usize,
        height: usize,
        coarse: usize,
        shared: Option<Arc<Vec<f32>>>,
        logit_at: &mut impl FnMut(usize, usize) -> f32,
    ) -> Self {
        let coarse = coarse.clamp(1, width.min(height));
        let mut logits = if shared.is_none() {
            vec![f32::NEG_INFINITY; width * height]
        } else {
            Vec::new()
        };
        let mut legal = vec![0u64; (width * height).div_ceil(64)];
        // A cell is playable when its centre is legal, or when the centre is
        // only just illegal and projects onto legal area still inside the cell.
        // The second case is what keeps thin legal regions reachable: legality
        // is continuous but the grid is not, so a sliver narrower than a cell
        // contains no cell centre, and a group whose last liberty sits there
        // could never be answered.
        //
        // Projection is expensive -- it enumerates candidates over every stone
        // and the whole vertex set -- and this loop runs over every cell at
        // every node, so it is gated. A cell can only hold legal area its
        // centre misses when the centre sits just inside the exclusion radius
        // of its nearest stone, within half a cell diagonal of the boundary.
        // Cells deeper inside a stone cannot be rescued, and cells already
        // clear need no rescuing, so only that thin band pays for a projection.
        let mut placement_overrides = Vec::with_capacity(Self::PLACEMENT_OVERRIDE_CAPACITY);
        // The vertex set depends only on the position but is ~78% of a single
        // projection's cost, and this loop projects several hundred cells.
        // Computing it once per grid rather than once per cell is the
        // difference between snapping being affordable and not.
        let index = position.legal_index();
        let half_diagonal =
            0.5 * ((1.0 / width as f64).powi(2) + (1.0 / height as f64).powi(2)).sqrt();
        let exclusion = 2.0 * position.radius();
        // Exactly `clear_of_stones`'s threshold, so the two agree bit for bit.
        let minimum = exclusion - COORDINATE_EPSILON;
        let minimum_squared = minimum * minimum;
        let half_x = 0.5 / width as f64;
        let half_y = 0.5 / height as f64;
        // Distance to the nearest stone, for every cell, scattered rather than
        // searched.
        //
        // Both predicates below are `nearest^2 >= threshold^2` for a threshold
        // under `2r` -- legality wants `2r - epsilon`, the boundary band wants
        // `2r - half_diagonal`. Asked per cell against every stone that is
        // O(cells * stones), and a grid is built per *node*, so it is multiplied
        // by the simulation count on every move. Measured at 128 square: 1.0 ms
        // at 28 stones and 8.6 ms at 255, which on a standard board is seconds
        // per move and was why a multi-radius shard produced no games at all.
        //
        // A stone can only bring a cell under `2r`, so scattering its own
        // neighbourhood touches about `16 r^2` of the grid and the total is
        // `stones * 16 r^2 * cells`. Stone count goes as `1/r^2`, so that
        // product is constant: O(cells), whatever the board size. Cells no
        // stone reaches keep infinity, which satisfies both thresholds exactly
        // as an untouched cell should.
        let snap_reach = half_diagonal + exclusion + SNAP_REACH_MARGIN;
        let snap_reach_square = snap_reach * snap_reach;
        let mut borrowed_anchor = SNAP_ANCHOR.take();
        let anchor = &mut borrowed_anchor;
        let reads_answers = anchor.prepare(position, width, height);
        NEAREST_SQUARES.with_borrow_mut(|nearest_squared| {
            // Per axis, not per cell: the board's edges and each cell's centre are
            // products of one row and one column, so a grid per node need not
            // divide or compare them 16,384 times.
            let radius = position.radius();
            let xs: Vec<f64> = (0..width)
                .map(|col| (col as f64 + 0.5) / width as f64)
                .collect();
            nearest_stone_squares(position, &xs, height, exclusion, nearest_squared);
            let inset = |value: f64| {
                value >= radius - COORDINATE_EPSILON && value <= 1.0 - radius + COORDINATE_EPSILON
            };
            let side_slack = |value: f64| (value - radius).min(1.0 - radius - value);
            let columns: Vec<(bool, f64)> = xs.iter().map(|&x| (inset(x), side_slack(x))).collect();
            let threshold = exclusion - half_diagonal;
            let threshold_squared = threshold * threshold;
            for row in 0..height {
                let y = (row as f64 + 0.5) / height as f64;
                let (row_inset, row_slack) = (inset(y), side_slack(y));
                let base = row * width;
                let cells = &nearest_squared[base..base + width];
                for (col, (&nearest, &(column_inset, column_slack))) in
                    cells.iter().zip(&columns).enumerate()
                {
                    let idx = base + col;
                    let placement_override =
                        if row_inset && column_inset && nearest >= minimum_squared {
                            None
                        } else if column_slack.min(row_slack) >= -half_diagonal
                            && (threshold <= 0.0 || nearest >= threshold_squared)
                        {
                            // Only a point inside this cell is kept, and such a point
                            // is within half a diagonal of its centre.
                            let point = Point::new(xs[col], y);
                            // Stones this far away cannot change what snapping
                            // finds in this cell: [E18].
                            let saved = (reads_answers
                                && anchor.stones.differing_beyond(point, snap_reach_square))
                            .then(|| &mut anchor.answers[idx]);
                            let answer = saved.as_deref().copied().unwrap_or_default();
                            let answer = if let SavedSnap::Unknown = answer {
                                let snapped = nearest_legal_placement_indexed_within(
                                    position,
                                    point,
                                    index,
                                    half_diagonal,
                                );
                                let inside = snapped.legal
                                    && (snapped.point.x - point.x).abs() <= half_x
                                    && (snapped.point.y - point.y).abs() <= half_y;
                                let answer = if inside {
                                    SavedSnap::Kept(snapped.point)
                                } else {
                                    SavedSnap::Rejected
                                };
                                if let Some(saved) = saved {
                                    *saved = answer;
                                }
                                answer
                            } else {
                                answer
                            };
                            match answer {
                                SavedSnap::Kept(point) => Some(point),
                                _ => continue,
                            }
                        } else {
                            continue;
                        };
                    legal[idx / 64] |= 1u64 << (idx % 64);
                    if shared.is_none() {
                        logits[idx] = logit_at(row, col);
                    }
                    if let Some(point) = placement_override {
                        placement_overrides.push((idx, point));
                    }
                }
            }
        });
        SNAP_ANCHOR.set(borrowed_anchor);
        // Scratch headroom is useful while snapping; it need not live in every
        // retained tree node. Empty grids also release the initial 6 KiB slab.
        placement_overrides.shrink_to_fit();
        Self {
            width,
            height,
            coarse,
            logits: shared.unwrap_or_else(|| Arc::new(logits)),
            legal,
            placement_overrides,
            coarse_distribution: OnceLock::new(),
        }
    }

    /// Grid dimensions, for callers that need to map points back to cells.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Whether two points fall in the same fine cell of this grid.
    #[must_use]
    pub fn same_cell(&self, a: Point, b: Point) -> bool {
        self.cell_of(a) == self.cell_of(b)
    }

    fn cell_of(&self, p: Point) -> (usize, usize) {
        let col = ((p.x * self.width as f64).floor() as usize).min(self.width - 1);
        let row = ((p.y * self.height as f64).floor() as usize).min(self.height - 1);
        (row, col)
    }

    fn coarse_dims(&self) -> (usize, usize) {
        // ceil division so partial edge cells are covered.
        (
            self.width.div_ceil(self.coarse),
            self.height.div_ceil(self.coarse),
        )
    }

    /// Max fine logit within a coarse cell, and whether it contains any legal cell.
    fn coarse_max(&self, crow: usize, ccol: usize) -> Option<f32> {
        let r0 = crow * self.coarse;
        let c0 = ccol * self.coarse;
        let mut best: Option<f32> = None;
        for row in r0..(r0 + self.coarse).min(self.height) {
            for col in c0..(c0 + self.coarse).min(self.width) {
                let idx = row * self.width + col;
                if self.legal_at(idx) {
                    let v = self.logits[idx];
                    best = Some(best.map_or(v, |b| b.max(v)));
                }
            }
        }
        best
    }

    fn placement_at(&self, row: usize, col: usize) -> Point {
        let index = row * self.width + col;
        match self
            .placement_overrides
            .binary_search_by_key(&index, |&(cell, _)| cell)
        {
            Ok(override_index) => self.placement_overrides[override_index].1,
            Err(_) => cell_center(row, col, self.width, self.height),
        }
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use vgo_core::{Color, Stone};

    #[test]
    fn shared_logits_preserve_mask_snapping_and_noisy_sampling() {
        let parent = Position::new(0.1, vec![Stone::new(0.3, 0.3, Color::Black)], Color::White);
        for (width, height, coarse) in [(17, 23, 4), (32, 32, 8)] {
            let values = Arc::new(
                (0..width * height + 1)
                    .map(|i| ((i * 19) % 97) as f32 / 13.0)
                    .collect::<Vec<_>>(),
            );
            let copy =
                FineGrid::build(&parent, width, height, coarse, |r, c| values[r * width + c]);
            let shared =
                FineGrid::from_shared_logits(&parent, width, height, coarse, Arc::clone(&values));
            assert!(Arc::ptr_eq(&values, &shared.logits));
            assert_eq!(copy.legal, shared.legal);
            assert_eq!(copy.placement_overrides, shared.placement_overrides);
            assert!(copy.unpacked_legal().iter().any(|&v| !v));
            assert!(!copy.placement_overrides.is_empty());
            for noise in [0.0, 0.25, 1.0] {
                let rng = || {
                    let mut state = 19u64;
                    move || {
                        state = crate::candidates::splitmix64(state);
                        crate::candidates::unit_f64(state)
                    }
                };
                let expected = sample_candidates_noisy(&copy, 128, noise, rng());
                let actual = sample_candidates_noisy(&shared, 128, noise, rng());
                assert_eq!(expected.len(), actual.len());
                for (a, b) in expected.iter().zip(&actual) {
                    assert_eq!(
                        (a.point.x.to_bits(), a.point.y.to_bits(), a.beta.to_bits()),
                        (b.point.x.to_bits(), b.point.y.to_bits(), b.beta.to_bits())
                    );
                }
            }
        }
    }
}

/// Draw `count` candidates with replacement via coarse->fine factored sampling.
/// `rng` yields uniform f64 in [0, 1). Returns fewer than `count` only if the
/// board has no legal placement at all.
pub fn sample_candidates(
    grid: &FineGrid,
    count: usize,
    rng: impl FnMut() -> f64,
) -> Vec<CandidateSample> {
    sample_candidates_with(grid, count, 0.0, rng, |row, col| {
        grid.placement_at(row, col)
    })
}

/// Sample candidates with `noise` uniform mass mixed into the proposal.
///
/// The policy's softmax decides which placements get proposed at all, so a
/// move it rates near zero is never drawn however long the search runs.
/// Blending in a uniform component over the *legal* cells floors that rate,
/// which is the role Dirichlet noise plays in AlphaZero -- adapted to a sampler
/// rather than an enumeration, since blending into priors would only reshuffle
/// visits among candidates already drawn.
///
/// The returned `beta` is the probability under the mixture, not under the
/// softmax, so the importance correction on the policy target stays exact.
#[must_use]
pub fn sample_candidates_noisy(
    grid: &FineGrid,
    count: usize,
    noise: f64,
    rng: impl FnMut() -> f64,
) -> Vec<CandidateSample> {
    sample_candidates_with(grid, count, noise, rng, |row, col| {
        grid.placement_at(row, col)
    })
}

/// Blend a distribution with the uniform one over the same support.
fn mix_uniform(probs: &mut [f64], noise: f64) {
    if noise <= 0.0 || probs.is_empty() {
        return;
    }
    let uniform = 1.0 / probs.len() as f64;
    for p in probs.iter_mut() {
        *p = (1.0 - noise) * *p + noise * uniform;
    }
}

fn sample_candidates_with(
    grid: &FineGrid,
    count: usize,
    noise: f64,
    mut rng: impl FnMut() -> f64,
    mut placement_at: impl FnMut(usize, usize) -> Point,
) -> Vec<CandidateSample> {
    // Coarse distribution: softmax over per-coarse-cell max logits (legal only).
    let distribution = grid.coarse_distribution();
    let coarse_cells = &distribution.cells;
    if coarse_cells.is_empty() {
        return Vec::new();
    }
    let mixed;
    let coarse_probs = if noise > 0.0 {
        let mut probabilities = distribution.probabilities.clone();
        mix_uniform(&mut probabilities, noise);
        mixed = probabilities;
        &mixed
    } else {
        &distribution.probabilities
    };

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Stage 1: sample a coarse cell.
        let ci = sample_index(coarse_probs, rng());
        let (crow, ccol, _) = coarse_cells[ci];
        let p_coarse = coarse_probs[ci];

        // Stage 2: sample a fine cell within it, softmax over legal fine logits.
        let r0 = crow * grid.coarse;
        let c0 = ccol * grid.coarse;
        let mut fine: Vec<(usize, usize, f32)> = Vec::new();
        for row in r0..(r0 + grid.coarse).min(grid.height) {
            for col in c0..(c0 + grid.coarse).min(grid.width) {
                let idx = row * grid.width + col;
                if grid.legal_at(idx) {
                    fine.push((row, col, grid.logits[idx]));
                }
            }
        }
        let mut fine_probs = softmax(fine.iter().map(|&(_, _, v)| v));
        mix_uniform(&mut fine_probs, noise);
        let fi = sample_index(&fine_probs, rng());
        let (row, col, _) = fine[fi];
        let p_fine = fine_probs[fi];

        out.push(CandidateSample {
            point: placement_at(row, col),
            beta: p_coarse * p_fine,
        });
    }
    out
}

/// Whether a cell's illegal centre is close enough to the legal boundary that
/// the cell might still contain legal area.
///
/// Cheap rejection for the projection above. A centre excluded by a stone can
/// only have legal area within its own cell if it sits within half a cell
/// diagonal of that stone's exclusion circle; deeper than that, every point in
/// the cell is excluded too. Board-edge exclusion is treated the same way.
/// Squared distance to the nearest stone per cell, or infinity beyond `reach`.
///
/// Only correct for callers comparing against a threshold at or below `reach`,
/// which is what makes the bounded scatter sound: a cell no stone comes within
/// `reach` of keeps infinity, and infinity satisfies any such threshold just as
/// its true distance would.
fn nearest_stone_squares(
    position: &Position,
    xs: &[f64],
    height: usize,
    reach: f64,
    nearest: &mut Vec<f64>,
) {
    let width = xs.len();
    nearest.clear();
    nearest.resize(width * height, f64::INFINITY);
    if reach <= 0.0 {
        return;
    }
    for stone in position.stones() {
        // Cells whose centre can lie within `reach` of this stone.
        let low_col = (((stone.x - reach) * width as f64 - 0.5).ceil()).max(0.0) as usize;
        let low_row = (((stone.y - reach) * height as f64 - 0.5).ceil()).max(0.0) as usize;
        let high_col = (((stone.x + reach) * width as f64 - 0.5).floor()).max(0.0) as usize;
        let high_row = (((stone.y + reach) * height as f64 - 0.5).floor()).max(0.0) as usize;
        for row in low_row..=high_row.min(height - 1) {
            let y = (row as f64 + 0.5) / height as f64;
            let dy = y - stone.y;
            let dy_squared = dy * dy;
            let base = row * width;
            for col in low_col..=high_col.min(width - 1) {
                let dx = xs[col] - stone.x;
                let squared = dx.mul_add(dx, dy_squared);
                let slot = &mut nearest[base + col];
                if squared < *slot {
                    *slot = squared;
                }
            }
        }
    }
}

/// The original per-cell predicate, kept as the reference the scattered path is
/// tested against. Production applies it per axis to the stone distances already
/// computed for the whole grid.
#[cfg(test)]
fn straddles_boundary(
    position: &Position,
    point: Point,
    exclusion: f64,
    half_diagonal: f64,
) -> bool {
    let radius = position.radius();
    let edge_slack = (point.x - radius)
        .min(1.0 - radius - point.x)
        .min(point.y - radius)
        .min(1.0 - radius - point.y);
    if edge_slack < -half_diagonal {
        return false;
    }
    // Compared as squares: `hypot` is a libm call that cannot be inlined or
    // vectorized, and this runs once per (cell, stone) over the whole coarse
    // grid. A negative threshold is satisfied by every distance, so it is
    // handled before squaring -- squaring it would flip the comparison.
    let threshold = exclusion - half_diagonal;
    if threshold <= 0.0 {
        return true;
    }
    let threshold_squared = threshold * threshold;
    position.stones().iter().all(|stone| {
        let dx = point.x - stone.x;
        let dy = point.y - stone.y;
        dx.mul_add(dx, dy * dy) >= threshold_squared
    })
}

fn cell_center(row: usize, col: usize, width: usize, height: usize) -> Point {
    Point::new(
        (col as f64 + 0.5) / width as f64,
        (row as f64 + 0.5) / height as f64,
    )
}

/// Numerically stable softmax of an iterator of logits (already legal-only).
fn softmax(logits: impl Iterator<Item = f32> + Clone) -> Vec<f64> {
    let max = logits.clone().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.map(|l| f64::from(l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        let n = exps.len().max(1);
        return vec![1.0 / n as f64; exps.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Inverse-CDF sample: given probabilities and a uniform draw, return the index.
fn sample_index(probs: &[f64], u: f64) -> usize {
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod scatter_tests {
    use vgo_core::{
        Color, Point, Position, Stone, is_legal_placement, legal_set_vertices,
        nearest_legal_placement_with,
    };

    use super::{FineGrid, straddles_boundary};

    fn lattice(count: usize, radius: f64, seed: u64) -> Position {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let spacing = 2.0 * radius * 1.06;
        let per_row = ((0.88_f64 / spacing).floor() as usize).max(1);
        let jitter = (spacing - 2.0 * radius) * 0.4;
        let mut stones = Vec::new();
        for index in 0..count {
            let (row, col) = (index / per_row, index % per_row);
            let x = radius + 0.02 + col as f64 * spacing + (next() - 0.5) * jitter;
            let y = radius + 0.02 + row as f64 * spacing + (next() - 0.5) * jitter;
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

    /// The reference: the per-cell scan the scatter replaced. Kept here so the
    /// optimisation is pinned to the definition rather than to itself.
    fn reference_legal(position: &Position, width: usize, height: usize) -> Vec<bool> {
        let radius = position.radius();
        let exclusion = 2.0 * radius;
        let half_diagonal =
            0.5 * ((1.0 / width as f64).powi(2) + (1.0 / height as f64).powi(2)).sqrt();
        let known = legal_set_vertices(position);
        let (half_x, half_y) = (0.5 / width as f64, 0.5 / height as f64);
        let mut legal = vec![false; width * height];
        for row in 0..height {
            for col in 0..width {
                let point = Point::new(
                    (col as f64 + 0.5) / width as f64,
                    (row as f64 + 0.5) / height as f64,
                );
                if is_legal_placement(position, point.x, point.y) {
                    legal[row * width + col] = true;
                } else if straddles_boundary(position, point, exclusion, half_diagonal) {
                    let snapped = nearest_legal_placement_with(position, point, Some(&known));
                    if snapped.legal
                        && (snapped.point.x - point.x).abs() <= half_x
                        && (snapped.point.y - point.y).abs() <= half_y
                    {
                        legal[row * width + col] = true;
                    }
                }
            }
        }
        legal
    }

    /// A faster legality mask that disagrees anywhere is a different game: the
    /// search would propose moves the rules refuse, or refuse moves they allow.
    #[test]
    fn the_scattered_grid_matches_the_per_cell_scan() {
        for (radius, count) in [
            (1.0 / 18.0, 0usize),
            (1.0 / 18.0, 1),
            (1.0 / 18.0, 28),
            (1.0 / 18.0, 52),
            (1.0 / 38.0, 120),
            (1.0 / 38.0, 300),
        ] {
            let position = lattice(count, radius, 19);
            let grid = FineGrid::build(&position, 64, 64, 16, |_, _| 0.0);
            let expected = reference_legal(&position, 64, 64);
            let actual = grid.unpacked_legal();
            assert_eq!(
                actual,
                expected,
                "1/{} with {} stones",
                (1.0 / radius).round(),
                position.stones().len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgo_core::{Color, Position, Stone, is_legal_placement};

    fn empty_board() -> Position {
        Position::new(1.0 / 6.0, Vec::new(), Color::Black)
    }

    fn lattice(radius: f64) -> Vec<(f64, f64)> {
        let step = 2.3 * radius;
        let mut points = Vec::new();
        let mut y = 2.0 * radius;
        while y < 1.0 - 2.0 * radius {
            let mut x = 2.0 * radius;
            while x < 1.0 - 2.0 * radius {
                points.push((x, y));
                x += step;
            }
            y += step;
        }
        points
    }

    fn position_of(radius: f64, points: &[(f64, f64)], picks: &[usize]) -> Position {
        let stones = picks.iter().enumerate().map(|(turn, &pick)| {
            let color = if turn % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            Stone::new(points[pick].0, points[pick].1, color)
        });
        Position::new(radius, stones.collect(), Color::Black)
    }

    /// A grid built on the shared anchor, and one with none.
    fn both_grids(position: &Position, size: usize) -> (FineGrid, FineGrid) {
        let logits = || Arc::new(vec![0.0_f32; size * size]);
        let anchored = FineGrid::from_shared_logits(position, size, size, 4, logits());
        let kept = SNAP_ANCHOR.take();
        let fresh = FineGrid::from_shared_logits(position, size, size, 4, logits());
        SNAP_ANCHOR.set(kept);
        (anchored, fresh)
    }

    fn assert_same_grid(anchored: &FineGrid, fresh: &FineGrid, context: &str) {
        assert_eq!(anchored.legal, fresh.legal, "{context}");
        assert_eq!(
            anchored.placement_overrides.len(),
            fresh.placement_overrides.len(),
            "{context}"
        );
        for (a, b) in anchored
            .placement_overrides
            .iter()
            .zip(&fresh.placement_overrides)
        {
            assert_eq!(a.0, b.0, "{context}");
            assert_eq!(a.1.x.to_bits(), b.1.x.to_bits(), "{context}");
            assert_eq!(a.1.y.to_bits(), b.1.y.to_bits(), "{context}");
        }
    }

    /// Every path through the anchor against a grid snapped from nothing: nodes
    /// branching off a root, captures from the middle of the list, a jump to another
    /// game, and a change of grid size on the same thread.
    #[test]
    fn grids_on_the_anchor_match_grids_from_nothing() {
        let radius = 1.0 / 38.0;
        let points = lattice(radius);
        assert!(points.len() > 120);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |limit: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % limit as u64) as usize
        };
        let (mut overrides, mut saved) = (0usize, 0usize);
        for game in 0..12 {
            let mut root: Vec<usize> = (0..points.len()).collect();
            for at in 0..root.len() {
                root.swap(at, at + next(points.len() - at));
            }
            let mut pool = root.split_off(30 + game * 6);
            for node in 0..40 {
                let mut picks = root.clone();
                let depth = next(9);
                picks.extend(pool.iter().skip(next(20)).take(depth));
                if node % 7 == 3 {
                    picks.remove(next(picks.len()));
                }
                if node % 13 == 12 {
                    let by = 1 + next(20);
                    root.extend(pool.drain(..by));
                }
                if node % 13 == 1 {
                    next_grid_is_search_root();
                }
                let size = [128, 128, 128, 64][game % 4];
                let position = position_of(radius, &points, &picks);
                assert!(position.validate().is_playable());
                let (anchored, fresh) = both_grids(&position, size);
                assert_same_grid(&anchored, &fresh, &format!("game {game} node {node}"));
                overrides += fresh.placement_overrides.len();
                saved += SNAP_ANCHOR.with_borrow(|anchor| {
                    anchor
                        .answers
                        .iter()
                        .filter(|answer| !matches!(answer, SavedSnap::Unknown))
                        .count()
                });
            }
        }
        // A run with nothing to snap, or that never saved an answer, proved nothing.
        assert!(overrides > 1000, "{overrides} overrides");
        assert!(saved > 1000, "saved {saved} answers");
    }

    /// An answer belongs to its radius and its grid as well as to its stones.
    #[test]
    fn the_anchor_keeps_to_its_board() {
        let wide = 1.0 / 38.0;
        let points = lattice(wide);
        let picks: Vec<usize> = (0..90).map(|index| index * 2).collect();
        // The close pair puts both radii's rings on the same cells.
        for (radius, size) in [
            (wide, 128),
            (0.8 * wide, 128),
            (wide, 64),
            (wide, 128),
            (0.999 * wide, 128),
        ] {
            let position = position_of(radius, &points, &picks);
            for announced in [false, true, false] {
                if announced {
                    next_grid_is_search_root();
                }
                let (anchored, fresh) = both_grids(&position, size);
                assert_same_grid(&anchored, &fresh, &format!("radius {radius} size {size}"));
            }
        }
    }

    fn narrow_sliver_position() -> Position {
        let radius = 1.0 / 18.0;
        let gap = 4.0 * radius + 0.004;
        Position::new(
            radius,
            vec![
                Stone::new(0.5 - gap / 2.0, 0.5, Color::Black),
                Stone::new(0.5 + gap / 2.0, 0.5, Color::Black),
            ],
            Color::White,
        )
    }

    /// A deterministic uniform stream for reproducible sampling in tests.
    fn stream(values: Vec<f64>) -> impl FnMut() -> f64 {
        let mut i = 0;
        move || {
            let v = values[i % values.len()];
            i += 1;
            v
        }
    }

    /// The dense placement table used before snapped cells became sparse.
    /// Keeping the former construction here makes the representation change a
    /// bit-for-bit regression test rather than merely a legality check.
    fn dense_placement_reference(
        position: &Position,
        width: usize,
        height: usize,
    ) -> (Vec<bool>, Vec<Point>) {
        let mut legal = vec![false; width * height];
        let mut placement = vec![Point::new(0.0, 0.0); width * height];
        let known_vertices = vgo_core::legal_set_vertices(position);
        let half_diagonal =
            0.5 * ((1.0 / width as f64).powi(2) + (1.0 / height as f64).powi(2)).sqrt();
        let exclusion = 2.0 * position.radius();
        let half_x = 0.5 / width as f64;
        let half_y = 0.5 / height as f64;
        for row in 0..height {
            for col in 0..width {
                let point = cell_center(row, col, width, height);
                let index = row * width + col;
                let resolved = if is_legal_placement(position, point.x, point.y) {
                    Some(point)
                } else if straddles_boundary(position, point, exclusion, half_diagonal) {
                    let snapped = vgo_core::nearest_legal_placement_with(
                        position,
                        point,
                        Some(&known_vertices),
                    );
                    let inside = snapped.legal
                        && (snapped.point.x - point.x).abs() <= half_x
                        && (snapped.point.y - point.y).abs() <= half_y;
                    inside.then_some(snapped.point)
                } else {
                    None
                };
                if let Some(resolved) = resolved {
                    legal[index] = true;
                    placement[index] = resolved;
                }
            }
        }
        (legal, placement)
    }

    #[test]
    fn sparse_placements_and_samples_match_dense_semantics_bit_for_bit() {
        let position = narrow_sliver_position();
        let width = 32;
        let height = 32;
        let (expected_legal, expected_placement) =
            dense_placement_reference(&position, width, height);
        let mut grid = FineGrid::build(&position, width, height, 4, |row, col| {
            (row * width + col) as f32 * 0.03125 - 7.0
        });

        assert_eq!(grid.unpacked_legal(), expected_legal);
        let mut expected_overrides = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let index = row * width + col;
                let expected_logit = if expected_legal[index] {
                    index as f32 * 0.03125 - 7.0
                } else {
                    f32::NEG_INFINITY
                };
                assert_eq!(
                    grid.logits[index].to_bits(),
                    expected_logit.to_bits(),
                    "logit changed at cell ({row},{col})"
                );
                if !expected_legal[index] {
                    continue;
                }
                let expected = expected_placement[index];
                let actual = grid.placement_at(row, col);
                assert_eq!(
                    (actual.x.to_bits(), actual.y.to_bits()),
                    (expected.x.to_bits(), expected.y.to_bits()),
                    "placement changed at cell ({row},{col})"
                );
                let centre = cell_center(row, col, width, height);
                if (expected.x.to_bits(), expected.y.to_bits())
                    != (centre.x.to_bits(), centre.y.to_bits())
                {
                    expected_overrides.push((index, expected));
                }
            }
        }
        assert!(
            !expected_overrides.is_empty(),
            "fixture must exercise snapped legal cells"
        );
        assert_eq!(grid.placement_overrides.len(), expected_overrides.len());
        for (&(actual_index, actual), &(expected_index, expected)) in
            grid.placement_overrides.iter().zip(&expected_overrides)
        {
            assert_eq!(actual_index, expected_index);
            assert_eq!(
                (actual.x.to_bits(), actual.y.to_bits()),
                (expected.x.to_bits(), expected.y.to_bits())
            );
        }
        assert!(
            grid.placement_overrides
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0),
            "placement overrides must stay in row-major order"
        );

        // Force sampling through one snapped sliver cell, then compare the
        // complete sampled output with the former dense placement lookup.
        let snapped_index = expected_overrides[0].0;
        for (index, logit) in Arc::make_mut(&mut grid.logits).iter_mut().enumerate() {
            if grid.legal[index / 64] & (1u64 << (index % 64)) != 0 {
                *logit = -20.0;
            }
        }
        Arc::make_mut(&mut grid.logits)[snapped_index] = 20.0;
        let random_values = vec![0.13, 0.73, 0.41, 0.89, 0.27, 0.61];
        let expected =
            sample_candidates_with(&grid, 64, 0.0, stream(random_values.clone()), |row, col| {
                expected_placement[row * width + col]
            });
        let actual = sample_candidates(&grid, 64, stream(random_values));
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(
                (actual.point.x.to_bits(), actual.point.y.to_bits()),
                (expected.point.x.to_bits(), expected.point.y.to_bits())
            );
            assert_eq!(actual.beta.to_bits(), expected.beta.to_bits());
        }
        let snapped = expected_placement[snapped_index];
        assert!(actual.iter().all(|sample| {
            (sample.point.x.to_bits(), sample.point.y.to_bits())
                == (snapped.x.to_bits(), snapped.y.to_bits())
        }));
    }

    #[test]
    fn beta_is_a_valid_factored_probability() {
        // With uniform logits the coarse maxes are equal so P_coarse = 1/(#legal
        // coarse cells) and within a cell P_fine = 1/(#legal fine cells there).
        // We don't hardcode the counts (edge cells are illegal on a bordered
        // board); instead we check beta = P_coarse * P_fine holds by deriving the
        // two factors from the grid's own legality, and that every beta is a valid
        // probability in (0, 1].
        let width = 12;
        let height = 12;
        let coarse = 4; // 3x3 coarse cells
        let grid = FineGrid::build(&empty_board(), width, height, coarse, |_, _| 0.0);

        // Reconstruct the uniform-logit stage probabilities independently.
        let (cw, ch) = grid.coarse_dims();
        let legal_coarse = (0..ch)
            .flat_map(|cr| (0..cw).map(move |cc| (cr, cc)))
            .filter(|&(cr, cc)| grid.coarse_max(cr, cc).is_some())
            .count();
        let p_coarse = 1.0 / legal_coarse as f64;

        let samples = sample_candidates(&grid, 50, stream(vec![0.05, 0.27, 0.51, 0.73, 0.95]));
        assert_eq!(samples.len(), 50);
        for s in &samples {
            assert!(
                s.beta > 0.0 && s.beta <= 1.0,
                "beta {} out of range",
                s.beta
            );
            // Recover which coarse cell the point fell in and its legal-fine count.
            let col = (s.point.x * width as f64).floor() as usize;
            let row = (s.point.y * height as f64).floor() as usize;
            let (crow, ccol) = (row / coarse, col / coarse);
            let r0 = crow * coarse;
            let c0 = ccol * coarse;
            let mut legal_fine = 0;
            for r in r0..(r0 + coarse).min(height) {
                for c in c0..(c0 + coarse).min(width) {
                    if grid.legal_at(r * width + c) {
                        legal_fine += 1;
                    }
                }
            }
            let p_fine = 1.0 / legal_fine as f64;
            assert!(
                (s.beta - p_coarse * p_fine).abs() < 1e-9,
                "beta {} != P_coarse {} * P_fine {}",
                s.beta,
                p_coarse,
                p_fine
            );
        }
    }

    #[test]
    fn a_sharp_peak_dominates_the_samples() {
        // One coarse region has a huge logit; nearly every draw should land there.
        let width = 8;
        let height = 8;
        let coarse = 4;
        let peak_row = 1;
        let peak_col = 1; // inside coarse cell (0,0)
        let grid = FineGrid::build(&empty_board(), width, height, coarse, |r, c| {
            if r == peak_row && c == peak_col {
                20.0
            } else {
                0.0
            }
        });
        let mut hits = 0;
        let mut rng = {
            // pseudo-random stream via a small LCG so the test is deterministic
            let mut state = 0x2545_f491_4f6c_dd1d_u64;
            move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 11) as f64 / (1u64 << 53) as f64
            }
        };
        let n = 500;
        for _ in 0..n {
            let s = sample_candidates(&grid, 1, &mut rng);
            let p = s[0].point;
            let col = (p.x * width as f64).floor() as usize;
            let row = (p.y * height as f64).floor() as usize;
            if row == peak_row && col == peak_col {
                hits += 1;
            }
        }
        assert!(hits as f64 / n as f64 > 0.8, "peak got only {hits}/{n}");
    }

    #[test]
    fn coarse_cells_partition_distinct_regions() {
        // Two separated peaks in different coarse cells should both be reachable,
        // and each sample's point must lie in exactly one coarse cell.
        let width = 8;
        let height = 8;
        let coarse = 4; // 2x2 coarse cells
        let grid = FineGrid::build(&empty_board(), width, height, coarse, |r, c| {
            if (r, c) == (1, 1) || (r, c) == (5, 5) {
                10.0
            } else {
                0.0
            }
        });
        let mut seen_regions = std::collections::HashSet::new();
        let mut rng = {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 11) as f64 / (1u64 << 53) as f64
            }
        };
        for _ in 0..200 {
            let s = sample_candidates(&grid, 1, &mut rng);
            let p = s[0].point;
            let col = (p.x * width as f64).floor() as usize / coarse;
            let row = (p.y * height as f64).floor() as usize / coarse;
            seen_regions.insert((row, col));
        }
        // both peak regions (0,0) and (1,1) should appear
        assert!(seen_regions.contains(&(0, 0)));
        assert!(seen_regions.contains(&(1, 1)));
    }

    #[test]
    fn no_legal_cells_yields_no_candidates() {
        // A board where nothing is legal (simulate by an always-illegal predicate
        // isn't available; instead use a 1x1 grid whose only cell is legal to show
        // the happy path returns something, and rely on empty coarse handling).
        let grid = FineGrid::build(&empty_board(), 4, 4, 2, |_, _| 0.0);
        let s = sample_candidates(&grid, 3, stream(vec![0.5]));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn a_legal_sliver_narrower_than_a_cell_stays_playable() {
        // Two stones leaving a legal gap between them that is thinner than one
        // grid cell, so no cell centre lands inside it. Before projection the
        // sampler could not name any point there, which is what let a group
        // with its last liberty in such a gap survive indefinitely.
        let position = narrow_sliver_position();
        assert!(
            is_legal_placement(&position, 0.5, 0.5),
            "the gap between the stones must be legal"
        );

        let width = 32;
        let height = 32;
        let grid = FineGrid::build(&position, width, height, 4, |_, _| 0.0);

        // Some cell overlapping the gap is now playable, and it places inside
        // the gap rather than at its own illegal centre.
        let row = (0.5 * height as f64) as usize;
        let mut reached = false;
        for col in 0..width {
            let idx = row * width + col;
            if !grid.legal_at(idx) {
                continue;
            }
            let placement = grid.placement_at(row, col);
            if is_legal_placement(&position, placement.x, placement.y)
                && (placement.y - 0.5).abs() < 0.05
                && (placement.x - 0.5).abs() < 0.05
            {
                reached = true;
                break;
            }
        }
        assert!(reached, "the sliver between the stones must be reachable");
    }

    #[test]
    fn projection_never_places_outside_the_cell_that_named_it() {
        // A cell stands in for its own logit, so it must not resolve to a
        // placement elsewhere on the board: that would let the sampler reach a
        // point whose logit it never read.
        let position = Position::new(
            1.0 / 18.0,
            vec![
                Stone::new(0.30, 0.30, Color::Black),
                Stone::new(0.55, 0.42, Color::White),
                Stone::new(0.70, 0.65, Color::Black),
            ],
            Color::White,
        );
        let width = 24;
        let height = 24;
        let grid = FineGrid::build(&position, width, height, 4, |_, _| 0.0);
        let half_x = 0.5 / width as f64;
        let half_y = 0.5 / height as f64;
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                if !grid.legal_at(idx) {
                    continue;
                }
                let centre = cell_center(row, col, width, height);
                let placement = grid.placement_at(row, col);
                assert!(
                    (placement.x - centre.x).abs() <= half_x + 1.0e-12
                        && (placement.y - centre.y).abs() <= half_y + 1.0e-12,
                    "cell ({row},{col}) placed outside itself"
                );
                assert!(
                    is_legal_placement(&position, placement.x, placement.y),
                    "cell ({row},{col}) marked playable but places illegally"
                );
            }
        }
    }
}

#[cfg(test)]
mod noise_tests {
    use super::*;

    #[test]
    fn mixing_uniform_preserves_normalisation() {
        for noise in [0.0, 0.1, 0.25, 0.9] {
            let mut probs = vec![0.7, 0.2, 0.09, 0.01];
            mix_uniform(&mut probs, noise);
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "noise {noise} left the mass at {total}"
            );
        }
    }

    #[test]
    fn mixing_floors_the_tail_without_reordering() {
        // The point of the noise is that a cell the policy has written off
        // still gets proposed sometimes -- and that it does not overturn the
        // policy's ranking while doing so.
        let mut probs = vec![0.97, 0.02, 0.009, 0.001];
        let before = probs.clone();
        mix_uniform(&mut probs, 0.25);
        assert!(
            probs[3] > before[3] * 10.0,
            "the tail should be lifted substantially, {} -> {}",
            before[3],
            probs[3]
        );
        for pair in probs.windows(2) {
            assert!(pair[0] > pair[1], "the policy's ordering must survive");
        }
    }

    #[test]
    fn zero_noise_is_exactly_the_policy() {
        let mut probs = vec![0.5, 0.3, 0.2];
        let before = probs.clone();
        mix_uniform(&mut probs, 0.0);
        assert_eq!(probs, before, "the default must not perturb anything");
    }
}
