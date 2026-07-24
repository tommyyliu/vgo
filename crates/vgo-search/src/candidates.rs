use std::collections::HashSet;

use vgo_core::{Point, Position, is_legal_placement, legal_set_vertices, pass, place};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Pass,
    Place(Point),
}

impl Action {
    pub fn apply(self, position: &Position) -> vgo_core::MoveResult {
        match self {
            Self::Pass => pass(position).expect("pass candidate must be legal"),
            Self::Place(point) => {
                place(position, point.x, point.y).expect("placement candidate must be legal")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateSource {
    Pass,
    LegalVertex,
    AreaSequence,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub action: Action,
    pub source: CandidateSource,
}

pub struct CandidateSequence {
    position: Position,
    anchors: Vec<Point>,
    anchor_index: usize,
    area_index: u64,
    shift_x: f64,
    shift_y: f64,
    emitted_pass: bool,
    seen: HashSet<(i64, i64)>,
}

impl CandidateSequence {
    #[must_use]
    pub fn new(position: &Position, match_seed: u64) -> Self {
        let mut anchors = legal_set_vertices(position);
        anchors.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
        let state_seed = splitmix64(match_seed ^ position_hash(position));
        Self {
            position: position.clone(),
            anchors,
            anchor_index: 0,
            area_index: 1,
            shift_x: unit_f64(splitmix64(state_seed)),
            shift_y: unit_f64(splitmix64(state_seed ^ 0x9e37_79b9_7f4a_7c15)),
            emitted_pass: false,
            seen: HashSet::new(),
        }
    }

    pub fn next_candidate(&mut self) -> Option<Candidate> {
        if !self.emitted_pass {
            self.emitted_pass = true;
            return Some(Candidate {
                action: Action::Pass,
                source: CandidateSource::Pass,
            });
        }

        while let Some(point) = self.anchors.get(self.anchor_index).copied() {
            self.anchor_index += 1;
            if self.mark_new(point) {
                return Some(Candidate {
                    action: Action::Place(point),
                    source: CandidateSource::LegalVertex,
                });
            }
        }

        let radius = self.position.radius();
        let width = 1.0 - 2.0 * radius;
        for _ in 0..100_000 {
            let index = self.area_index;
            self.area_index += 1;
            let unit_x = (radical_inverse(index, 2) + self.shift_x).fract();
            let unit_y = (radical_inverse(index, 3) + self.shift_y).fract();
            let point = Point::new(radius + width * unit_x, radius + width * unit_y);
            if is_legal_placement(&self.position, point.x, point.y) && self.mark_new(point) {
                return Some(Candidate {
                    action: Action::Place(point),
                    source: CandidateSource::AreaSequence,
                });
            }
        }
        None
    }

    fn mark_new(&mut self, point: Point) -> bool {
        self.seen.insert((
            (point.x * 1.0e12).round() as i64,
            (point.y * 1.0e12).round() as i64,
        ))
    }
}

#[must_use]
pub fn generate_candidates(position: &Position, budget: usize, match_seed: u64) -> Vec<Candidate> {
    let mut sequence = CandidateSequence::new(position, match_seed);
    std::iter::from_fn(|| sequence.next_candidate())
        .take(budget)
        .collect()
}

fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let inverse_base = 1.0 / base as f64;
    let mut factor = inverse_base;
    let mut value = 0.0;
    while index > 0 {
        value += (index % base) as f64 * factor;
        index /= base;
        factor *= inverse_base;
    }
    value
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn unit_f64(value: u64) -> f64 {
    (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

fn hash_word(mut hash: u64, word: u64) -> u64 {
    for byte in word.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(crate) fn position_hash(position: &Position) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    hash = hash_word(hash, position.radius().to_bits());
    hash = hash_word(hash, u64::from(position.consecutive_passes()));
    let mut stones = position
        .stones()
        .iter()
        .map(|stone| {
            (
                stone.x.to_bits(),
                stone.y.to_bits(),
                u64::from(stone.color == position.to_move()),
            )
        })
        .collect::<Vec<_>>();
    stones.sort_unstable();
    for (x, y, relative_color) in stones {
        hash = hash_word(hash, x);
        hash = hash_word(hash, y);
        hash = hash_word(hash, relative_color);
    }
    hash
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Position, Stone, is_legal_placement};

    use super::{Action, CandidateSource, generate_candidates};

    #[test]
    fn larger_budgets_extend_the_same_candidate_prefix() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let small = generate_candidates(&position, 10, 42);
        let large = generate_candidates(&position, 100, 42);
        assert_eq!(small, large[..small.len()]);
    }

    #[test]
    fn every_generated_placement_is_legal() {
        let position = Position::new(1.0 / 6.0, Vec::new(), Color::Black);
        let candidates = generate_candidates(&position, 100, 7);
        assert_eq!(candidates[0].source, CandidateSource::Pass);
        for candidate in candidates {
            if let Action::Place(point) = candidate.action {
                assert!(is_legal_placement(&position, point.x, point.y));
            }
        }
    }

    #[test]
    fn candidate_prefix_ignores_storage_order_and_absolute_color() {
        let stones = vec![
            Stone::new(0.25, 0.25, Color::Black),
            Stone::new(0.75, 0.75, Color::White),
        ];
        let mut reversed = stones.clone();
        reversed.reverse();
        let swapped = stones
            .iter()
            .map(|stone| Stone::new(stone.x, stone.y, stone.color.other()))
            .collect();
        let first = Position::new(0.1, stones, Color::Black);
        let reordered = Position::new(0.1, reversed, Color::Black);
        let color_swapped = Position::new(0.1, swapped, Color::White);

        assert_eq!(
            generate_candidates(&first, 20, 11),
            generate_candidates(&reordered, 20, 11)
        );
        assert_eq!(
            generate_candidates(&first, 20, 11),
            generate_candidates(&color_swapped, 20, 11)
        );
    }
}
