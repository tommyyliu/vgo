#![forbid(unsafe_code)]

use std::collections::HashSet;

use vgo_core::{Color, GameEvent, Outcome, Phase, Position};
use vgo_search::{Action, SearchResult, SearchStats};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlayoutStats {
    pub plies: u32,
    pub captures: u64,
    pub self_captures: u64,
    pub passes: u64,
    pub repetitions: u64,
    pub repetition_avoids: u64,
    pub search: SearchStats,
}

#[derive(Clone, Debug)]
pub struct PlayoutReport {
    pub final_position: Position,
    pub outcome: Option<Outcome>,
    pub stats: PlayoutStats,
}

impl PlayoutReport {
    #[must_use]
    pub const fn completed(&self) -> bool {
        self.outcome.is_some()
    }
}

pub struct PlayoutStep<'a> {
    pub ply: u32,
    pub position: &'a Position,
    pub search: &'a SearchResult,
    pub action: Action,
    pub transition: &'a vgo_core::MoveResult,
    pub repetition_avoids: u32,
}

pub fn play_game<E>(
    initial: Position,
    maximum_plies: u32,
    mut search: impl FnMut(&Position, u32) -> Result<SearchResult, E>,
    mut observe: impl FnMut(PlayoutStep<'_>),
) -> Result<PlayoutReport, E> {
    assert!(maximum_plies > 0, "maximum plies must be positive");
    assert_eq!(initial.phase(), Phase::Playing, "game must start playable");

    let mut position = initial;
    let mut seen = HashSet::new();
    let mut stats = PlayoutStats::default();
    seen.insert(position_fingerprint(&position));

    for ply in 0..maximum_plies {
        let result = search(&position, ply)?;
        accumulate_search_stats(&mut stats.search, result.stats);
        let mut selected = None;
        let mut repetition_avoids = 0_u32;
        for action in result.actions_by_preference(position.to_move()) {
            let transition = action.apply(&position);
            if transition.position.phase() == Phase::Finished
                || !seen.contains(&position_fingerprint(&transition.position))
            {
                selected = Some((action, transition));
                break;
            }
            repetition_avoids += 1;
        }
        let (action, transition) = selected.unwrap_or_else(|| {
            let action = Action::Pass;
            (action, action.apply(&position))
        });
        observe(PlayoutStep {
            ply,
            position: &position,
            search: &result,
            action,
            transition: &transition,
            repetition_avoids,
        });

        stats.plies = ply + 1;
        stats.captures += transition.captured as u64;
        stats.self_captures += transition
            .events
            .iter()
            .filter_map(|event| match event {
                GameEvent::SelfCapture { count, .. } => Some(*count as u64),
                _ => None,
            })
            .sum::<u64>();
        stats.passes += u64::from(action == Action::Pass);
        stats.repetition_avoids += u64::from(repetition_avoids);

        position = transition.position;
        if position.phase() == Phase::Finished {
            return Ok(PlayoutReport {
                final_position: position,
                outcome: Some(transition.analysis.outcome),
                stats,
            });
        }
        let inserted = seen.insert(position_fingerprint(&position));
        stats.repetitions += u64::from(!inserted);
    }

    Ok(PlayoutReport {
        final_position: position,
        outcome: None,
        stats,
    })
}

#[must_use]
pub fn position_fingerprint(position: &Position) -> u64 {
    let mut hash = hash_word(0xcbf2_9ce4_8422_2325, position.radius().to_bits());
    hash = hash_word(hash, u64::from(position.consecutive_passes()));
    hash = hash_word(hash, u64::from(position.to_move() == Color::White));
    hash = hash_word(hash, u64::from(position.phase() == Phase::Finished));
    let mut stones = position
        .stones()
        .iter()
        .map(|stone| {
            (
                stone.x.to_bits(),
                stone.y.to_bits(),
                u64::from(stone.color == Color::White),
            )
        })
        .collect::<Vec<_>>();
    stones.sort_unstable();
    for (x, y, color) in stones {
        hash = hash_word(hash, x);
        hash = hash_word(hash, y);
        hash = hash_word(hash, color);
    }
    hash
}

fn hash_word(mut hash: u64, word: u64) -> u64 {
    for byte in word.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn accumulate_search_stats(total: &mut SearchStats, next: SearchStats) {
    total.simulations += next.simulations;
    total.evaluations += next.evaluations;
    total.expanded_nodes += next.expanded_nodes;
    total.generated_candidates += next.generated_candidates;
    total.terminal_leaves += next.terminal_leaves;
    total.depth_limited_leaves += next.depth_limited_leaves;
    total.maximum_depth = total.maximum_depth.max(next.maximum_depth);
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::{play_game, position_fingerprint};
    use vgo_core::{Color, Position, Stone};
    use vgo_search::{SearchConfig, search};

    #[test]
    fn short_playout_is_bounded_and_accumulates_search() {
        let report = play_game(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            4,
            |position, _| Ok::<_, Infallible>(search(position, SearchConfig::canary(2), 1)),
            |_| {},
        )
        .unwrap();
        assert!(report.stats.plies <= 4);
        assert_eq!(report.stats.search.simulations, report.stats.plies * 2);
    }

    #[test]
    fn fingerprint_is_order_independent_but_color_absolute() {
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
            position_fingerprint(&first),
            position_fingerprint(&reordered)
        );
        assert_ne!(
            position_fingerprint(&first),
            position_fingerprint(&color_swapped)
        );
    }
}
