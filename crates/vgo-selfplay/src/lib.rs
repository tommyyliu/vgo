#![forbid(unsafe_code)]

pub mod generation;
/// The replay record format, shared by every generator.
///
/// Lived inside `generate_demo` when there was one generator. A second one that
/// wrote its own records would be a second format the moment either changed,
/// and the readers -- the Python loader and `render_shard` -- already have to
/// agree with this file exactly.
pub mod replay_stream;
pub mod tail;
pub mod telemetry;

use vgo_core::{Analysis, Color, GameEvent, Outcome, Phase, Position};

/// Awards a truncated game on its final position alone.
///
/// [`adjudicate_positions`] can refuse -- on a margin, or on the lead changing
/// inside its window -- and a refused game is discarded. That is the right
/// trade for self-play, where a mislabelled game teaches the wrong thing and
/// throwing one away costs only its samples.
///
/// It is the wrong trade for rating. A refused arena game is not a discarded
/// sample, it is a missing result: twelve games between two close models
/// produced three usable outcomes, because nine ended inside the 0.10 margin.
/// Rating two similar models is exactly the case where the margin rejects
/// everything, and a run whose komi balances the game makes that the norm
/// rather than the exception.
///
/// Voronoi area is continuous and totals 1.0, so an exact tie is measure-zero.
/// The final position is enough to name a winner, and a hundred-odd plies is
/// far past the point where the board is decided.
#[must_use]
pub fn award_by_area(position: &Position) -> Outcome {
    let analysis = Analysis::new(position);
    let delta = analysis.score.black - analysis.score.white - position.komi();
    Outcome {
        winner: if delta.abs() <= f64::EPSILON {
            None
        } else if delta > 0.0 {
            Some(Color::Black)
        } else {
            Some(Color::White)
        },
        margin: delta.abs(),
    }
}

use vgo_search::{Action, SearchResult, SearchStats};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlayoutStats {
    pub plies: u32,
    pub captures: u64,
    pub self_captures: u64,
    pub passes: u64,
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
    let mut stats = PlayoutStats::default();

    for ply in 0..maximum_plies {
        let result = search(&position, ply)?;
        accumulate_search_stats(&mut stats.search, result.stats);
        let (action, transition) = result
            .actions_by_preference(position.to_move())
            .into_iter()
            .next()
            .map_or_else(
                || {
                    let action = Action::Pass;
                    (action, action.apply(&position))
                },
                |action| {
                    let transition = action.apply(&position);
                    (action, transition)
                },
            );
        observe(PlayoutStep {
            ply,
            position: &position,
            search: &result,
            action,
            transition: &transition,
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

        position = transition.position;
        if position.phase() == Phase::Finished {
            return Ok(PlayoutReport {
                final_position: position,
                outcome: Some(transition.analysis.outcome),
                stats,
            });
        }
    }

    Ok(PlayoutReport {
        final_position: position,
        outcome: None,
        stats,
    })
}

pub(crate) fn accumulate_search_stats(total: &mut SearchStats, next: SearchStats) {
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

    use super::play_game;
    use vgo_core::{Color, Position};
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
}
