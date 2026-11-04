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

/// Ends a one-sided pass streak at random, then awards the game by area.
///
/// Once a side has passed `after` times running -- its own turns, with the
/// opponent playing in between, since a pass answered by a pass already ends
/// the game -- each further pass by that side ends the game with
/// `probability`. A bootstrap for the stage where the value head cannot yet
/// support resignation: in the first fresh run 72% of big-board games with a
/// long streak ran to the ply cap while the side behind repeated moves that
/// changed nothing, and each was then scored as "who leads at the cap".
///
/// Neither the model nor the search sees it. It decides only where a played
/// game stops, like the cap it mostly replaces, so turning it off later is a
/// setting. The draw comes from the game seed and the ply, so a game replays
/// identically.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomStop {
    pub after: u32,
    pub probability: f64,
}

impl RandomStop {
    /// Whether a pass at `ply`, the side's `streak`-th running, ends the game.
    #[must_use]
    pub fn fires(&self, seed: u64, ply: u32, streak: u32) -> bool {
        streak > self.after && unit_draw(seed, ply) < self.probability
    }
}

/// A uniform draw in [0, 1) from a seed and a ply: SplitMix64's finalizer.
fn unit_draw(seed: u64, ply: u32) -> f64 {
    let mut z = seed ^ (u64::from(ply).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

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
    /// The game ended on a [`RandomStop`]; `outcome` is then the area award.
    pub random_stop: bool,
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
    random_stop: Option<(RandomStop, u64)>,
    mut search: impl FnMut(&Position, u32) -> Result<SearchResult, E>,
    mut observe: impl FnMut(PlayoutStep<'_>),
) -> Result<PlayoutReport, E> {
    assert!(maximum_plies > 0, "maximum plies must be positive");
    assert_eq!(initial.phase(), Phase::Playing, "game must start playable");

    let mut position = initial;
    let mut stats = PlayoutStats::default();
    // Running passes by each side, indexed by `Color::Black == 0`.
    let mut streaks = [0_u32; 2];

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
        let side = usize::from(position.to_move() != Color::Black);
        streaks[side] = if action == Action::Pass {
            streaks[side] + 1
        } else {
            0
        };

        position = transition.position;
        if position.phase() == Phase::Finished {
            return Ok(PlayoutReport {
                final_position: position,
                outcome: Some(transition.analysis.outcome),
                stats,
                random_stop: false,
            });
        }
        if let Some((rule, seed)) = random_stop
            && action == Action::Pass
            && rule.fires(seed, ply, streaks[side])
        {
            let outcome = award_by_area(&position);
            return Ok(PlayoutReport {
                final_position: position,
                outcome: Some(outcome),
                stats,
                random_stop: true,
            });
        }
    }

    Ok(PlayoutReport {
        final_position: position,
        outcome: None,
        stats,
        random_stop: false,
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

    use super::{RandomStop, play_game};
    use vgo_core::{Color, Point, Position};
    use vgo_search::{
        Action, CandidateSource, ChildSummary, SearchConfig, SearchResult, SearchStats, search,
    };

    fn chosen(action: Action, to_move: Color) -> SearchResult {
        let child = ChildSummary {
            action,
            source: CandidateSource::AreaSequence,
            prior: 1.0,
            visits: 1,
            black_value: 0.0,
            proposal_count: 0,
            beta: None,
        };
        SearchResult::from_children(action, vec![child], SearchStats::default(), to_move)
    }

    /// Black places on a spread grid, White always passes: a one-sided streak.
    fn streak_game(rule: Option<(RandomStop, u64)>, cap: u32) -> super::PlayoutReport {
        play_game(
            Position::new(1.0 / 38.0, Vec::new(), Color::Black),
            cap,
            rule,
            |position, ply| {
                let action = if position.to_move() == Color::Black {
                    let index = f64::from(ply / 2);
                    Action::Place(Point {
                        x: 0.05 + 0.1 * (index % 9.0),
                        y: 0.05 + 0.1 * (index / 9.0).floor(),
                    })
                } else {
                    Action::Pass
                };
                Ok::<_, std::convert::Infallible>(chosen(action, position.to_move()))
            },
            |_| {},
        )
        .unwrap()
    }

    #[test]
    fn a_certain_stop_ends_on_the_pass_after_the_threshold() {
        // White passes at plies 1, 3, 5, 7, 9: the fifth is the first past four.
        let report = streak_game(
            Some((
                RandomStop {
                    after: 4,
                    probability: 1.0,
                },
                7,
            )),
            60,
        );
        assert!(report.random_stop);
        assert_eq!(report.stats.plies, 10);
        assert!(report.outcome.is_some());
        assert_eq!(report.stats.passes, 5);
    }

    #[test]
    fn no_rule_or_zero_probability_runs_to_the_cap() {
        for rule in [
            None,
            Some((
                RandomStop {
                    after: 4,
                    probability: 0.0,
                },
                7,
            )),
        ] {
            let report = streak_game(rule, 40);
            assert!(!report.random_stop);
            assert!(report.outcome.is_none());
            assert_eq!(report.stats.plies, 40);
        }
    }

    #[test]
    fn stops_are_seeded_and_average_one_over_the_probability() {
        let rule = RandomStop {
            after: 4,
            probability: 0.25,
        };
        let lengths: Vec<u32> = (0..400)
            .map(|seed| streak_game(Some((rule, seed)), 400).stats.passes as u32)
            .collect();
        assert_eq!(
            lengths[3],
            streak_game(Some((rule, 3)), 400).stats.passes as u32
        );
        // Passes beyond the threshold are geometric with mean 1 / 0.25 = 4.
        let mean = lengths.iter().map(|&n| f64::from(n - 4)).sum::<f64>() / 400.0;
        assert!((3.4..4.6).contains(&mean), "mean extra passes {mean}");
        assert!(lengths.iter().all(|&n| n >= 5));
    }

    #[test]
    fn short_playout_is_bounded_and_accumulates_search() {
        let report = play_game(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            4,
            None,
            |position, _| Ok::<_, Infallible>(search(position, SearchConfig::canary(2), 1)),
            |_| {},
        )
        .unwrap();
        assert!(report.stats.plies <= 4);
        assert_eq!(report.stats.search.simulations, report.stats.plies * 2);
    }
}
