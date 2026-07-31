#![forbid(unsafe_code)]

use std::collections::HashSet;

pub mod render_svg;

use vgo_core::{Analysis, Color, GameEvent, Outcome, Phase, Position};

/// Closing plies whose area leader must agree before a truncated game is
/// awarded.
///
/// Short, because the area leader is not a noisy quantity: measured on the
/// three longest games in a real shard it settled by ply 10, 44, and 48 and
/// never changed again across the remaining 180-240 plies. A handful of plies
/// is enough to reject a position genuinely oscillating around even.
pub const ADJUDICATION_PLIES: usize = 8;

/// How far apart the two areas must be, as a fraction of the whole board.
///
/// Those same games sat at margins of 0.49-0.80 from their halfway point on, so
/// this rejects only positions that really are close. The board totals 1.0.
pub const ADJUDICATION_MARGIN: f64 = 0.10;

/// Awards a truncated game to whoever holds the board.
///
/// A game that runs out of plies has no played-out result, but by that point
/// the territory is usually long settled: the late game is capture-and-replace
/// churn in a contested pocket, not development. Across a shard the average
/// game gains 7.4 stones over its entire second half while producing 65 stone
/// changes -- nine changes per net stone -- so the area has stopped moving well
/// before the cap.
///
/// This scores the position the same way a finished game is scored, by Voronoi
/// area, rather than asking the network. The value head agrees with outcomes
/// only ~76% of the time; the area is ground truth under the same rule that
/// decides a real result.
///
/// Returns `None` -- leaving the game undecided -- unless the same player leads
/// by at least `margin` on every one of the closing `plies`.
///
/// Shared with the arena rather than living in the generator. An arena that
/// discards what self-play adjudicates does not merely lose games, it loses
/// them selectively: 69% of arena games were dropped at a 100-ply cap, and the
/// survivors are the ones that happened to resolve quickly, which is a property
/// of playing style rather than of strength.
#[must_use]
pub fn adjudicate_positions(
    positions: &[Position],
    plies: usize,
    margin: f64,
) -> Option<Outcome> {
    if positions.len() < plies {
        return None;
    }
    let mut leader: Option<Color> = None;
    for position in &positions[positions.len() - plies..] {
        let analysis = Analysis::new(position);
        let delta = analysis.score.black - analysis.score.white;
        if delta.abs() < margin {
            return None;
        }
        let ply_leader = if delta > 0.0 {
            Color::Black
        } else {
            Color::White
        };
        match leader {
            None => leader = Some(ply_leader),
            Some(current) if current == ply_leader => {}
            // The lead changed hands inside the window: not settled.
            Some(_) => return None,
        }
    }
    leader.map(|winner| Outcome {
        winner: Some(winner),
        // No final count was played out, so there is no margin to report.
        margin: 0.0,
    })
}
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

/// When a playout may concede rather than play a decided game to the end.
///
/// A single ply's root value is noisy -- the value head saturates past |v|>0.99
/// on about a third of positions while agreeing with the outcome only ~76% of
/// the time -- so one bad evaluation must not end a game. The rule instead
/// requires the *least* confident evaluation over a window of consecutive plies
/// to stay past the threshold: every ply in the window has to agree, which is
/// far harder to satisfy by noise than any single one.
///
/// `disable_fraction` leaves that many games unresigned, played to a real
/// finish. Those are the only games that can measure the rule's false-positive
/// rate -- how often it concedes a game the resigning side would have won --
/// and the threshold is meant to be calibrated from them, not guessed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResignRule {
    /// Concede when the mover's value stays at or below `-threshold`.
    pub threshold: f64,
    /// Consecutive plies that must all agree before conceding.
    pub window: u32,
    /// Fraction of games exempted, for calibration.
    pub disable_fraction: f64,
}

impl ResignRule {
    /// A rule that never fires.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            threshold: 1.0,
            window: u32::MAX,
            disable_fraction: 0.0,
        }
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.window != u32::MAX && self.threshold < 1.0
    }
}

#[derive(Clone, Debug)]
pub struct PlayoutReport {
    pub final_position: Position,
    pub outcome: Option<Outcome>,
    pub stats: PlayoutStats,
    /// Set when the game ended by resignation rather than by play. The outcome
    /// is still real -- the conceding side loses -- so these samples train
    /// normally; this only records how the result was reached.
    pub resigned: bool,
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
    search: impl FnMut(&Position, u32) -> Result<SearchResult, E>,
    observe: impl FnMut(PlayoutStep<'_>),
) -> Result<PlayoutReport, E> {
    play_game_with_resignation(initial, maximum_plies, ResignRule::disabled(), search, observe)
}

/// [`play_game`], with the option to concede a decided game.
pub fn play_game_with_resignation<E>(
    initial: Position,
    maximum_plies: u32,
    resign: ResignRule,
    mut search: impl FnMut(&Position, u32) -> Result<SearchResult, E>,
    mut observe: impl FnMut(PlayoutStep<'_>),
) -> Result<PlayoutReport, E> {
    assert!(maximum_plies > 0, "maximum plies must be positive");
    assert_eq!(initial.phase(), Phase::Playing, "game must start playable");

    let mut position = initial;
    let mut seen = HashSet::new();
    let mut stats = PlayoutStats::default();
    seen.insert(position_fingerprint(&position));
    // Consecutive *own* turns whose root value has stayed past the threshold,
    // counted per seat. Reset by any of that seat's turns that does not, so the
    // window measures the least confident evaluation in a run rather than the
    // most confident.
    //
    // One shared counter cannot work: `mover_value` is relative to the side to
    // move, so a decided game alternates -1, +1, -1, +1 as the seat changes, and
    // the loser's increment is undone by the winner's reset on the very next
    // ply. The streak never exceeds 1 and no threshold ever fires -- measured
    // on ddrnet-vs shard 4, where games sat 80 plies past the point of decision
    // and resign_calibration reported zero firings at every threshold down to
    // 0.7. Indexed by seat, a window of 5 means five consecutive turns of one's
    // own, which is the ten plies of game the rule was written to describe.
    let mut losing_streak = [0_u32; 2];

    for ply in 0..maximum_plies {
        let result = search(&position, ply)?;
        accumulate_search_stats(&mut stats.search, result.stats);

        if resign.is_enabled() {
            // Root value is Black-relative; flip it to the mover's view so the
            // same threshold means the same thing for either side.
            let mover_value = if position.to_move() == Color::Black {
                result.root_black_value()
            } else {
                -result.root_black_value()
            };
            let seat = usize::from(position.to_move() == Color::White);
            if mover_value <= -resign.threshold {
                losing_streak[seat] += 1;
            } else {
                losing_streak[seat] = 0;
            }
            if losing_streak[seat] >= resign.window {
                // The mover concedes, so the opponent wins. This is a real
                // result, not a truncation: the samples carry a genuine outcome
                // and train exactly as a played-out game would.
                let winner = position.to_move().other();
                stats.plies = ply + 1;
                return Ok(PlayoutReport {
                    final_position: position,
                    // Margin zero: a conceded game has no area result to
                    // report, and black_utility only reads `winner`.
                    outcome: Some(Outcome {
                        winner: Some(winner),
                        margin: 0.0,
                    }),
                    stats,
                    resigned: true,
                });
            }
        }
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
                resigned: false,
            });
        }
        let inserted = seen.insert(position_fingerprint(&position));
        stats.repetitions += u64::from(!inserted);
    }

    Ok(PlayoutReport {
        final_position: position,
        outcome: None,
        stats,
        resigned: false,
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

    use super::{ResignRule, play_game, play_game_with_resignation, position_fingerprint};
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

    /// A search result whose root value is fixed, for driving the resign rule.
    fn fixed_value(position: &vgo_core::Position, black_value: f64) -> vgo_search::SearchResult {
        use vgo_search::{Action, CandidateSource, ChildSummary, SearchResult, SearchStats};
        // Pass is always legal and needs no geometry. These tests exercise the
        // resign rule, which reads only the root value, so the action is
        // immaterial -- using Pass keeps the fixture from depending on where a
        // legal placement happens to be.
        let _ = position;
        let action = Action::Pass;
        SearchResult::from_children(
            action,
            vec![ChildSummary {
                action,
                source: CandidateSource::AreaSequence,
                prior: 1.0,
                visits: 8,
                black_value,
                proposal_count: 0,
                beta: None,
            }],
            SearchStats::default(),
            position.to_move(),
        )
    }

    /// Like `fixed_value`, but placing a stone so the game does not end by
    /// double-pass before a multi-ply window can be observed. The query walks
    /// across the board and is snapped to the nearest legal point, so the
    /// fixture does not have to know where the stones already are.
    fn placing_value(
        position: &vgo_core::Position,
        ply: u32,
        black_value: f64,
    ) -> vgo_search::SearchResult {
        use vgo_core::{Point, nearest_legal_placement};
        use vgo_search::{Action, CandidateSource, ChildSummary, SearchResult, SearchStats};
        let step = f64::from(ply + 1) * 0.13;
        let query = Point::new(0.1 + step % 0.8, 0.1 + (step * 1.7) % 0.8);
        let nearest = nearest_legal_placement(position, query);
        let action = if nearest.legal {
            Action::Place(nearest.point)
        } else {
            Action::Pass
        };
        SearchResult::from_children(
            action,
            vec![ChildSummary {
                action,
                source: CandidateSource::AreaSequence,
                prior: 1.0,
                visits: 8,
                black_value,
                proposal_count: 0,
                beta: None,
            }],
            SearchStats::default(),
            position.to_move(),
        )
    }

    #[test]
    fn a_single_bad_evaluation_does_not_end_a_game() {
        // The window exists because one ply's root value is noisy. A lone
        // hopeless evaluation surrounded by even ones must not concede.
        let rule = ResignRule { threshold: 0.9, window: 5, disable_fraction: 0.0 };
        let mut ply = 0;
        let report = play_game_with_resignation(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            12,
            rule,
            |position, _| {
                ply += 1;
                // One deeply losing ply, the rest neutral.
                let value = if ply == 3 { -1.0 } else { 0.0 };
                Ok::<_, Infallible>(fixed_value(position, value))
            },
            |_| {},
        )
        .unwrap();
        assert!(!report.resigned, "one bad ply must not trigger resignation");
    }

    #[test]
    fn a_sustained_loss_concedes_and_reports_a_real_winner() {
        // Black to move and losing on every ply: after `window` consecutive
        // plies the mover concedes, and White is recorded as the winner so the
        // samples carry a genuine outcome rather than being discarded.
        // Window 1 so the rule fires before the fixture's repeated Pass ends
        // the game by double-pass; the window's behaviour is covered by
        // `a_single_bad_evaluation_does_not_end_a_game`.
        let rule = ResignRule { threshold: 0.9, window: 1, disable_fraction: 0.0 };
        let report = play_game_with_resignation(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            40,
            rule,
            |position, _| Ok::<_, Infallible>(fixed_value(position, -1.0)),
            |_| {},
        )
        .unwrap();
        assert!(report.resigned);
        let outcome = report.outcome.expect("a resigned game still has a winner");
        assert_eq!(outcome.winner, Some(Color::White));
        assert!(report.stats.plies <= 2, "should concede promptly, got {}", report.stats.plies);
    }

    #[test]
    fn a_multi_ply_window_fires_despite_the_alternating_seat() {
        // The regression the per-seat counter exists for. `mover_value` is
        // relative to the side to move, so a game Black is losing reads -1 on
        // Black's plies and +1 on White's. A single shared counter is reset by
        // every one of White's plies, the streak never exceeds 1, and no
        // window above 1 can ever fire -- which is why the production run
        // reported zero resignations at every threshold while games sat 80
        // plies past the point of decision.
        //
        // `a_sustained_loss_concedes_and_reports_a_real_winner` uses window 1,
        // the one setting where the bug is invisible.
        // Placing rather than passing: two passes finish the game at ply 2,
        // before Black reaches a third turn.
        let rule = ResignRule { threshold: 0.9, window: 3, disable_fraction: 0.0 };
        let report = play_game_with_resignation(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            40,
            rule,
            |position, ply| {
                Ok::<_, Infallible>(placing_value(position, ply, -1.0))
            },
            |_| {},
        )
        .unwrap();
        assert!(
            report.resigned,
            "a window above 1 must still fire when one seat is consistently lost"
        );
        assert_eq!(
            report.outcome.and_then(|outcome| outcome.winner),
            Some(Color::White)
        );
        // Three of Black's own turns, which are plies 0, 2 and 4.
        assert_eq!(
            report.stats.plies, 5,
            "should concede on Black's third turn, got {}",
            report.stats.plies
        );
    }

    #[test]
    fn resignation_is_relative_to_the_side_to_move() {
        // The root value is Black-relative. A strongly positive value means
        // Black is winning, so Black must never concede on it however long it
        // persists -- only the side actually losing may resign.
        let rule = ResignRule { threshold: 0.9, window: 3, disable_fraction: 0.0 };
        let report = play_game_with_resignation(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            8,
            rule,
            |position, _| Ok::<_, Infallible>(fixed_value(position, 1.0)),
            |_| {},
        )
        .unwrap();
        // Black never resigns while ahead; if anyone concedes it is White, and
        // never on the first ply.
        assert!(report.stats.plies > 1);
    }

    #[test]
    fn a_disabled_rule_never_fires() {
        let report = play_game_with_resignation(
            Position::new(1.0 / 6.0, Vec::new(), Color::Black),
            6,
            ResignRule::disabled(),
            |position, _| Ok::<_, Infallible>(fixed_value(position, -1.0)),
            |_| {},
        )
        .unwrap();
        assert!(!report.resigned);
    }
}
