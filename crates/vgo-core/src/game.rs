use std::collections::HashSet;

use crate::{Analysis, Color, Phase, Position, Settlement, Stone, legal_set};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveError {
    Finished,
    InvalidPosition,
    IllegalPlacement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameEvent {
    Capture { color: Color, count: usize },
    SelfCapture { color: Color, count: usize },
    Pass,
    GameFinished,
}

#[derive(Clone, Debug)]
pub struct MoveResult {
    pub position: Position,
    pub analysis: Analysis,
    pub events: Vec<GameEvent>,
    pub captured: usize,
}

/// Returns the surviving position and the indices removed from `position`.
///
/// Callers need the identities, not just the tally: telling a no-op placement
/// from a genuine one turns on *which* stone died, not how many.
fn remove_settled(
    position: &Position,
    settlement: &Settlement,
    color: Color,
) -> (Position, HashSet<usize>) {
    let doomed: HashSet<usize> = position
        .stones()
        .iter()
        .enumerate()
        .filter(|(index, stone)| {
            stone.color == color
                && settlement
                    .settled_groups
                    .contains(&settlement.geometry.groups[*index])
        })
        .map(|(index, _)| index)
        .collect();
    if doomed.is_empty() {
        return (position.clone(), doomed);
    }
    let stones = position
        .stones()
        .iter()
        .enumerate()
        .filter(|(index, _)| !doomed.contains(index))
        .map(|(_, stone)| *stone)
        .collect();
    (position.with_stones(stones), doomed)
}

pub fn place(position: &Position, x: f64, y: f64) -> Result<MoveResult, MoveError> {
    if position.phase() != Phase::Playing {
        return Err(MoveError::Finished);
    }
    if !position.validate().is_playable() {
        return Err(MoveError::InvalidPosition);
    }
    if !legal_set::contains(position, x, y) {
        return Err(MoveError::IllegalPlacement);
    }

    let mover = position.to_move();
    let mut stones = position.stones().to_vec();
    stones.push(Stone::new(x, y, mover));
    let mut provisional = position.with_stones(stones);
    // Settlement first: captures may still change either provisional board, so
    // scoring it eagerly would usually be throwaway work. If self-capture leaves
    // the final one unchanged, it is promoted into the committed analysis below.
    let mut current_settlement = Settlement::new(&provisional);

    let (after_enemy, enemy_doomed) =
        remove_settled(&provisional, &current_settlement, mover.other());
    let enemy_count = enemy_doomed.len();
    provisional = after_enemy;
    if enemy_count > 0 {
        current_settlement = Settlement::new(&provisional);
    }

    // The placed stone was appended last and belongs to the mover, so it is
    // never in `enemy_doomed`; order-preserving removal leaves it last here.
    let placed_index = provisional.stones().len() - 1;
    let (after_self, self_doomed) = remove_settled(&provisional, &current_settlement, mover);
    let self_count = self_doomed.len();
    // A placement that leaves the board exactly as it was is a pass, not a
    // move, or it would be a better stall than passing: two passes end the
    // game and score it, two no-op suicides end nothing, and both sides
    // learned to abuse that.
    //
    // The board is unchanged exactly when the stone just placed is the only
    // one removed. Stone counts do not state this -- they collide on every
    // even trade, so a placement that captures one enemy stone reads as a
    // no-op while having changed the board completely. Two such trades in a
    // row ended games at an arbitrary point, scoring a position neither
    // player had finished. Self-capture is legal and global here, so unlike
    // in Go a static count does not imply a capture happened.
    //
    // Must match `place` in reference/src/engine/game.js.
    let unchanged =
        enemy_count == 0 && self_doomed.len() == 1 && self_doomed.contains(&placed_index);
    let changed = !unchanged;
    let committed = position.after_placement(after_self.stones().to_vec(), changed);
    // With no self-capture, `committed` has exactly the radius and stones that
    // `current_settlement` analysed; committing only changed turn/pass metadata.
    // Promote those already-computed geometry and liveness results instead of
    // building the same diagram for a second time. Self-capture changes stones,
    // so that branch still needs a fresh analysis.
    let analysis = if self_count == 0 {
        current_settlement.into_analysis(&committed)
    } else {
        Analysis::new(&committed)
    };
    let mut events = Vec::new();
    if enemy_count > 0 {
        events.push(GameEvent::Capture {
            color: mover.other(),
            count: enemy_count,
        });
    }
    if self_count > 0 {
        events.push(GameEvent::SelfCapture {
            color: mover,
            count: self_count,
        });
    }
    Ok(MoveResult {
        position: committed,
        analysis,
        events,
        captured: enemy_count + self_count,
    })
}

pub fn pass(position: &Position) -> Result<MoveResult, MoveError> {
    if position.phase() != Phase::Playing {
        return Err(MoveError::Finished);
    }
    if !position.validate().is_playable() {
        return Err(MoveError::InvalidPosition);
    }
    let next = position.after_pass();
    let analysis = Analysis::new(&next);
    let event = if next.phase() == Phase::Finished {
        GameEvent::GameFinished
    } else {
        GameEvent::Pass
    };
    Ok(MoveResult {
        position: next,
        analysis,
        events: vec![event],
        captured: 0,
    })
}

#[cfg(test)]
mod tests {
    use crate::{Analysis, Color, GameEvent, MoveResult, Phase, Position, Stone};

    use super::{MoveError, pass, place};

    /// Promoting a settlement must be indistinguishable from recomputing the
    /// committed position, including every floating-point bit in its geometry.
    fn assert_analysis_matches_fresh(result: &MoveResult) {
        let actual = &result.analysis;
        let expected = Analysis::new(&result.position);

        assert_eq!(actual.validation, expected.validation);
        assert_eq!(actual.alive_groups, expected.alive_groups);
        assert_eq!(actual.settled_groups, expected.settled_groups);
        assert_eq!(actual.geometry.adjacency, expected.geometry.adjacency);
        assert_eq!(actual.geometry.groups, expected.geometry.groups);
        assert_eq!(actual.geometry.diagnostics, expected.geometry.diagnostics);
        assert_eq!(actual.geometry.cells.len(), expected.geometry.cells.len());

        let point_bits = |point: crate::Point| (point.x.to_bits(), point.y.to_bits());
        assert_eq!(
            actual
                .legal_vertices
                .iter()
                .copied()
                .map(point_bits)
                .collect::<Vec<_>>(),
            expected
                .legal_vertices
                .iter()
                .copied()
                .map(point_bits)
                .collect::<Vec<_>>()
        );
        for (actual_cell, expected_cell) in
            actual.geometry.cells.iter().zip(&expected.geometry.cells)
        {
            assert_eq!(actual_cell.area.to_bits(), expected_cell.area.to_bits());
            assert_eq!(
                actual_cell
                    .polygon
                    .iter()
                    .copied()
                    .map(point_bits)
                    .collect::<Vec<_>>(),
                expected_cell
                    .polygon
                    .iter()
                    .copied()
                    .map(point_bits)
                    .collect::<Vec<_>>()
            );
            assert_eq!(actual_cell.edges.len(), expected_cell.edges.len());
            for (actual_edge, expected_edge) in actual_cell.edges.iter().zip(&expected_cell.edges) {
                assert_eq!(
                    point_bits(actual_edge.start),
                    point_bits(expected_edge.start)
                );
                assert_eq!(point_bits(actual_edge.end), point_bits(expected_edge.end));
                assert_eq!(actual_edge.source, expected_edge.source);
            }
        }
        assert_eq!(actual.score.black.to_bits(), expected.score.black.to_bits());
        assert_eq!(actual.score.white.to_bits(), expected.score.white.to_bits());
        assert_eq!(actual.outcome.winner, expected.outcome.winner);
        assert_eq!(
            actual.outcome.margin.to_bits(),
            expected.outcome.margin.to_bits()
        );
    }

    #[test]
    fn promoted_analysis_matches_fresh_after_an_ordinary_placement() {
        let position = Position::new(0.1, Vec::new(), Color::Black).with_komi(0.12);
        let result = place(&position, 0.5, 0.5).expect("legal move");
        assert!(result.events.is_empty(), "fixture must not capture");
        assert_analysis_matches_fresh(&result);
    }

    #[test]
    fn promoted_analysis_matches_fresh_after_enemy_capture() {
        // Deterministic four-ply position found by the generation sampler. This
        // move removes both Black groups and no White group, exercising the
        // post-enemy-removal settlement rather than the initial provisional one.
        let position = Position::new(
            0.2,
            vec![
                Stone::new(0.799_999, 0.394_301_338_356_336_74, Color::Black),
                Stone::new(0.450_359_755_708_616_2, 0.200_001, Color::White),
                Stone::new(0.200_001, 0.590_089_667_194_192_1, Color::Black),
            ],
            Color::White,
        )
        .with_komi(0.12);
        let result =
            place(&position, 0.604_780_272_776_142_7, 0.799_999).expect("legal capturing move");
        assert_eq!(
            result.events,
            vec![GameEvent::Capture {
                color: Color::Black,
                count: 2,
            }],
            "fixture must capture enemies without self-capture"
        );
        assert_analysis_matches_fresh(&result);
    }

    /// A placement that leaves the board unchanged counts as a pass.
    ///
    /// Without this a no-op self-capture is a *better* stall than passing: two
    /// passes end the game and score it, two no-op suicides end nothing. Both
    /// sides learned to abuse it -- in arena games between close models half a
    /// game could be one side suiciding while the other passed, and in
    /// self-play 3.8% of transitions were no-ops, twice the pass rate, with
    /// the games containing a run of them reaching the ply cap 88% of the time
    /// against 0% for games without one.
    ///
    /// The position is lifted from a real shard rather than constructed,
    /// because a constructed one silently had no legal no-op at all and the
    /// test passed without exercising anything. Here 8 of 20 legal placements
    /// are no-ops.
    #[test]
    fn a_no_op_self_capture_counts_as_a_pass() {
        use crate::{Point, Stone, legal_set_vertices, nearest_legal_placement};
        let position = Position::new(
            0.055714285714285716,
            vec![
                Stone::new(0.52734375, 0.55859375, Color::Black),
                Stone::new(0.68359375, 0.58984375, Color::White),
                Stone::new(0.66015625, 0.45703125, Color::Black),
                Stone::new(0.60546875, 0.68359375, Color::White),
                Stone::new(0.35546875, 0.55859375, Color::Black),
                Stone::new(0.76953125, 0.49609375, Color::White),
                Stone::new(0.44921875, 0.40234375, Color::Black),
                Stone::new(0.81640625, 0.37890625, Color::White),
                Stone::new(0.32421875, 0.68359375, Color::Black),
                Stone::new(0.33984375, 0.30859375, Color::Black),
                Stone::new(0.23828125, 0.42578125, Color::Black),
                Stone::new(0.82421875, 0.73046875, Color::White),
                Stone::new(0.18359375, 0.71484375, Color::Black),
                Stone::new(0.42578125, 0.20703125, Color::White),
                Stone::new(0.46484375, 0.77734375, Color::Black),
                Stone::new(0.64453125, 0.07421875, Color::White),
                Stone::new(0.59765625, 0.29296875, Color::Black),
                Stone::new(0.48046875, 0.66015625, Color::White),
                Stone::new(0.19921875, 0.17578125, Color::Black),
                Stone::new(0.72265625, 0.26953125, Color::White),
                Stone::new(0.70703125, 0.78515625, Color::Black),
                Stone::new(0.5960043538276335, 0.7946208004428714, Color::Black),
                Stone::new(0.69140625, 0.94140625, Color::White),
                Stone::new(0.28515625, 0.80078125, Color::Black),
                Stone::new(0.84765625, 0.23046875, Color::White),
                Stone::new(0.44921875, 0.92578125, Color::Black),
                Stone::new(0.6055952782457028, 0.1818223545601611, Color::White),
                Stone::new(0.32421875, 0.10546875, Color::Black),
                Stone::new(0.80078125, 0.85546875, Color::White),
                Stone::new(0.76953125, 0.05859375, Color::Black),
                Stone::new(0.07421875, 0.17578125, Color::Black),
                Stone::new(0.89453125, 0.09765625, Color::White),
                Stone::new(0.06640625, 0.70703125, Color::Black),
                Stone::new(0.47265625, 0.10546875, Color::White),
                Stone::new(0.89453125, 0.6387733214285715, Color::White),
                Stone::new(0.2157783978145951, 0.3162727288836293, Color::Black),
                Stone::new(0.15301135558185364, 0.5514488555818536, Color::Black),
                Stone::new(0.9180080332469719, 0.9261099115477557, Color::White),
                Stone::new(0.07421875, 0.35546875, Color::Black),
                Stone::new(0.93359375, 0.43359375, Color::White),
                Stone::new(0.29296875, 0.94140625, Color::Black),
                Stone::new(0.56640625, 0.94140625, Color::White),
                Stone::new(0.06463485456016112, 0.9180952782457028, Color::Black),
                Stone::new(0.17565472175429722, 0.8259901454398388, Color::White),
                Stone::new(0.11191055388951679, 0.07092015181722182, Color::Black),
                Stone::new(0.9282210508939471, 0.7704696349592104, Color::White),
            ],
            Color::Black,
        );
        assert!(position.validate().is_playable());
        // Classify by the board itself, never by its stone count. An earlier
        // version of this test asked whether the count had changed -- the same
        // predicate the implementation used -- so it agreed with any behaviour
        // that predicate produced and could not fail on the bug it guarded.
        let before: Vec<Stone> = position.stones().to_vec();
        let mut no_ops = 0;
        let mut real_moves = 0;
        for vertex in legal_set_vertices(&position) {
            let snapped = nearest_legal_placement(&position, Point::new(vertex.x, vertex.y));
            if !snapped.legal {
                continue;
            }
            let Ok(result) = place(&position, snapped.point.x, snapped.point.y) else {
                continue;
            };
            if result.position.stones() == before.as_slice() {
                no_ops += 1;
                assert_eq!(
                    result.position.consecutive_passes(),
                    1,
                    "a placement that left the board identical must count as a pass"
                );
            } else {
                real_moves += 1;
                assert_eq!(
                    result.position.consecutive_passes(),
                    0,
                    "a placement that changed the board must reset the pass count, \
                     even when it removes as many stones as it adds"
                );
            }
        }
        assert!(no_ops > 0, "fixture must contain a no-op placement");
        assert!(real_moves > 0, "fixture must contain a real placement");
    }

    /// The bug this pins: a placement that captures exactly one enemy stone
    /// leaves the stone count untouched while changing the board completely.
    /// Read as a pass, two of these in a row ended a real game at ply 44 --
    /// scoring a position neither player had passed on, at the one moment
    /// Black happened to be ahead.
    ///
    /// This is the position after ply 42 of that game; Black to play
    /// (0.121094, 0.714844), which captures the white stone at
    /// (0.07421875, 0.60546875).
    #[test]
    fn an_even_trade_is_not_a_pass() {
        let position = Position::new(
            0.055714285714285716,
            vec![
                Stone::new(0.519531, 0.496094, Color::Black),
                Stone::new(0.53515625, 0.71484375, Color::White),
                Stone::new(0.378906, 0.519531, Color::Black),
                Stone::new(0.59765625, 0.23828125, Color::White),
                Stone::new(0.660156, 0.488281, Color::Black),
                Stone::new(0.36328125, 0.25390625, Color::White),
                Stone::new(0.347656, 0.667969, Color::Black),
                Stone::new(0.37890625, 0.81640625, Color::White),
                Stone::new(0.230469, 0.792969, Color::Black),
                Stone::new(0.24155634060202594, 0.9038455992765371, Color::White),
                Stone::new(0.113281, 0.886719, Color::Black),
                Stone::new(0.69140625, 0.70703125, Color::White),
                Stone::new(0.769531, 0.613281, Color::Black),
                Stone::new(0.8440734679885532, 0.6961059349829543, Color::White),
                Stone::new(0.917969, 0.589844, Color::Black),
                Stone::new(0.9442857142857143, 0.7448279240914601, Color::White),
                Stone::new(0.261719, 0.417969, Color::Black),
                Stone::new(0.18359375, 0.28515625, Color::White),
                Stone::new(0.097656, 0.371094, Color::Black),
                Stone::new(0.74609375, 0.26953125, Color::White),
                Stone::new(0.097656, 0.207031, Color::Black),
                Stone::new(0.1815156102469017, 0.1336544699785221, Color::White),
                Stone::new(0.089844, 0.066406, Color::Black),
                Stone::new(0.36328125, 0.08203125, Color::White),
                Stone::new(0.792969, 0.402344, Color::Black),
                Stone::new(0.89453125, 0.33984375, Color::White),
                Stone::new(0.442074, 0.332699, Color::Black),
                Stone::new(0.4864982006504679, 0.23050780769204193, Color::White),
                Stone::new(0.894531, 0.451273, Color::Black),
                Stone::new(0.4431630138148883, 0.6105672604713251, Color::White),
                Stone::new(0.832031, 0.121094, Color::Black),
                Stone::new(0.5396564125988921, 0.3864969752419616, Color::White),
                Stone::new(0.199219, 0.605469, Color::Black),
                Stone::new(0.07421875, 0.60546875, Color::White),
                Stone::new(0.792969, 0.886719, Color::Black),
                Stone::new(0.66796875, 0.88671875, Color::White),
                Stone::new(0.628906, 0.082031, Color::Black),
                Stone::new(0.7226365745353382, 0.14228984819701174, Color::White),
                Stone::new(0.472656, 0.917969, Color::Black),
                Stone::new(0.6839238463317968, 0.3794157673984005, Color::White),
                Stone::new(0.113281, 0.488281, Color::Black),
                Stone::new(0.6589316771545608, 0.5997038436377342, Color::White),
            ],
            Color::Black,
        );
        assert!(position.validate().is_playable());
        let before = position.stones().len();

        let result = place(&position, 0.121094, 0.714844).expect("legal placement");
        assert_eq!(
            result.captured, 1,
            "fixture must capture exactly one stone, or it does not exercise the collision"
        );
        assert_eq!(
            result.position.stones().len(),
            before,
            "fixture must be an even trade, or it does not exercise the collision"
        );
        assert!(
            !result
                .position
                .stones()
                .contains(&Stone::new(0.07421875, 0.60546875, Color::White)),
            "the white stone must actually be captured"
        );
        assert_eq!(
            result.position.consecutive_passes(),
            0,
            "an even trade changes the board and must reset the pass count"
        );
        assert_eq!(result.position.phase(), Phase::Playing);
    }

    #[test]
    fn placement_is_a_pure_transaction() {
        let before = Position::new(0.05, Vec::new(), Color::Black);
        let result = place(&before, 0.5, 0.5).expect("legal move");
        assert!(before.stones().is_empty());
        assert_eq!(before.to_move(), Color::Black);
        assert_eq!(result.position.stones().len(), 1);
        assert_eq!(result.position.to_move(), Color::White);
    }

    #[test]
    fn two_passes_finish_and_block_further_actions() {
        let start = Position::new(0.05, Vec::new(), Color::Black);
        let first = pass(&start).expect("first pass");
        let second = pass(&first.position).expect("second pass");
        assert_eq!(second.position.phase(), Phase::Finished);
        assert_eq!(pass(&second.position).unwrap_err(), MoveError::Finished);
        assert_eq!(
            place(&second.position, 0.5, 0.5).unwrap_err(),
            MoveError::Finished
        );
    }
}
