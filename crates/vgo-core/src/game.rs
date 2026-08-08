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

fn remove_settled(position: &Position, settlement: &Settlement, color: Color) -> (Position, usize) {
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
        return (position.clone(), 0);
    }
    let stones = position
        .stones()
        .iter()
        .enumerate()
        .filter(|(index, _)| !doomed.contains(index))
        .map(|(_, stone)| *stone)
        .collect();
    (position.with_stones(stones), doomed.len())
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
    // Settlement only: both of these describe a board that captures are about
    // to change, so the score and outcome a full analysis would compute are
    // discarded unread. Only the committed position below needs those.
    let mut current_settlement = Settlement::new(&provisional);

    let (after_enemy, enemy_count) =
        remove_settled(&provisional, &current_settlement, mover.other());
    provisional = after_enemy;
    if enemy_count > 0 {
        current_settlement = Settlement::new(&provisional);
    }

    let (after_self, self_count) = remove_settled(&provisional, &current_settlement, mover);
    // A placement that leaves the board exactly as it was is a pass, not a
    // move. Comparing the stone counts rather than the capture counts states
    // that directly: a lone stone that self-captures is a no-op, while a
    // placement that takes a group of its own down with it is not, even though
    // both report self_count > 0.
    let changed = after_self.stones().len() != position.stones().len();
    let committed = position.after_placement(after_self.stones().to_vec(), changed);
    let analysis = Analysis::new(&committed);
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
    use crate::{Color, Phase, Position};

    use super::{MoveError, pass, place};

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
        let before = position.stones().len();
        let mut seen = 0;
        for vertex in legal_set_vertices(&position) {
            let snapped = nearest_legal_placement(&position, Point::new(vertex.x, vertex.y));
            if !snapped.legal {
                continue;
            }
            let Ok(result) = place(&position, snapped.point.x, snapped.point.y) else {
                continue;
            };
            if result.position.stones().len() != before {
                // A real move: the pass counter resets.
                assert_eq!(result.position.consecutive_passes(), 0);
                continue;
            }
            seen += 1;
            assert_eq!(
                result.position.consecutive_passes(),
                1,
                "a placement that changed nothing must count as a pass"
            );
        }
        assert!(seen > 0, "fixture must contain a no-op placement");
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
