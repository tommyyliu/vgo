use std::collections::HashSet;

use crate::{Analysis, Color, Phase, Position, Stone, legal_set};

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

fn remove_settled(position: &Position, analysis: &Analysis, color: Color) -> (Position, usize) {
    let doomed: HashSet<usize> = position
        .stones()
        .iter()
        .enumerate()
        .filter(|(index, stone)| {
            stone.color == color
                && analysis
                    .settled_groups
                    .contains(&analysis.geometry.groups[*index])
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
    let mut current_analysis = Analysis::new(&provisional);

    let (after_enemy, enemy_count) = remove_settled(&provisional, &current_analysis, mover.other());
    provisional = after_enemy;
    if enemy_count > 0 {
        current_analysis = Analysis::new(&provisional);
    }

    let (after_self, self_count) = remove_settled(&provisional, &current_analysis, mover);
    let committed = position.after_placement(after_self.stones().to_vec());
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
