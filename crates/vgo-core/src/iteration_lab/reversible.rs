//! Compact path history experiment. Forward moves still use the rebuild engine;
//! this is not incremental geometry and is not installed in production MCTS.
use crate::{GameEvent, MoveError, MoveResult, Position, Stone};

struct Undo {
    metadata: Position, // Empty stone vector: only the old game metadata.
    removed: Vec<(usize, Stone)>,
    added_survived: bool,
}

pub struct ReversiblePosition {
    position: Position,
    history: Vec<Undo>,
}

#[derive(Debug)]
pub struct TransitionSummary {
    pub captured: usize,
    pub events: Vec<GameEvent>,
}

impl ReversiblePosition {
    pub fn new(position: Position) -> Self {
        Self {
            position,
            history: Vec::new(),
        }
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    /// Stack depth; valid only as an ancestor depth of the current traversal.
    pub fn depth(&self) -> usize {
        self.history.len()
    }
    /// Reserved undo payload, excluding allocator headers and the current board.
    pub fn undo_bytes(&self) -> usize {
        self.history.capacity() * size_of::<Undo>()
            + self
                .history
                .iter()
                .map(|u| u.removed.capacity() * size_of::<(usize, Stone)>())
                .sum::<usize>()
    }
    pub fn place(&mut self, x: f64, y: f64) -> Result<TransitionSummary, MoveError> {
        let result = crate::place(&self.position, x, y)?;
        Ok(self.commit(result))
    }
    pub fn pass(&mut self) -> Result<TransitionSummary, MoveError> {
        let result = crate::pass(&self.position)?;
        Ok(self.commit(result))
    }
    fn commit(&mut self, result: MoveResult) -> TransitionSummary {
        // Captures preserve old stone order; a surviving insertion is last.
        let mut cursor = 0;
        let mut removed = Vec::new();
        for (index, stone) in self.position.stones().iter().enumerate() {
            if result.position.stones().get(cursor) == Some(stone) {
                cursor += 1;
            } else {
                removed.push((index, *stone));
            }
        }
        let added_survived = result.position.stones().len() > cursor;
        debug_assert!(result.position.stones().len() <= cursor + 1);
        let mut metadata = std::mem::replace(&mut self.position, result.position);
        drop(metadata.take_stones_for_lab());
        self.history.push(Undo {
            metadata,
            removed,
            added_survived,
        });
        TransitionSummary {
            captured: result.captured,
            events: result.events,
        }
    }
    /// Restore coordinates, order, and metadata from saved values, never by
    /// reversing floating-point constructions. Returns false at the root.
    pub fn undo(&mut self) -> bool {
        let Some(undo) = self.history.pop() else {
            return false;
        };
        let mut survivors = self.position.take_stones_for_lab();
        if undo.added_survived {
            survivors.pop();
        }
        let mut restored = Vec::with_capacity(survivors.len() + undo.removed.len());
        let mut survivors = survivors.into_iter();
        let mut removed = undo.removed.into_iter().peekable();
        while survivors.len() > 0 || removed.peek().is_some() {
            let next = if removed
                .peek()
                .is_some_and(|(index, _)| *index == restored.len())
            {
                removed.next().unwrap().1
            } else {
                survivors.next().expect("undo indices preserve old order")
            };
            restored.push(next);
        }
        self.position = undo.metadata.with_stones(restored);
        true
    }
    pub fn rollback(&mut self, depth: usize) {
        assert!(depth <= self.depth());
        while self.depth() > depth {
            self.undo();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Color, Ruleset};
    fn same(a: &Position, b: &Position) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        for (a, b) in a.stones().iter().zip(b.stones()) {
            assert_eq!(
                (a.x.to_bits(), a.y.to_bits()),
                (b.x.to_bits(), b.y.to_bits())
            );
        }
    }
    #[test]
    fn capture_and_no_op_undo_are_exact() {
        let ring = (0..3)
            .flat_map(|y| {
                (0..3).filter_map(move |x| {
                    (x != 1 || y != 1).then(|| {
                        Stone::new(0.3 + x as f64 * 0.2, 0.3 + y as f64 * 0.2, Color::White)
                    })
                })
            })
            .collect();
        let no_op = Position::new(0.1, ring, Color::Black);
        let mut state = ReversiblePosition::new(no_op.clone());
        state.place(0.5, 0.5).unwrap();
        assert_eq!(state.position().stones(), no_op.stones());
        assert_eq!(state.position().consecutive_passes(), 1);
        state.undo();
        same(state.position(), &no_op);
        let root = Position::new(
            0.25,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::White),
                Stone::new(0.25, 0.75, Color::White),
            ],
            Color::Black,
        );
        let mut state = ReversiblePosition::new(root.clone());
        let result = state.place(0.75, 0.75).unwrap();
        assert_eq!(result.captured, 2);
        assert!(state.undo());
        same(state.position(), &root);
        // Three friendly stones: filling the last gap removes all four.
        let root = Position::new(
            0.25,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::Black),
                Stone::new(0.25, 0.75, Color::Black),
            ],
            Color::Black,
        );
        let mut state = ReversiblePosition::new(root.clone());
        state.place(0.75, 0.75).unwrap();
        assert!(state.position().stones().is_empty());
        state.undo();
        same(state.position(), &root);
        let refused = root.with_ruleset(Ruleset::Official);
        let mut state = ReversiblePosition::new(refused.clone());
        assert!(matches!(
            state.place(0.75, 0.75),
            Err(MoveError::SelfCapture)
        ));
        assert_eq!(state.depth(), 0);
        same(state.position(), &refused);
    }
    #[test]
    fn undo_restores_captures_passes_refusals_and_sibling_branches() {
        for rules in [Ruleset::Vgo, Ruleset::Official] {
            for radius in [0.25, 0.1, 1.0 / 38.0] {
                let root = Position::new(radius, vec![], Color::Black)
                    .with_ruleset(rules)
                    .with_komi(0.104);
                let mut state = ReversiblePosition::new(root.clone());
                let mut parents = Vec::new();
                for step in 0..64 {
                    let vertices = crate::legal_set_vertices(state.position());
                    let Some(p) = vertices.get(step % vertices.len().max(1)) else {
                        break;
                    };
                    let parent = state.position().clone();
                    let baseline = crate::place(&parent, p.x, p.y);
                    let actual = state.place(p.x, p.y);
                    match (baseline, actual) {
                        (Ok(expected), Ok(actual)) => {
                            assert_eq!(actual.events, expected.events);
                            assert_eq!(actual.captured, expected.captured);
                            same(state.position(), &expected.position);
                            assert!(state.undo());
                            same(state.position(), &parent);
                            state.place(p.x, p.y).unwrap();
                            parents.push(parent);
                        }
                        (Err(expected), Err(actual)) => {
                            assert_eq!(expected, actual);
                            same(state.position(), &parent);
                            break;
                        }
                        _ => panic!("transition changed"),
                    }
                    if state.position().phase() == crate::Phase::Finished {
                        break;
                    }
                }
                while let Some(parent) = parents.pop() {
                    assert!(state.undo());
                    same(state.position(), &parent);
                }
                same(state.position(), &root);
                assert!(!state.undo());
                assert!(state.place(-1.0, 0.5).is_err());
                assert_eq!(state.depth(), 0);
                state.pass().unwrap();
                state.pass().unwrap();
                assert!(state.pass().is_err());
                assert_eq!(state.depth(), 2);
                state.rollback(0);
                same(state.position(), &root);
            }
        }
    }
}
