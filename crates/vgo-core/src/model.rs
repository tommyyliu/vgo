use crate::numeric::COORDINATE_EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Color {
    Black,
    White,
}

impl Color {
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stone {
    pub x: f64,
    pub y: f64,
    pub color: Color,
}

impl Stone {
    #[must_use]
    pub const fn new(x: f64, y: f64, color: Color) -> Self {
        Self { x, y, color }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Playing,
    Finished,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    radius: f64,
    stones: Vec<Stone>,
    to_move: Color,
    consecutive_passes: u32,
    phase: Phase,
    /// Area subtracted from Black's lead when the game is scored.
    ///
    /// Voronoi area is a fraction of the board, so komi is one too rather than
    /// a stone count: 0.18 means White is spotted eighteen percent of the
    /// board. Measured on games played to the ply cap, Black led by a median
    /// of 0.181 and won twelve of thirteen, so an unbalanced game is close to
    /// decided before either side moves.
    ///
    /// Part of the position because it changes who won: the same stones under
    /// different komi score differently, so a search, a shard, and a model
    /// that disagree about it are not playing the same game.
    komi: f64,
}

impl Position {
    #[must_use]
    pub fn new(radius: f64, stones: Vec<Stone>, to_move: Color) -> Self {
        Self {
            radius,
            stones,
            to_move,
            consecutive_passes: 0,
            phase: Phase::Playing,
            komi: 0.0,
        }
    }

    /// The same position scored with `komi` subtracted from Black's lead.
    #[must_use]
    pub fn with_komi(mut self, komi: f64) -> Self {
        self.komi = komi;
        self
    }

    /// The same position with `passes` consecutive passes already played.
    ///
    /// For reconstructing a position from an external record, where the stones
    /// are known but the pass count is carried separately. It is not optional
    /// information: a search that believes nobody has passed does not know that
    /// passing now would end the game, so it cannot pass to close out a win and
    /// cannot see that passing while behind loses on the spot. `new` starts at
    /// zero, which is the assumption that reads as "no passes yet" and is wrong
    /// exactly when the endgame starts.
    ///
    /// Two passes end the game, so that count moves the phase to
    /// [`Phase::Finished`] and the turn stays with whoever was to move, matching
    /// what playing the second pass would have produced.
    #[must_use]
    pub fn with_passes(mut self, passes: u32) -> Self {
        self.consecutive_passes = passes;
        self.phase = if passes >= 2 {
            Phase::Finished
        } else {
            Phase::Playing
        };
        self
    }

    #[must_use]
    pub const fn komi(&self) -> f64 {
        self.komi
    }

    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    #[must_use]
    pub fn stones(&self) -> &[Stone] {
        &self.stones
    }

    #[must_use]
    pub const fn to_move(&self) -> Color {
        self.to_move
    }

    #[must_use]
    pub const fn consecutive_passes(&self) -> u32 {
        self.consecutive_passes
    }

    #[must_use]
    pub const fn phase(&self) -> Phase {
        self.phase
    }

    #[must_use]
    pub fn validate(&self) -> Validation {
        let mut issues = Vec::new();
        let radius = self.radius;
        if !radius.is_finite() || radius <= 0.0 || radius >= 0.5 {
            issues.push(ValidationIssue::InvalidRadius);
        }

        for (index, stone) in self.stones.iter().enumerate() {
            if !stone.x.is_finite() || !stone.y.is_finite() {
                issues.push(ValidationIssue::NonFiniteStone { index });
                continue;
            }
            if stone.x < radius - COORDINATE_EPSILON
                || stone.x > 1.0 - radius + COORDINATE_EPSILON
                || stone.y < radius - COORDINATE_EPSILON
                || stone.y > 1.0 - radius + COORDINATE_EPSILON
            {
                issues.push(ValidationIssue::StoneOutsideBoard { index });
            }
            for (other_index, other) in self.stones[..index].iter().enumerate() {
                let distance = (stone.x - other.x).hypot(stone.y - other.y);
                if distance < 2.0 * radius - COORDINATE_EPSILON {
                    issues.push(ValidationIssue::OverlappingStones {
                        first: other_index,
                        second: index,
                    });
                }
            }
        }
        Validation { issues }
    }

    pub(crate) fn with_stones(&self, stones: Vec<Stone>) -> Self {
        let mut next = self.clone();
        next.stones = stones;
        next
    }

    /// The position after a placement.
    ///
    /// `changed` is false when the move left the board exactly as it was: a
    /// stone placed with no liberties, self-captured in the same move, taking
    /// nothing with it. That is a pass in every respect except that it used to
    /// reset the pass counter, which made it a *better* stall than passing --
    /// two passes end the game and score it, while two no-op suicides end
    /// nothing.
    ///
    /// Both sides learned to abuse that. In arena games between two close
    /// models, half the game could be one side suiciding while the other
    /// passed, neither able to end it. In self-play 3.8% of all transitions
    /// were no-ops -- twice the pass rate -- and the games containing a run of
    /// four or more reached the ply cap 88% of the time against 0% for games
    /// without one, with a median of 34 wasted plies out of 100.
    ///
    /// Counting a no-op as a pass closes it: two in a row now end the game and
    /// score it, so stalling loses to whoever is ahead on the board.
    pub(crate) fn after_placement(&self, stones: Vec<Stone>, changed: bool) -> Self {
        let passes = if changed {
            0
        } else {
            self.consecutive_passes + 1
        };
        let finished = passes >= 2;
        Self {
            radius: self.radius,
            stones,
            to_move: if finished {
                self.to_move
            } else {
                self.to_move.other()
            },
            consecutive_passes: passes,
            phase: if finished {
                Phase::Finished
            } else {
                Phase::Playing
            },
            // Carried, not reset: komi belongs to the game, not the ply.
            komi: self.komi,
        }
    }

    pub(crate) fn after_pass(&self) -> Self {
        let passes = self.consecutive_passes + 1;
        let finished = passes >= 2;
        Self {
            radius: self.radius,
            stones: self.stones.clone(),
            to_move: if finished {
                self.to_move
            } else {
                self.to_move.other()
            },
            consecutive_passes: passes,
            phase: if finished {
                Phase::Finished
            } else {
                Phase::Playing
            },
            komi: self.komi,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationIssue {
    InvalidRadius,
    NonFiniteStone { index: usize },
    StoneOutsideBoard { index: usize },
    OverlappingStones { first: usize, second: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Validation {
    issues: Vec<ValidationIssue>,
}

impl Validation {
    #[must_use]
    pub fn is_playable(&self) -> bool {
        self.issues.is_empty()
    }

    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }
}

#[cfg(test)]
mod tests {
    use super::{Color, Phase, Position, Stone, ValidationIssue};

    #[test]
    fn tangent_stones_are_playable() {
        let position = Position::new(
            0.1,
            vec![
                Stone::new(0.4, 0.5, Color::Black),
                Stone::new(0.6, 0.5, Color::White),
            ],
            Color::Black,
        );
        assert!(position.validate().is_playable());
    }

    #[test]
    fn duplicate_centers_are_rejected() {
        let position = Position::new(
            0.05,
            vec![
                Stone::new(0.5, 0.5, Color::Black),
                Stone::new(0.5, 0.5, Color::White),
            ],
            Color::Black,
        );
        assert_eq!(
            position.validate().issues(),
            &[ValidationIssue::OverlappingStones {
                first: 0,
                second: 1,
            }]
        );
    }

    #[test]
    fn stones_must_fit_inside_the_board() {
        let position = Position::new(0.1, vec![Stone::new(0.02, 0.5, Color::Black)], Color::White);
        assert_eq!(
            position.validate().issues(),
            &[ValidationIssue::StoneOutsideBoard { index: 0 }]
        );
    }

    /// A reconstructed position must know how many passes it inherited, or the
    /// search cannot see that passing ends the game. Two passes must also land
    /// on the same state playing the second pass would have produced -- finished,
    /// with the turn left where it was rather than handed over.
    #[test]
    fn a_reconstructed_position_carries_its_pass_count() {
        let base = Position::new(0.1, Vec::new(), Color::Black);
        assert_eq!(base.consecutive_passes(), 0);

        let one = base.clone().with_passes(1);
        assert_eq!(one.consecutive_passes(), 1);
        assert_eq!(one.phase(), Phase::Playing);
        assert_eq!(one.to_move(), Color::Black);

        let two = base.clone().with_passes(2);
        assert_eq!(two.phase(), Phase::Finished);
        assert_eq!(two.to_move(), Color::Black);

        // Reconstructing must agree with playing it out: one pass from `one`
        // ends the game, and lands where `with_passes(2)` says it does.
        let played = crate::pass(&one).expect("a pass is legal here").position;
        assert_eq!(played.phase(), two.phase());
        assert_eq!(played.consecutive_passes(), two.consecutive_passes());
        assert_eq!(played.to_move(), two.to_move());
    }
}
