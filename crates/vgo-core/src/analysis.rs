use std::collections::HashSet;

use crate::{Color, Point, Position, Validation, legal_set, numeric, voronoi};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Score {
    pub black: f64,
    pub white: f64,
}

impl Score {
    #[must_use]
    pub const fn for_color(self, color: Color) -> f64 {
        match color {
            Color::Black => self.black,
            Color::White => self.white,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    pub winner: Option<Color>,
    pub margin: f64,
}

impl Outcome {
    #[must_use]
    pub const fn is_tie(self) -> bool {
        self.winner.is_none()
    }

    #[must_use]
    pub const fn black_utility(self) -> f64 {
        match self.winner {
            Some(Color::Black) => 1.0,
            Some(Color::White) => -1.0,
            None => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub validation: Validation,
    pub geometry: voronoi::Geometry,
    pub legal_vertices: Vec<Point>,
    pub alive_groups: HashSet<usize>,
    pub settled_groups: HashSet<usize>,
    pub score: Score,
    pub outcome: Outcome,
}

/// Which groups are settled, without the scoring a full `Analysis` carries.
///
/// `place` applies captures before it can know the committed position, so it
/// analyses the provisional board once, and again if removing enemy stones
/// changed it. Intermediate capture checks read `settled_groups` and
/// `geometry.groups`; when no self-capture changes the last provisional board,
/// the remaining fields let that same work become the committed `Analysis`.
///
/// Sharing `settlement` between them keeps the liveness search in one place,
/// skips provisional scoring, and avoids rebuilding the final geometry.
#[derive(Clone, Debug)]
pub struct Settlement {
    pub geometry: voronoi::Geometry,
    pub settled_groups: HashSet<usize>,
    validation: Validation,
    legal_vertices: Vec<Point>,
    alive_groups: HashSet<usize>,
}

/// Groups with a witness that a placement can still reach them.
///
/// The shared core of `Analysis::new` and `Settlement::new`: everything either
/// one needs before it can say which groups are settled.
fn alive_groups_of(
    position: &Position,
    geometry: &voronoi::Geometry,
    legal_vertices: &[Point],
    playable: bool,
) -> HashSet<usize> {
    let mut alive_groups = HashSet::new();
    if !playable {
        return alive_groups;
    }
    for (stone_index, cell) in geometry.cells.iter().enumerate() {
        let group = geometry.groups[stone_index];
        if alive_groups.contains(&group) {
            continue;
        }
        let stone = position.stones()[stone_index];
        let stone_point = Point::new(stone.x, stone.y);
        for &vertex in &cell.polygon {
            if legal_set::escape_witness(position, vertex, stone_point, Some(legal_vertices))
                .is_some()
            {
                alive_groups.insert(group);
                break;
            }
        }
    }
    alive_groups
}

fn settled_from(geometry: &voronoi::Geometry, alive_groups: &HashSet<usize>) -> HashSet<usize> {
    geometry
        .groups
        .iter()
        .copied()
        .filter(|group| !alive_groups.contains(group))
        .collect()
}

impl Settlement {
    /// Settlement alone, skipping the score and outcome a full analysis builds.
    #[must_use]
    pub fn new(position: &Position) -> Self {
        let validation = position.validate();
        let geometry = voronoi::compute(position);
        let legal_vertices = legal_set::vertices(position);
        let alive_groups = alive_groups_of(
            position,
            &geometry,
            &legal_vertices,
            validation.is_playable(),
        );
        let settled_groups = settled_from(&geometry, &alive_groups);
        Self {
            geometry,
            settled_groups,
            validation,
            legal_vertices,
            alive_groups,
        }
    }

    /// Finish an analysis when only position metadata changed since settlement.
    ///
    /// A placement commits by changing the turn and pass state after capture
    /// resolution. Neither field enters validation, geometry, the legal set, or
    /// group liveness, so a settlement of the same radius and stones already
    /// contains all of the expensive parts of the committed analysis.
    pub(crate) fn into_analysis(self, position: &Position) -> Analysis {
        Analysis::from_parts(
            position,
            self.validation,
            self.geometry,
            self.legal_vertices,
            self.alive_groups,
            self.settled_groups,
        )
    }
}

impl Analysis {
    #[must_use]
    pub fn new(position: &Position) -> Self {
        Settlement::new(position).into_analysis(position)
    }

    fn from_parts(
        position: &Position,
        validation: Validation,
        geometry: voronoi::Geometry,
        legal_vertices: Vec<Point>,
        alive_groups: HashSet<usize>,
        settled_groups: HashSet<usize>,
    ) -> Self {
        let mut score = Score::default();
        for (index, stone) in position.stones().iter().enumerate() {
            match stone.color {
                Color::Black => score.black += geometry.cells[index].area,
                Color::White => score.white += geometry.cells[index].area,
            }
        }
        // Komi is subtracted from Black's lead, so a positive value
        // compensates White for moving second. Measured on thirteen games
        // played to the ply cap, Black led by a median of 0.181 of the board
        // and won twelve of them; the margins cluster tightly enough that the
        // balance point is sharp, with komi 0.15 leaving Black 10/13 and 0.20
        // leaving Black 4/13.
        //
        // Held on the position rather than passed to this call because it is
        // part of what a game *is*: two positions with the same stones and
        // different komi have different winners, so a search, a shard, and a
        // model that disagree about it are not playing the same game.
        let delta = score.black - score.white - position.komi();
        let winner = if delta.abs() <= numeric::COMPARISON_EPSILON {
            None
        } else if delta > 0.0 {
            Some(Color::Black)
        } else {
            Some(Color::White)
        };
        Self {
            validation,
            geometry,
            legal_vertices,
            alive_groups,
            settled_groups,
            score,
            outcome: Outcome {
                winner,
                margin: delta.abs(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Color, Position, Stone};

    use super::Analysis;

    /// Komi decides the winner, and survives the moves of a game.
    ///
    /// Measured on thirteen games at the ply cap, Black led by a median of
    /// 0.181 of the board and won twelve; without komi the game is close to
    /// decided before either side moves.
    #[test]
    fn komi_shifts_the_winner_and_is_carried_through_play() {
        // Black holds more area than White.
        let stones = vec![
            Stone::new(0.25, 0.25, Color::Black),
            Stone::new(0.35, 0.25, Color::Black),
            Stone::new(0.80, 0.80, Color::White),
        ];
        let plain = Position::new(0.05, stones.clone(), Color::Black);
        let lead = {
            let analysis = Analysis::new(&plain);
            analysis.score.black - analysis.score.white
        };
        assert!(lead > 0.0, "fixture must have Black ahead, got {lead}");
        assert_eq!(Analysis::new(&plain).outcome.winner, Some(Color::Black));

        // Komi just over the lead hands the game to White.
        let compensated = plain.clone().with_komi(lead + 0.01);
        assert_eq!(
            Analysis::new(&compensated).outcome.winner,
            Some(Color::White),
            "komi above Black's lead must make White the winner"
        );

        // And it is a property of the game, not of one ply: a move must not
        // reset it, or a search would score its own leaves differently from
        // the position it was handed.
        let after = compensated.after_pass();
        assert_eq!(after.komi(), lead + 0.01);
    }

    #[test]
    fn covered_legal_space_settles_every_isolated_group() {
        let position = Position::new(
            0.25,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::White),
                Stone::new(0.75, 0.75, Color::Black),
                Stone::new(0.25, 0.75, Color::White),
            ],
            Color::Black,
        );
        let analysis = Analysis::new(&position);
        assert!(analysis.validation.is_playable());
        assert!(analysis.legal_vertices.is_empty());
        assert_eq!(analysis.settled_groups.len(), 4);
        assert!((analysis.score.black - 0.5).abs() < 1.0e-12);
        assert!((analysis.score.white - 0.5).abs() < 1.0e-12);
    }
}
