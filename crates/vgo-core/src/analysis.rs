use std::collections::{HashMap, HashSet};

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
pub struct SurvivalEvidence {
    pub stone: usize,
    pub vertex: Point,
    pub free_distance: f64,
    pub influence_radius: f64,
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub validation: Validation,
    pub geometry: voronoi::Geometry,
    pub legal_vertices: Vec<Point>,
    pub alive_groups: HashSet<usize>,
    pub settled_groups: HashSet<usize>,
    pub survival_evidence: HashMap<usize, SurvivalEvidence>,
    pub score: Score,
    pub outcome: Outcome,
}

impl Analysis {
    #[must_use]
    pub fn new(position: &Position) -> Self {
        let validation = position.validate();
        let geometry = voronoi::compute(position);
        let legal_vertices = legal_set::vertices(position);
        let mut alive_groups = HashSet::new();
        let mut survival_evidence = HashMap::new();

        if validation.is_playable() {
            for (stone_index, cell) in geometry.cells.iter().enumerate() {
                let group = geometry.groups[stone_index];
                if alive_groups.contains(&group) {
                    continue;
                }
                let stone = position.stones()[stone_index];
                for &vertex in &cell.polygon {
                    let stone_point = Point::new(stone.x, stone.y);
                    if let Some(witness) = legal_set::escape_witness(
                        position,
                        vertex,
                        stone_point,
                        Some(&legal_vertices),
                    ) {
                        alive_groups.insert(group);
                        survival_evidence.insert(
                            group,
                            SurvivalEvidence {
                                stone: stone_index,
                                vertex,
                                free_distance: vertex.distance(witness),
                                influence_radius: vertex.distance(stone_point),
                            },
                        );
                        break;
                    }
                }
            }
        }

        let settled_groups = geometry
            .groups
            .iter()
            .copied()
            .filter(|group| !alive_groups.contains(group))
            .collect();
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
            survival_evidence,
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
