//! Opt-in experiments, measured against the normal full transition contract.
//! No strategy here is selected by the production search or client.

use crate::{
    Analysis, MoveError, MoveResult, Point, Position, Ruleset, Settlement, Stone, numeric,
};
use std::collections::HashSet;

mod support_points;
mod reversible;
#[cfg(test)]
mod locality;
pub use reversible::{ReversiblePosition, TransitionSummary};
pub use support_points::{SupportPolicy, SupportPosition, SupportStats};

struct Certificate {
    owner: Stone,
    point: Point,
    legal: Point,
}

/// An experimental cache for sibling moves from one immutable parent.
/// Construction cost is paid once per parent, not once per candidate.
pub struct PreparedPosition<'a> {
    position: &'a Position,
    certificates: Vec<Certificate>,
}

impl<'a> PreparedPosition<'a> {
    pub fn new(position: &'a Position) -> Self {
        let analysis = Analysis::new(position);
        Self::from_analysis(position, &analysis)
    }

    /// The caller may reuse an analysis already computed for this exact parent.
    /// Kept private until there is a checked public association between them.
    fn from_analysis(position: &'a Position, analysis: &Analysis) -> Self {
        let mut certificates = Vec::new();
        let mut seen = HashSet::new();
        if analysis.validation.is_playable() && position.ruleset() == Ruleset::Vgo {
            for (i, cell) in analysis.geometry.cells.iter().enumerate() {
                let group = analysis.geometry.groups[i];
                if seen.contains(&group) || !analysis.alive_groups.contains(&group) {
                    continue;
                }
                let owner = position.stones()[i];
                let site = Point::new(owner.x, owner.y);
                for &point in &cell.polygon {
                    let Some(legal) = crate::legal_set::escape_witness(
                        position,
                        point,
                        site,
                        Some(&analysis.legal_vertices),
                    ) else {
                        continue;
                    };
                    // Certify against the actual sites and placement predicate,
                    // rather than assuming a rounded polygon vertex is inside.
                    if !crate::is_legal_placement(position, legal.x, legal.y)
                        || position.stones().iter().any(|s| {
                            numeric::strictly_closer(point, Point::new(s.x, s.y), site)
                                .is_strictly_less
                        })
                    {
                        continue;
                    }
                    certificates.push(Certificate {
                        owner,
                        point,
                        legal,
                    });
                    seen.insert(group);
                    break;
                }
            }
        }
        Self {
            position,
            certificates,
        }
    }

    pub fn certificate_count(&self) -> usize {
        self.certificates.len()
    }

    pub fn place(&self, x: f64, y: f64) -> Result<MoveResult, MoveError> {
        let added = Stone::new(x, y, self.position.to_move());
        crate::game::place_with(self.position, x, y, |current| {
            Settlement::build_seeded::<false, false, false>(current, |geometry| {
                let mut alive = HashSet::new();
                let added_survives = current.stones().contains(&added);
                let minimum = 2.0 * current.radius() - numeric::COORDINATE_EPSILON;
                for certificate in &self.certificates {
                    let Some(owner) = current
                        .stones()
                        .iter()
                        .position(|s| *s == certificate.owner)
                    else {
                        continue;
                    };
                    if added_survives {
                        let dx = certificate.legal.x - added.x;
                        let dy = certificate.legal.y - added.y;
                        if dx.mul_add(dx, dy * dy) < minimum * minimum {
                            continue;
                        }
                        if numeric::strictly_closer(
                            certificate.point,
                            Point::new(x, y),
                            Point::new(certificate.owner.x, certificate.owner.y),
                        )
                        .is_strictly_less
                        {
                            continue;
                        }
                    }
                    // All remaining sites are a subset of the parent plus
                    // `added`. Removal only expands legal space and this cell.
                    // Mapping via the surviving owner handles group splits.
                    alive.insert(geometry.groups[owner]);
                }
                alive
            })
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Strategy {
    Baseline,
    WitnessFirst,
    Indexed,
    QueryFirst,
}

pub fn place(
    position: &Position,
    x: f64,
    y: f64,
    strategy: Strategy,
) -> Result<MoveResult, MoveError> {
    crate::game::place_with(position, x, y, |position| settlement(position, strategy))
}

/// Separate component timings are diagnostic; their sum is not a move time.
pub fn geometry(position: &Position) -> crate::Geometry {
    crate::voronoi::compute(position)
}

pub fn analysis(position: &Position, strategy: Strategy) -> Analysis {
    settlement(position, strategy).into_analysis(position)
}

fn settlement(position: &Position, strategy: Strategy) -> Settlement {
    match strategy {
        Strategy::Baseline => Settlement::new(position),
        Strategy::WitnessFirst => Settlement::build::<true, false, false>(position),
        Strategy::Indexed => Settlement::build::<false, true, false>(position),
        Strategy::QueryFirst => Settlement::build::<false, true, true>(position),
    }
}

/// Compare the complete observable transition, including geometry and metadata.
pub fn assert_same(
    expected: &Result<MoveResult, MoveError>,
    actual: &Result<MoveResult, MoveError>,
) {
    match (expected, actual) {
        (Err(a), Err(b)) => assert_eq!(a, b),
        (Ok(a), Ok(b)) => {
            assert_eq!(a.position, b.position);
            assert_eq!(a.events, b.events);
            assert_eq!(a.captured, b.captured);
            assert_eq!(a.analysis.validation, b.analysis.validation);
            assert_eq!(a.analysis.geometry, b.analysis.geometry);
            assert_eq!(a.analysis.legal_vertices, b.analysis.legal_vertices);
            assert_eq!(a.analysis.alive_groups, b.analysis.alive_groups);
            assert_eq!(a.analysis.settled_groups, b.analysis.settled_groups);
            assert_eq!(a.analysis.score, b.analysis.score);
            assert_eq!(a.analysis.outcome, b.analysis.outcome);
        }
        _ => panic!("transition disagreement: expected {expected:?}, actual {actual:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Color, Point, Ruleset, Stone};

    #[test]
    fn strategies_agree_at_tangencies_and_game_boundaries() {
        for radius in [0.05, 0.25, 0.4, 0.499] {
            let mut setups = vec![Position::new(radius, vec![], Color::Black)];
            if radius <= 0.25 {
                for delta in [0.0, 1e-12, 1e-8] {
                    let spacing = 2.0 * radius + delta;
                    let stones = (0..2)
                        .flat_map(|y| {
                            (0..2).map(move |x| {
                                Stone::new(
                                    radius + x as f64 * spacing,
                                    radius + y as f64 * spacing,
                                    if (x + y) % 2 == 0 {
                                        Color::Black
                                    } else {
                                        Color::White
                                    },
                                )
                            })
                        })
                        .collect();
                    let p = Position::new(radius, stones, Color::White);
                    if p.validate().is_playable() {
                        setups.push(p);
                    }
                }
            }
            for setup in setups {
                for rules in [Ruleset::Vgo, Ruleset::Official] {
                    for passes in 0..=2 {
                        let p = setup.clone().with_ruleset(rules).with_passes(passes);
                        let prepared = PreparedPosition::new(&p);
                        let supports = SupportPosition::new(&p, SupportPolicy::All);
                        let diverse = SupportPosition::new(&p, SupportPolicy::DiverseFour);
                        let negative = SupportPosition::new(&p, SupportPolicy::AllWithNegatives);
                        let points = crate::legal_set_vertices(&p).into_iter().chain([
                            Point::new(0.5, 0.5),
                            Point::new(-1.0, 0.5),
                            Point::new(f64::NAN, 0.5),
                        ]);
                        for q in points {
                            let expected = crate::place(&p, q.x, q.y);
                            assert_same(&expected, &prepared.place(q.x, q.y));
                            assert_same(&expected, &supports.place(q.x, q.y));
                            assert_same(&expected, &diverse.place(q.x, q.y));
                            assert_same(&expected, &negative.place(q.x, q.y));
                            for strategy in [
                                Strategy::WitnessFirst,
                                Strategy::Indexed,
                                Strategy::QueryFirst,
                            ] {
                                assert_same(&expected, &place(&p, q.x, q.y, strategy));
                            }
                        }
                    }
                }
            }
        }
    }
}
