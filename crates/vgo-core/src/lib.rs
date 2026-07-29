#![forbid(unsafe_code)]

mod analysis;
mod game;
mod legal_set;
mod model;
mod numeric;
mod voronoi;

pub use analysis::{Analysis, Outcome, Score};
pub use game::{GameEvent, MoveError, MoveResult, pass, place};
pub use legal_set::{
    Nearest, contains as is_legal_placement, distance as distance_to_legal_set,
    nearest as nearest_legal_placement, vertices as legal_set_vertices,
};
pub use model::{Color, Phase, Position, Stone, Validation, ValidationIssue};
pub use numeric::COORDINATE_EPSILON;
pub use voronoi::{Cell, Edge, EdgeSource, Geometry, Point};
