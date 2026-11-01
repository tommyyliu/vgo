#![forbid(unsafe_code)]

mod analysis;
mod connectivity;
mod game;
mod legal_set;
mod model;
mod numeric;
// `!(t > 0.0)` is deliberate: a NaN root must take the rejecting branch.
#[allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]
mod settled;
mod voronoi;

pub use analysis::{Analysis, Outcome, Score, Settlement};
pub use connectivity::{
    CutKind, MAX_PAIR_CUT_DISTANCE, SAFE_PAIR_DISTANCE, connected_pairs, pair_color, pair_cut,
};
pub use game::{GameEvent, MoveError, MoveResult, pass, place};
pub use legal_set::{
    LegalSetIndex, Nearest, contains as is_legal_placement, distance as distance_to_legal_set,
    in_inset as is_inside_legal_inset, nearest as nearest_legal_placement,
    nearest_indexed as nearest_legal_placement_indexed,
    nearest_indexed_within as nearest_legal_placement_indexed_within,
    nearest_with as nearest_legal_placement_with, none_closer_than as no_legal_point_closer_than,
    none_closer_than_except_vertices,
    none_closer_than_indexed as no_legal_point_closer_than_indexed, vertices as legal_set_vertices,
};
pub use model::{Color, Phase, Position, Ruleset, Stone, Validation, ValidationIssue};
pub use numeric::{COORDINATE_EPSILON, distance_to_segment, length as planar_length, length_below};
pub use settled::SettledRegion;
pub use voronoi::{Cell, Edge, EdgeSource, Geometry, Point};
