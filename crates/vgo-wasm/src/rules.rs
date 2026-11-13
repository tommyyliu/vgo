//! The rules, as stateless functions over plain data, for the board client.
//!
//! `client/` draws whatever it likes, but every rule it shows comes from here:
//! which cells the stones own, where a stone may go, what a move removes, and
//! who is ahead. Nothing is kept between calls. A position goes in as a plain
//! object and comes back out as one, so the client can cache positions by game
//! tree node without owning anything in wasm memory.
//!
//! Each call rebuilds the Rust `Position` it is given. That repeats work a
//! long-lived handle would keep (the legal-set index, the cells), and is the
//! price of not managing handles from JavaScript.

use serde::{Deserialize, Serialize};
use vgo_core::{
    Analysis, Color, EdgeSource, MoveError, Phase, Point, Position, Ruleset, Stone,
    nearest_legal_placement, nearest_legal_placement_with, pass, place,
};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct PointIo {
    pub x: f64,
    pub y: f64,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct StoneIo {
    pub x: f64,
    pub y: f64,
    /// "B" or "W".
    pub c: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionIo {
    /// "vgo" or "official".
    pub ruleset: String,
    pub radius: f64,
    pub komi: f64,
    pub stones: Vec<StoneIo>,
    pub to_move: String,
    /// Consecutive passes so far. Whether that ends the game is the rules'
    /// call, reported as `finished` on a played move, never sent in.
    pub passes: u32,
}

#[derive(Serialize)]
pub struct EdgeIo {
    pub a: PointIo,
    pub b: PointIo,
    /// The stone across this edge, or -1 for the board's edge. An edge the
    /// geometry could not classify also reads -1 and is counted in
    /// `unclassifiedEdges`, so a caller can refuse to trust it.
    pub across: i64,
}

#[derive(Serialize)]
pub struct CellIo {
    pub polygon: Vec<PointIo>,
    pub edges: Vec<EdgeIo>,
    pub area: f64,
}

#[derive(Serialize)]
pub struct ScoreIo {
    #[serde(rename = "B")]
    pub black: f64,
    #[serde(rename = "W")]
    pub white: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisIo {
    pub playable: bool,
    pub cells: Vec<CellIo>,
    pub group: Vec<usize>,
    pub settled_groups: Vec<usize>,
    pub legal_vertices: Vec<PointIo>,
    pub score: ScoreIo,
    pub winner: Option<String>,
    pub margin: f64,
    pub unclassified_edges: usize,
    /// No stone fits anywhere, so every point is settled.
    pub legal_set_empty: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum MoveIo {
    Place { x: f64, y: f64 },
    Pass,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayIo {
    /// Absent when the move was played; one of "finished", "invalid-position",
    /// "illegal-placement", "self-capture" when it was refused.
    pub refused: Option<String>,
    pub after: Option<PositionIo>,
    /// Whether the move ended the game.
    pub finished: bool,
    /// Every stone the move took off, the opponent's first, then the mover's
    /// own. Stones are named by value: indices shift once stones go.
    pub removed: Vec<StoneIo>,
    pub analysis: Option<AnalysisIo>,
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn colour_from(text: &str) -> Result<Color, JsValue> {
    match text {
        "B" => Ok(Color::Black),
        "W" => Ok(Color::White),
        other => Err(js_error(format!("colour must be B or W, got {other:?}"))),
    }
}

fn colour_to(colour: Color) -> String {
    match colour {
        Color::Black => "B",
        Color::White => "W",
    }
    .to_string()
}

fn ruleset_from(text: &str) -> Result<Ruleset, JsValue> {
    match text {
        "vgo" => Ok(Ruleset::Vgo),
        "official" => Ok(Ruleset::Official),
        other => Err(js_error(format!(
            "ruleset must be vgo or official, got {other:?}"
        ))),
    }
}

fn ruleset_to(ruleset: Ruleset) -> String {
    match ruleset {
        Ruleset::Vgo => "vgo",
        Ruleset::Official => "official",
    }
    .to_string()
}

fn stone_to(stone: &Stone) -> StoneIo {
    StoneIo {
        x: stone.x,
        y: stone.y,
        c: colour_to(stone.color),
    }
}

fn position_from(io: &PositionIo) -> Result<Position, JsValue> {
    let stones = io
        .stones
        .iter()
        .map(|stone| Ok(Stone::new(stone.x, stone.y, colour_from(&stone.c)?)))
        .collect::<Result<Vec<_>, JsValue>>()?;
    Ok(Position::new(io.radius, stones, colour_from(&io.to_move)?)
        .with_ruleset(ruleset_from(&io.ruleset)?)
        .with_komi(io.komi)
        .with_passes(io.passes))
}

fn position_to(position: &Position) -> PositionIo {
    PositionIo {
        ruleset: ruleset_to(position.ruleset()),
        radius: position.radius(),
        komi: position.komi(),
        stones: position.stones().iter().map(stone_to).collect(),
        to_move: colour_to(position.to_move()),
        passes: position.consecutive_passes(),
    }
}

fn point_to(point: Point) -> PointIo {
    PointIo {
        x: point.x,
        y: point.y,
    }
}

fn analysis_to(position: &Position, analysis: &Analysis) -> AnalysisIo {
    let mut settled: Vec<usize> = analysis.settled_groups.iter().copied().collect();
    settled.sort_unstable();
    let cells = analysis
        .geometry
        .cells
        .iter()
        .map(|cell| CellIo {
            polygon: cell.polygon.iter().copied().map(point_to).collect(),
            edges: cell
                .edges
                .iter()
                .map(|edge| EdgeIo {
                    a: point_to(edge.start),
                    b: point_to(edge.end),
                    across: match edge.source {
                        Some(EdgeSource::Bisector { other, .. }) => other as i64,
                        _ => -1,
                    },
                })
                .collect(),
            area: cell.area,
        })
        .collect();
    AnalysisIo {
        playable: analysis.validation.is_playable(),
        cells,
        group: analysis.geometry.groups.clone(),
        settled_groups: settled,
        legal_vertices: analysis
            .legal_vertices
            .iter()
            .copied()
            .map(point_to)
            .collect(),
        score: ScoreIo {
            black: analysis.score.black,
            white: analysis.score.white,
        },
        winner: analysis.outcome.winner.map(colour_to),
        margin: analysis.outcome.margin,
        unclassified_edges: analysis.geometry.diagnostics.unclassified_edges,
        legal_set_empty: !nearest_legal_placement_with(
            position,
            Point::new(0.5, 0.5),
            Some(&analysis.legal_vertices),
        )
        .legal,
    }
}

fn refused(reason: &str) -> PlayIo {
    PlayIo {
        refused: Some(reason.to_string()),
        after: None,
        finished: false,
        removed: Vec::new(),
        analysis: None,
    }
}

/// Cells, groups, legal-set corners and the score of a position.
#[wasm_bindgen(js_name = rulesAnalyze)]
pub fn rules_analyze(position: JsValue) -> Result<JsValue, JsValue> {
    let io: PositionIo = serde_wasm_bindgen::from_value(position).map_err(js_error)?;
    let position = position_from(&io)?;
    serde_wasm_bindgen::to_value(&analysis_to(&position, &Analysis::new(&position)))
        .map_err(js_error)
}

/// The legal stone centre nearest `(x, y)`, or null when no stone fits
/// anywhere. A point that is already legal comes back unchanged.
#[wasm_bindgen(js_name = rulesSnap)]
pub fn rules_snap(position: JsValue, x: f64, y: f64) -> Result<JsValue, JsValue> {
    let io: PositionIo = serde_wasm_bindgen::from_value(position).map_err(js_error)?;
    let position = position_from(&io)?;
    let nearest = nearest_legal_placement(&position, Point::new(x, y));
    if !nearest.legal {
        return Ok(JsValue::NULL);
    }
    serde_wasm_bindgen::to_value(&point_to(nearest.point)).map_err(js_error)
}

/// Plays `move` for the side to move: the position after it, the stones it
/// removed and the analysis of the result, or the reason it was refused.
#[wasm_bindgen(js_name = rulesPlay)]
pub fn rules_play(position: JsValue, chosen: JsValue) -> Result<JsValue, JsValue> {
    let io: PositionIo = serde_wasm_bindgen::from_value(position).map_err(js_error)?;
    let chosen: MoveIo = serde_wasm_bindgen::from_value(chosen).map_err(js_error)?;
    let position = position_from(&io)?;
    // `place` checks this only in debug builds; a record or a setup from
    // outside can hold overlapping stones, so the door checks it every time.
    let result = if position.validate().is_playable() {
        let outcome = match chosen {
            MoveIo::Place { x, y } => place(&position, x, y),
            MoveIo::Pass => pass(&position),
        };
        match outcome {
            Ok(result) => PlayIo {
                refused: None,
                finished: result.position.phase() != Phase::Playing,
                removed: result.removed.iter().map(stone_to).collect(),
                analysis: Some(analysis_to(&result.position, &result.analysis)),
                after: Some(position_to(&result.position)),
            },
            Err(MoveError::Finished) => refused("finished"),
            Err(MoveError::InvalidPosition) => refused("invalid-position"),
            Err(MoveError::IllegalPlacement) => refused("illegal-placement"),
            Err(MoveError::SelfCapture) => refused("self-capture"),
        }
    } else {
        refused("invalid-position")
    };
    serde_wasm_bindgen::to_value(&result).map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_round_trip_through_the_plain_shape() {
        let position = Position::new(0.05, vec![Stone::new(0.3, 0.7, Color::White)], Color::Black)
            .with_ruleset(Ruleset::Official)
            .with_komi(0.1)
            .with_passes(1);
        let back =
            position_from(&position_to(&position)).unwrap_or_else(|_| panic!("round trip failed"));
        assert_eq!(back, position);
    }

    #[test]
    fn a_full_board_has_no_legal_spot_left() {
        let open = Position::new(0.05, vec![Stone::new(0.5, 0.5, Color::Black)], Color::White);
        assert!(!analysis_to(&open, &Analysis::new(&open)).legal_set_empty);
        // Radius 0.3 on a unit board: one stone at the centre leaves no centre 0.6 away
        // inside the inset square [0.3, 0.7].
        let full = Position::new(0.3, vec![Stone::new(0.5, 0.5, Color::Black)], Color::White);
        assert!(analysis_to(&full, &Analysis::new(&full)).legal_set_empty);
    }
}
