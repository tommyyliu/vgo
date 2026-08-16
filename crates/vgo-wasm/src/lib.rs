//! The browser bot's Rust half.
//!
//! Exists so the client does not become a fourth implementation of the rules.
//! `crates/vgo-core/src/game.rs` and `reference/src/engine/game.js` already
//! carry mutual "must match" comments and have diverged in practice: a
//! placement capturing exactly one stone read as a pass and ended live games,
//! and the fix had to be found twice. Everything below is the same code
//! self-play runs.
//!
//! ## What is settled and what is not
//!
//! Rules, scoring and rasterization are straightforward to expose: they are
//! synchronous, pure, and already exactly what the client needs.
//!
//! **Search is not**, and the shape of this crate's eventual API depends on a
//! decision that has not been made. `vgo_search::search_with_evaluator` is
//! synchronous — it takes `&dyn Evaluator` and runs to completion — while
//! `session.run()` in onnxruntime-web is asynchronous. Three ways out, none
//! free:
//!
//!   1. **Stepped search.** Turn the search loop inside out so it yields a
//!      batch of leaves, takes results back, and resumes. Cleanest at runtime,
//!      no deployment constraints, but it means restructuring 1,573 lines of
//!      MCTS.
//!   2. **Asyncify** (`wasm-opt --asyncify`). No search changes at all; the
//!      cost is roughly double the binary and a speed penalty on every call
//!      that crosses the boundary.
//!   3. **`Atomics.wait` on a `SharedArrayBuffer`,** search in one worker and
//!      inference in another. Fastest, but requires COOP/COEP headers on the
//!      host — a constraint on a community site nobody here controls.
//!
//! [`search_naive`] below runs the real MCTS against the built-in evaluator, so
//! search is proven to work under WASM before that decision is taken. It is not
//! the bot; it is the evidence that the hard part is only the seam.

use vgo_core::{Analysis, Color, Phase, Position, Stone, is_legal_placement, pass, place};
use vgo_raster::{DensePolicy, RasterConfig, RasterKind, rasterize_any_into};
use vgo_search::{Action, Evaluation, NaiveEvaluator, Policy, SearchConfig, SteppedSearch, search_at_ply};
use wasm_bindgen::prelude::*;

/// Board radius and komi are per-game; everything else here is per-call.
#[wasm_bindgen]
pub struct Game {
    position: Position,
    ply: u32,
}

/// One stone, as the client sees it. Colours are absolute, unlike the raster.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StoneView {
    pub x: f64,
    pub y: f64,
    /// "black" or "white".
    pub color: String,
}

#[wasm_bindgen]
impl Game {
    /// An empty board.
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f64, komi: f64) -> Game {
        Game {
            position: Position::new(radius, Vec::new(), Color::Black).with_komi(komi),
            ply: 0,
        }
    }

    /// Place a stone for the side to move.
    ///
    /// Returns false when the placement is rejected. A placement that leaves
    /// the board unchanged is a *pass*, not an error, and is accepted — that
    /// distinction is the one the two implementations got wrong.
    pub fn place(&mut self, x: f64, y: f64) -> bool {
        match place(&self.position, x, y) {
            Ok(result) => {
                self.position = result.position;
                self.ply += 1;
                true
            }
            Err(_) => false,
        }
    }

    /// Returns false when the game has already finished.
    pub fn pass(&mut self) -> bool {
        match pass(&self.position) {
            Ok(result) => {
                self.position = result.position;
                self.ply += 1;
                true
            }
            Err(_) => false,
        }
    }

    /// Whether a stone centre may legally be placed here.
    pub fn is_legal(&self, x: f64, y: f64) -> bool {
        is_legal_placement(&self.position, x, y)
    }

    #[wasm_bindgen(getter)]
    pub fn ply(&self) -> u32 {
        self.ply
    }

    #[wasm_bindgen(getter)]
    pub fn radius(&self) -> f64 {
        self.position.radius()
    }

    /// "black" or "white".
    #[wasm_bindgen(getter, js_name = toMove)]
    pub fn to_move(&self) -> String {
        colour_name(self.position.to_move())
    }

    /// True once the game has ended, by two passes or otherwise.
    #[wasm_bindgen(getter)]
    pub fn finished(&self) -> bool {
        self.position.phase() != Phase::Playing
    }

    /// Area held by Black minus area held by White. Komi is carried on the
    /// position and is the client's business to apply when reporting a result.
    #[wasm_bindgen(js_name = blackMargin)]
    pub fn black_margin(&self) -> f64 {
        let score = Analysis::new(&self.position).score;
        score.black - score.white
    }

    /// Every stone on the board, as `StoneView[]`.
    pub fn stones(&self) -> Result<JsValue, JsValue> {
        let stones: Vec<StoneView> = self
            .position
            .stones()
            .iter()
            .map(|stone| StoneView {
                x: stone.x,
                y: stone.y,
                color: colour_name(stone.color),
            })
            .collect();
        serde_wasm_bindgen::to_value(&stones).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Replace the position wholesale, for loading a game from the server.
    #[wasm_bindgen(js_name = setStones)]
    pub fn set_stones(&mut self, stones: JsValue, to_move: &str, ply: u32) -> Result<(), JsValue> {
        let views: Vec<StoneView> = serde_wasm_bindgen::from_value(stones)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let parsed = views
            .iter()
            .map(|view| {
                Ok(Stone::new(view.x, view.y, parse_colour(&view.color)?))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let position = Position::new(self.position.radius(), parsed, parse_colour(to_move)?)
            .with_komi(self.position.komi());
        if !position.validate().is_playable() {
            return Err(JsValue::from_str("position is not playable"));
        }
        self.position = position;
        self.ply = ply;
        Ok(())
    }

    /// The model input for the current position: `channels * size * size`
    /// floats, channel-major, exactly what self-play feeds the network.
    ///
    /// Built with the distance-transform `settled` channel, which differs from
    /// the training-time rasterizer on at most a pixel or two of 16384 — far
    /// below what fp16 inference rounds away, and measured *closer* to the
    /// definition than the implementation the model was trained against.
    pub fn raster(&self, size: usize) -> Result<Vec<f32>, JsValue> {
        if size == 0 {
            return Err(JsValue::from_str("raster size must be positive"));
        }
        let config = RasterConfig::square_of(size, RasterKind::Compact);
        let mut data = vec![0.0_f32; config.channels() * config.pixels()];
        rasterize_any_into(&self.position, config, &mut data);
        Ok(data)
    }

    /// Begin a network-driven search that JavaScript drives.
    ///
    /// `policySize` is the model's policy output width, `resolution² + 1`.
    pub fn search(
        &self,
        simulations: u32,
        seed: u64,
        policy_size: usize,
    ) -> Result<Search, JsValue> {
        if self.position.phase() != Phase::Playing {
            return Err(JsValue::from_str("cannot search a finished position"));
        }
        if policy_size < 2 {
            return Err(JsValue::from_str("policy size must include the pass entry"));
        }
        let resolution = (((policy_size - 1) as f64).sqrt()).round() as usize;
        if resolution * resolution + 1 != policy_size {
            return Err(JsValue::from_str(
                "policy size must be a square grid plus the pass entry",
            ));
        }
        let mut config = SearchConfig::canary(simulations);
        config.temperature = 0.0;
        Ok(Search {
            inner: SteppedSearch::new(self.position.clone(), config, seed, self.ply),
            outstanding: 0,
            policy_size,
        })
    }

    /// Run the real MCTS against the built-in evaluator, with no network.
    ///
    /// This exists to prove search runs under WASM, and as the weakest possible
    /// difficulty. It is *not* the bot: without the network the search has no
    /// prior worth speaking of, and the project's own measurements put the
    /// policy head as the ceiling even when it is present.
    ///
    /// Returns `[x, y]`, or an empty array when the best action is a pass.
    #[wasm_bindgen(js_name = searchNaive)]
    pub fn search_naive(&self, simulations: u32, seed: u64) -> Result<Vec<f64>, JsValue> {
        if self.position.phase() != Phase::Playing {
            return Err(JsValue::from_str("cannot search a finished position"));
        }
        let mut config = SearchConfig::canary(simulations);
        config.temperature = 0.0;
        let result = search_at_ply(&self.position, config, seed, &NaiveEvaluator, self.ply)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(match result.action {
            Action::Place(point) => vec![point.x, point.y],
            Action::Pass => Vec::new(),
        })
    }
}

/// A search the JavaScript side drives, so it can `await` inference.
///
/// The loop lives in JS:
///
/// ```js
/// const search = game.search(simulations, seed);
/// while (!search.finished) {
///   const batch = search.nextBatch(rasterSize);   // Float32Array, n * C*H*W
///   const { values, policies } = await infer(batch);
///   search.submit(values, policies);
/// }
/// const move = search.best();                      // [x, y] or []
/// ```
///
/// Stopping early is allowed and expected: call `best()` whenever the time
/// budget runs out. That is the point of this shape — a fixed simulation count
/// is the wrong contract when the same code runs on a desktop and a phone.
#[wasm_bindgen]
pub struct Search {
    inner: SteppedSearch,
    /// Size of the last batch handed out, so `submit` can check the counts.
    outstanding: usize,
    policy_size: usize,
}

#[wasm_bindgen]
impl Search {
    /// Positions needing evaluation, rasterized and concatenated:
    /// `count * channels * size * size` floats. Empty when the search is done.
    #[wasm_bindgen(js_name = nextBatch)]
    pub fn next_batch(&mut self, size: usize) -> Result<Vec<f32>, JsValue> {
        if size == 0 {
            return Err(JsValue::from_str("raster size must be positive"));
        }
        let config = RasterConfig::square_of(size, RasterKind::Compact);
        let stride = config.channels() * config.pixels();
        let batch = self.inner.next_batch();
        self.outstanding = batch.len();
        let mut data = vec![0.0_f32; batch.len() * stride];
        for (index, position) in batch.iter().enumerate() {
            rasterize_any_into(position, config, &mut data[index * stride..(index + 1) * stride]);
        }
        Ok(data)
    }

    /// Hand back one value per position and one policy row per position.
    ///
    /// `policies` is `count * policySize` floats, in the same order as the
    /// batch. Values are from the side to move at each position.
    pub fn submit(&mut self, values: Vec<f32>, policies: Vec<f32>) -> Result<(), JsValue> {
        if values.len() != self.outstanding {
            return Err(JsValue::from_str(&format!(
                "expected {} values, got {}",
                self.outstanding,
                values.len()
            )));
        }
        if policies.len() != self.outstanding * self.policy_size {
            return Err(JsValue::from_str(&format!(
                "expected {} policy floats, got {}",
                self.outstanding * self.policy_size,
                policies.len()
            )));
        }
        let evaluations = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let row = &policies[index * self.policy_size..(index + 1) * self.policy_size];
                Evaluation::new(
                    f64::from(*value).clamp(-1.0, 1.0),
                    Box::new(DensePolicy::new(
                        RasterConfig::square(self.policy_resolution()),
                        row.to_vec(),
                    )),
                )
            })
            .collect::<Vec<_>>();
        self.outstanding = 0;
        self.inner
            .submit(evaluations)
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    fn policy_resolution(&self) -> usize {
        // policy_size is resolution^2 + 1, the extra entry being the pass.
        (((self.policy_size - 1) as f64).sqrt()).round() as usize
    }

    #[wasm_bindgen(getter)]
    pub fn finished(&self) -> bool {
        self.inner.finished()
    }

    #[wasm_bindgen(getter)]
    pub fn simulations(&self) -> u32 {
        self.inner.simulations()
    }

    /// The chosen move as `[x, y]`, or an empty array for a pass.
    ///
    /// May be called before the simulation budget is spent — stopping on a
    /// deadline is the expected use, not an error.
    pub fn best(self) -> Result<Vec<f64>, JsValue> {
        let action = self
            .inner
            .best_action()
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(match action {
            Action::Place(point) => vec![point.x, point.y],
            Action::Pass => Vec::new(),
        })
    }
}

fn colour_name(colour: Color) -> String {
    match colour {
        Color::Black => "black".to_string(),
        Color::White => "white".to_string(),
    }
}

fn parse_colour(name: &str) -> Result<Color, JsValue> {
    match name {
        "black" => Ok(Color::Black),
        "white" => Ok(Color::White),
        other => Err(JsValue::from_str(&format!("unknown colour {other}"))),
    }
}
