//! The browser bot's Rust half.
//!
//! Exists so the client does not become a fourth implementation of the rules.
//! `crates/vgo-core/src/game.rs` and `reference/src/engine/game.js` already
//! carry mutual "must match" comments and have diverged in practice: a
//! placement capturing exactly one stone read as a pass and ended live games,
//! and the fix had to be found twice. Everything below is the same code
//! self-play runs.
//!
//! ## The asynchronous seam
//!
//! Rules, scoring and rasterization expose directly: they are synchronous,
//! pure, and already what the client needs.
//!
//! Search is the one hard part. `vgo_search::search_at_ply` is synchronous — it
//! takes `&dyn Evaluator` and runs to completion — while `session.run()` in
//! onnxruntime-web returns a promise, and the thread that would block on it is
//! the thread that must run the event loop to resolve it. Three ways across,
//! and this crate takes the first:
//!
//!   1. **Stepped search**, `vgo_search::SteppedSearch`: turn the loop inside
//!      out so it yields a batch of leaves, takes results back, and resumes.
//!      Chosen. Nothing to deploy, and the caller owning the loop is what
//!      allows a *time budget* instead of a fixed simulation count.
//!   2. **Asyncify** (`wasm-opt --asyncify`). No search changes, at the cost of
//!      roughly double the binary and a penalty on every boundary crossing.
//!   3. **`Atomics.wait` on a `SharedArrayBuffer`,** search in one worker and
//!      inference in another. Fastest, but needs COOP/COEP on the host — a
//!      constraint on a community site nobody here controls.
//!
//! The stepped path is proven identical to the batched one rather than assumed
//! to be: `stepped_search_matches_the_batched_search` asserts bit-identical
//! visits, priors and values across stone counts, leaf batches and seeds.
//!
//! [`Game::search_naive`] runs the real MCTS against the built-in evaluator with
//! no network at all. It is not the bot -- without a policy the search has no
//! prior worth speaking of -- but it is the weakest possible difficulty and a
//! way to exercise the engine with nothing else present.

use vgo_core::{Analysis, Color, Phase, Position, Stone, is_legal_placement, pass, place};
use vgo_raster::{DensePolicy, RasterConfig, RasterKind, rasterize_any_into};
use vgo_search::{Action, Evaluation, NaiveEvaluator, SearchConfig, SteppedSearch, search_at_ply};
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

    /// Replace the position wholesale, for loading a game from elsewhere.
    ///
    /// `passes` is how many consecutive passes precede this position, and is
    /// not decoration. A search that believes nobody has passed does not know
    /// that passing now would end the game: it cannot pass to close out a win,
    /// and cannot see that passing while behind hands over the result. Hosts
    /// that track their own pass count must pass it through, because a board
    /// full of stones does not carry it.
    #[wasm_bindgen(js_name = setStones)]
    pub fn set_stones(
        &mut self,
        stones: JsValue,
        to_move: &str,
        ply: u32,
        passes: u32,
    ) -> Result<(), JsValue> {
        let views: Vec<StoneView> = serde_wasm_bindgen::from_value(stones)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let parsed = views
            .iter()
            .map(|view| {
                Ok(Stone::new(view.x, view.y, parse_colour(&view.color)?))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let position = Position::new(self.position.radius(), parsed, parse_colour(to_move)?)
            .with_komi(self.position.komi())
            .with_passes(passes);
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
    ///
    /// `coarsePool` and `leafBatch` are not optional in practice and must match
    /// how the model is served elsewhere, because `SearchConfig::canary` is a
    /// test default rather than a playing one and gets both wrong for this use:
    ///
    ///   * `coarse_pool = 0` makes the search draw candidate moves from the
    ///     legacy quasi-random sequence instead of the network's own policy
    ///     map. The search still plays legal moves, so nothing looks broken --
    ///     it is simply no longer guided by the policy head, which is the part
    ///     of the model that decides *where* to look. `vgo-serve-move` uses 4.
    ///   * `leaf_batch = 1` evaluates one position per network call. Correct,
    ///     and about eight times slower than it needs to be in a browser, where
    ///     one inference costs roughly the same whether it carries 1 position
    ///     or 8.
    pub fn search(
        &self,
        simulations: u32,
        seed: u64,
        policy_size: usize,
        coarse_pool: usize,
        leaf_batch: usize,
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
        config.coarse_pool = coarse_pool;
        config.leaf_batch = leaf_batch.max(1);
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
/// const search = game.search(simulations, seed, policySize, coarsePool, leafBatch);
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
