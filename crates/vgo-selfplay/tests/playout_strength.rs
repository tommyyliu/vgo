//! Diagnostic: with an untrained (naive) evaluator, more MCTS playouts should
//! beat fewer. Search adds strength independent of policy quality, so if this
//! fails, something is wrong in search itself.
//!
//! Run with: `cargo test -p vgo-selfplay --test playout_strength -- --ignored --nocapture`

use std::time::Duration;

use vgo_core::{Color, Position};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::RasterConfig;
use vgo_search::{Evaluator, NaiveEvaluator, SearchConfig, search_with_evaluator};
use vgo_selfplay::play_game;

/// Play `pairs * 2` color-swapped games. In each pair, the high-playout config
/// plays black once and white once. Returns the high-playout side's score
/// (win = 1, tie = 0.5) over all games.
fn duel<E: Evaluator>(
    evaluator: &E,
    high: u32,
    low: u32,
    coarse_pool: usize,
    pairs: usize,
    radius: f64,
    maximum_plies: u32,
) -> (f64, usize) {
    let mut high_points = 0.0_f64;
    let mut completed = 0usize;

    let config = |sims: u32| {
        let mut c = SearchConfig::canary(sims);
        c.coarse_pool = coarse_pool;
        c
    };

    for pair in 0..pairs {
        for high_is_black in [true, false] {
            let seed = 0xC0FFEE ^ ((pair as u64) << 8) ^ u64::from(high_is_black);
            let report = play_game(
                Position::new(radius, Vec::new(), Color::Black),
                maximum_plies,
                |position, _ply| {
                    let high_to_move = (position.to_move() == Color::Black) == high_is_black;
                    let sims = if high_to_move { high } else { low };
                    search_with_evaluator(position, config(sims), seed, evaluator)
                },
                |_step| {},
            )
            .expect("playout runs");

            let Some(outcome) = report.outcome else {
                // Truncated / unfinished game: skip (not a win for either side).
                continue;
            };
            completed += 1;
            let high_color = if high_is_black { Color::Black } else { Color::White };
            high_points += match outcome.winner {
                Some(w) if w == high_color => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
        }
    }
    (high_points, completed)
}

#[test]
#[ignore = "slow strength diagnostic; run explicitly"]
fn more_playouts_beat_fewer_with_naive_net() {
    let radius = 1.0 / 6.0;
    let maximum_plies = 80;
    let pairs = 12; // 24 games

    let (points, completed) = duel(&NaiveEvaluator, 128, 8, 0, pairs, radius, maximum_plies);
    let score = points / completed as f64;
    println!(
        "legacy naive: high(128) vs low(8): score={score:.3} over {completed} completed games ({points} pts)"
    );

    assert!(completed >= pairs, "too many unfinished games: {completed}");
    assert!(
        score > 0.5,
        "more playouts did not beat fewer (score {score:.3})"
    );
}

/// Same test, but on the coarse->fine path with a real trained net (so the
/// value head actually drives backup and the fine grid actually exists).
/// Requires ORT_DYLIB_PATH / LD_LIBRARY_PATH and an ONNX model; CPU provider.
#[test]
#[ignore = "needs ONNX runtime + model; run explicitly"]
fn more_playouts_beat_fewer_coarse_fine() {
    let model = std::env::var("VGO_TEST_ONNX").expect("set VGO_TEST_ONNX to a candidate.onnx");
    // Match the training run's board geometry so the net sees in-distribution
    // rasters (radius/resolution mismatch produces NaN "invalid inference value").
    let radius = 1.0 / 18.0;
    let resolution = 128;
    let coarse_pool = 8;
    let maximum_plies = 120;
    let pairs = 8; // 16 games (ONNX CPU is slow)

    let service = OnnxBatchService::load(&OnnxServiceConfig {
        model: model.into(),
        raster: RasterConfig::square(resolution),
        // This fixture predates the decoupled placement grid; its model is
        // raster-coupled.
        policy: None,
        maximum_batch: 16,
        provider: OnnxProvider::Cpu,
        device_id: 0,
        fp16: false,
        cache_directory: std::env::temp_dir(),
    })
    .expect("onnx model loads");
    let evaluator = BatchedEvaluator::spawn(
        BrokerConfig {
            maximum_delay: Duration::from_millis(1),
            queue_capacity: 64,
        },
        service,
    )
    .expect("evaluator spawns");

    let (points, completed) = duel(
        &evaluator,
        128,
        8,
        coarse_pool,
        pairs,
        radius,
        maximum_plies,
    );
    let score = points / completed as f64;
    println!(
        "coarse->fine (pool={coarse_pool}): high(128) vs low(8): score={score:.3} over {completed} completed games ({points} pts)"
    );

    assert!(completed >= pairs, "too many unfinished games: {completed}");
    assert!(
        score > 0.5,
        "more playouts did not beat fewer on coarse->fine path (score {score:.3})"
    );
}
