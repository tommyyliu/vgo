#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::{ArgAction, Parser};
use vgo_core::{Color, Outcome, Point, Position, Ruleset};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind};
use vgo_search::{Action, EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, search_with_evaluator};
use vgo_selfplay::{ADJUDICATION_PLIES, award_by_area, play_game as run_playout};

#[derive(Debug, Parser)]
#[command(about = "Run a color-swapped held-out arena for an ONNX candidate")]
struct Arguments {
    #[arg(long)]
    candidate: PathBuf,
    /// Channel layout the candidate was exported with. Each model reads only
    /// its own layout, so a semantic and an RGB model can still play: both see
    /// the same positions, each rendered the way it was trained to read.
    #[arg(long, default_value = "semantic")]
    candidate_raster_kind: RasterKind,
    /// Layout for every --opponent. Defaults to the candidate's.
    #[arg(long)]
    opponent_raster_kind: Option<RasterKind>,
    /// Repeatable. Each opponent plays `--pairs` color-swapped pairs against the
    /// same loaded candidate and emits its own JSON record. Batching them here
    /// amortizes the provider's per-process model load and warmup, which is
    /// ~21s against ~0.93s per additional pair: six opponents in one process
    /// costs about what one does. Omit entirely to play the naive evaluator.
    #[arg(long)]
    opponent: Vec<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pairs: usize,
    #[arg(long, default_value_t = 16)]
    simulations: u32,
    /// Simulations for the opponent seat, when it should differ from the
    /// candidate's. Ratings need both seats equal, so this is not for measuring
    /// strength -- it is for asking what search itself is worth: play a model
    /// against *itself* at two budgets and the score is how much the extra
    /// search buys on top of the network's own priors. A plateau where doubling
    /// search changes nothing says the targets can no longer improve on the
    /// policy that generated them. Defaults to `--simulations`.
    #[arg(long)]
    opponent_simulations: Option<u32>,
    /// Fine cells per coarse sampling region; zero uses legacy candidates.
    #[arg(long, default_value_t = 0)]
    coarse_pool: usize,
    /// Leaves evaluated together per simulation round; above one a single game
    /// keeps that many evaluations in flight. Both seats must use the same value
    /// for a fair comparison, since it changes which nodes get explored.
    #[arg(long, default_value_t = 1)]
    leaf_batch: usize,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// Placement grid the policy head emits; independent of the render
    /// resolution. Must match the value the model was exported with.
    #[arg(long, default_value_t = 32)]
    policy_resolution: usize,
    #[arg(long, default_value_t = 96)]
    resolution: usize,
    #[arg(long, default_value_t = 1.0 / 6.0)]
    radius: f64,
    /// Komi every arena game is played at.
    ///
    /// Rating at komi 0 measures a model outside the distribution it trained
    /// on: generation draws komi per game, and at 0 the game is not close to
    /// balanced -- Black took 85% of the lowest bucket in the run this was
    /// built for. Fixed rather than drawn so both seats of a colour-swapped
    /// pair face the same game.
    #[arg(long, default_value_t = 0.0)]
    komi: f64,
    /// Which rules to play. Must match the models being compared: a result
    /// measured under one ruleset says nothing about play under the other.
    #[arg(long, default_value = "vgo")]
    ruleset: Ruleset,
    #[arg(long, default_value_t = 900_001)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
    /// Write each game as SGF here. Off by default; a rating pass is cheap
    /// enough to record and its games are the evidence behind the number.
    #[arg(long)]
    sgf_directory: Option<PathBuf>,
}

#[derive(Clone)]
struct GameResult {
    completed: bool,
    outcome: Option<Outcome>,
    candidate_color: Color,
    plies: u32,
    /// The moves played, kept only when `--sgf-directory` asks for them.
    ///
    /// Arena games are few -- a rating pass is hundreds where self-play is
    /// hundreds of thousands -- and they are the ones worth reading, because
    /// they are what a rating is computed from. An undecided rate has no
    /// explanation without them.
    moves: Vec<(Color, Option<Point>)>,
}

fn load_model(
    model: PathBuf,
    kind: RasterKind,
    arguments: &Arguments,
) -> Result<BatchedEvaluator, EvaluationError> {
    let service = OnnxBatchService::load(&OnnxServiceConfig {
        model,
        raster: RasterConfig::square_of(arguments.resolution, kind),
        policy: Some(RasterConfig::square(arguments.policy_resolution)),
        maximum_batch: arguments.maximum_batch,
        provider: arguments.provider,
        device_id: arguments.device_id,
        fp16: arguments.fp16,
        cache_directory: arguments.cache_directory.clone(),
    })?;
    BatchedEvaluator::spawn(
        BrokerConfig {
            maximum_delay: Duration::from_millis(arguments.delay_ms),
            queue_capacity: (arguments.threads * 4).max(arguments.maximum_batch * 2),
        },
        service,
    )
}

fn search_config(simulations: u32, coarse_pool: usize, leaf_batch: usize) -> SearchConfig {
    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = coarse_pool;
    config.leaf_batch = leaf_batch.max(1);
    config
}

fn validate_arguments(arguments: &Arguments) -> Result<(), &'static str> {
    if arguments.pairs == 0
        || arguments.simulations == 0
        || arguments.maximum_plies == 0
        || arguments.threads == 0
        || arguments.maximum_batch == 0
        || arguments.resolution == 0
        || arguments.policy_resolution == 0
        || arguments.device_id < 0
    {
        return Err("arena counts, simulations, and dimensions must be positive");
    }
    if arguments.opponent_simulations == Some(0) {
        return Err("--opponent-simulations must be positive");
    }
    if arguments.coarse_pool > arguments.policy_resolution {
        return Err("--coarse-pool must not exceed --policy-resolution");
    }
    if !arguments.radius.is_finite() || arguments.radius <= 0.0 || arguments.radius >= 0.5 {
        return Err("--radius must be finite and between zero and one half");
    }
    Ok(())
}

/// One game as SGF, in the dialect `reference/src/io/vgo-sgf.js` reads.
///
/// Coordinates print with `{}` rather than a fixed precision: placements come
/// from a continuous space and a snapped move can sit within 1e-6 of a
/// constraint, so a rounded file would not reload as the position that was
/// played. Rust's float Display is already the shortest round-tripping form.
fn game_sgf(result: &GameResult, radius: f64, komi: f64) -> String {
    let mut text = format!(
        "(;FF[4]GM[VGO]SZ[1]RA[{radius}]KM[{komi}]PL[B]"
    );
    for (colour, action) in &result.moves {
        let tag = match colour {
            Color::Black => "B",
            Color::White => "W",
        };
        match action {
            Some(point) => {
                text.push_str(&format!(";{tag}[{},{}]", point.x, point.y));
            }
            None => text.push_str(&format!(";{tag}[]")),
        }
    }
    // Result in SGF's own vocabulary, so a reader knows how the game ended
    // rather than inferring it from the move count.
    let outcome = match result.outcome {
        Some(outcome) => match outcome.winner {
            Some(Color::Black) => format!("B+{:.3}", outcome.margin),
            Some(Color::White) => format!("W+{:.3}", outcome.margin),
            None => "0".to_owned(),
        },
        None => "Void".to_owned(),
    };
    text.push_str(&format!("RE[{outcome}])"));
    text
}

fn play_game(
    candidate: &BatchedEvaluator,
    opponent: Option<&BatchedEvaluator>,
    candidate_color: Color,
    seed: u64,
    arguments: &Arguments,
) -> Result<GameResult, EvaluationError> {
    let naive = NaiveEvaluator;
    // The closing positions, for adjudicating a game that reaches the cap.
    // Only the last ADJUDICATION_PLIES are needed, so this stays bounded rather
    // than retaining the whole game.
    let mut closing: VecDeque<Position> = VecDeque::with_capacity(ADJUDICATION_PLIES);
    let record_moves = arguments.sgf_directory.is_some();
    let mut moves: Vec<(Color, Option<Point>)> = Vec::new();
    let playout = run_playout(
        Position::new(arguments.radius, Vec::new(), Color::Black)
            .with_ruleset(arguments.ruleset)
            .with_komi(arguments.komi),
        arguments.maximum_plies,
        |position, _| {
            let is_candidate = position.to_move() == candidate_color;
            let evaluator: &dyn Evaluator = if is_candidate {
                candidate
            } else if let Some(opponent) = opponent {
                opponent
            } else {
                &naive
            };
            let simulations = if is_candidate {
                arguments.simulations
            } else {
                arguments
                    .opponent_simulations
                    .unwrap_or(arguments.simulations)
            };
            search_with_evaluator(
                position,
                search_config(simulations, arguments.coarse_pool, arguments.leaf_batch),
                seed,
                evaluator,
            )
        },
        |step| {
            if closing.len() == ADJUDICATION_PLIES {
                closing.pop_front();
            }
            closing.push_back(step.transition.position.clone());
            if record_moves {
                moves.push((
                    step.position.to_move(),
                    match step.action {
                        Action::Place(point) => Some(point),
                        Action::Pass => None,
                    },
                ));
            }
        },
    )?;
    // A game that runs out of plies is awarded by area, the same rule self-play
    // uses. Discarding it instead is not neutral: at a 100-ply cap 69% of arena
    // games went undecided, and the ones that survive are those that resolve
    // quickly, which reflects playing style rather than strength.
    // A rating pass cannot afford to refuse: a discarded arena game is a
    // missing result, not a lost sample, and the games it refuses are exactly
    // the close ones between similar models that a rating most needs.
    let outcome = playout
        .outcome
        .or_else(|| closing.back().map(award_by_area));
    Ok(GameResult {
        completed: outcome.is_some(),
        outcome,
        candidate_color,
        plies: playout.stats.plies,
        moves,
    })
}

fn wilson_interval(points: f64, games: usize) -> (f64, f64) {
    if games == 0 {
        return (0.0, 1.0);
    }
    let count = games as f64;
    let proportion = points / count;
    let z = 1.959_963_984_540_054;
    let denominator = 1.0 + z * z / count;
    let center = (proportion + z * z / (2.0 * count)) / denominator;
    let margin = z
        * ((proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)).sqrt())
        / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arc::new(Arguments::parse());
    if let Err(message) = validate_arguments(&arguments) {
        return Err(message.into());
    }
    let candidate = load_model(
        arguments.candidate.clone(),
        arguments.candidate_raster_kind,
        &arguments,
    )?;
    // One record per opponent, or a single naive-evaluator record when none are
    // given. The candidate is loaded once and reused across every match.
    let opponents: Vec<Option<PathBuf>> = if arguments.opponent.is_empty() {
        vec![None]
    } else {
        arguments.opponent.iter().cloned().map(Some).collect()
    };
    for (index, opponent_path) in opponents.into_iter().enumerate() {
        let opponent = opponent_path
            .clone()
            .map(|model| {
                load_model(
                    model,
                    arguments
                        .opponent_raster_kind
                        .unwrap_or(arguments.candidate_raster_kind),
                    &arguments,
                )
            })
            .transpose()?;
        // Distinct seeds per opponent, or every match would replay one game set.
        let seed_base = arguments.seed + (index as u64) * 1_000_003;
        run_match(
            &arguments,
            &candidate,
            opponent.as_ref(),
            opponent_path.as_deref(),
            seed_base,
        )?;
    }
    Ok(())
}

fn run_match(
    arguments: &Arc<Arguments>,
    candidate: &BatchedEvaluator,
    opponent: Option<&BatchedEvaluator>,
    opponent_path: Option<&std::path::Path>,
    seed_base: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let games = arguments.pairs * 2;
    let next_game = Arc::new(AtomicUsize::new(0));
    // Broker counters are cumulative over the process, and the candidate is
    // shared across every opponent, so report this match's delta.
    let baseline = candidate.metrics();
    let started = Instant::now();
    let mut handles = Vec::with_capacity(arguments.threads);
    for _ in 0..arguments.threads {
        let arguments = Arc::clone(arguments);
        let candidate = candidate.clone();
        let opponent = opponent.cloned();
        let next_game = Arc::clone(&next_game);
        handles.push(thread::spawn(move || {
            let mut results = Vec::new();
            loop {
                let game = next_game.fetch_add(1, Ordering::Relaxed);
                if game >= arguments.pairs * 2 {
                    break;
                }
                let candidate_color = if game.is_multiple_of(2) {
                    Color::Black
                } else {
                    Color::White
                };
                results.push(play_game(
                    &candidate,
                    opponent.as_ref(),
                    candidate_color,
                    seed_base + (game / 2) as u64,
                    &arguments,
                )?);
            }
            Ok::<_, EvaluationError>(results)
        }));
    }
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("arena worker"))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let elapsed = started.elapsed().as_secs_f64();
    if let Some(directory) = arguments.sgf_directory.as_ref() {
        std::fs::create_dir_all(directory)?;
        // The opponent names the match; no path is the naive player.
        let label = opponent_path
            .and_then(|path| path.parent())
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("naive")
            .to_owned();
        for (index, result) in results.iter().enumerate() {
            let seat = match result.candidate_color {
                Color::Black => "B",
                Color::White => "W",
            };
            // Every game is decided now, so "decided" says nothing. What is
            // worth finding in a directory listing is which games stalled to
            // the cap rather than ending on their own.
            let status = if result.plies >= arguments.maximum_plies {
                "capped"
            } else {
                "natural"
            };
            let path = directory.join(format!(
                "{label}-game{index:03}-cand{seat}-{status}.sgf"
            ));
            std::fs::write(path, game_sgf(result, arguments.radius, arguments.komi))?;
        }
    }
    let mut wins = 0_usize;
    let mut losses = 0_usize;
    let mut draws = 0_usize;
    let mut completed = 0_usize;
    let mut points = 0.0;
    let mut plies = 0_u64;
    // Games that ran out of plies rather than ending on their own. A hundred
    // plies is far past where the board settles, so a high share means the
    // sides are stalling rather than that the game is long.
    let mut capped = 0_usize;
    for result in &results {
        plies += u64::from(result.plies);
        if result.plies >= arguments.maximum_plies {
            capped += 1;
        }
        if !result.completed {
            continue;
        }
        completed += 1;
        match result.outcome.expect("completed game has outcome").winner {
            Some(winner) if winner == result.candidate_color => {
                wins += 1;
                points += 1.0;
            }
            Some(_) => losses += 1,
            None => {
                draws += 1;
                points += 0.5;
            }
        }
    }
    // No decided game is no evidence, not a defeat -- but `candidate_score`
    // has to stay a JSON number, so `completed` is what callers must test.
    // Reporting 0.0 and letting a consumer read it as a result fed the
    // Bradley-Terry fit a fabricated shutout: one ddrnet-vs pairing decided
    // none of its 16 games and was recorded as the candidate losing all 16.
    let score = if completed == 0 {
        0.0
    } else {
        points / completed as f64
    };
    let interval = wilson_interval(points, completed);
    let current = candidate.metrics();
    let candidate_metrics = current.delta_since(baseline);
    let opponent_name = if opponent_path.is_some() {
        "onnx"
    } else {
        "naive"
    };
    let opponent_model = opponent_path
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    println!(
        concat!(
            "{{\n",
            "  \"schema\": \"vgo.arena.v1\",\n",
            "  \"opponent\": \"{}\",\n",
            "  \"opponent_model\": \"{}\",\n",
            "  \"pairs\": {},\n",
            "  \"games\": {},\n",
            "  \"completed\": {},\n",
            "  \"truncated\": {},\n",
            "  \"reached_ply_cap\": {},\n",
            "  \"candidate_wins\": {},\n",
            "  \"candidate_losses\": {},\n",
            "  \"draws\": {},\n",
            "  \"candidate_score\": {:.6},\n",
            "  \"score_ci95\": [{:.6}, {:.6}],\n",
            "  \"simulations_per_move\": {},\n",
            "  \"opponent_simulations_per_move\": {},\n",
            "  \"coarse_pool\": {},\n",
            "  \"average_plies\": {:.3},\n",
            "  \"wall_seconds\": {:.6},\n",
            "  \"model_evaluations\": {},\n",
            "  \"model_batches\": {},\n",
            "  \"encoding_seconds\": {:.3},\n",
            "  \"queue_seconds\": {:.3},\n",
            "  \"inference_seconds\": {:.3},\n",
            "  \"failures\": {}\n",
            "}}"
        ),
        opponent_name,
        opponent_model,
        arguments.pairs,
        games,
        completed,
        games - completed,
        capped,
        wins,
        losses,
        draws,
        score,
        interval.0,
        interval.1,
        arguments.simulations,
        arguments
            .opponent_simulations
            .unwrap_or(arguments.simulations),
        arguments.coarse_pool,
        plies as f64 / games as f64,
        elapsed,
        candidate_metrics.requests,
        candidate_metrics.batches,
        candidate_metrics.encoding_nanoseconds as f64 / 1e9,
        candidate_metrics.queue_nanoseconds as f64 / 1e9,
        candidate_metrics.inference_nanoseconds as f64 / 1e9,
        candidate_metrics.failures,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Arguments, search_config, validate_arguments, wilson_interval};

    #[test]
    fn empty_arena_interval_is_uninformative() {
        assert_eq!(wilson_interval(0.0, 0), (0.0, 1.0));
    }

    #[test]
    fn coarse_pool_cli_defaults_to_legacy_and_accepts_an_override() {
        let default = Arguments::try_parse_from(["vgo-arena", "--candidate", "candidate.onnx"])
            .expect("default CLI parses");
        assert_eq!(default.coarse_pool, 0);

        let configured = Arguments::try_parse_from([
            "vgo-arena",
            "--candidate",
            "candidate.onnx",
            "--coarse-pool",
            "8",
        ])
        .expect("coarse sampling options parse");
        assert_eq!(configured.coarse_pool, 8);
    }

    #[test]
    fn coarse_sampling_is_applied_to_search_config() {
        let configured = search_config(19, 8, 1);
        assert_eq!(configured.simulations, 19);
        assert_eq!(configured.coarse_pool, 8);
    }

    #[test]
    fn invalid_coarse_sampling_config_is_rejected_before_arena() {
        // The pool counts fine cells per coarse region on the placement grid,
        // which is now independent of the render resolution.
        let oversized_pool = Arguments::try_parse_from([
            "vgo-arena",
            "--candidate",
            "candidate.onnx",
            "--policy-resolution",
            "16",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(
            validate_arguments(&oversized_pool),
            Err("--coarse-pool must not exceed --policy-resolution")
        );

        // A pool larger than the raster but within the placement grid is fine:
        // the two resolutions are unrelated.
        let coarse_raster = Arguments::try_parse_from([
            "vgo-arena",
            "--candidate",
            "candidate.onnx",
            "--resolution",
            "16",
            "--policy-resolution",
            "32",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(validate_arguments(&coarse_raster), Ok(()));
    }

    #[test]
    fn invalid_radius_is_rejected_before_arena() {
        for radius in ["0", "0.5", "NaN"] {
            let arguments = Arguments::try_parse_from([
                "vgo-arena",
                "--candidate",
                "candidate.onnx",
                "--radius",
                radius,
            ])
            .expect("CLI syntax parses");
            assert_eq!(
                validate_arguments(&arguments),
                Err("--radius must be finite and between zero and one half")
            );
        }
    }
}
