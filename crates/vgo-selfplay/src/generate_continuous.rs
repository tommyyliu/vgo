//! Self-play that never stops to fill a shard.
//!
//! `vgo-generate-demo` plays toward a sample target and then drains: actors that
//! finish a game stop taking new ones while the slowest game plays out. That
//! tail is a fixed cost per shard, and it grew teeth when games did. Measured on
//! a multi-radius run at 38 units: 57 minutes reaching the target with 32 actors
//! busy, then 65 minutes with *two* actors busy and thirty cores idle. Half the
//! wall clock at 6% utilisation, for a shard that had already collected 1.66x
//! the samples it asked for.
//!
//! Nothing tuned that away. `--no-drain-tail` trades idle cores for ~32
//! discarded part-games; a second overlapping generator hides the tail behind
//! another shard's productive phase; a larger shard amortises it. All three are
//! working around the same thing: a shard is a batch, and a batch ends when its
//! slowest member does.
//!
//! Here a *game* is the unit. Each completed game is written as its own
//! one-game shard and the actor immediately starts another. There is no target
//! to overshoot, nothing in flight to wait for, and no point at which the
//! machine is asked to do less work than it has cores. The tail cannot exist,
//! rather than being made small.
//!
//! ## Games are grouped by the model that produced them
//!
//! Output goes to `<root>/<label>/game-000000001/`, where the label names the
//! model generation. That is not filing for its own sake:
//!
//!   * **Handoff needs no coordination.** A new model means a new process
//!     writing to its own directory while the old one finishes into its own.
//!     No shared counter, no name races, no locking -- which was most of what
//!     made a hot model swap awkward, and this avoids swapping an
//!     `Arc<dyn Evaluator>` on the per-leaf hot path entirely.
//!   * **Staleness becomes visible.** A replay window spanning thirty model
//!     generations currently says so nowhere; here it is a directory listing.
//!   * **Retirement gets a unit** -- drop a generation, not a file list.
//!
//! Each game directory has the same shape as a shard from the old generator, so
//! the Python loader, the replay cache, komi fitting and retirement all read it
//! unchanged. A game simply is a shard that happens to hold one game.

use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use clap::{ArgAction, Parser};
use sha2::{Digest, Sha256};
use vgo_core::Ruleset;
use vgo_inference::{
    BatchedEvaluatorPool, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind};
use vgo_search::{Evaluator, NaiveEvaluator};
use vgo_selfplay::generation::{
    GameSamples, GameSettings, KOMI_AREA_COEFFICIENT, generate_game, parse_board_mix,
    replay_capacity_for,
};
use vgo_selfplay::replay_stream::{ReplayStream, stone_capacity_for_radius};

#[derive(Parser, Debug, Clone)]
#[command(about = "Continuous self-play: one directory per game, no shard tail")]
struct Config {
    /// Directory that holds one subdirectory per model generation.
    #[arg(long)]
    output_root: PathBuf,
    /// Names this generation, and the directory its games land in. Convention is
    /// `gen-<update>-<model sha prefix>`, which sorts chronologically and says
    /// which weights produced the data.
    #[arg(long)]
    label: String,
    /// Stop cleanly when this path appears: actors finish the game in hand and
    /// the process exits. A signal would race the writer mid-record; a file is
    /// checked between games, where stopping is free.
    #[arg(long)]
    stop_file: Option<PathBuf>,
    /// Stop after this many games. Zero runs until the stop file appears.
    #[arg(long, default_value_t = 0)]
    maximum_games: u64,
    /// First game index, so a restarted generation does not reuse seeds.
    #[arg(long, default_value_t = 0)]
    first_game: u64,

    #[arg(long, default_value_t = 128)]
    resolution: usize,
    #[arg(long, default_value_t = 128)]
    policy_resolution: usize,
    #[arg(long, default_value = "compact-radius")]
    raster_kind: RasterKind,
    #[arg(long, default_value_t = 1600)]
    simulations: u32,
    #[arg(long, default_value_t = 16)]
    coarse_pool: usize,
    #[arg(long, default_value_t = 4)]
    leaf_batch: usize,
    #[arg(long, default_value_t = 321)]
    maximum_candidates: usize,
    #[arg(long, default_value_t = 4.0)]
    widening_coefficient: f64,
    #[arg(long, default_value_t = 0.0)]
    root_exploration_noise: f64,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    #[arg(long, default_value_t = 30)]
    temperature_plies: u32,
    #[arg(long = "max-plies", default_value_t = 70)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 1.0 / 18.0)]
    radius: f64,
    #[arg(long = "board-mix")]
    board_mix: Vec<String>,
    #[arg(long, default_value_t = KOMI_AREA_COEFFICIENT)]
    komi_area_coefficient: f64,
    #[arg(long, default_value_t = 0.017)]
    komi_low: f64,
    #[arg(long, default_value_t = 0.137)]
    komi_high: f64,
    #[arg(long, default_value_t = 1.0)]
    ply_sample_rate: f64,
    #[arg(long, default_value = "vgo")]
    ruleset: Ruleset,
    #[arg(long, default_value_t = 0.0)]
    resign_threshold: f64,
    #[arg(long, default_value_t = 5)]
    resign_window: u32,
    #[arg(long, default_value_t = 20)]
    resign_minimum_ply: u32,
    #[arg(long, default_value_t = 2400)]
    resign_soft_simulations: u32,
    #[arg(long, default_value_t = 0.0)]
    resign_disable_fraction: f64,
    #[arg(long, default_value_t = 50_001)]
    seed: u64,

    #[arg(long, default_value_t = 16)]
    actors: usize,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value_t = 2)]
    inference_slots: usize,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
}

impl Config {
    fn game_settings(&self) -> GameSettings {
        GameSettings {
            policy_resolution: self.policy_resolution,
            simulations: self.simulations,
            coarse_pool: self.coarse_pool,
            temperature: self.temperature,
            temperature_plies: self.temperature_plies,
            leaf_batch: self.leaf_batch,
            maximum_candidates: self.maximum_candidates,
            root_exploration_noise: self.root_exploration_noise,
            widening_coefficient: self.widening_coefficient,
            seed: self.seed,
            radius: self.radius,
            board_mix: parse_board_mix(&self.board_mix).unwrap_or_default(),
            komi_low: self.komi_low,
            komi_high: self.komi_high,
            komi_area_coefficient: self.komi_area_coefficient,
            maximum_plies: self.maximum_plies,
            ruleset: self.ruleset,
            ply_sample_rate: self.ply_sample_rate,
            resign_threshold: self.resign_threshold,
            resign_window: self.resign_window,
            resign_minimum_ply: self.resign_minimum_ply,
            resign_soft_simulations: self.resign_soft_simulations,
            resign_disable_fraction: self.resign_disable_fraction,
        }
    }

    /// The smallest radius this run can play, which is the board that holds the
    /// most stones and therefore sizes the record.
    ///
    /// Taken from `--radius` alone it would fit the mini board and fail on a
    /// standard one *after* the games are played.
    fn smallest_radius(&self) -> f64 {
        parse_board_mix(&self.board_mix)
            .unwrap_or_default()
            .iter()
            .map(|band| 1.0 / band.high_units)
            .fold(self.radius, f64::min)
    }
}

fn file_sha256(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path)?);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Writes one completed game as a one-game shard.
///
/// The directory is built under a temporary name and renamed once the manifest
/// is in place, so a reader can never observe a game that is missing its
/// checksum -- the same discipline the shard writer uses, at a finer grain.
fn write_game(
    root: &Path,
    game_id: u64,
    game: GameSamples,
    config: &Config,
    raster: RasterConfig,
    policy_size: usize,
    model_sha256: Option<&str>,
) -> io::Result<usize> {
    let name = format!("game-{game_id:09}");
    let staging = root.join(format!("{name}.staging"));
    let final_path = root.join(&name);
    if final_path.exists() {
        return Ok(0);
    }
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    let dataset = staging.join("dataset.vgo");
    let mut stream = ReplayStream::create(
        &dataset,
        game.samples.len(),
        raster,
        policy_size,
        0,
        replay_capacity_for(config.maximum_candidates, policy_size),
        stone_capacity_for_radius(config.smallest_radius()),
    )?;
    stream.write_game(game.samples)?;
    let published = stream.publish()?;

    if let Some(record) = game.record.as_ref() {
        let mut writer = BufWriter::new(fs::File::create(staging.join("games.jsonl"))?);
        writeln!(
            writer,
            r#"{{"game":{},"komi":{:.6},"radius":{:.8},"plies":{},"black_utility":{},"reached_ply_cap":{},"resigned":{},"first_sample":0,"sample_count":{}}}"#,
            record.game,
            record.komi,
            record.radius,
            record.plies,
            record.black_utility,
            record.reached_ply_cap,
            record.resigned,
            published.samples,
        )?;
        writer.flush()?;
    }

    let manifest = staging.join("manifest.json");
    let mut writer = BufWriter::new(fs::File::create(&manifest)?);
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema\": \"vgo.replay-shard.v1\",")?;
    writeln!(writer, "  \"dataset\": \"dataset.vgo\",")?;
    writeln!(writer, "  \"games\": \"games.jsonl\",")?;
    writeln!(writer, "  \"samples\": {},", published.samples)?;
    writeln!(writer, "  \"completed_games\": 1,")?;
    writeln!(writer, "  \"channels\": {},", raster.channels())?;
    writeln!(writer, "  \"height\": {},", raster.height)?;
    writeln!(writer, "  \"width\": {},", raster.width)?;
    writeln!(writer, "  \"policy_size\": {},", policy_size)?;
    writeln!(writer, "  \"simulations\": {},", config.simulations)?;
    writeln!(writer, "  \"radius\": {},", config.radius)?;
    writeln!(writer, "  \"ply_sample_rate\": {},", config.ply_sample_rate)?;
    if let Some(sha) = model_sha256 {
        writeln!(writer, "  \"behavior_model_sha256\": \"{sha}\",")?;
    }
    writeln!(writer, "  \"dataset_sha256\": \"{}\"", published.sha256)?;
    writeln!(writer, "}}")?;
    writer.flush()?;

    fs::rename(&staging, &final_path)?;
    Ok(published.samples)
}

fn main() -> io::Result<()> {
    let config = Config::parse();
    if config.actors == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--actors must be positive",
        ));
    }
    let raster = RasterConfig::square_of(config.resolution, config.raster_kind);
    let policy_size = RasterConfig::square(config.policy_resolution).pixels() + 1;
    let root = config.output_root.join(&config.label);
    fs::create_dir_all(&root)?;

    let model_sha256 = match &config.model {
        Some(path) => Some(file_sha256(path)?),
        None => None,
    };

    let (evaluator, _broker): (Arc<dyn Evaluator>, Option<BatchedEvaluatorPool>) =
        match &config.model {
            None => (Arc::new(NaiveEvaluator), None),
            Some(model) => {
                let mut services = Vec::with_capacity(config.inference_slots);
                // Sequential loads: TensorRT reuses its engine cache rather than
                // racing two builders over the same directory.
                for _slot in 0..config.inference_slots {
                    services.push(
                        OnnxBatchService::load(&OnnxServiceConfig {
                            policy: Some(RasterConfig::square(config.policy_resolution)),
                            model: model.clone(),
                            raster,
                            maximum_batch: config.maximum_batch,
                            provider: config.provider,
                            device_id: config.device_id,
                            fp16: config.fp16,
                            cache_directory: config.cache_directory.clone(),
                        })
                        .map_err(io::Error::other)?,
                    );
                }
                let pool = BatchedEvaluatorPool::spawn(
                    BrokerConfig {
                        maximum_delay: Duration::from_millis(config.delay_ms),
                        queue_capacity: (config.actors * 4).max(config.maximum_batch * 2),
                    },
                    services,
                )
                .map_err(io::Error::other)?;
                (Arc::new(pool.clone()), Some(pool))
            }
        };

    // Two signals, not one. `stopping` means take no new games -- the game in
    // hand plays to its end, which is the whole point of making a game the unit.
    // `cancelled` aborts mid-game and is only set when something has actually
    // failed; `generate_game` reports it as an error, so conflating the two
    // turns an orderly stop into a spurious failure in every other actor.
    let stopping = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let next_game = Arc::new(AtomicU64::new(config.first_game));
    let games_written = Arc::new(AtomicU64::new(0));
    let samples_written = Arc::new(AtomicU64::new(0));
    let settings = config.game_settings();

    // Each actor owns the whole cycle: play a game, write it, take the next.
    // There is no collector and no channel, because there is nothing to batch --
    // which is also why no actor can ever be left waiting on another's game.
    let mut handles = Vec::with_capacity(config.actors);
    for _ in 0..config.actors {
        let evaluator = Arc::clone(&evaluator);
        let stopping = Arc::clone(&stopping);
        let cancelled = Arc::clone(&cancelled);
        let next_game = Arc::clone(&next_game);
        let games_written = Arc::clone(&games_written);
        let samples_written = Arc::clone(&samples_written);
        let settings = settings.clone();
        let config = config.clone();
        let root = root.clone();
        let model_sha256 = model_sha256.clone();
        handles.push(thread::spawn(move || -> io::Result<()> {
            loop {
                if stopping.load(Ordering::Acquire) || cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                if let Some(path) = &config.stop_file {
                    if path.exists() {
                        stopping.store(true, Ordering::Release);
                        return Ok(());
                    }
                }
                if config.maximum_games > 0
                    && games_written.load(Ordering::Relaxed) >= config.maximum_games
                {
                    stopping.store(true, Ordering::Release);
                    return Ok(());
                }
                let index = next_game.fetch_add(1, Ordering::Relaxed);
                let game = match generate_game(&settings, evaluator.as_ref(), index, &cancelled) {
                    Ok(game) => game,
                    Err(error) => {
                        cancelled.store(true, Ordering::Release);
                        return Err(io::Error::other(error));
                    }
                };
                // A game with no terminal label teaches nothing; it is dropped
                // rather than written, and the actor moves on.
                if !game.completed || game.samples.is_empty() {
                    continue;
                }
                let written = write_game(
                    &root,
                    index,
                    game,
                    &config,
                    raster,
                    policy_size,
                    model_sha256.as_deref(),
                )?;
                if written > 0 {
                    games_written.fetch_add(1, Ordering::Relaxed);
                    samples_written.fetch_add(written as u64, Ordering::Relaxed);
                }
            }
        }));
    }

    let mut failure = None;
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                failure.get_or_insert(error);
            }
            Err(_) => {
                failure.get_or_insert(io::Error::other("generation actor panicked"));
            }
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{{")?;
    writeln!(out, "  \"label\": \"{}\",", config.label)?;
    writeln!(out, "  \"games\": {},", games_written.load(Ordering::Relaxed))?;
    writeln!(
        out,
        "  \"samples\": {}",
        samples_written.load(Ordering::Relaxed)
    )?;
    writeln!(out, "}}")?;
    Ok(())
}
