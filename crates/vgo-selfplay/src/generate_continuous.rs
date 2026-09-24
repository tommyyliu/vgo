//! Self-play that never stops to fill a shard.
//!
//! The shard generator this replaced (`vgo-generate-demo`) played toward a
//! sample target and then drained: actors that finished a game stopped taking
//! new ones while the slowest game played out. That tail was a fixed cost per
//! shard, and it grew teeth when games did. Measured on
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
    /// Block rather than spin while waiting for the GPU.
    ///
    /// CUDA's default busy-polls, which costs a core per inference lane for the
    /// whole 7.6 ms a session takes. Blocking wakes in tens of microseconds --
    /// under a percent of that -- and frees the core. Measured at 43% of all
    /// CPU samples across two lanes, both pinned at 100% while the GPU was the
    /// thing actually working.
    ///
    /// Measured, 32 actors, two lanes, GPU pinned at 100% either way:
    ///
    ///     spin    5.0 cores total, 2.0 in the inference threads
    ///     block   2.8 cores total, 0.2 in the inference threads
    ///
    /// On by default. Must be set before any session exists, so it happens
    /// first in `main`; a failure is reported and the run continues, spinning.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    blocking_sync: bool,
    /// Write a flamegraph here on exit, sampling every thread at 199 Hz.
    ///
    /// Requires the `profiling` feature; without it this errors rather than
    /// running unprofiled, because a profiling run that silently produced no
    /// profile would be indistinguishable from one that found nothing.
    ///
    /// 199 rather than 200: a prime sampling rate cannot beat against a loop
    /// that happens to run at a round frequency, which is how a hot function
    /// hides from a profiler entirely.
    #[arg(long)]
    profile_output: Option<PathBuf>,
    /// Write the profile after this many seconds and exit.
    ///
    /// Without it the profile lands when the process does, and this process
    /// exits only after every actor finishes the game in hand -- which on a
    /// 38-unit board is over an hour. A profile of the search does not need a
    /// whole game, it needs representative positions, so this takes a bounded
    /// slice and stops. Zero waits for a normal exit.
    #[arg(long, default_value_t = 0)]
    profile_seconds: u64,
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
    mut game: GameSamples,
    config: &Config,
    raster: RasterConfig,
    policy_size: usize,
    model_sha256: Option<&str>,
) -> io::Result<usize> {
    let name = format!("game-{game_id:09}");
    let staging = root.join(format!("{name}.staging"));
    let final_path = root.join(&name);
    if final_path.exists() {
        // Already written, so the game in hand is a duplicate of it and there is
        // nothing to do. Say so: reaching here means the index range is being
        // replayed, and every game the run finishes from now on is discarded
        // after paying its full search cost, with nothing else looking wrong.
        eprintln!(
            "warning: {name} already exists in {}; discarding the finished game. \
             Game indices are being reused -- check --first-game.",
            root.display()
        );
        return Ok(0);
    }
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    // Taken before `game.samples` is moved into the stream below.
    //
    // One row per (threshold, window) the rule could have used, describing what
    // it would have done to *this* game: whether it would have conceded, whether
    // that concession would have been wrong, and how many plies it would have
    // skipped. Under soft resignation every game calibrates, because the game
    // plays on to a real terminal state and the counterfactual is therefore
    // known rather than sampled.
    //
    // Written raw, per game, rather than pre-pooled. The generator does not know
    // which window a future run will pool over, and a shard-level aggregate
    // cannot be un-summed; 36 rows of small integers per game is cheaper than
    // regenerating.
    //
    // `scripts/resign-calibration.py` is the consumer: it pools these over the
    // trailing window, requires at least thirty firings before trusting a rate,
    // and picks the lowest threshold whose false-positive rate stays under the
    // target. Nothing qualifying leaves the fallback in place, which is what
    // makes adaptation safe to leave on while the value head is still learning.
    let calibration = std::mem::take(&mut game.calibration);

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
        // The final board and both areas are what the score and ownership
        // targets are built from. They cannot be recovered afterwards -- the
        // last recorded position is one move short of the end -- so every
        // game written without them is score data lost for good.
        writeln!(
            writer,
            concat!(
                r#"{{"game":{},"komi":{:.6},"radius":{:.8},"plies":{},"passes":{},"#,
                r#""self_captures":{},"black_utility":{},"margin":{:.6},"#,
                r#""black_area":{:.8},"white_area":{:.8},"#,
                r#""reached_ply_cap":{},"resigned":{},"soft_resign_ply":{},"#,
                r#""first_sample":0,"sample_count":{},"final_stones":[{}]}}"#
            ),
            record.game,
            record.komi,
            record.radius,
            record.plies,
            record.passes,
            record.self_captures,
            record.black_utility,
            record.margin,
            record.black_area,
            record.white_area,
            record.reached_ply_cap,
            record.resigned,
            record
                .soft_resign_ply
                .map_or_else(|| "null".to_owned(), |ply| ply.to_string()),
            published.samples,
            record
                .final_stones
                .iter()
                .map(|(x, y, colour)| format!("[{x:.9},{y:.9},{colour}]"))
                .collect::<Vec<_>>()
                .join(","),
        )?;
        writer.flush()?;
    }

    if !calibration.is_empty() {
        let mut writer =
            BufWriter::new(fs::File::create(staging.join("resign-calibration.jsonl"))?);
        for trial in &calibration {
            writeln!(
                writer,
                r#"{{"threshold":{:.4},"window":{},"fired":{},"wrong":{},"plies_saved":{},"confidence":{:.4}}}"#,
                trial.threshold,
                trial.window,
                u32::from(trial.fired),
                u32::from(trial.wrong),
                trial.plies_saved,
                trial.fired_confidence,
            )?;
        }
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
    if let Some(sha) = model_sha256 {
        writeln!(writer, "  \"behavior_model_sha256\": \"{sha}\",")?;
    }
    writeln!(writer, "  \"dataset_sha256\": \"{}\"", published.sha256)?;
    writeln!(writer, "}}")?;
    writer.flush()?;

    fs::rename(&staging, &final_path)?;
    Ok(published.samples)
}

/// Stop the run after `--profile-seconds` so the profile can be written.
///
/// Drives the generator's own `stopping` and `cancelled` flags rather than
/// calling `exit`: the flamegraph is written at the end of `main`, and exiting
/// from a side thread would skip it. `cancelled` is the right one here -- a
/// profiling run wants a bounded slice of search, not the games, and waiting
/// for 32 actors to finish a 38-unit board would take over an hour.
#[cfg(feature = "profiling")]
fn arm_profile_deadline(
    config: &Config,
    stopping: &Arc<AtomicBool>,
    cancelled: &Arc<AtomicBool>,
) {
    let seconds = config.profile_seconds;
    if config.profile_output.is_none() || seconds == 0 {
        return;
    }
    let stopping = Arc::clone(stopping);
    let cancelled = Arc::clone(cancelled);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(seconds));
        eprintln!("[profile] {seconds}s elapsed, abandoning games in flight");
        stopping.store(true, Ordering::Release);
        cancelled.store(true, Ordering::Release);
    });
}

#[cfg(not(feature = "profiling"))]
fn arm_profile_deadline(_: &Config, _: &Arc<AtomicBool>, _: &Arc<AtomicBool>) {}

#[cfg(feature = "profiling")]
type Profiler = Option<pprof::ProfilerGuard<'static>>;
#[cfg(not(feature = "profiling"))]
type Profiler = ();

#[cfg(feature = "profiling")]
fn start_profiler(output: Option<&std::path::Path>) -> io::Result<Profiler> {
    let Some(_) = output else { return Ok(None) };
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(199)
        // Skip the runtime's own frames, which otherwise dominate a stack that
        // is mostly blocked on inference.
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .map_err(|error| io::Error::other(format!("start profiler: {error}")))?;
    Ok(Some(guard))
}

#[cfg(feature = "profiling")]
fn finish_profiler(profiler: Profiler, output: Option<&std::path::Path>) -> io::Result<()> {
    let (Some(guard), Some(path)) = (profiler, output) else {
        return Ok(());
    };
    let report = guard
        .report()
        .build()
        .map_err(|error| io::Error::other(format!("build profile: {error}")))?;
    let file = fs::File::create(path)?;
    report
        .flamegraph(file)
        .map_err(|error| io::Error::other(format!("write flamegraph: {error}")))?;
    eprintln!("[profile] wrote {}", path.display());
    Ok(())
}

#[cfg(not(feature = "profiling"))]
fn start_profiler(output: Option<&std::path::Path>) -> io::Result<Profiler> {
    if output.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "--profile-output needs the `profiling` feature: \
             cargo build --release -p vgo-selfplay --features profiling",
        ));
    }
    Ok(())
}

#[cfg(not(feature = "profiling"))]
fn finish_profiler(_: Profiler, _: Option<&std::path::Path>) -> io::Result<()> {
    Ok(())
}

/// Ask the driver to block, rather than spin, while waiting for the GPU.
///
/// CUDA offers two waiting strategies. Spinning busy-polls a memory location
/// until the GPU signals: a microsecond or two of latency, and a core pinned at
/// 100% for the whole wait. Blocking sleeps on a driver semaphore and is woken
/// by an interrupt: tens of microseconds to wake, and the core is free
/// meanwhile. The default is "auto", which spins when threads do not outnumber
/// cores.
///
/// For this generator that default is exactly backwards. A profile of the
/// generator put 43% of all CPU samples inside `ort::session::run` across two
/// inference threads, both at 100% CPU, while the GPU sat pinned at 296 W --
/// they were not computing, they were spinning. Sessions take ~7.6 ms, so a
/// 50-microsecond wakeup is under a percent of added latency, and it buys back
/// two whole cores. On this box those cores are idle anyway; on the 8-vCPU
/// machines a rented GPU comes attached to, it is a quarter of the CPU.
///
/// # Ordering
///
/// This sets a flag on the device's *primary* context, which the driver
/// refuses to change while that context is active. ONNX Runtime retains the
/// same primary context, so this has to run before any session is built --
/// hence `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE` is reported plainly rather than
/// swallowed: silently failing would leave the threads spinning while the log
/// said the flag was set.
fn use_blocking_sync(device: usize) -> Result<(), String> {
    use cudarc::driver::sys;
    // SAFETY: `cuInit` is idempotent and `device` is validated by `device::get`
    // before use; both are the documented preconditions.
    unsafe {
        sys::cuInit(0)
            .result()
            .map_err(|e| e.to_string())?;
        let handle = cudarc::driver::result::device::get(device as i32)
            .map_err(|e| e.to_string())?;
        sys::cuDevicePrimaryCtxSetFlags_v2(
            handle,
            sys::CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC as u32,
        )
        .result()
        .map_err(|e| format!("set blocking sync: {e}"))?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let config = Config::parse();
    // Before anything touches CUDA: the driver refuses to change this flag once
    // the primary context is active, and ONNX Runtime retains that same
    // context when the first session is built.
    if config.blocking_sync {
        match use_blocking_sync(config.device_id.max(0) as usize) {
            Ok(()) => eprintln!("[cuda] waiting on the GPU will block, not spin"),
            Err(error) => eprintln!("[cuda] blocking sync unavailable, continuing: {error}"),
        }
    }
    let profiler = start_profiler(config.profile_output.as_deref())?;
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
    arm_profile_deadline(&config, &stopping, &cancelled);
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
    // Before the failure check, not after: a profiling run ends by cancelling
    // the actors, so the error path is the *expected* one there and returning
    // early would discard the profile the run existed to produce.
    finish_profiler(profiler, config.profile_output.as_deref())?;

    if let Some(error) = failure {
        // A deliberate profiling stop is not a failure. Anything else is.
        if config.profile_seconds > 0 && config.profile_output.is_some() {
            eprintln!("[profile] games in flight were abandoned, as intended");
        } else {
            return Err(error);
        }
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
