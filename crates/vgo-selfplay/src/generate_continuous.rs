#![recursion_limit = "256"]

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
//!   * **Handoff needs no coordination.** A new model can mean a new process
//!     writing to its own directory while the old one finishes into its own.
//!     Or, with `--swap-file`, the running process loads the new model beside
//!     the old one and switches: see "Swapping models in place" below.
//!   * **Staleness becomes visible.** A replay window spanning thirty model
//!     generations currently says so nowhere; here it is a directory listing.
//!   * **Retirement gets a unit** -- drop a generation, not a file list.
//!
//! Each game directory has the same shape as a shard from the old generator, so
//! the Python loader, the replay cache, komi fitting and retirement all read it
//! unchanged. A game simply is a shard that happens to hold one game.
//!
//! ## Swapping models in place
//!
//! Starting a new process for each model cost about 4% of generation in the
//! loop: for the first few minutes the new process ramps up while the old one
//! drains, and its dwindling actors send partial batches. With `--swap-file`,
//! the loop instead writes a request naming the next generation's label, model
//! and schedules. A watcher thread loads the model beside the current one, then
//! switches, and every actor moves to it at its next move. Games in progress
//! finish on the new model.
//!
//! The unit of switching is a move, not a leaf: [`SwappingEvaluator`] pins the
//! current generation when a search begins, so one move's search never mixes
//! two models. A game lands in the directory of the generation that played its
//! last move. When more than one model played a game, its manifest lists each
//! with the ply it took over from (`behavior_models`); `behavior_model_sha256`
//! names the last. A game with one model is written exactly as before.

use std::cell::RefCell;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

mod telemetry_writer;

use clap::{ArgAction, Parser};
use sha2::{Digest, Sha256};
use vgo_core::Position;
use vgo_core::Ruleset;
use vgo_inference::{
    BatchedEvaluatorPool, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{RasterConfig, RasterKind};
use vgo_search::{Evaluation, EvaluationError, Evaluator, NaiveEvaluator};
use vgo_selfplay::generation::{
    GameRecord, GameSamples, GameSettings, KOMI_AREA_COEFFICIENT, generate_game_observed,
    parse_board_mix, replay_capacity_for,
};
use vgo_selfplay::replay_stream::{PublishedReplay, ReplayStream, stone_capacity_for_radius};
use vgo_selfplay::tail::{TailSchedule, TailSettings};
use vgo_selfplay::telemetry::GenerationTelemetry;

// Rasters are allocated by actors and freed by the broker, and a move churns
// position state: mimalloc measured 3-9% faster than glibc here, and no worse on
// 32 threads.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    /// Watch this path for a request to move to a new model generation. The
    /// request is JSON: `{"label": ..., "model": ..., "tail_schedule": ...,
    /// "komi_schedule": ...}`, the last two optional, written whole (rename it
    /// into place). The generator loads the model beside the current one,
    /// switches, and answers with `<swap-file>.done` holding the new label, or
    /// `<swap-file>.failed` holding the error and carries on as it was.
    #[arg(long)]
    swap_file: Option<PathBuf>,
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
    /// Stop after this many seconds of generation, abandoning incomplete games.
    /// Independent of profiling and measured after model sessions are ready.
    #[arg(long, default_value_t = 0)]
    run_seconds: u64,
    /// First game index, so a restarted generation does not reuse seeds.
    #[arg(long, default_value_t = 0)]
    first_game: u64,
    /// Seconds between progress reports; zero writes only startup and final records.
    /// Appends interval metrics to <output-root>/<label>/performance.jsonl.
    #[arg(long, default_value_t = 30)]
    telemetry_seconds: u64,

    #[arg(long, default_value_t = 128)]
    resolution: usize,
    #[arg(long, default_value_t = 128)]
    policy_resolution: usize,
    #[arg(long, default_value = "compact-radius")]
    raster_kind: RasterKind,
    #[arg(long, default_value_t = 1600)]
    simulations: u32,
    /// Full-search probability; fast turns are omitted from training replay.
    #[arg(long, default_value_t = 1.0)]
    full_search_fraction: f64,
    #[arg(long, default_value_t = 200)]
    fast_simulations: u32,
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
    /// A refitted komi centre and spread, from
    /// `training/loop/komi-calibration.py`. Replaces the coefficient and the
    /// low/high range above.
    #[arg(long)]
    komi_schedule: Option<PathBuf>,
    #[arg(long, default_value = "vgo")]
    ruleset: Ruleset,
    /// Thresholds from `training/loop/tail-calibration.py`. Without one, every
    /// game gets full search.
    #[arg(long)]
    tail_schedule: Option<PathBuf>,
    #[arg(long, default_value_t = 400)]
    tail_simulations: u32,
    #[arg(long, default_value_t = 5)]
    tail_window: u32,
    #[arg(long, default_value_t = 0.6)]
    tail_exit_value: f64,
    #[arg(long, default_value_t = 2)]
    tail_exit_window: u32,
    /// After a side has passed this many times running, each further pass by
    /// it ends the game with `--random-stop-probability`; see
    /// `vgo_selfplay::RandomStop`. 0 turns the rule off.
    #[arg(long, default_value_t = 0)]
    random_stop_after: u32,
    #[arg(long, default_value_t = 0.25)]
    random_stop_probability: f64,
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
    /// Fails on a malformed `--board-mix` rather than dropping it: a run that
    /// ignores its mix plays every game on the fallback board with nothing else
    /// looking wrong.
    fn game_settings(&self) -> Result<GameSettings, String> {
        let schedule = self
            .tail_schedule
            .as_ref()
            .map(|path| {
                let text = fs::read_to_string(path)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                TailSchedule::parse(&text)
            })
            .transpose()?;
        let tail = TailSettings {
            schedule,
            simulations: self.tail_simulations,
            window: self.tail_window,
            exit_value: self.tail_exit_value,
            exit_window: self.tail_exit_window,
        };
        tail.validate()?;
        Ok(GameSettings {
            policy_resolution: self.policy_resolution,
            simulations: self.simulations,
            full_search_fraction: self.full_search_fraction,
            fast_simulations: self.fast_simulations,
            coarse_pool: self.coarse_pool,
            temperature: self.temperature,
            temperature_plies: self.temperature_plies,
            leaf_batch: self.leaf_batch,
            maximum_candidates: self.maximum_candidates,
            root_exploration_noise: self.root_exploration_noise,
            widening_coefficient: self.widening_coefficient,
            seed: self.seed,
            radius: self.radius,
            board_mix: parse_board_mix(&self.board_mix)?,
            komi_low: self.komi_low,
            komi_high: self.komi_high,
            komi_area_coefficient: self.komi_area_coefficient,
            maximum_plies: self.maximum_plies,
            ruleset: self.ruleset,
            tail,
            random_stop: random_stop(self.random_stop_after, self.random_stop_probability)?,
            komi_schedule: self
                .komi_schedule
                .as_ref()
                .map(|path| {
                    let text = fs::read_to_string(path)
                        .map_err(|error| format!("{}: {error}", path.display()))?;
                    vgo_selfplay::generation::KomiSchedule::parse(&text)
                        .map_err(|error| format!("{}: {error}", path.display()))
                })
                .transpose()?,
        })
    }
}

fn random_stop(after: u32, probability: f64) -> Result<Option<vgo_selfplay::RandomStop>, String> {
    if !(0.0..=1.0).contains(&probability) {
        return Err(format!(
            "--random-stop-probability must be in [0, 1], got {probability}"
        ));
    }
    Ok((after > 0).then_some(vgo_selfplay::RandomStop { after, probability }))
}

fn file_sha256(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path)?);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Everything a game's directory shares with the rest of the run.
struct GameWriter {
    root: PathBuf,
    raster: RasterConfig,
    policy_size: usize,
    replay_capacity: usize,
    stone_capacity: usize,
    simulations: u32,
    model_sha256: Option<String>,
}

impl GameWriter {
    fn new(
        config: &Config,
        settings: &GameSettings,
        root: PathBuf,
        raster: RasterConfig,
        policy_size: usize,
        model_sha256: Option<String>,
    ) -> Self {
        // The smallest radius this run can play, which is the board that holds
        // the most stones and therefore sizes the record. Taken from `--radius`
        // alone it would fit the mini board and fail on a standard one *after*
        // the games are played.
        let smallest_radius = settings
            .board_mix
            .iter()
            .map(|band| 1.0 / band.high_units)
            .fold(settings.radius, f64::min);
        Self {
            root,
            raster,
            policy_size,
            replay_capacity: replay_capacity_for(config.maximum_candidates, policy_size),
            stone_capacity: stone_capacity_for_radius(smallest_radius),
            simulations: config.simulations,
            model_sha256,
        }
    }

    /// Writes one completed game as a one-game shard.
    ///
    /// The directory is built under a temporary name and renamed once the manifest
    /// is in place, so a reader can never observe a game that is missing its
    /// checksum -- the same discipline the shard writer uses, at a finer grain.
    /// `models` lists the generations that played this game, with the ply each
    /// took over from; the last is this writer's. Empty means this writer's
    /// model played every move.
    fn write(&self, game_id: u64, game: GameSamples, models: &[ModelSpan]) -> io::Result<usize> {
        let name = format!("game-{game_id:09}");
        let staging = self.root.join(format!("{name}.staging"));
        let final_path = self.root.join(&name);
        if final_path.exists() {
            // Already written, so the game in hand is a duplicate of it and there is
            // nothing to do. Say so: reaching here means the index range is being
            // replayed, and every game the run finishes from now on is discarded
            // after paying its full search cost, with nothing else looking wrong.
            eprintln!(
                "warning: {name} already exists in {}; discarding the finished game. \
                 Game indices are being reused -- check --first-game.",
                self.root.display()
            );
            return Ok(0);
        }
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        let GameSamples { samples, record } = game;
        let mut stream = ReplayStream::create(
            &staging.join("dataset.vgo"),
            samples.len(),
            self.raster,
            self.policy_size,
            0,
            self.replay_capacity,
            self.stone_capacity,
        )?;
        stream.write_game(samples)?;
        let published = stream.publish()?;
        write_game_record(&staging.join("games.jsonl"), &record, published.samples)?;
        self.write_manifest(&staging.join("manifest.json"), &published, &record, models)?;
        fs::rename(&staging, &final_path)?;
        Ok(published.samples)
    }

    fn write_manifest(
        &self,
        path: &Path,
        published: &PublishedReplay,
        record: &GameRecord,
        models: &[ModelSpan],
    ) -> io::Result<()> {
        let mut writer = BufWriter::new(fs::File::create(path)?);
        writeln!(writer, "{{")?;
        writeln!(writer, "  \"schema\": \"vgo.replay-shard.v1\",")?;
        writeln!(writer, "  \"dataset\": \"dataset.vgo\",")?;
        writeln!(writer, "  \"games\": \"games.jsonl\",")?;
        writeln!(writer, "  \"samples\": {},", published.samples)?;
        writeln!(writer, "  \"completed_games\": 1,")?;
        writeln!(writer, "  \"channels\": {},", self.raster.channels())?;
        writeln!(writer, "  \"height\": {},", self.raster.height)?;
        writeln!(writer, "  \"width\": {},", self.raster.width)?;
        writeln!(writer, "  \"policy_size\": {},", self.policy_size)?;
        writeln!(writer, "  \"simulations\": {},", self.simulations)?;
        writeln!(writer, "  \"radius\": {},", record.radius)?;
        writeln!(writer, "  \"tail_samples\": {},", record.tail_samples)?;
        if let Some(sha) = &self.model_sha256 {
            writeln!(writer, "  \"behavior_model_sha256\": \"{sha}\",")?;
        }
        if models.len() > 1 {
            let spans: Vec<String> = models
                .iter()
                .map(|span| {
                    format!(
                        "{{\"from_ply\": {}, \"sha256\": {}}}",
                        span.from_ply,
                        span.sha256
                            .as_ref()
                            .map_or_else(|| "null".to_owned(), |sha| format!("\"{sha}\""))
                    )
                })
                .collect();
            writeln!(writer, "  \"behavior_models\": [{}],", spans.join(", "))?;
        }
        writeln!(writer, "  \"dataset_sha256\": \"{}\"", published.sha256)?;
        writeln!(writer, "}}")?;
        writer.flush()
    }
}

/// The game's sidecar line.
///
/// The final board and both areas are what the score and ownership targets are
/// built from. They cannot be recovered afterwards -- the last recorded position
/// is one move short of the end -- so every game written without them is score
/// data lost for good.
fn write_game_record(path: &Path, record: &GameRecord, samples: usize) -> io::Result<()> {
    let mut writer = BufWriter::new(fs::File::create(path)?);
    writeln!(
        writer,
        concat!(
            r#"{{"game":{},"komi":{:.6},"radius":{:.8},"plies":{},"passes":{},"#,
            r#""self_captures":{},"black_utility":{},"margin":{:.6},"#,
            r#""black_area":{:.8},"white_area":{:.8},"#,
            r#""reached_ply_cap":{},"random_stop":{},"maximum_plies":{},"tail_spans":{},"tail_samples":{},"#,
            r#""first_sample":0,"sample_count":{},"final_stones":[{}],"search_budgets":{},"#,
            r#""policy_eligible":{},"root_values":{},"search_kl":{},"top_gap":{}}}"#
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
        record.random_stop,
        record.maximum_plies,
        serde_json::to_string(&record.tail_spans)?,
        record.tail_samples,
        samples,
        record
            .final_stones
            .iter()
            .map(|(x, y, colour)| format!("[{x:.9},{y:.9},{colour}]"))
            .collect::<Vec<_>>()
            .join(","),
        serde_json::to_string(&record.search_budgets)?,
        serde_json::to_string(&record.policy_eligible)?,
        rounded(&record.searches.root_values),
        rounded(&record.searches.kl),
        rounded(&record.searches.top_gap),
    )?;
    writer.flush()
}

/// Four places: enough to read a saturated value, at a third of the bytes.
fn rounded(values: &[f32]) -> String {
    let values: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", values.join(","))
}

/// Stop the run after `--profile-seconds` so the profile can be written.
///
/// Drives the generator's own `stopping` and `cancelled` flags rather than
/// calling `exit`: the flamegraph is written at the end of `main`, and exiting
/// from a side thread would skip it. `cancelled` is the right one here -- a
/// profiling run wants a bounded slice of search, not the games, and waiting
/// for 32 actors to finish a 38-unit board would take over an hour.
fn deadline_seconds(config: &Config) -> Option<u64> {
    [
        config.run_seconds,
        if config.profile_output.is_some() {
            config.profile_seconds
        } else {
            0
        },
    ]
    .into_iter()
    .filter(|seconds| *seconds > 0)
    .min()
}

fn expected_deadline_cancellation(error: &vgo_search::EvaluationError, deadline: bool) -> bool {
    deadline && error.to_string() == "replay generation cancelled"
}

fn arm_deadline(config: &Config, run: &Arc<RunState>) {
    let Some(seconds) = deadline_seconds(config) else {
        return;
    };
    let run = Arc::clone(run);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(seconds));
        eprintln!("[deadline] {seconds}s elapsed, abandoning games in flight");
        run.deadline.store(true, Ordering::Release);
        run.stopping.store(true, Ordering::Release);
        run.cancelled.store(true, Ordering::Release);
    });
}

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
        sys::cuInit(0).result().map_err(|e| e.to_string())?;
        let handle =
            cudarc::driver::result::device::get(device as i32).map_err(|e| e.to_string())?;
        sys::cuDevicePrimaryCtxSetFlags_v2(
            handle,
            sys::CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC as u32,
        )
        .result()
        .map_err(|e| format!("set blocking sync: {e}"))?;
    }
    Ok(())
}

/// Load the model's lanes behind one broker, or fall back to the naive evaluator.
fn load_evaluator(
    config: &Config,
    raster: RasterConfig,
) -> io::Result<(Arc<dyn Evaluator>, Option<BatchedEvaluatorPool>)> {
    let Some(model) = &config.model else {
        return Ok((Arc::new(NaiveEvaluator), None));
    };
    let mut services = Vec::with_capacity(config.inference_slots);
    // Sequential loads: TensorRT reuses its engine cache rather than racing two
    // builders over the same directory.
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
    Ok((Arc::new(pool.clone()), Some(pool)))
}

/// Which model played a game from which ply on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelSpan {
    from_ply: u32,
    sha256: Option<String>,
}

/// Everything that changes when the model does: the weights, the schedules
/// fitted for them, and the directory their games land in.
struct Generation {
    label: String,
    root: PathBuf,
    model_sha256: Option<String>,
    evaluator: Arc<dyn Evaluator>,
    pool: Option<BatchedEvaluatorPool>,
    settings: Arc<GameSettings>,
    writer: Arc<GameWriter>,
}

impl Generation {
    fn load(config: &Config, raster: RasterConfig, policy_size: usize) -> io::Result<Self> {
        let settings = Arc::new(
            config
                .game_settings()
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?,
        );
        let root = config.output_root.join(&config.label);
        fs::create_dir_all(&root)?;
        let model_sha256 = config.model.as_deref().map(file_sha256).transpose()?;
        let (evaluator, pool) = load_evaluator(config, raster)?;
        let writer = Arc::new(GameWriter::new(
            config,
            &settings,
            root.clone(),
            raster,
            policy_size,
            model_sha256.clone(),
        ));
        Ok(Self {
            label: config.label.clone(),
            root,
            model_sha256,
            evaluator,
            pool,
            settings,
            writer,
        })
    }
}

/// The generation new searches use. Swapped whole, never edited.
struct Generations(RwLock<Arc<Generation>>);

impl Generations {
    fn new(first: Generation) -> Self {
        Self(RwLock::new(Arc::new(first)))
    }

    fn current(&self) -> Arc<Generation> {
        Arc::clone(&self.0.read().unwrap_or_else(|poison| poison.into_inner()))
    }

    fn replace(&self, next: Generation) {
        *self.0.write().unwrap_or_else(|poison| poison.into_inner()) = Arc::new(next);
    }
}

thread_local! {
    /// The generation this thread's current search started with. One process
    /// holds one `Generations`, so a single slot per thread is enough.
    static PINNED: RefCell<Option<Arc<Generation>>> = const { RefCell::new(None) };
}

/// Evaluates with the generation that was current when this thread's search
/// began, so a move is searched by one model even if a swap lands mid-search.
/// The old generation, and its lanes, are freed once no thread holds it.
struct SwappingEvaluator {
    generations: Arc<Generations>,
}

impl SwappingEvaluator {
    /// The generation pinned on this thread, or the current one before any search.
    fn pinned(&self) -> Arc<Generation> {
        PINNED.with(|slot| {
            slot.borrow()
                .clone()
                .unwrap_or_else(|| self.generations.current())
        })
    }

    fn with_pinned<T>(&self, run: impl FnOnce(&dyn Evaluator) -> T) -> T {
        PINNED.with(|slot| match &*slot.borrow() {
            Some(generation) => run(generation.evaluator.as_ref()),
            None => run(self.generations.current().evaluator.as_ref()),
        })
    }
}

impl Evaluator for SwappingEvaluator {
    fn begin_search(&self, root: &Position) {
        let generation = self.generations.current();
        generation.evaluator.begin_search(root);
        PINNED.with(|slot| *slot.borrow_mut() = Some(generation));
    }

    fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
        self.with_pinned(|evaluator| evaluator.evaluate(position))
    }

    fn evaluate_batch(&self, positions: &[Position]) -> Result<Vec<Evaluation>, EvaluationError> {
        self.with_pinned(|evaluator| evaluator.evaluate_batch(positions))
    }
}

/// A request to move to the next generation, as the loop writes it.
#[derive(Debug, PartialEq)]
struct SwapRequest {
    label: String,
    model: Option<PathBuf>,
    tail_schedule: Option<PathBuf>,
    komi_schedule: Option<PathBuf>,
}

impl SwapRequest {
    /// Unknown fields are refused: a misspelt `komi_schedule` would otherwise
    /// drop the refit in silence.
    fn parse(text: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|error| error.to_string())?;
        let object = value.as_object().ok_or("a swap request is a JSON object")?;
        for key in object.keys() {
            if !["label", "model", "tail_schedule", "komi_schedule"].contains(&key.as_str()) {
                return Err(format!("unknown field {key:?} in swap request"));
            }
        }
        let path = |key: &str| -> Result<Option<PathBuf>, String> {
            match object.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::String(text)) => Ok(Some(PathBuf::from(text))),
                Some(_) => Err(format!("{key} must be a path")),
            }
        };
        Ok(Self {
            label: object
                .get("label")
                .and_then(serde_json::Value::as_str)
                .ok_or("a swap request needs a label")?
                .to_owned(),
            model: path("model")?,
            tail_schedule: path("tail_schedule")?,
            komi_schedule: path("komi_schedule")?,
        })
    }

    /// The process's configuration with this request's generation in place.
    fn apply(&self, config: &Config) -> Result<Config, String> {
        if self.label.is_empty() || self.label.contains('/') || self.label == config.label {
            return Err(format!(
                "swap label {:?} is not a new directory name",
                self.label
            ));
        }
        if config.model.is_some() != self.model.is_some() {
            // A naive generator searches on a different budget; the loop starts a
            // process for that change instead.
            return Err("a swap cannot move between the naive evaluator and a model".into());
        }
        let mut next = config.clone();
        next.label.clone_from(&self.label);
        next.model.clone_from(&self.model);
        next.tail_schedule.clone_from(&self.tail_schedule);
        next.komi_schedule.clone_from(&self.komi_schedule);
        Ok(next)
    }
}

/// Write `text` to `path` whole, so a reader never sees half of it.
fn write_whole(path: &Path, text: &str) -> io::Result<()> {
    let staging = path.with_extension("staging");
    fs::write(&staging, text)?;
    fs::rename(&staging, path)
}

fn answer_path(request: &Path, suffix: &str) -> PathBuf {
    let mut name = request.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Load the requested generation and make it current. The request is removed
/// before the answer appears, so the loop can never see a stale request and a
/// fresh answer together.
fn take_swap(
    path: &Path,
    config: &Config,
    current_label: &str,
    raster: RasterConfig,
    policy_size: usize,
    generations: &Generations,
) -> Result<String, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let request =
        SwapRequest::parse(&text).map_err(|error| format!("parse {}: {error}", path.display()))?;
    if request.label == current_label {
        return Err(format!("already generating as {current_label}"));
    }
    let next_config = request.apply(config)?;
    let started = Instant::now();
    let next =
        Generation::load(&next_config, raster, policy_size).map_err(|error| error.to_string())?;
    generations.replace(next);
    eprintln!(
        "[swap] now {} with {} after {:.1}s loading",
        request.label,
        request.model.as_ref().map_or_else(
            || "the naive evaluator".into(),
            |model| model.display().to_string()
        ),
        started.elapsed().as_secs_f64()
    );
    Ok(request.label)
}

/// Poll for swap requests until the run stops.
fn watch_for_swaps(
    path: PathBuf,
    config: Config,
    raster: RasterConfig,
    policy_size: usize,
    generations: Arc<Generations>,
    run: Arc<RunState>,
) {
    thread::Builder::new()
        .name("vgo-swap".into())
        .spawn(move || {
            while !run.stopping.load(Ordering::Acquire) && !run.cancelled.load(Ordering::Acquire) {
                if path.exists() {
                    let label = generations.current().label.clone();
                    let outcome =
                        take_swap(&path, &config, &label, raster, policy_size, &generations);
                    let _ = fs::remove_file(&path);
                    let answer = match &outcome {
                        Ok(label) => write_whole(&answer_path(&path, ".done"), label),
                        Err(message) => {
                            eprintln!("[swap] refused: {message}");
                            write_whole(&answer_path(&path, ".failed"), message)
                        }
                    };
                    if let Err(error) = answer {
                        eprintln!("[swap] could not answer {}: {error}", path.display());
                    }
                }
                thread::sleep(Duration::from_millis(500));
            }
        })
        .expect("start the swap watcher");
}

/// The flags and counters every actor shares.
///
/// Two stop signals, not one. `stopping` means take no new games -- the game in
/// hand plays to its end, which is the whole point of making a game the unit.
/// `cancelled` aborts mid-game and is only set when something has actually
/// failed; `generate_game` reports it as an error, so conflating the two turns an
/// orderly stop into a spurious failure in every other actor.
#[derive(Default)]
struct RunState {
    stopping: AtomicBool,
    cancelled: Arc<AtomicBool>,
    deadline: AtomicBool,
    next_game: AtomicU64,
    games_written: AtomicU64,
    samples_written: AtomicU64,
}

/// One actor owns the whole cycle: play a game, write it, take the next. There is
/// no collector and no channel, because there is nothing to batch -- which is
/// also why no actor can ever be left waiting on another's game.
struct Actor {
    index: usize,
    stop_file: Option<PathBuf>,
    maximum_games: u64,
    evaluator: Arc<SwappingEvaluator>,
    telemetry: Arc<GenerationTelemetry>,
    run: Arc<RunState>,
}

impl Actor {
    fn should_stop(&self) -> bool {
        let run = &self.run;
        if run.stopping.load(Ordering::Acquire) || run.cancelled.load(Ordering::Acquire) {
            return true;
        }
        let stop_file = self.stop_file.as_ref().is_some_and(|path| path.exists());
        let enough = self.maximum_games > 0
            && run.games_written.load(Ordering::Relaxed) >= self.maximum_games;
        if stop_file || enough {
            run.stopping.store(true, Ordering::Release);
        }
        stop_file || enough
    }

    /// Play and publish games until the run stops. A cancellation the deadline
    /// asked for is a normal end; any other failure stops every actor.
    fn run(&self) -> io::Result<()> {
        let result = self.play_games();
        self.telemetry.phase(self.index, "done");
        if result.is_err() {
            self.run.cancelled.store(true, Ordering::Release);
        }
        result
    }

    fn play_games(&self) -> io::Result<()> {
        let (run, telemetry, actor) = (&self.run, &self.telemetry, self.index);
        while !self.should_stop() {
            let index = run.next_game.fetch_add(1, Ordering::Relaxed);
            telemetry.phase(actor, "playing");
            // Settings come from the generation current as the game starts: komi
            // is drawn once, and the tail schedule stays the one it began under.
            let settings = Arc::clone(&self.evaluator.generations.current().settings);
            let mut played: Vec<(u32, Arc<Generation>)> = Vec::new();
            let game = match generate_game_observed(
                &settings,
                self.evaluator.as_ref(),
                index,
                &run.cancelled,
                |progress| {
                    let generation = self.evaluator.pinned();
                    if played
                        .last()
                        .is_none_or(|(_, last)| !Arc::ptr_eq(last, &generation))
                    {
                        played.push((progress.ply, generation));
                    }
                    telemetry.searched(actor, progress);
                },
            ) {
                Ok(game) => game,
                Err(error) => {
                    let deadline = run.deadline.load(Ordering::Acquire);
                    if expected_deadline_cancellation(&error, deadline) {
                        telemetry.discarded(actor);
                        return Ok(());
                    }
                    return Err(io::Error::other(error));
                }
            };
            if game.samples.is_empty() {
                telemetry.discarded(actor);
                continue;
            }
            let capped = game.record.reached_ply_cap;
            let radius = game.samples[0].position.radius();
            let reached_tail = !game.record.tail_spans.is_empty();
            telemetry.phase(actor, "writing");
            let writing_started = Instant::now();
            let (writer, models) = match played.last() {
                Some((_, last)) => (
                    Arc::clone(&last.writer),
                    played
                        .iter()
                        .map(|(ply, generation)| ModelSpan {
                            from_ply: *ply,
                            sha256: generation.model_sha256.clone(),
                        })
                        .collect(),
                ),
                None => (
                    Arc::clone(&self.evaluator.generations.current().writer),
                    Vec::new(),
                ),
            };
            drop(played);
            let written = writer.write(index, game, &models)?;
            telemetry.published(
                actor,
                radius,
                written,
                capped,
                reached_tail,
                writing_started.elapsed(),
            );
            if written > 0 {
                run.games_written.fetch_add(1, Ordering::Relaxed);
                run.samples_written
                    .fetch_add(written as u64, Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

fn join_actors(handles: Vec<thread::JoinHandle<io::Result<()>>>) -> Option<io::Error> {
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
    failure
}

fn print_summary(label: &str, run: &RunState) -> io::Result<()> {
    let mut out = io::stdout().lock();
    writeln!(out, "{{")?;
    writeln!(out, "  \"label\": \"{label}\",")?;
    let games = run.games_written.load(Ordering::Relaxed);
    writeln!(out, "  \"games\": {games},")?;
    let samples = run.samples_written.load(Ordering::Relaxed);
    writeln!(out, "  \"samples\": {samples}")?;
    writeln!(out, "}}")
}

fn main() -> io::Result<()> {
    let config = Config::parse();
    if config.profile_seconds > 0 && config.profile_output.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--profile-seconds needs --profile-output; use --run-seconds for an unprofiled timed run",
        ));
    }
    if config.actors == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--actors must be positive",
        ));
    }
    // Fail on bad settings before touching CUDA.
    config
        .game_settings()
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    // Before anything touches CUDA: the driver refuses to change this flag once
    // the primary context is active, and ONNX Runtime retains that same
    // context when the first session is built.
    if config.blocking_sync {
        match use_blocking_sync(config.device_id.max(0) as usize) {
            Ok(()) => eprintln!("[cuda] waiting on the GPU will block, not spin"),
            Err(error) => eprintln!("[cuda] blocking sync unavailable, continuing: {error}"),
        }
    }
    // A unit without the `profiling` feature, which is the point of the alias.
    #[allow(clippy::let_unit_value)]
    let profiler = start_profiler(config.profile_output.as_deref())?;
    let raster = RasterConfig::square_of(config.resolution, config.raster_kind);
    let policy_size = RasterConfig::square(config.policy_resolution).pixels() + 1;
    let generations = Arc::new(Generations::new(Generation::load(
        &config,
        raster,
        policy_size,
    )?));
    let evaluator = Arc::new(SwappingEvaluator {
        generations: Arc::clone(&generations),
    });

    let run = Arc::new(RunState {
        next_game: AtomicU64::new(config.first_game),
        ..RunState::default()
    });
    arm_deadline(&config, &run);
    let telemetry = Arc::new(GenerationTelemetry::new(config.actors));
    let reporter = telemetry_writer::Reporter::start(
        &config,
        Arc::clone(&generations),
        Arc::clone(&telemetry),
        Arc::clone(&run.cancelled),
    )?;
    if let Some(path) = &config.swap_file {
        watch_for_swaps(
            path.clone(),
            config.clone(),
            raster,
            policy_size,
            Arc::clone(&generations),
            Arc::clone(&run),
        );
    }
    let handles = (0..config.actors)
        .map(|index| {
            let actor = Actor {
                index,
                stop_file: config.stop_file.clone(),
                maximum_games: config.maximum_games,
                evaluator: Arc::clone(&evaluator),
                telemetry: Arc::clone(&telemetry),
                run: Arc::clone(&run),
            };
            thread::spawn(move || actor.run())
        })
        .collect();
    let failure = join_actors(handles);
    reporter.finish()?;
    // Preserve the profile even when a genuine actor/backend failure occurred.
    finish_profiler(profiler, config.profile_output.as_deref())?;
    match failure {
        Some(error) => Err(error),
        None => print_summary(&generations.current().label, &run),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgo_selfplay::generation::generate_game;

    fn tiny_config(label: &str, output_root: &Path) -> Config {
        Config::parse_from([
            "generator",
            "--output-root",
            output_root.to_str().unwrap(),
            "--label",
            label,
            "--simulations",
            "8",
            "--max-plies",
            "9",
            "--policy-resolution",
            "16",
        ])
    }

    /// Counts its calls, so a test can tell which generation answered.
    #[derive(Default)]
    struct Counting(std::sync::atomic::AtomicUsize);

    impl Evaluator for Counting {
        fn evaluate(&self, position: &Position) -> Result<Evaluation, EvaluationError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            NaiveEvaluator.evaluate(position)
        }
    }

    fn generation_with(label: &str, evaluator: Arc<dyn Evaluator>, root: &Path) -> Generation {
        let config = tiny_config(label, root);
        let settings = Arc::new(config.game_settings().unwrap());
        let raster = RasterConfig::square_of(config.resolution, config.raster_kind);
        let writer = Arc::new(GameWriter::new(
            &config,
            &settings,
            root.join(label),
            raster,
            RasterConfig::square(16).pixels() + 1,
            Some(format!("sha-{label}")),
        ));
        Generation {
            label: label.into(),
            root: root.join(label),
            model_sha256: Some(format!("sha-{label}")),
            evaluator,
            pool: None,
            settings,
            writer,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("vgo-swap-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn a_search_keeps_the_generation_it_began_with_and_the_old_one_is_freed() {
        let root = scratch("pin");
        let (first, second) = (Arc::new(Counting::default()), Arc::new(Counting::default()));
        let generations = Arc::new(Generations::new(generation_with("a", first.clone(), &root)));
        let evaluator = SwappingEvaluator {
            generations: Arc::clone(&generations),
        };
        let position = Position::new(1.0 / 6.0, Vec::new(), vgo_core::Color::Black);
        evaluator.begin_search(&position);
        let old = Arc::downgrade(&generations.current());
        generations.replace(generation_with("b", second.clone(), &root));
        evaluator.evaluate(&position).unwrap();
        evaluator
            .evaluate_batch(&[position.clone(), position.clone()])
            .unwrap();
        assert_eq!(
            (
                first.0.load(Ordering::SeqCst),
                second.0.load(Ordering::SeqCst)
            ),
            (3, 0)
        );
        assert_eq!(evaluator.pinned().label, "a");
        assert!(
            old.upgrade().is_some(),
            "a search in progress holds its generation"
        );
        evaluator.begin_search(&position);
        evaluator.evaluate(&position).unwrap();
        assert_eq!(
            (
                first.0.load(Ordering::SeqCst),
                second.0.load(Ordering::SeqCst)
            ),
            (3, 1)
        );
        assert_eq!(evaluator.pinned().label, "b");
        assert!(
            old.upgrade().is_none(),
            "nothing holds the old generation once searches move on"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_swap_request_names_a_new_generation_and_nothing_else() {
        let request = SwapRequest::parse(
            r#"{"label": "gen-2", "model": "m.onnx", "tail_schedule": "t.json", "komi_schedule": null}"#,
        )
        .unwrap();
        assert_eq!(request.model, Some(PathBuf::from("m.onnx")));
        assert_eq!(request.komi_schedule, None);
        assert!(
            SwapRequest::parse(r#"{"label": "x", "komi_shedule": "k.json"}"#)
                .unwrap_err()
                .contains("komi_shedule")
        );
        assert!(SwapRequest::parse(r#"{"model": "m.onnx"}"#).is_err());
        assert!(SwapRequest::parse(r#"{"label": "x", "model": 3}"#).is_err());

        let mut config = tiny_config("gen-1", Path::new("unused"));
        config.model = Some("old.onnx".into());
        let next = request.apply(&config).unwrap();
        assert_eq!(
            (next.label.as_str(), next.model),
            ("gen-2", Some("m.onnx".into()))
        );
        assert_eq!(next.tail_schedule, Some("t.json".into()));
        assert_eq!(next.komi_schedule, None);
        assert_eq!(next.simulations, config.simulations);
        let refuse = |text: &str| SwapRequest::parse(text).unwrap().apply(&config).is_err();
        assert!(refuse(r#"{"label": "gen-1", "model": "m.onnx"}"#));
        assert!(refuse(r#"{"label": "a/b", "model": "m.onnx"}"#));
        assert!(
            refuse(r#"{"label": "gen-2"}"#),
            "a model generator cannot go naive"
        );
    }

    #[test]
    fn only_a_game_played_by_two_models_lists_them() {
        let root = scratch("manifest");
        let generation = generation_with("a", Arc::new(NaiveEvaluator), &root);
        fs::create_dir_all(&generation.root).unwrap();
        let cancelled = AtomicBool::new(false);
        let play = |index| {
            generate_game(&generation.settings, &NaiveEvaluator, index, &cancelled).unwrap()
        };
        let manifest = |index: u64| {
            fs::read_to_string(
                generation
                    .root
                    .join(format!("game-{index:09}/manifest.json")),
            )
            .unwrap()
        };
        let one = [ModelSpan {
            from_ply: 0,
            sha256: Some("sha-a".into()),
        }];
        generation.writer.write(1, play(1), &[]).unwrap();
        generation.writer.write(2, play(2), &one).unwrap();
        assert!(!manifest(1).contains("behavior_models"));
        assert!(!manifest(2).contains("behavior_models"));
        let two = [
            ModelSpan {
                from_ply: 0,
                sha256: Some("sha-x".into()),
            },
            ModelSpan {
                from_ply: 4,
                sha256: Some("sha-a".into()),
            },
        ];
        generation.writer.write(3, play(3), &two).unwrap();
        let text = manifest(3);
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["behavior_model_sha256"], "sha-a");
        assert_eq!(value["behavior_models"][0]["sha256"], "sha-x");
        assert_eq!(value["behavior_models"][1]["from_ply"], 4);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn timed_run_does_not_require_a_profile_and_uses_the_earlier_deadline() {
        let mut config = Config::parse_from([
            "generator",
            "--output-root",
            "unused",
            "--label",
            "test",
            "--run-seconds",
            "7",
        ]);
        assert_eq!(super::deadline_seconds(&config), Some(7));
        config.profile_output = Some("unused.svg".into());
        config.profile_seconds = 3;
        assert_eq!(super::deadline_seconds(&config), Some(3));
        config.run_seconds = 0;
        config.profile_seconds = 0;
        assert_eq!(super::deadline_seconds(&config), None);
    }

    #[test]
    fn a_deadline_does_not_hide_genuine_backend_failures() {
        let cancellation = vgo_search::EvaluationError::new("replay generation cancelled");
        assert!(super::expected_deadline_cancellation(&cancellation, true));
        assert!(!super::expected_deadline_cancellation(&cancellation, false));
        assert!(!super::expected_deadline_cancellation(
            &vgo_search::EvaluationError::new("inference failed"),
            true
        ));
    }

    #[test]
    fn randomized_budgets_keep_only_full_targets_but_record_every_search() {
        let config = Config::parse_from([
            "generator",
            "--output-root",
            "unused",
            "--label",
            "test",
            "--simulations",
            "16",
            "--fast-simulations",
            "4",
            "--full-search-fraction",
            "0.25",
            "--max-plies",
            "24",
            "--policy-resolution",
            "16",
        ]);
        let cancelled = AtomicBool::new(false);
        let mut updates = Vec::new();
        let game = generate_game_observed(
            &config.game_settings().unwrap(),
            &NaiveEvaluator,
            99,
            &cancelled,
            |update| updates.push(update),
        )
        .unwrap();
        let record = game.record;
        assert_eq!(record.search_budgets.len(), updates.len());
        assert_eq!(
            record.policy_eligible.iter().filter(|&&keep| keep).count(),
            game.samples.len()
        );
        assert!(record.policy_eligible[0]);
        assert!(record.policy_eligible.iter().any(|&keep| !keep));
        for sample in &game.samples {
            assert!(record.policy_eligible[sample.ply as usize]);
            let expected = if sample.position.to_move() == vgo_core::Color::Black {
                record.black_utility
            } else {
                -record.black_utility
            };
            assert_eq!(sample.value, expected);
        }
        for (index, update) in updates.iter().enumerate() {
            assert_eq!(
                update.simulations,
                if record.policy_eligible[index] { 16 } else { 4 }
            );
        }
        let again = generate_game(
            &config.game_settings().unwrap(),
            &NaiveEvaluator,
            99,
            &cancelled,
        )
        .unwrap();
        assert_eq!(again.record.policy_eligible, record.policy_eligible);
    }

    #[test]
    fn a_malformed_board_mix_is_an_error_rather_than_the_default_board() {
        let config = Config::parse_from([
            "generator",
            "--output-root",
            "unused",
            "--label",
            "test",
            "--board-mix",
            "1:9",
        ]);
        let error = config.game_settings().unwrap_err();
        assert!(error.contains("goes below"), "{error}");
    }

    #[test]
    fn observing_progress_preserves_the_game_and_replay_targets() {
        let config = Config::parse_from([
            "generator",
            "--output-root",
            "unused",
            "--label",
            "test",
            "--simulations",
            "8",
            "--max-plies",
            "9",
            "--policy-resolution",
            "16",
        ]);
        let settings = config.game_settings().unwrap();
        let cancelled = AtomicBool::new(false);
        let plain = generate_game(&settings, &NaiveEvaluator, 123, &cancelled).unwrap();
        let mut updates = Vec::new();
        let observed =
            generate_game_observed(&settings, &NaiveEvaluator, 123, &cancelled, |progress| {
                updates.push(progress)
            })
            .unwrap();
        assert_eq!(updates.len(), observed.samples.len());
        assert_eq!(plain.samples.len(), observed.samples.len());
        assert_eq!(plain.record.final_stones, observed.record.final_stones);
        for (index, (left, right)) in plain.samples.iter().zip(&observed.samples).enumerate() {
            assert_eq!(left.position, right.position);
            assert_eq!(left.policy, right.policy);
            assert_eq!(left.visits, right.visits);
            assert_eq!(left.beta, right.beta);
            assert_eq!(left.proposal_counts, right.proposal_counts);
            assert_eq!(left.value, right.value);
            assert_eq!(left.selected_action, right.selected_action);
            assert_eq!(updates[index].ply, index as u32);
            assert_eq!(updates[index].simulations, 8);
        }
    }
}
