#![forbid(unsafe_code)]

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use clap::{ArgAction, Parser};
use sha2::{Digest, Sha256};
use vgo_core::{Color, Position};
use vgo_inference::{
    BatchedEvaluator, BrokerConfig, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
};
use vgo_raster::{CHANNEL_COUNT, CHANNELS, RasterConfig, SemanticRaster, action_pixel, rasterize};
use vgo_search::{
    Action, EvaluationError, Evaluator, NaiveEvaluator, SearchConfig, SearchResult,
    search_at_ply,
};
use vgo_selfplay::play_game as run_playout;

const REPLAY_MAGIC: [u8; 8] = *b"VGORPLY1";
// v2 added raw per-cell visit counts and coarse->fine sampling probability
// (beta). v3 additionally records the empirical proposal multiplicity per cell,
// so training can correct using both beta and beta-hat without regenerating
// replay.
const REPLAY_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationRuntime {
    Naive,
    Onnx,
}

impl GenerationRuntime {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Onnx => "onnx",
        }
    }
}

impl std::str::FromStr for GenerationRuntime {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "naive" => Ok(Self::Naive),
            "onnx" => Ok(Self::Onnx),
            _ => Err(format!("unsupported generation runtime: {value}")),
        }
    }
}

struct PendingSample {
    raster: SemanticRaster,
    policy: Vec<f32>,
    policy_mask: Vec<f32>,
    visits: Vec<f32>,
    beta: Vec<f32>,
    proposal_counts: Vec<u32>,
    to_move: Color,
    selected_action: u32,
    game: u64,
    ply: u32,
    seed: u64,
}

struct LabeledSample {
    raster: SemanticRaster,
    policy: Vec<f32>,
    policy_mask: Vec<f32>,
    visits: Vec<f32>,
    beta: Vec<f32>,
    proposal_counts: Vec<u32>,
    value: f32,
    selected_action: u32,
    game: u64,
    ply: u32,
    seed: u64,
}

struct GameSamples {
    index: u64,
    samples: Vec<LabeledSample>,
    completed: bool,
}

#[derive(Clone, Debug, Parser)]
#[command(about = "Generate a labeled semantic-raster dataset from self-play")]
struct Config {
    #[arg(long, default_value_t = 96)]
    samples: usize,
    #[arg(long, default_value_t = 96)]
    resolution: usize,
    /// Placement grid the policy head emits, independent of the render
    /// resolution. The board is only ~9 stones across, so 128x128 of placement
    /// precision mostly splits single moves across many cells while spreading
    /// the fixed proposal budget too thin to ever revisit one. Rendering stays
    /// at `--resolution` so the Voronoi boundary channels keep their detail.
    #[arg(long, default_value_t = 32)]
    policy_resolution: usize,
    #[arg(long, default_value_t = 256)]
    simulations: u32,
    /// Fine cells per coarse sampling region; zero uses legacy candidates.
    #[arg(long, default_value_t = 0)]
    coarse_pool: usize,
    /// Softmax temperature on root visit counts for the opening plies. Zero is
    /// deterministic argmax, which makes every game from a given position
    /// identical; a positive value is what gives self-play its diversity.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Plies over which `--temperature` applies; selection is argmax afterwards.
    #[arg(long, default_value_t = 30)]
    temperature_plies: u32,
    #[arg(long = "max-plies", default_value_t = 48)]
    maximum_plies: u32,
    #[arg(long, default_value_t = 1.0 / 6.0)]
    radius: f64,
    #[arg(long, default_value_t = 50_001)]
    seed: u64,
    #[arg(long, default_value_t = 4)]
    examples: usize,
    #[arg(long, default_value = "artifacts/raster-demo")]
    output: PathBuf,
    #[arg(long, default_value = "naive")]
    runtime: GenerationRuntime,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "tensorrt")]
    provider: OnnxProvider,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    fp16: bool,
    #[arg(long, default_value_t = 8)]
    maximum_batch: usize,
    #[arg(long, default_value_t = 1)]
    delay_ms: u64,
    #[arg(long, default_value_t = 8)]
    actors: usize,
    #[arg(long)]
    maximum_games: Option<u64>,
    #[arg(long, default_value = "artifacts/onnx-cache")]
    cache_directory: PathBuf,
}

struct PolicyTarget {
    /// Normalized visit distribution over cells (the legacy target).
    policy: Vec<f32>,
    /// 1.0 for any cell that received a candidate, else 0.0.
    mask: Vec<f32>,
    /// Raw visit counts per cell (unnormalized), for off-policy reweighting.
    visits: Vec<f32>,
    /// Coarse->fine sampling probability beta per cell; 0.0 for legacy/pass
    /// candidates (which have no factored sampling probability).
    beta: Vec<f32>,
    /// Number of raw coarse->fine proposal draws landing in each cell. Legacy
    /// candidates and pass have zero multiplicity.
    proposal_counts: Vec<u32>,
}

fn policy_target(result: &SearchResult, config: RasterConfig) -> PolicyTarget {
    let size = config.pixels() + 1;
    let mut policy = vec![0.0_f32; size];
    let mut mask = vec![0.0_f32; size];
    let mut visits = vec![0.0_f32; size];
    let mut beta = vec![0.0_f32; size];
    let mut proposal_counts = vec![0_u32; size];
    let total = result
        .children
        .iter()
        .map(|child| child.visits)
        .sum::<u32>();
    for child in &result.children {
        let index = match child.action {
            Action::Pass => config.pixels(),
            Action::Place(point) => action_pixel(point.x, point.y, config),
        };
        policy[index] += child.visits as f32 / total as f32;
        mask[index] = 1.0;
        visits[index] += child.visits as f32;
        if let Some(b) = child.beta {
            beta[index] = b as f32;
        }
        proposal_counts[index] = proposal_counts[index]
            .checked_add(child.proposal_count)
            .expect("proposal multiplicity must fit in u32");
    }
    PolicyTarget {
        policy,
        mask,
        visits,
        beta,
        proposal_counts,
    }
}

fn action_index(action: Action, config: RasterConfig) -> u32 {
    match action {
        Action::Pass => config.pixels() as u32,
        Action::Place(point) => action_pixel(point.x, point.y, config) as u32,
    }
}

fn search_config(
    simulations: u32,
    coarse_pool: usize,
    temperature: f64,
    temperature_plies: u32,
) -> SearchConfig {
    let mut config = SearchConfig::canary(simulations);
    config.coarse_pool = coarse_pool;
    config.temperature = temperature;
    config.temperature_plies = temperature_plies;
    config
}

fn validate_config(config: &Config) -> Result<(), &'static str> {
    if config.samples == 0
        || config.resolution == 0
        || config.simulations == 0
        || config.maximum_plies == 0
        || config.maximum_batch == 0
        || config.actors == 0
        || config.maximum_games.is_some_and(|games| games == 0)
    {
        return Err("generation counts, simulations, and dimensions must be positive");
    }
    if config.policy_resolution == 0 {
        return Err("--policy-resolution must be positive");
    }
    // The pool counts fine cells per coarse region on the policy grid, which is
    // what the sampler actually walks -- not the render raster.
    if config.coarse_pool > config.policy_resolution {
        return Err("--coarse-pool must not exceed --policy-resolution");
    }
    if !config.radius.is_finite() || config.radius <= 0.0 || config.radius >= 0.5 {
        return Err("--radius must be finite and between zero and one half");
    }
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err("--temperature must be finite and not negative");
    }
    Ok(())
}

fn generate_game(
    config: &Config,
    evaluator: &dyn Evaluator,
    game_index: u64,
) -> Result<GameSamples, EvaluationError> {
    let raster_config = RasterConfig::square(config.resolution);
    // Policy targets, the recorded action index, and the replay policy vector all
    // live on the placement grid, which may be coarser than the render raster.
    let policy_config = RasterConfig::square(config.policy_resolution);
    let search_config = search_config(
        config.simulations,
        config.coarse_pool,
        config.temperature,
        config.temperature_plies,
    );
    let game_seed = config.seed.wrapping_add(game_index);
    let mut pending = Vec::new();
    let playout = run_playout(
        Position::new(config.radius, Vec::new(), Color::Black),
        config.maximum_plies,
        |position, ply| search_at_ply(position, search_config, game_seed, evaluator, ply),
        |step| {
            let target = policy_target(step.search, policy_config);
            pending.push(PendingSample {
                raster: rasterize(step.position, raster_config),
                policy: target.policy,
                policy_mask: target.mask,
                visits: target.visits,
                beta: target.beta,
                proposal_counts: target.proposal_counts,
                to_move: step.position.to_move(),
                selected_action: action_index(step.action, policy_config),
                game: game_index,
                ply: step.ply,
                seed: game_seed,
            });
        },
    )?;
    let Some(outcome) = playout.outcome else {
        return Ok(GameSamples {
            index: game_index,
            samples: Vec::new(),
            completed: false,
        });
    };
    let black_value = outcome.black_utility() as f32;
    let samples = pending
        .into_iter()
        .map(|sample| LabeledSample {
            raster: sample.raster,
            policy: sample.policy,
            policy_mask: sample.policy_mask,
            visits: sample.visits,
            beta: sample.beta,
            proposal_counts: sample.proposal_counts,
            value: if sample.to_move == Color::Black {
                black_value
            } else {
                -black_value
            },
            selected_action: sample.selected_action,
            game: sample.game,
            ply: sample.ply,
            seed: sample.seed,
        })
        .collect();
    Ok(GameSamples {
        index: game_index,
        samples,
        completed: true,
    })
}

fn generate(
    config: &Config,
    evaluator: Arc<dyn Evaluator>,
) -> Result<(Vec<LabeledSample>, usize, usize), EvaluationError> {
    let maximum_games = config
        .maximum_games
        .unwrap_or_else(|| (config.samples as u64).saturating_mul(8));
    let next_game = Arc::new(AtomicU64::new(0));
    let stopped = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    let mut handles = Vec::with_capacity(config.actors);
    for _ in 0..config.actors {
        let config = config.clone();
        let evaluator = Arc::clone(&evaluator);
        let next_game = Arc::clone(&next_game);
        let stopped = Arc::clone(&stopped);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let index = next_game.fetch_add(1, Ordering::Relaxed);
                if index >= maximum_games {
                    break;
                }
                if sender
                    .send(generate_game(&config, evaluator.as_ref(), index))
                    .is_err()
                {
                    break;
                }
            }
        }));
    }
    drop(sender);

    let mut samples = Vec::with_capacity(config.samples);
    let mut completed_games = 0;
    let mut discarded_games = 0;
    let mut expected_game = 0_u64;
    let mut pending = BTreeMap::new();
    while samples.len() < config.samples {
        let game = receiver.recv().map_err(|_| {
            EvaluationError::new(format!(
                "replay exhausted {maximum_games} game attempts after {completed_games} completed and {discarded_games} discarded games"
            ))
        })??;
        pending.insert(game.index, game);
        while let Some(game) = pending.remove(&expected_game) {
            expected_game += 1;
            if game.completed {
                completed_games += 1;
                samples.extend(game.samples);
                if samples.len() >= config.samples {
                    samples.truncate(config.samples);
                    break;
                }
            } else {
                discarded_games += 1;
            }
        }
    }
    stopped.store(true, Ordering::Relaxed);
    drop(receiver);
    for handle in handles {
        handle.join().expect("replay worker");
    }
    Ok((samples, completed_games, discarded_games))
}

fn write_f32(writer: &mut impl Write, value: f32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_dataset(path: &Path, samples: &[LabeledSample]) -> std::io::Result<()> {
    let temporary = path.with_extension("vgo.tmp");
    if path.exists() || temporary.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("replay output already exists: {}", path.display()),
        ));
    }
    let config = samples[0].raster.config();
    // The policy vector lives on the placement grid, which may be coarser than
    // the raster, so its own length is the authority for the header -- not
    // `config.pixels() + 1`.
    let policy_size = samples[0].policy.len();
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(&REPLAY_MAGIC)?;
    for value in [
        REPLAY_VERSION,
        samples.len() as u32,
        CHANNEL_COUNT as u32,
        config.height as u32,
        config.width as u32,
        policy_size as u32,
    ] {
        writer.write_all(&value.to_le_bytes())?;
    }
    for sample in samples {
        for &value in sample.raster.data() {
            write_f32(&mut writer, value)?;
        }
        for &value in &sample.policy {
            write_f32(&mut writer, value)?;
        }
        for &value in &sample.policy_mask {
            write_f32(&mut writer, value)?;
        }
        for &value in &sample.visits {
            write_f32(&mut writer, value)?;
        }
        for &value in &sample.beta {
            write_f32(&mut writer, value)?;
        }
        for &value in &sample.proposal_counts {
            writer.write_all(&value.to_le_bytes())?;
        }
        write_f32(&mut writer, sample.value)?;
        writer.write_all(&sample.selected_action.to_le_bytes())?;
        writer.write_all(&sample.game.to_le_bytes())?;
        writer.write_all(&sample.ply.to_le_bytes())?;
        writer.write_all(&sample.seed.to_le_bytes())?;
    }
    writer.flush()?;
    writer.into_inner()?.sync_all()?;
    fs::rename(temporary, path)
}

fn write_manifest(
    path: &Path,
    config: &Config,
    samples: &[LabeledSample],
    completed_games: usize,
    discarded_games: usize,
    dataset_sha256: &str,
    model_sha256: Option<&str>,
) -> std::io::Result<()> {
    let temporary = path.with_extension("json.tmp");
    if path.exists() || temporary.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("replay manifest already exists: {}", path.display()),
        ));
    }
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema\": \"vgo.replay-shard.v1\",")?;
    writeln!(writer, "  \"replay_version\": {REPLAY_VERSION},")?;
    writeln!(writer, "  \"dataset\": \"dataset.vgo\",")?;
    writeln!(writer, "  \"dataset_sha256\": \"{dataset_sha256}\",")?;
    writeln!(writer, "  \"samples\": {},", samples.len())?;
    writeln!(writer, "  \"completed_games\": {completed_games},")?;
    writeln!(writer, "  \"discarded_games\": {discarded_games},")?;
    writeln!(writer, "  \"channels\": {},", CHANNEL_COUNT)?;
    writeln!(writer, "  \"height\": {},", config.resolution)?;
    writeln!(writer, "  \"width\": {},", config.resolution)?;
    writeln!(
        writer,
        "  \"policy_size\": {},",
        config.policy_resolution * config.policy_resolution + 1
    )?;
    writeln!(writer, "  \"simulations\": {},", config.simulations)?;
    writeln!(writer, "  \"coarse_pool\": {},", config.coarse_pool)?;
    writeln!(writer, "  \"temperature\": {},", config.temperature)?;
    writeln!(
        writer,
        "  \"temperature_plies\": {},",
        config.temperature_plies
    )?;
    writeln!(writer, "  \"radius\": {},", config.radius)?;
    writeln!(writer, "  \"seed\": {},", config.seed)?;
    writeln!(writer, "  \"maximum_plies\": {},", config.maximum_plies)?;
    writeln!(writer, "  \"actors\": {},", config.actors)?;
    writeln!(writer, "  \"evaluator\": \"{}\",", config.runtime.as_str())?;
    match model_sha256 {
        Some(digest) => writeln!(writer, "  \"model_sha256\": \"{digest}\",")?,
        None => writeln!(writer, "  \"model_sha256\": null,")?,
    }
    writeln!(
        writer,
        "  \"orientation\": \"row 0 samples y near 0; column 0 samples x near 0\","
    )?;
    writeln!(writer, "  \"perspective\": \"current player\",")?;
    writeln!(
        writer,
        "  \"policy_target\": \"MCTS root visits aggregated by pixel; pass is last\","
    )?;
    writeln!(
        writer,
        "  \"policy_mask\": \"sampled candidate pixels and pass; training derives the full legal denominator from legal_clearance\","
    )?;
    writeln!(
        writer,
        "  \"raw_visits\": \"f32 MCTS root visits aggregated by policy cell\","
    )?;
    writeln!(
        writer,
        "  \"sampling_beta\": \"f32 exact per-draw coarse-to-fine proposal probability for sampled placements; zero for pass and legacy candidates\","
    )?;
    writeln!(
        writer,
        "  \"proposal_counts\": \"u32 raw coarse-to-fine proposal multiplicity aggregated by policy cell; zero for pass and legacy candidates\","
    )?;
    writeln!(
        writer,
        "  \"value_target\": \"terminal utility in [-1, 1] for current player\","
    )?;
    writeln!(writer, "  \"channel_names\": [")?;
    for (index, channel) in CHANNELS.iter().enumerate() {
        let comma = if index + 1 == CHANNEL_COUNT { "" } else { "," };
        writeln!(writer, "    \"{}\"{}", channel.name, comma)?;
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;
    writer.flush()?;
    writer.into_inner()?.sync_all()?;
    fs::rename(temporary, path)
}

fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_bmp(
    path: &Path,
    width: usize,
    height: usize,
    rgb: &[u8],
    scale: usize,
) -> std::io::Result<()> {
    let output_width = width * scale;
    let output_height = height * scale;
    let row_bytes = output_width * 3;
    let row_stride = (row_bytes + 3) & !3;
    let pixel_bytes = row_stride * output_height;
    let file_bytes = 54 + pixel_bytes;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"BM")?;
    writer.write_all(&(file_bytes as u32).to_le_bytes())?;
    writer.write_all(&[0; 4])?;
    writer.write_all(&54_u32.to_le_bytes())?;
    writer.write_all(&40_u32.to_le_bytes())?;
    writer.write_all(&(output_width as i32).to_le_bytes())?;
    writer.write_all(&(output_height as i32).to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&24_u16.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&(pixel_bytes as u32).to_le_bytes())?;
    writer.write_all(&2_835_i32.to_le_bytes())?;
    writer.write_all(&2_835_i32.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    let padding = vec![0_u8; row_stride - row_bytes];
    for row in (0..height).rev() {
        for _ in 0..scale {
            for column in 0..width {
                let start = (row * width + column) * 3;
                for _ in 0..scale {
                    writer.write_all(&[rgb[start + 2], rgb[start + 1], rgb[start]])?;
                }
            }
            writer.write_all(&padding)?;
        }
    }
    writer.flush()
}

fn write_examples(
    directory: &Path,
    samples: &[LabeledSample],
    count: usize,
) -> std::io::Result<()> {
    fs::create_dir_all(directory)?;
    for (sample_index, sample) in samples.iter().take(count).enumerate() {
        let config = sample.raster.config();
        write_bmp(
            &directory.join(format!("sample-{sample_index:03}-overview.bmp")),
            config.width,
            config.height,
            &sample.raster.overview_rgb(),
            6,
        )?;
        for (channel_index, channel) in CHANNELS.iter().enumerate() {
            write_bmp(
                &directory.join(format!(
                    "sample-{sample_index:03}-{channel_index:02}-{}.bmp",
                    channel.name
                )),
                config.width,
                config.height,
                &sample.raster.channel_rgb(channel_index),
                6,
            )?;
        }
    }
    Ok(())
}

fn write_json_string(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
    writer.write_all(b"\"")?;
    for character in value.chars() {
        match character {
            '"' => writer.write_all(b"\\\"")?,
            '\\' => writer.write_all(b"\\\\")?,
            '\u{08}' => writer.write_all(b"\\b")?,
            '\u{0c}' => writer.write_all(b"\\f")?,
            '\n' => writer.write_all(b"\\n")?,
            '\r' => writer.write_all(b"\\r")?,
            '\t' => writer.write_all(b"\\t")?,
            control if control <= '\u{1f}' => write!(writer, "\\u{:04x}", control as u32)?,
            ordinary => write!(writer, "{ordinary}")?,
        }
    }
    writer.write_all(b"\"")
}

fn main() -> std::io::Result<()> {
    let config = Config::parse();
    validate_config(&config)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    fs::create_dir_all(&config.output)?;

    let raster = RasterConfig::square(config.resolution);
    let model_path = config.model.as_deref();
    let evaluator: Arc<dyn Evaluator> = match config.runtime {
        GenerationRuntime::Naive => Arc::new(NaiveEvaluator),
        GenerationRuntime::Onnx => {
            let model = model_path.ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--model is required for ONNX generation",
                )
            })?;
            let service = OnnxBatchService::load(&OnnxServiceConfig {
                policy: Some(RasterConfig::square(config.policy_resolution)),
                model: model.to_path_buf(),
                raster,
                maximum_batch: config.maximum_batch,
                provider: config.provider,
                device_id: 0,
                fp16: config.fp16,
                cache_directory: config.cache_directory.clone(),
            })
            .map_err(std::io::Error::other)?;
            Arc::new(
                BatchedEvaluator::spawn(
                    BrokerConfig {
                        maximum_delay: Duration::from_millis(config.delay_ms),
                        queue_capacity: (config.actors * 4).max(config.maximum_batch * 2),
                    },
                    service,
                )
                .map_err(std::io::Error::other)?,
            )
        }
    };
    let model_sha256 = if config.runtime == GenerationRuntime::Onnx {
        model_path.map(file_sha256).transpose()?
    } else {
        None
    };
    let (samples, completed_games, discarded_games) =
        generate(&config, evaluator).map_err(std::io::Error::other)?;
    let dataset_path = config.output.join("dataset.vgo");
    write_dataset(&dataset_path, &samples)?;
    let dataset_sha256 = file_sha256(&dataset_path)?;
    write_examples(&config.output.join("images"), &samples, config.examples)?;
    write_manifest(
        &config.output.join("manifest.json"),
        &config,
        &samples,
        completed_games,
        discarded_games,
        &dataset_sha256,
        model_sha256.as_deref(),
    )?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    write!(output, "{{\n  \"dataset\": ")?;
    write_json_string(&mut output, &dataset_path.to_string_lossy())?;
    writeln!(
        output,
        concat!(
            ",\n",
            "  \"samples\": {},\n",
            "  \"completed_games\": {},\n",
            "  \"discarded_games\": {},\n",
            "  \"dataset_sha256\": \"{}\",\n",
            "  \"evaluator\": \"{}\",\n",
            "  \"actors\": {},\n",
            "  \"coarse_pool\": {},\n",
            "  \"temperature\": {},\n",
            "  \"temperature_plies\": {},\n",
            "  \"channels\": {},\n",
            "  \"resolution\": {},\n",
            "  \"policy_size\": {},\n",
            "  \"examples\": {}\n",
            "}}"
        ),
        samples.len(),
        completed_games,
        discarded_games,
        dataset_sha256,
        config.runtime.as_str(),
        config.actors,
        config.coarse_pool,
        config.temperature,
        config.temperature_plies,
        CHANNEL_COUNT,
        config.resolution,
        config.policy_resolution * config.policy_resolution + 1,
        config.examples.min(samples.len()),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use vgo_core::Point;
    use vgo_raster::RasterConfig;
    use vgo_search::{Action, CandidateSource, ChildSummary, SearchResult, SearchStats};

    use super::{Config, policy_target, search_config, validate_config};

    #[test]
    fn malformed_cli_values_are_rejected() {
        assert!(Config::try_parse_from(["vgo-generate-demo", "--resolution", "large"]).is_err());
    }

    #[test]
    fn coarse_pool_cli_defaults_to_legacy_and_accepts_an_override() {
        let default = Config::try_parse_from(["vgo-generate-demo"]).expect("default CLI parses");
        assert_eq!(default.coarse_pool, 0);

        let configured = Config::try_parse_from(["vgo-generate-demo", "--coarse-pool", "8"])
            .expect("coarse sampling options parse");
        assert_eq!(configured.coarse_pool, 8);
    }

    #[test]
    fn coarse_sampling_is_applied_to_search_config() {
        let configured = search_config(37, 8, 1.0, 30);
        assert_eq!(configured.simulations, 37);
        assert_eq!(configured.coarse_pool, 8);
        assert_eq!(configured.temperature, 1.0);
        assert_eq!(configured.temperature_plies, 30);
    }

    /// Generation defaults to sampled opening moves. A zero default here would
    /// silently reproduce the deterministic self-play this change exists to fix.
    #[test]
    fn generation_defaults_to_a_positive_opening_temperature() {
        let default = Config::try_parse_from(["vgo-generate-demo"]).expect("default CLI parses");
        assert!(default.temperature > 0.0);
        assert!(default.temperature_plies > 0);
    }

    #[test]
    fn negative_temperature_is_rejected() {
        let configured =
            Config::try_parse_from(["vgo-generate-demo", "--temperature=-1"]).expect("CLI parses");
        assert!(validate_config(&configured).is_err());
    }

    #[test]
    fn invalid_coarse_sampling_config_is_rejected_before_generation() {
        let oversized_pool = Config::try_parse_from([
            "vgo-generate-demo",
            "--policy-resolution",
            "16",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&oversized_pool),
            Err("--coarse-pool must not exceed --policy-resolution")
        );

        // The placement grid is independent of the render raster, so a pool
        // exceeding the raster is legitimate as long as it fits the policy grid.
        let decoupled = Config::try_parse_from([
            "vgo-generate-demo",
            "--resolution",
            "16",
            "--policy-resolution",
            "32",
            "--coarse-pool",
            "17",
        ])
        .expect("CLI syntax parses");
        assert_eq!(validate_config(&decoupled), Ok(()));
    }

    #[test]
    fn zero_maximum_plies_is_rejected_before_generation() {
        let config = Config::try_parse_from(["vgo-generate-demo", "--max-plies", "0"])
            .expect("CLI syntax parses");
        assert_eq!(
            validate_config(&config),
            Err("generation counts, simulations, and dimensions must be positive")
        );
    }

    #[test]
    fn invalid_radius_is_rejected_before_generation() {
        for radius in ["0", "0.5", "NaN"] {
            let config = Config::try_parse_from(["vgo-generate-demo", "--radius", radius])
                .expect("CLI syntax parses");
            assert_eq!(
                validate_config(&config),
                Err("--radius must be finite and between zero and one half")
            );
        }
    }

    #[test]
    fn policy_target_aggregates_proposal_multiplicity_by_pixel() {
        let sampled = |point, visits, proposal_count| ChildSummary {
            action: Action::Place(point),
            source: CandidateSource::AreaSequence,
            prior: 0.25,
            visits,
            black_value: 0.0,
            beta: Some(0.125),
            proposal_count,
        };
        let result = SearchResult::from_children(
            Action::Pass,
            vec![
                sampled(Point::new(0.1, 0.1), 2, 2),
                sampled(Point::new(0.2, 0.2), 1, 3),
                ChildSummary {
                    action: Action::Pass,
                    source: CandidateSource::Pass,
                    prior: 0.5,
                    visits: 1,
                    black_value: 0.0,
                    beta: None,
                    proposal_count: 0,
                },
            ],
            SearchStats::default(),
            vgo_core::Color::Black,
        );

        let target = policy_target(&result, RasterConfig::square(2));

        assert_eq!(target.proposal_counts, vec![5, 0, 0, 0, 0]);
        assert_eq!(target.proposal_counts[4], 0, "pass is not proposed");
    }
}
