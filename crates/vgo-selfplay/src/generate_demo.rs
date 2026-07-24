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
    search_with_evaluator,
};
use vgo_selfplay::play_game as run_playout;

const REPLAY_MAGIC: [u8; 8] = *b"VGORPLY1";
// v2 adds raw per-cell visit counts and the coarse->fine sampling probability
// (beta) after the policy mask, so training can apply an off-policy importance
// correction independent of how candidates were drawn. See docs/POLICY_REDESIGN.md.
const REPLAY_VERSION: u32 = 2;

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
    #[arg(long, default_value_t = 128)]
    resolution: usize,
    #[arg(long, default_value_t = 100)]
    simulations: u32,
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
}

fn policy_target(result: &SearchResult, config: RasterConfig) -> PolicyTarget {
    let size = config.pixels() + 1;
    let mut policy = vec![0.0_f32; size];
    let mut mask = vec![0.0_f32; size];
    let mut visits = vec![0.0_f32; size];
    let mut beta = vec![0.0_f32; size];
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
    }
    PolicyTarget {
        policy,
        mask,
        visits,
        beta,
    }
}

fn action_index(action: Action, config: RasterConfig) -> u32 {
    match action {
        Action::Pass => config.pixels() as u32,
        Action::Place(point) => action_pixel(point.x, point.y, config) as u32,
    }
}

fn generate_game(
    config: &Config,
    evaluator: &dyn Evaluator,
    game_index: u64,
) -> Result<GameSamples, EvaluationError> {
    let raster_config = RasterConfig::square(config.resolution);
    let game_seed = config.seed.wrapping_add(game_index);
    let mut pending = Vec::new();
    let playout = run_playout(
        Position::new(config.radius, Vec::new(), Color::Black),
        config.maximum_plies,
        |position, _ply| {
            search_with_evaluator(
                position,
                SearchConfig::canary(config.simulations),
                game_seed,
                evaluator,
            )
        },
        |step| {
            let target = policy_target(step.search, raster_config);
            pending.push(PendingSample {
                raster: rasterize(step.position, raster_config),
                policy: target.policy,
                policy_mask: target.mask,
                visits: target.visits,
                beta: target.beta,
                to_move: step.position.to_move(),
                selected_action: action_index(step.action, raster_config),
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
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(&REPLAY_MAGIC)?;
    for value in [
        REPLAY_VERSION,
        samples.len() as u32,
        CHANNEL_COUNT as u32,
        config.height as u32,
        config.width as u32,
        (config.pixels() + 1) as u32,
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
        config.resolution * config.resolution + 1
    )?;
    writeln!(writer, "  \"simulations\": {},", config.simulations)?;
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
        "  \"policy_mask\": \"sampled candidate pixels and pass; unsampled pixels are excluded from policy loss\","
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
    assert!(config.samples > 0);
    assert!(config.resolution > 0);
    assert!(config.simulations > 0);
    assert!(config.maximum_batch > 0);
    assert!(config.actors > 0);
    assert!(config.maximum_games.is_none_or(|games| games > 0));
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
        CHANNEL_COUNT,
        config.resolution,
        config.resolution * config.resolution + 1,
        config.examples.min(samples.len()),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Config;

    #[test]
    fn malformed_cli_values_are_rejected() {
        assert!(Config::try_parse_from(["vgo-generate-demo", "--resolution", "large"]).is_err());
    }
}
