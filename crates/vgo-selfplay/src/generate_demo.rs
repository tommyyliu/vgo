#![forbid(unsafe_code)]

use std::{
    convert::Infallible,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use vgo_core::{Color, Position};
use vgo_raster::{
    CHANNEL_COUNT, CHANNELS, DATASET_MAGIC, DATASET_VERSION, RasterConfig, SemanticRaster,
    action_pixel, rasterize,
};
use vgo_search::{Action, SearchConfig, SearchResult, search};
use vgo_selfplay::play_game as run_playout;

struct PendingSample {
    raster: SemanticRaster,
    policy: Vec<f32>,
    policy_mask: Vec<f32>,
    to_move: Color,
}

struct LabeledSample {
    raster: SemanticRaster,
    policy: Vec<f32>,
    policy_mask: Vec<f32>,
    value: f32,
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
}

fn policy_target(result: &SearchResult, config: RasterConfig) -> (Vec<f32>, Vec<f32>) {
    let mut policy = vec![0.0_f32; config.pixels() + 1];
    let mut policy_mask = vec![0.0_f32; config.pixels() + 1];
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
        policy_mask[index] = 1.0;
    }
    (policy, policy_mask)
}

fn generate(config: &Config) -> (Vec<LabeledSample>, usize, usize) {
    let raster_config = RasterConfig::square(config.resolution);
    let mut samples = Vec::with_capacity(config.samples);
    let mut completed_games = 0;
    let mut discarded_games = 0;
    let mut game_index = 0_u64;

    while samples.len() < config.samples {
        let game_seed = config.seed.wrapping_add(game_index);
        game_index += 1;
        let mut pending = Vec::new();
        let playout = run_playout(
            Position::new(config.radius, Vec::new(), Color::Black),
            config.maximum_plies,
            |position, _ply| {
                Ok::<_, Infallible>(search(
                    position,
                    SearchConfig::canary(config.simulations),
                    game_seed,
                ))
            },
            |step| {
                let (policy, policy_mask) = policy_target(step.search, raster_config);
                pending.push(PendingSample {
                    raster: rasterize(step.position, raster_config),
                    policy,
                    policy_mask,
                    to_move: step.position.to_move(),
                });
            },
        )
        .expect("infallible search");

        if let Some(outcome) = playout.outcome {
            let black_value = outcome.black_utility() as f32;
            completed_games += 1;
            for sample in pending {
                let value = if sample.to_move == Color::Black {
                    black_value
                } else {
                    -black_value
                };
                samples.push(LabeledSample {
                    raster: sample.raster,
                    policy: sample.policy,
                    policy_mask: sample.policy_mask,
                    value,
                });
                if samples.len() == config.samples {
                    break;
                }
            }
        } else {
            discarded_games += 1;
        }
    }

    (samples, completed_games, discarded_games)
}

fn write_f32(writer: &mut impl Write, value: f32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_dataset(path: &Path, samples: &[LabeledSample]) -> std::io::Result<()> {
    let config = samples[0].raster.config();
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&DATASET_MAGIC)?;
    for value in [
        DATASET_VERSION,
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
        write_f32(&mut writer, sample.value)?;
    }
    writer.flush()
}

fn write_manifest(path: &Path, config: &Config, samples: &[LabeledSample]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema\": \"vgo.raster-dataset.v2\",")?;
    writeln!(writer, "  \"samples\": {},", samples.len())?;
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
    writer.flush()
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
    fs::create_dir_all(&config.output)?;

    let (samples, completed_games, discarded_games) = generate(&config);
    let dataset_path = config.output.join("dataset.vgo");
    write_dataset(&dataset_path, &samples)?;
    write_manifest(&config.output.join("manifest.json"), &config, &samples)?;
    write_examples(&config.output.join("images"), &samples, config.examples)?;
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
            "  \"channels\": {},\n",
            "  \"resolution\": {},\n",
            "  \"policy_size\": {},\n",
            "  \"examples\": {}\n",
            "}}"
        ),
        samples.len(),
        completed_games,
        discarded_games,
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
