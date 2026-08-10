use std::{env, hint::black_box, path::PathBuf, thread, time::Instant};

use vgo_core::{Color, Position, Stone};
use vgo_inference::{
    BatchService, InferenceInput, PythonBatchService, PythonProcessConfig, TorchDevice,
    encode_request_frame,
};
use vgo_raster::{CHANNEL_COUNT, RasterConfig, SemanticRaster, rasterize, rasterize_into};

/// Path to the training venv's interpreter, relative to the repo root.
/// The layout differs by platform: `bin/` on Unix, `Scripts/` on Windows.
#[cfg(windows)]
const VENV_PYTHON: &str = "training/.venv/Scripts/python.exe";
#[cfg(not(windows))]
const VENV_PYTHON: &str = "training/.venv/bin/python3";

fn value_argument<T>(arguments: &[String], name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(pair) = arguments.windows(2).find(|pair| pair[0] == name) else {
        return Ok(default);
    };
    pair[1]
        .parse()
        .map_err(|error| format!("invalid value for {name}: {error}"))
}

fn path_argument(arguments: &[String], name: &str, default: PathBuf) -> PathBuf {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map_or(default, |pair| PathBuf::from(&pair[1]))
}

fn fixture_positions() -> Vec<Position> {
    let radius = 1.0 / 6.0;
    let coordinates = [radius, 0.5, 1.0 - radius];
    let stones = coordinates
        .into_iter()
        .flat_map(|y| coordinates.into_iter().map(move |x| (x, y)))
        .enumerate()
        .map(|(index, (x, y))| {
            Stone::new(
                x,
                y,
                if index % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                },
            )
        })
        .collect::<Vec<_>>();
    (0..=stones.len())
        .map(|count| Position::new(radius, stones[..count].to_vec(), Color::Black))
        .collect()
}

fn raster_throughput(
    positions: &[Position],
    config: RasterConfig,
    samples: usize,
    threads: usize,
) -> (f64, f64) {
    let started = Instant::now();
    let checksum = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for worker in 0..threads {
            handles.push(scope.spawn(move || {
                let mut checksum = 0.0_f64;
                for index in (worker..samples).step_by(threads) {
                    let raster = rasterize(&positions[index % positions.len()], config);
                    checksum += f64::from(raster.data()[index % raster.data().len()]);
                    black_box(&raster);
                }
                checksum
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("raster worker"))
            .sum::<f64>()
    });
    let seconds = started.elapsed().as_secs_f64();
    black_box(checksum);
    (seconds, samples as f64 / seconds)
}

fn raster_into_throughput(
    positions: &[Position],
    config: RasterConfig,
    samples: usize,
    threads: usize,
) -> (f64, f64) {
    let started = Instant::now();
    let checksum = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for worker in 0..threads {
            handles.push(scope.spawn(move || {
                let mut data = vec![f32::NAN; CHANNEL_COUNT * config.pixels()];
                let mut checksum = 0.0_f64;
                for index in (worker..samples).step_by(threads) {
                    rasterize_into(&positions[index % positions.len()], config, &mut data);
                    checksum += f64::from(data[index % data.len()]);
                    black_box(&data);
                }
                checksum
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("raster worker"))
            .sum::<f64>()
    });
    let seconds = started.elapsed().as_secs_f64();
    black_box(checksum);
    (seconds, samples as f64 / seconds)
}

fn make_inputs(rasters: &[SemanticRaster], batch: usize) -> Vec<InferenceInput> {
    (0..batch)
        .map(|index| InferenceInput::new(index as u64 + 1, rasters[index % rasters.len()].clone()))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let root = env::current_dir()?;
    let python = path_argument(&arguments, "--python", root.join(VENV_PYTHON));
    let working_directory = path_argument(&arguments, "--training", root.join("training"));
    let checkpoint = path_argument(
        &arguments,
        "--checkpoint",
        root.join("artifacts/raster-demo/model.pt"),
    );
    let resolution = value_argument(&arguments, "--resolution", 128_usize)?;
    let policy_resolution = value_argument(&arguments, "--policy-resolution", resolution)?;
    let batch = value_argument(&arguments, "--batch", 16_usize)?;
    let raster_samples = value_argument(&arguments, "--raster-samples", 10_000_usize)?;
    let raster_threads = value_argument(&arguments, "--raster-threads", 16_usize)?;
    let frame_batches = value_argument(&arguments, "--frame-batches", 500_usize)?;
    let service_batches = value_argument(&arguments, "--service-batches", 500_usize)?;
    let device = value_argument(&arguments, "--device", TorchDevice::Cuda)?;
    let compile = value_argument(&arguments, "--compile", true)?;
    if resolution == 0
        || policy_resolution == 0
        || batch == 0
        || raster_samples == 0
        || raster_threads == 0
        || frame_batches == 0
        || service_batches == 0
    {
        return Err("benchmark counts must be positive".into());
    }

    let raster_config = RasterConfig::square(resolution);
    let policy_config = RasterConfig::square(policy_resolution);
    let positions = fixture_positions();
    let rasters = positions
        .iter()
        .map(|position| rasterize(position, raster_config))
        .collect::<Vec<_>>();
    let inputs = make_inputs(&rasters, batch);

    let (single_raster_seconds, single_raster_per_second) =
        raster_throughput(&positions, raster_config, raster_samples, 1);
    let (parallel_raster_seconds, parallel_raster_per_second) =
        raster_throughput(&positions, raster_config, raster_samples, raster_threads);
    let (parallel_raster_into_seconds, parallel_raster_into_per_second) =
        raster_into_throughput(&positions, raster_config, raster_samples, raster_threads);

    let frame = encode_request_frame(&inputs)?;
    let framing_started = Instant::now();
    let mut framing_checksum = 0_usize;
    for _ in 0..frame_batches {
        let encoded = encode_request_frame(&inputs)?;
        framing_checksum ^= encoded.len();
        black_box(encoded);
    }
    let framing_seconds = framing_started.elapsed().as_secs_f64();
    black_box(framing_checksum);

    let service_config = PythonProcessConfig {
        python,
        working_directory,
        checkpoint,
        raster: raster_config,
        policy: Some(policy_config),
        maximum_batch: batch,
        torch_threads: 1,
        device,
        compile,
    };
    let mut service = PythonBatchService::spawn(&service_config)?;
    service.infer(&inputs)?;
    let service_started = Instant::now();
    let mut service_checksum = 0.0_f64;
    for _ in 0..service_batches {
        let outputs = service.infer(&inputs)?;
        service_checksum += outputs[0].current_value();
        black_box(outputs);
    }
    let service_seconds = service_started.elapsed().as_secs_f64();
    black_box(service_checksum);

    let framed_positions = frame_batches * batch;
    let service_positions = service_batches * batch;
    println!(
        concat!(
            "{{\n",
            "  \"resolution\": {},\n",
            "  \"policy_resolution\": {},\n",
            "  \"channels\": 10,\n",
            "  \"batch\": {},\n",
            "  \"device\": \"{}\",\n",
            "  \"compiled\": {},\n",
            "  \"raster_samples\": {},\n",
            "  \"raster_single_seconds\": {:.6},\n",
            "  \"raster_single_positions_per_second\": {:.3},\n",
            "  \"raster_parallel_threads\": {},\n",
            "  \"raster_parallel_seconds\": {:.6},\n",
            "  \"raster_parallel_positions_per_second\": {:.3},\n",
            "  \"raster_into_parallel_seconds\": {:.6},\n",
            "  \"raster_into_parallel_positions_per_second\": {:.3},\n",
            "  \"frame_bytes\": {},\n",
            "  \"framing_seconds\": {:.6},\n",
            "  \"framing_positions_per_second\": {:.3},\n",
            "  \"framing_gibibytes_per_second\": {:.3},\n",
            "  \"service_seconds\": {:.6},\n",
            "  \"service_batches_per_second\": {:.3},\n",
            "  \"service_positions_per_second\": {:.3}\n",
            "}}"
        ),
        resolution,
        policy_resolution,
        batch,
        device.as_str(),
        compile,
        raster_samples,
        single_raster_seconds,
        single_raster_per_second,
        raster_threads,
        parallel_raster_seconds,
        parallel_raster_per_second,
        parallel_raster_into_seconds,
        parallel_raster_into_per_second,
        frame.len(),
        framing_seconds,
        framed_positions as f64 / framing_seconds,
        frame.len() as f64 * frame_batches as f64 / framing_seconds / 1024.0_f64.powi(3),
        service_seconds,
        service_batches as f64 / service_seconds,
        service_positions as f64 / service_seconds,
    );
    Ok(())
}
