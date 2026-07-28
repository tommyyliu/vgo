#![forbid(unsafe_code)]

use std::{
    env,
    hint::black_box,
    path::{Path, PathBuf},
    time::Instant,
};

use vgo_core::{Color, Position, Stone};
use vgo_inference::{
    BatchService, InferenceInput, OnnxBatchService, OnnxProvider, OnnxServiceConfig,
    PythonBatchService, PythonProcessConfig, TorchDevice,
};
use vgo_raster::{RasterConfig, rasterize};

fn argument_value<'a>(arguments: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    let Some(index) = arguments.iter().position(|argument| argument == name) else {
        return Ok(None);
    };
    let Some(value) = arguments.get(index + 1) else {
        return Err(format!("missing value for {name}"));
    };
    if value.starts_with("--") {
        return Err(format!("missing value for {name}"));
    }
    Ok(Some(value))
}

fn value_argument<T>(arguments: &[String], name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = argument_value(arguments, name)? else {
        return Ok(default);
    };
    value
        .parse()
        .map_err(|error| format!("invalid value for {name}: {error}"))
}

fn path_argument(arguments: &[String], name: &str, default: PathBuf) -> Result<PathBuf, String> {
    let Some(value) = argument_value(arguments, name)? else {
        return Ok(default);
    };
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(PathBuf::from(value))
}

#[derive(Debug, Eq, PartialEq)]
struct Arguments {
    model: PathBuf,
    checkpoint: PathBuf,
    python: PathBuf,
    training: PathBuf,
    cache_directory: PathBuf,
    provider: OnnxProvider,
    resolution: usize,
    policy_resolution: usize,
    batch: usize,
    warmup: usize,
    iterations: usize,
    device_id: i32,
    fp16: bool,
    compare_python: bool,
    python_device: TorchDevice,
    python_compile: bool,
}

fn parse_arguments(arguments: &[String], root: &Path) -> Result<Arguments, String> {
    let model = path_argument(
        arguments,
        "--model",
        root.join("artifacts/raster-demo/model.onnx"),
    )?;
    let checkpoint = path_argument(
        arguments,
        "--checkpoint",
        root.join("artifacts/raster-demo/model.pt"),
    )?;
    let python = path_argument(
        arguments,
        "--python",
        root.join("training/.venv/Scripts/python.exe"),
    )?;
    let training = path_argument(arguments, "--training", root.join("training"))?;
    let cache_directory = path_argument(
        arguments,
        "--cache-directory",
        root.join("artifacts/onnx-cache"),
    )?;
    let provider = value_argument(arguments, "--provider", OnnxProvider::Cuda)?;
    let resolution = value_argument(arguments, "--resolution", 128_usize)?;
    let policy_resolution = value_argument(arguments, "--policy-resolution", resolution)?;
    let batch = value_argument(arguments, "--batch", 8_usize)?;
    let warmup = value_argument(arguments, "--warmup", 10_usize)?;
    let iterations = value_argument(arguments, "--iterations", 200_usize)?;
    let device_id = value_argument(arguments, "--device-id", 0_i32)?;
    let fp16 = value_argument(arguments, "--fp16", true)?;
    let compare_python = value_argument(arguments, "--compare-python", true)?;
    let python_device = value_argument(arguments, "--python-device", TorchDevice::Cpu)?;
    let python_compile = value_argument(arguments, "--python-compile", false)?;

    if resolution == 0 || policy_resolution == 0 || batch == 0 || warmup == 0 || iterations == 0 {
        return Err("raster and policy resolutions and benchmark counts must be positive".into());
    }
    if device_id < 0 {
        return Err("--device-id must be nonnegative".into());
    }

    Ok(Arguments {
        model,
        checkpoint,
        python,
        training,
        cache_directory,
        provider,
        resolution,
        policy_resolution,
        batch,
        warmup,
        iterations,
        device_id,
        fp16,
        compare_python,
        python_device,
        python_compile,
    })
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

fn compare(
    expected: &[vgo_inference::InferenceOutput],
    actual: &[vgo_inference::InferenceOutput],
) -> Result<(f32, f64, usize), std::io::Error> {
    if expected.len() != actual.len()
        || expected
            .iter()
            .zip(actual)
            .any(|(left, right)| left.policy().len() != right.policy().len())
    {
        return Err(std::io::Error::other(
            "inference parity output shapes differ",
        ));
    }
    let mut policy_maximum = 0.0_f32;
    let mut value_maximum = 0.0_f64;
    let mut top_one_equal = 0;
    for (expected, actual) in expected.iter().zip(actual) {
        value_maximum =
            value_maximum.max((expected.current_value() - actual.current_value()).abs());
        for (&left, &right) in expected.policy().iter().zip(actual.policy()) {
            policy_maximum = policy_maximum.max((left - right).abs());
        }
        let expected_top = expected
            .policy()
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index);
        let actual_top = actual
            .policy()
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index);
        top_one_equal += usize::from(expected_top == actual_top);
    }
    Ok((policy_maximum, value_maximum, top_one_equal))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = env::current_dir()?;
    let Arguments {
        model,
        checkpoint,
        python,
        training,
        cache_directory,
        provider,
        resolution,
        policy_resolution,
        batch,
        warmup,
        iterations,
        device_id,
        fp16,
        compare_python,
        python_device,
        python_compile,
    } = parse_arguments(&env::args().collect::<Vec<_>>(), &root)?;
    let raster_config = RasterConfig::square(resolution);
    let policy_config = RasterConfig::square(policy_resolution);
    let positions = fixture_positions();
    let inputs = (0..batch)
        .map(|index| {
            InferenceInput::new(
                index as u64 + 1,
                rasterize(&positions[index % positions.len()], raster_config),
            )
        })
        .collect::<Vec<_>>();
    let load_started = Instant::now();
    let mut service = OnnxBatchService::load(&OnnxServiceConfig {
        model,
        raster: raster_config,
        policy: Some(policy_config),
        maximum_batch: batch,
        provider,
        device_id,
        fp16,
        cache_directory,
    })?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    for _ in 0..warmup {
        black_box(service.infer(&inputs)?);
    }
    let started = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..iterations {
        let outputs = service.infer(&inputs)?;
        checksum += outputs[0].current_value();
        black_box(outputs);
    }
    let seconds = started.elapsed().as_secs_f64();
    black_box(checksum);

    let parity = if compare_python {
        let mut python_service = PythonBatchService::spawn(&PythonProcessConfig {
            python,
            working_directory: training,
            checkpoint,
            raster: raster_config,
            policy: Some(policy_config),
            maximum_batch: batch,
            torch_threads: 1,
            device: python_device,
            compile: python_compile,
        })?;
        let expected = python_service.infer(&inputs)?;
        let actual = service.infer(&inputs)?;
        Some(compare(&expected, &actual)?)
    } else {
        None
    };
    let parity_json = parity.map_or_else(
        || String::from("null"),
        |(policy, value, top_one)| {
            format!(
                concat!(
                    "{{\"policy_max_absolute_difference\":{:.9},",
                    "\"value_max_absolute_difference\":{:.9},",
                    "\"top_one_equal\":{},",
                    "\"top_one_total\":{}}}"
                ),
                policy, value, top_one, batch
            )
        },
    );
    println!(
        concat!(
            "{{\n",
            "  \"provider\": \"{}\",\n",
            "  \"resolution\": {},\n",
            "  \"policy_resolution\": {},\n",
            "  \"batch\": {},\n",
            "  \"fp16\": {},\n",
            "  \"warmup\": {},\n",
            "  \"iterations\": {},\n",
            "  \"load_seconds\": {:.6},\n",
            "  \"seconds\": {:.6},\n",
            "  \"milliseconds_per_batch\": {:.6},\n",
            "  \"positions_per_second\": {:.3},\n",
            "  \"parity\": {}\n",
            "}}"
        ),
        service.provider().as_str(),
        resolution,
        policy_resolution,
        batch,
        fp16,
        warmup,
        iterations,
        load_seconds,
        seconds,
        seconds * 1000.0 / iterations as f64,
        batch as f64 * iterations as f64 / seconds,
        parity_json,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::parse_arguments;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn device_and_cache_defaults_match_the_existing_benchmark() {
        let root = Path::new("/workspace/vgo");
        let parsed = parse_arguments(&arguments(&["vgo-onnx-bench"]), root).unwrap();

        assert_eq!(parsed.device_id, 0);
        assert_eq!(parsed.cache_directory, root.join("artifacts/onnx-cache"));
    }

    #[test]
    fn device_and_cache_overrides_are_preserved() {
        let parsed = parse_arguments(
            &arguments(&[
                "vgo-onnx-bench",
                "--device-id",
                "2",
                "--cache-directory",
                "/mnt/tensorrt-cache",
            ]),
            Path::new("/workspace/vgo"),
        )
        .unwrap();

        assert_eq!(parsed.device_id, 2);
        assert_eq!(parsed.cache_directory, PathBuf::from("/mnt/tensorrt-cache"));
    }

    #[test]
    fn invalid_device_ids_are_rejected_before_loading_a_model() {
        for value in ["-1", "gpu"] {
            let error = parse_arguments(
                &arguments(&["vgo-onnx-bench", "--device-id", value]),
                Path::new("/workspace/vgo"),
            )
            .unwrap_err();
            assert!(error.contains("--device-id"));
        }
    }

    #[test]
    fn cache_directory_requires_a_nonempty_value() {
        for values in [
            &["vgo-onnx-bench", "--cache-directory"][..],
            &["vgo-onnx-bench", "--cache-directory", ""][..],
            &["vgo-onnx-bench", "--cache-directory", "--device-id", "0"][..],
        ] {
            let error =
                parse_arguments(&arguments(values), Path::new("/workspace/vgo")).unwrap_err();
            assert!(error.contains("--cache-directory"));
        }
    }
}
