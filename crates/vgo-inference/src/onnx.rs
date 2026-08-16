use std::{
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use sha2::{Digest, Sha256};
use vgo_raster::RasterConfig;
use vgo_search::EvaluationError;

use crate::{BatchContract, BatchService, InferenceInput, InferenceOutput, InferenceStageMetrics};

const MODEL_SCHEMA: &str = "vgo.raster-policy-value.onnx.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxProvider {
    Cpu,
    Cuda,
    TensorRt,
}

impl OnnxProvider {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
        }
    }
}

impl std::str::FromStr for OnnxProvider {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" => Ok(Self::TensorRt),
            _ => Err(format!("unsupported ONNX provider: {value}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnnxServiceConfig {
    pub model: PathBuf,
    pub raster: RasterConfig,
    /// Placement grid the model's policy head emits, which need not match the
    /// render resolution. A ~9-across board does not need 128x128 of placement
    /// precision, and a coarser policy grid concentrates the fixed number of
    /// coarse->fine proposal draws over far fewer cells. `None` means the policy
    /// grid equals the raster, which is the pre-decoupling behaviour.
    pub policy: Option<RasterConfig>,
    pub maximum_batch: usize,
    pub provider: OnnxProvider,
    pub device_id: i32,
    pub fp16: bool,
    pub cache_directory: PathBuf,
}

pub struct OnnxBatchService {
    session: Session,
    raster: RasterConfig,
    policy: RasterConfig,
    maximum_batch: usize,
    provider: OnnxProvider,
    /// One broker thread owns the synchronous session, so its packed input can
    /// reuse a maximum-batch allocation across every call. Reallocating and
    /// freeing several megabytes for each short GPU run creates avoidable host
    /// allocator pressure on the inference critical path.
    states: Vec<f32>,
    last_stages: InferenceStageMetrics,
}

impl OnnxBatchService {
    pub fn load(config: &OnnxServiceConfig) -> Result<Self, EvaluationError> {
        if config.maximum_batch == 0 {
            return Err(EvaluationError::new("maximum batch must be positive"));
        }
        let model = std::path::absolute(&config.model)
            .map_err(|error| evaluation_error("resolve ONNX model path", error))?;
        let mut builder = Session::builder()
            .map_err(|error| evaluation_error("create ONNX session builder", error))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|error| evaluation_error("configure ONNX graph optimization", error))?
            .with_intra_threads(1)
            .map_err(|error| evaluation_error("configure ONNX threads", error))?;

        builder = match config.provider {
            OnnxProvider::Cpu => builder,
            OnnxProvider::Cuda => builder
                .with_execution_providers([cuda_provider(config.device_id)])
                .map_err(|error| evaluation_error("register ONNX CUDA provider", error))?,
            OnnxProvider::TensorRt => {
                let model_digest = file_sha256(&model)?;
                let cache_directory = scoped_cache_directory(config, &model_digest);
                fs::create_dir_all(&cache_directory)
                    .map_err(|error| evaluation_error("create TensorRT cache", error))?;
                // The engine is weight-specific, so its cache is keyed on the model
                // digest and every trained model rebuilds. The timing cache is not:
                // it records how fast each kernel tactic runs for a given layer shape
                // on this device, which is identical across RL iterations because only
                // the weights change. Leaving it inside the per-digest directory made
                // every rebuild re-benchmark tactics from scratch -- ~10.9s cold versus
                // ~0.3s warm. Hoisting it one level up shares those measurements.
                let timing_directory = timing_cache_directory(config);
                fs::create_dir_all(&timing_directory)
                    .map_err(|error| evaluation_error("create TensorRT timing cache", error))?;
                let profiles = profile_shapes(config);
                let cache = path_text(&cache_directory)?;
                let timing_cache = path_text(&timing_directory)?;
                let tensorrt = ep::TensorRT::default()
                    .with_device_id(config.device_id)
                    .with_fp16(config.fp16)
                    .with_engine_cache(true)
                    .with_engine_cache_path(cache)
                    .with_timing_cache(true)
                    .with_timing_cache_path(timing_cache)
                    .with_profile_min_shapes(&profiles.minimum)
                    .with_profile_opt_shapes(&profiles.optimum)
                    .with_profile_max_shapes(&profiles.maximum);
                // Knobs measured on this model 2026-08-16 and left at their
                // defaults, so nobody re-measures them:
                //
                //   trt_builder_optimization_level 5   13,119 pos/s
                //   trt_auxiliary_streams 1            13,462
                //   trt_auxiliary_streams 0            13,078
                //   default                            13,389
                //
                // All within noise. Batch and lane count are already at their
                // optimum too: single-session inference saturates at ~11.7k
                // pos/s by batch 16-32 (batch 48 and 64 are slower), and two
                // broker lanes reach ~14.4k while three and four are *worse*
                // -- 4.6ms of service time becomes 8.6 and 12.5, so the lanes
                // contend rather than overlap.
                //
                // trt_cuda_graph_enable is deliberately absent. It panics here
                // ("expected typeinfo_ptr to not be null") because ORT's
                // CUDA-graph mode needs outputs bound to pre-allocated device
                // buffers, while this path calls Run() and lets the allocator
                // produce them. Using it means IoBinding plus a fixed batch
                // shape; the upside is bounded by launch overhead, which
                // batch-1 throughput puts at roughly 0.5ms of the 2.7ms
                // batch-32 inference.
                builder
                    .with_execution_providers([
                        tensorrt.build().error_on_failure(),
                        cuda_provider(config.device_id),
                    ])
                    .map_err(|error| evaluation_error("register ONNX TensorRT provider", error))?
            }
        };

        let session = builder
            .commit_from_file(&model)
            .map_err(|error| evaluation_error("load ONNX model", error))?;
        validate_model(&session, config)?;
        let state_capacity = config
            .maximum_batch
            .checked_mul(config.raster.channels())
            .and_then(|value| value.checked_mul(config.raster.pixels()))
            .ok_or_else(|| EvaluationError::new("ONNX input allocation size overflow"))?;
        Ok(Self {
            session,
            raster: config.raster,
            policy: config.policy.unwrap_or(config.raster),
            maximum_batch: config.maximum_batch,
            provider: config.provider,
            states: Vec::with_capacity(state_capacity),
            last_stages: InferenceStageMetrics::default(),
        })
    }

    #[must_use]
    pub const fn provider(&self) -> OnnxProvider {
        self.provider
    }

    #[must_use]
    pub const fn policy_grid(&self) -> RasterConfig {
        self.policy
    }
}

impl BatchService for OnnxBatchService {
    fn contract(&self) -> BatchContract {
        BatchContract {
            raster: self.raster,
            policy: self.policy,
            maximum_batch: self.maximum_batch,
        }
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        self.last_stages = InferenceStageMetrics::default();
        if batch.is_empty() || batch.len() > self.maximum_batch {
            return Err(EvaluationError::new(format!(
                "ONNX batch size {} is outside supported range 1..{}",
                batch.len(),
                self.maximum_batch
            )));
        }
        if batch
            .iter()
            .any(|input| input.raster().config() != self.raster)
        {
            return Err(EvaluationError::new("ONNX batch raster shape mismatch"));
        }
        let packing_started = Instant::now();
        self.states.clear();
        for input in batch {
            self.states.extend_from_slice(input.raster().data());
        }
        let input = TensorRef::from_array_view((
            [
                batch.len(),
                self.raster.channels(),
                self.raster.height,
                self.raster.width,
            ],
            self.states.as_slice(),
        ));
        self.last_stages.input_packing_nanoseconds = packing_started.elapsed().as_nanos() as u64;
        let input =
            input.map_err(|error| evaluation_error("construct ONNX input tensor", error))?;

        let session_started = Instant::now();
        let outputs = self.session.run(ort::inputs! {"states" => input});
        self.last_stages.session_run_nanoseconds = session_started.elapsed().as_nanos() as u64;
        let outputs = outputs.map_err(|error| evaluation_error("run ONNX inference", error))?;

        let materialization_started = Instant::now();
        let result = (|| {
            let (policy_shape, policies) = outputs["policy_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|error| evaluation_error("extract ONNX policy output", error))?;
            let (value_shape, values) = outputs["values"]
                .try_extract_tensor::<f32>()
                .map_err(|error| evaluation_error("extract ONNX value output", error))?;
            let policy_size = self.policy.pixels() + 1;
            if **policy_shape != [batch.len() as i64, policy_size as i64]
                || **value_shape != [batch.len() as i64]
            {
                return Err(EvaluationError::new(format!(
                    "ONNX output shape mismatch: policy={policy_shape}, value={value_shape}"
                )));
            }
            batch
                .iter()
                .enumerate()
                .map(|(index, request)| {
                    let start = index * policy_size;
                    InferenceOutput::new(
                        request.id(),
                        f64::from(values[index]),
                        policies[start..start + policy_size].to_vec(),
                    )
                })
                .collect()
        })();
        self.last_stages.output_materialization_nanoseconds =
            materialization_started.elapsed().as_nanos() as u64;
        result
    }

    fn last_inference_stages(&self) -> InferenceStageMetrics {
        self.last_stages
    }
}

fn cuda_provider(device_id: i32) -> ort::ep::ExecutionProviderDispatch {
    // NOTE: with_fuse_conv_bias is intentionally disabled. On this Blackwell
    // (sm_120) onnxruntime build the fused conv+bias kernel corrupts state
    // across Run() calls — the first inference is correct, then outputs compound
    // every subsequent call (maxabs 5 -> 103 -> 455 -> ...) until they overflow
    // to NaN, which surfaced as `invalid inference value` in the arena. The
    // other options are safe. See docs/NVRTX_HANDOFF.md.
    ep::CUDA::default()
        .with_device_id(device_id)
        .with_tf32(true)
        .with_conv_max_workspace(true)
        .build()
        .error_on_failure()
}

struct ProfileShapes {
    minimum: String,
    optimum: String,
    maximum: String,
}

fn profile_shapes(config: &OnnxServiceConfig) -> ProfileShapes {
    let shape = |batch| {
        format!(
            "states:{batch}x{}x{}x{}",
            config.raster.channels(), config.raster.height, config.raster.width
        )
    };
    ProfileShapes {
        minimum: shape(1),
        optimum: shape(config.maximum_batch),
        maximum: shape(config.maximum_batch),
    }
}

/// Timing-cache location, scoped to everything the kernel measurements depend on
/// except the weights: provider, precision, raster shape, and batch profile. Two
/// models that differ only by training iteration share this directory.
fn timing_cache_directory(config: &OnnxServiceConfig) -> PathBuf {
    config.cache_directory.join(format!(
        "timing-{}-{}-{}x{}-batch{}",
        config.provider.as_str(),
        if config.fp16 { "fp16" } else { "fp32" },
        config.raster.width,
        config.raster.height,
        config.maximum_batch,
    ))
}

fn scoped_cache_directory(config: &OnnxServiceConfig, model_digest: &str) -> PathBuf {
    config.cache_directory.join(format!(
        "{}-{}-{}x{}-batch{}-{}",
        config.provider.as_str(),
        if config.fp16 { "fp16" } else { "fp32" },
        config.raster.width,
        config.raster.height,
        config.maximum_batch,
        &model_digest[..16]
    ))
}

fn file_sha256(path: &Path) -> Result<String, EvaluationError> {
    let file = fs::File::open(path).map_err(|error| evaluation_error("open ONNX model", error))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| evaluation_error("hash ONNX model", error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_model(session: &Session, config: &OnnxServiceConfig) -> Result<(), EvaluationError> {
    let metadata = session
        .metadata()
        .map_err(|error| evaluation_error("read ONNX metadata", error))?;
    if metadata.custom("vgo.schema").as_deref() != Some(MODEL_SCHEMA) {
        return Err(EvaluationError::new("unsupported ONNX model schema"));
    }
    let checkpoint_digest = metadata.custom("vgo.checkpoint_sha256");
    if !checkpoint_digest.as_deref().is_some_and(is_sha256) {
        return Err(EvaluationError::new(
            "ONNX checkpoint digest metadata is invalid",
        ));
    }
    let policy = config.policy.unwrap_or(config.raster);
    for (key, expected) in [
        ("vgo.channels", config.raster.channels()),
        ("vgo.height", config.raster.height),
        ("vgo.width", config.raster.width),
        ("vgo.policy_size", policy.pixels() + 1),
    ] {
        let actual = metadata
            .custom(key)
            .and_then(|value| value.parse::<usize>().ok());
        if actual != Some(expected) {
            return Err(EvaluationError::new(format!(
                "ONNX metadata {key} mismatch: expected {expected}, got {actual:?}"
            )));
        }
    }
    let exported_maximum = metadata
        .custom("vgo.maximum_batch")
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| EvaluationError::new("ONNX maximum batch metadata is invalid"))?;
    if config.maximum_batch > exported_maximum {
        return Err(EvaluationError::new(format!(
            "configured batch {} exceeds ONNX maximum {exported_maximum}",
            config.maximum_batch
        )));
    }
    let input_names = session
        .inputs()
        .iter()
        .map(|input| input.name())
        .collect::<Vec<_>>();
    let output_names = session
        .outputs()
        .iter()
        .map(|output| output.name())
        .collect::<Vec<_>>();
    if input_names != ["states"] || output_names != ["policy_logits", "values"] {
        return Err(EvaluationError::new(format!(
            "ONNX input/output contract mismatch: inputs={input_names:?}, outputs={output_names:?}"
        )));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn path_text(path: &Path) -> Result<String, EvaluationError> {
    path.to_str()
        .map(String::from)
        .ok_or_else(|| EvaluationError::new("ONNX cache path must be valid Unicode"))
}

fn evaluation_error(context: &str, error: impl std::fmt::Display) -> EvaluationError {
    EvaluationError::new(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use vgo_raster::{CHANNEL_COUNT, RasterConfig};

    use super::{OnnxProvider, OnnxServiceConfig, profile_shapes, scoped_cache_directory};

    #[test]
    fn providers_parse_without_implicit_fallback() {
        assert_eq!("cpu".parse(), Ok(OnnxProvider::Cpu));
        assert_eq!("cuda".parse(), Ok(OnnxProvider::Cuda));
        assert_eq!("tensorrt".parse(), Ok(OnnxProvider::TensorRt));
        assert!("tensorrt-rtx".parse::<OnnxProvider>().is_err());
        assert!("gpu".parse::<OnnxProvider>().is_err());
    }

    #[test]
    fn tensorrt_profile_spans_one_through_the_configured_batch() {
        let config = OnnxServiceConfig {
            model: PathBuf::from("model.onnx"),
            raster: RasterConfig::square(96),
            policy: Some(RasterConfig::square(48)),
            maximum_batch: 16,
            provider: OnnxProvider::TensorRt,
            device_id: 2,
            fp16: true,
            cache_directory: PathBuf::from("cache"),
        };

        let profiles = profile_shapes(&config);

        assert_eq!(profiles.minimum, format!("states:1x{CHANNEL_COUNT}x96x96"));
        assert_eq!(profiles.optimum, format!("states:16x{CHANNEL_COUNT}x96x96"));
        assert_eq!(profiles.maximum, profiles.optimum);
    }

    #[test]
    fn scoped_cache_uses_the_configured_base_directory() {
        let config = OnnxServiceConfig {
            model: PathBuf::from("model.onnx"),
            raster: RasterConfig::square(96),
            policy: None,
            maximum_batch: 16,
            provider: OnnxProvider::TensorRt,
            device_id: 0,
            fp16: true,
            cache_directory: PathBuf::from("/mnt/model-cache"),
        };

        assert_eq!(
            scoped_cache_directory(&config, &"a".repeat(64)),
            PathBuf::from("/mnt/model-cache/tensorrt-fp16-96x96-batch16-aaaaaaaaaaaaaaaa")
        );
    }
}
