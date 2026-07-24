use std::{
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use sha2::{Digest, Sha256};
use vgo_raster::{CHANNEL_COUNT, RasterConfig};
use vgo_search::EvaluationError;

use crate::{BatchContract, BatchService, InferenceInput, InferenceOutput};

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
                let profiles = profile_shapes(config);
                let cache = path_text(&cache_directory)?;
                builder
                    .with_execution_providers([
                        ep::TensorRT::default()
                            .with_device_id(config.device_id)
                            .with_fp16(config.fp16)
                            .with_engine_cache(true)
                            .with_engine_cache_path(cache)
                            .with_timing_cache(true)
                            .with_profile_min_shapes(&profiles.minimum)
                            .with_profile_opt_shapes(&profiles.optimum)
                            .with_profile_max_shapes(&profiles.maximum)
                            .build()
                            .error_on_failure(),
                        cuda_provider(config.device_id),
                    ])
                    .map_err(|error| evaluation_error("register ONNX TensorRT provider", error))?
            }
        };

        let session = builder
            .commit_from_file(&model)
            .map_err(|error| evaluation_error("load ONNX model", error))?;
        validate_model(&session, config)?;
        Ok(Self {
            session,
            raster: config.raster,
            policy: config.policy.unwrap_or(config.raster),
            maximum_batch: config.maximum_batch,
            provider: config.provider,
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
        let item_values = CHANNEL_COUNT * self.raster.pixels();
        let mut states = Vec::with_capacity(batch.len() * item_values);
        for input in batch {
            states.extend_from_slice(input.raster().data());
        }
        let input = TensorRef::from_array_view((
            [
                batch.len(),
                CHANNEL_COUNT,
                self.raster.height,
                self.raster.width,
            ],
            states.as_slice(),
        ))
        .map_err(|error| evaluation_error("construct ONNX input tensor", error))?;
        let outputs = self
            .session
            .run(ort::inputs! {"states" => input})
            .map_err(|error| evaluation_error("run ONNX inference", error))?;
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
            CHANNEL_COUNT, config.raster.height, config.raster.width
        )
    };
    ProfileShapes {
        minimum: shape(1),
        optimum: shape(config.maximum_batch),
        maximum: shape(config.maximum_batch),
    }
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
        ("vgo.channels", CHANNEL_COUNT),
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
    use super::OnnxProvider;

    #[test]
    fn providers_parse_without_implicit_fallback() {
        assert_eq!("cpu".parse(), Ok(OnnxProvider::Cpu));
        assert_eq!("cuda".parse(), Ok(OnnxProvider::Cuda));
        assert_eq!("tensorrt".parse(), Ok(OnnxProvider::TensorRt));
        assert!("tensorrt-rtx".parse::<OnnxProvider>().is_err());
        assert!("gpu".parse::<OnnxProvider>().is_err());
    }
}
