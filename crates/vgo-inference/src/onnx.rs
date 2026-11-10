use std::{
    fs,
    io::{BufReader, Read, Seek},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant},
};

use half::f16;
use ort::{
    ep,
    memory::Allocator,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session, builder::GraphOptimizationLevel},
    value::TensorElementType,
    value::{DynTensor, DynTensorValueType, DynValue, TensorRef},
};
use sha2::{Digest, Sha256};
use vgo_raster::RasterConfig;
use vgo_raster::packed::{bit_plane_bytes, packed_layout_for};
use vgo_search::EvaluationError;

use crate::{
    BatchContract, BatchService, FinishBatch, InferenceInput, InferenceOutput,
    InferenceStageMetrics, InputLayout, StagedBatch, Stager,
};

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
    description: String,
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
    /// Staging for a packed model, unused by a dense one. Same reason as
    /// `states`: one broker thread owns the session, so these are allocated
    /// once at maximum batch rather than per call.
    bits: Vec<u8>,
    halves: Vec<f16>,
    scalars: Vec<f16>,
    input_layout: InputLayout,
    last_stages: InferenceStageMetrics,
    /// Page-locked staging and output buffers for a packed model on the GPU;
    /// `None` copies through ordinary memory.
    pinned: Option<PinnedIo>,
}

/// Inputs and outputs through page-locked host memory.
///
/// From ordinary memory the driver stages every copy through a buffer of its own,
/// and the lane thread blocks while the CPU does that second copy. In a live
/// generator, with the actors using the memory bus, the 2 MB policy copy back ran
/// at 10 GB/s and the GPU sat idle a quarter of the time waiting on the lanes
/// (nsys, 2026-11-09). Page-locked memory is copied by DMA, without the CPU.
///
/// ONNX Runtime's own pinned allocator can't be used for this: it labels pinned
/// memory as device memory and then refuses to copy a bound input onto itself. So
/// the staging vectors are locked in place with the driver, the outputs stay on
/// the device, and the lane copies them into a pooled locked buffer, which travels
/// with the deferred finish and comes back to the pool.
struct PinnedIo {
    device: i32,
    binding: IoBinding,
    /// Locks on `bits`, `halves` and `scalars`, which never reallocate: they are
    /// sized for the maximum batch at load.
    _staging: Vec<vgo_pinned::Lock>,
    pool: (mpsc::Sender<HostOutputs>, mpsc::Receiver<HostOutputs>),
    /// This lane's copies, off the legacy default stream; see `vgo_pinned::CopyStream`.
    stream: vgo_pinned::CopyStream,
    /// Fixed device inputs for a session that replays a CUDA graph.
    graph: Option<GraphInputs>,
}

/// Device inputs at fixed addresses, bound once, with the outputs likewise bound
/// once to tensors of the full batch.
///
/// TensorRT's CUDA-graph mode records the first run's kernels with their buffer
/// addresses and replays them, so every run must read the same buffers at the same
/// shape. Each batch is copied into these and short batches run padded to the
/// maximum: rows are independent, the stale rows past the batch are never read,
/// and nearly every live batch is full anyway. A replay launches the whole net in
/// one call instead of one launch per kernel.
struct GraphInputs {
    bits: DynTensor,
    dense: DynTensor,
    scalars: DynTensor,
    _allocator: Allocator,
    /// Runs so far, to know when this session is still recording its graph.
    runs: u32,
}

/// Runs a session makes before its graph is surely recorded. ONNX Runtime records
/// on an early run; the margin costs only a few exclusive runs at start.
const RECORDING_RUNS: u32 = 4;

/// Held shared by every graph-mode run and exclusively by a session that may be
/// recording. While a stream records, CUDA rejects synchronous calls from any
/// thread, so another lane's copies broke the recording ("operation not permitted
/// when stream is capturing") and crashed about one start in two with two lanes.
static RECORDING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// A lane's hold on [`RECORDING`] for one run.
enum RecordingGuard {
    Exclusive(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>),
    Shared(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>),
}

/// Set `VGO_CUDA_GRAPH=0` to launch kernels one by one instead, for comparison.
fn cuda_graph_wanted() -> bool {
    std::env::var("VGO_CUDA_GRAPH").map_or(true, |value| value != "0")
}

/// One batch's outputs in locked host memory.
struct HostOutputs {
    // Declared first so the locks are released before the memory is freed.
    _locks: [vgo_pinned::Lock; 2],
    policy: Vec<f32>,
    values: Vec<f32>,
}

impl HostOutputs {
    fn new(device: i32, maximum_batch: usize, policy_size: usize) -> Result<Self, EvaluationError> {
        let mut policy = vec![0.0f32; maximum_batch * policy_size];
        let mut values = vec![0.0f32; maximum_batch];
        let lock = |error: vgo_pinned::Error| EvaluationError::new(error.0);
        let locks = [
            vgo_pinned::Lock::new(device, &mut policy).map_err(lock)?,
            vgo_pinned::Lock::new(device, &mut values).map_err(lock)?,
        ];
        Ok(Self {
            _locks: locks,
            policy,
            values,
        })
    }
}

/// Set `VGO_PINNED_IO=0` to copy through ordinary memory instead, for comparison.
fn pinned_io_wanted() -> bool {
    std::env::var("VGO_PINNED_IO").map_or(true, |value| value != "0")
}

fn pinned_io(
    session: &Session,
    config: &OnnxServiceConfig,
    graph: bool,
    bits: &mut Vec<u8>,
    halves: &mut Vec<f16>,
    scalars: &mut Vec<f16>,
) -> Result<PinnedIo, EvaluationError> {
    let device = config.device_id;
    let lock = |error: vgo_pinned::Error| EvaluationError::new(error.0);
    let ort = |error| evaluation_error("bind pinned ONNX buffers", error);
    let mut binding = session.create_binding().map_err(ort)?;
    let graph = if graph {
        let allocator = Allocator::new(
            session,
            MemoryInfo::new(
                AllocationDevice::CUDA,
                device,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(ort)?,
        )
        .map_err(ort)?;
        let layout = packed_layout_for(config.raster.kind)
            .expect("checked at load that the raster kind has a packed layout");
        let (raster, batch) = (config.raster, config.maximum_batch);
        let policy_size = config.policy.unwrap_or(config.raster).pixels() + 1;
        let tensor = |kind, shape: Vec<usize>| DynTensor::new(&allocator, kind, shape).map_err(ort);
        let bits = tensor(
            TensorElementType::Uint8,
            vec![batch, layout.binary.len(), bit_plane_bytes(raster.pixels())],
        )?;
        let dense = tensor(
            TensorElementType::Float16,
            vec![batch, layout.continuous.len(), raster.height, raster.width],
        )?;
        let scalars = tensor(TensorElementType::Float16, vec![batch, layout.scalar.len()])?;
        let policy = tensor(TensorElementType::Float32, vec![batch, policy_size])?;
        let values = tensor(TensorElementType::Float32, vec![batch])?;
        let inputs = GraphInputs {
            bits,
            dense,
            scalars,
            _allocator: allocator,
            runs: 0,
        };
        binding.bind_input("bits", &inputs.bits).map_err(ort)?;
        binding.bind_input("dense", &inputs.dense).map_err(ort)?;
        binding
            .bind_input("scalars", &inputs.scalars)
            .map_err(ort)?;
        binding.bind_output("policy_logits", policy).map_err(ort)?;
        binding.bind_output("values", values).map_err(ort)?;
        Some(inputs)
    } else {
        None
    };
    Ok(PinnedIo {
        device,
        stream: vgo_pinned::CopyStream::new(device).map_err(lock)?,
        binding,
        graph,
        // Staging is filled in place each batch and sized for the maximum batch at
        // load, so it never reallocates.
        _staging: vec![
            vgo_pinned::Lock::new(device, bits).map_err(lock)?,
            vgo_pinned::Lock::new(device, halves).map_err(lock)?,
            vgo_pinned::Lock::new(device, scalars).map_err(lock)?,
        ],
        pool: mpsc::channel(),
    })
}

impl OnnxBatchService {
    pub fn load(config: &OnnxServiceConfig) -> Result<Self, EvaluationError> {
        Self::load_with_timeout(
            config,
            crate::deadline_from_env("VGO_INFERENCE_STARTUP_SECONDS", 1800)?,
        )
    }

    /// A stalled native load is detached on timeout. Treat this as fatal to the
    /// owning process; retry in a fresh process so GPU/native state is released.
    pub fn load_with_timeout(
        config: &OnnxServiceConfig,
        timeout: Duration,
    ) -> Result<Self, EvaluationError> {
        if timeout.is_zero() {
            return Err(EvaluationError::new(
                "inference startup timeout must be positive",
            ));
        }
        let description = format!(
            "model={} provider={} device={} maximum_batch={} cache={} runtime={}",
            config.model.display(),
            config.provider.as_str(),
            config.device_id,
            config.maximum_batch,
            config.cache_directory.display(),
            std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "system lookup".into())
        );
        eprintln!(
            "[inference] loading {description}; startup limit {}s (TensorRT engine construction may take several minutes)",
            timeout.as_secs_f64()
        );
        let config = config.clone();
        let started = Instant::now();
        let service = crate::deadline::startup(&description, timeout, true, move || {
            Self::load_native(&config)
        })?;
        eprintln!(
            "[inference] ready {description}; elapsed {:.3}s",
            started.elapsed().as_secs_f64()
        );
        Ok(service)
    }

    fn load_native(config: &OnnxServiceConfig) -> Result<Self, EvaluationError> {
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

        // A graph replays fixed buffers, so it needs the pinned path's bound inputs,
        // which only a packed model has.
        let graph = config.provider == OnnxProvider::TensorRt
            && file_declares_packed_input(&model)
            && pinned_io_wanted()
            && cuda_graph_wanted();
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
                    .with_cuda_graph(graph)
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
                // All within noise. Batch is already at its optimum: single-session
                // inference saturates by batch 16-32, and batch 48 and 64 are slower.
                // With two lanes at 128, batch 64 gave 16.8k positions/s against
                // 18.8k at 32 (2026-11-01).
                // Lanes, measured 2026-11-01 over whole games: two and three both
                // gave 18.5k positions/s at 128 and 7.4k at 192, and one gave 13.4k
                // early in games. Three hung at 256 both times it ran, after about
                // 40 s, with the card at 100% and 60 W, until the watchdog ended it.
                // The cause is not known, so keep to two.
                //
                // trt_cuda_graph_enable is set when the pinned path runs: it needs
                // outputs bound to pre-allocated device buffers and a fixed shape
                // (see `GraphInputs`). Through plain Run() it panics ("expected
                // typeinfo_ptr to not be null").
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
        let input_layout = model_input_layout(&session);
        let layout = packed_layout_for(config.raster.kind);
        if input_layout == InputLayout::Packed && layout.is_none() {
            return Err(EvaluationError::new(format!(
                "model takes packed input but raster kind {:?} has no packed layout",
                config.raster.kind
            )));
        }
        let (mut bits, mut halves, mut scalars) = match input_layout {
            InputLayout::Packed => {
                let (bits, halves, scalars) = staging_sizes(config.raster, config.maximum_batch);
                (
                    Vec::with_capacity(bits),
                    Vec::with_capacity(halves),
                    Vec::with_capacity(scalars),
                )
            }
            InputLayout::Dense => (Vec::new(), Vec::new(), Vec::new()),
        };
        let states = match input_layout {
            InputLayout::Dense => Vec::with_capacity(state_capacity),
            InputLayout::Packed => Vec::new(),
        };
        let pinned = (config.provider != OnnxProvider::Cpu
            && input_layout == InputLayout::Packed
            && pinned_io_wanted())
        .then(|| match pinned_io(&session, config, graph, &mut bits, &mut halves, &mut scalars) {
            Ok(io) => Some(io),
            Err(error) => {
                eprintln!("[inference] pinned host buffers unavailable, staging through ordinary memory: {error}");
                None
            }
        })
        .flatten();
        if graph && pinned.is_none() {
            // The session would replay its first run's buffers on every call.
            return Err(EvaluationError::new(
                "CUDA graph mode needs pinned buffers; set VGO_CUDA_GRAPH=0",
            ));
        }
        Ok(Self {
            description: format!(
                "model={} provider={} device={}",
                model.display(),
                config.provider.as_str(),
                config.device_id
            ),
            session,
            raster: config.raster,
            policy: config.policy.unwrap_or(config.raster),
            maximum_batch: config.maximum_batch,
            provider: config.provider,
            states,
            bits,
            halves,
            scalars,
            input_layout,
            last_stages: InferenceStageMetrics::default(),
            pinned,
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

    #[must_use]
    pub const fn input_layout(&self) -> InputLayout {
        self.input_layout
    }
}

impl OnnxBatchService {
    /// Pack a batch and run it, returning the raw policy and value tensors.
    fn run_batch(&mut self, batch: &[InferenceInput]) -> Result<RawOutputs, EvaluationError> {
        self.last_stages = InferenceStageMetrics::default();
        if batch.is_empty() || batch.len() > self.maximum_batch {
            return Err(EvaluationError::new(format!(
                "ONNX batch size {} is outside supported range 1..{}",
                batch.len(),
                self.maximum_batch
            )));
        }
        if batch.iter().any(|input| input.config() != self.raster) {
            return Err(EvaluationError::new("ONNX batch raster shape mismatch"));
        }
        let packing_started = Instant::now();
        let session_started;
        let outputs = match self.input_layout {
            InputLayout::Dense => {
                self.states.clear();
                for input in batch {
                    let raster = input.raster().ok_or_else(|| {
                        EvaluationError::new(
                            "dense model received a packed raster; the generator and the \
                             model disagree about the input layout",
                        )
                    })?;
                    self.states.extend_from_slice(raster.data());
                }
                let input = TensorRef::from_array_view((
                    [
                        batch.len(),
                        self.raster.channels(),
                        self.raster.height,
                        self.raster.width,
                    ],
                    self.states.as_slice(),
                ))
                .map_err(|error| evaluation_error("construct ONNX input tensor", error))?;
                self.last_stages.input_packing_nanoseconds =
                    packing_started.elapsed().as_nanos() as u64;
                session_started = Instant::now();
                self.session.run(ort::inputs! {"states" => input})
            }
            InputLayout::Packed => {
                pack(batch, &mut self.bits, &mut self.halves, &mut self.scalars)?;
                if let Some(io) = self.pinned.as_mut() {
                    self.last_stages.input_packing_nanoseconds =
                        packing_started.elapsed().as_nanos() as u64;
                    let session_started = Instant::now();
                    let raw = run_pinned(
                        &mut self.session,
                        io,
                        self.raster,
                        (&self.bits, &self.halves, &self.scalars),
                        batch.len(),
                        self.maximum_batch,
                        self.policy.pixels() + 1,
                    );
                    self.last_stages.session_run_nanoseconds =
                        session_started.elapsed().as_nanos() as u64;
                    return raw;
                }
                let (bits, dense, scalars) = packed_tensors(
                    self.raster,
                    batch.len(),
                    &self.bits,
                    &self.halves,
                    &self.scalars,
                )?;
                self.last_stages.input_packing_nanoseconds =
                    packing_started.elapsed().as_nanos() as u64;
                session_started = Instant::now();
                self.session.run(ort::inputs! {
                    "bits" => bits,
                    "dense" => dense,
                    "scalars" => scalars,
                })
            }
        };
        self.last_stages.session_run_nanoseconds = session_started.elapsed().as_nanos() as u64;
        let mut outputs = outputs.map_err(|error| evaluation_error("run ONNX inference", error))?;
        let policy = outputs
            .remove("policy_logits")
            .ok_or_else(|| EvaluationError::new("ONNX output policy_logits is missing"))?;
        let values = outputs
            .remove("values")
            .ok_or_else(|| EvaluationError::new("ONNX output values is missing"))?;
        Ok(RawOutputs::Session { policy, values })
    }
}

/// Elements of packed staging for a full batch: bit bytes, dense halves, scalars.
fn staging_sizes(raster: RasterConfig, maximum_batch: usize) -> (usize, usize, usize) {
    let layout = packed_layout_for(raster.kind)
        .expect("checked at load that the raster kind has a packed layout");
    (
        maximum_batch * layout.binary.len() * bit_plane_bytes(raster.pixels()),
        maximum_batch * layout.continuous.len() * raster.pixels(),
        maximum_batch * layout.scalar.len(),
    )
}

/// Pack a batch's rasters end to end into the staging vectors.
fn pack(
    batch: &[InferenceInput],
    bits: &mut Vec<u8>,
    halves: &mut Vec<f16>,
    scalars: &mut Vec<f16>,
) -> Result<(), EvaluationError> {
    bits.clear();
    halves.clear();
    scalars.clear();
    for input in batch {
        let raster = input.packed_raster().ok_or_else(|| {
            EvaluationError::new(
                "packed model received a dense raster; the generator and the \
                 model disagree about the input layout",
            )
        })?;
        bits.extend_from_slice(raster.bits());
        halves.extend_from_slice(raster.dense());
        scalars.extend_from_slice(raster.scalars());
    }
    Ok(())
}

type PackedTensors<'a> = (TensorRef<'a, u8>, TensorRef<'a, f16>, TensorRef<'a, f16>);

/// Views of packed staging as the model's three inputs.
fn packed_tensors<'a>(
    raster: RasterConfig,
    size: usize,
    bits: &'a [u8],
    halves: &'a [f16],
    scalars: &'a [f16],
) -> Result<PackedTensors<'a>, EvaluationError> {
    let layout = packed_layout_for(raster.kind)
        .expect("checked at load that the raster kind has a packed layout");
    let bits = TensorRef::from_array_view((
        [size, layout.binary.len(), bit_plane_bytes(raster.pixels())],
        bits,
    ))
    .map_err(|error| evaluation_error("construct ONNX bits tensor", error))?;
    let dense = TensorRef::from_array_view((
        [size, layout.continuous.len(), raster.height, raster.width],
        halves,
    ))
    .map_err(|error| evaluation_error("construct ONNX dense tensor", error))?;
    let scalars = TensorRef::from_array_view(([size, layout.scalar.len()], scalars))
        .map_err(|error| evaluation_error("construct ONNX scalars tensor", error))?;
    Ok((bits, dense, scalars))
}

/// Run packed inputs from locked staging and copy the outputs into a pooled
/// locked buffer.
#[allow(clippy::too_many_arguments)]
fn run_pinned(
    session: &mut Session,
    io: &mut PinnedIo,
    raster: RasterConfig,
    (bits, halves, scalars): (&[u8], &[f16], &[f16]),
    size: usize,
    maximum_batch: usize,
    policy_size: usize,
) -> Result<RawOutputs, EvaluationError> {
    let ort = |error| evaluation_error("bind pinned ONNX buffers", error);
    // Held until the outputs are copied back: those copies are synchronous too.
    let _recording = io.graph.as_mut().map(|graph| {
        graph.runs = graph.runs.saturating_add(1);
        if graph.runs <= RECORDING_RUNS {
            RecordingGuard::Exclusive(RECORDING.write().unwrap_or_else(|e| e.into_inner()))
        } else {
            RecordingGuard::Shared(RECORDING.read().unwrap_or_else(|e| e.into_inner()))
        }
    });
    if let Some(graph) = io.graph.as_mut() {
        // Bound once at load; only the contents change.
        for result in [
            vgo_pinned::copy_to_device(&io.stream, bits, &mut graph.bits),
            vgo_pinned::copy_to_device(&io.stream, halves, &mut graph.dense),
            vgo_pinned::copy_to_device(&io.stream, scalars, &mut graph.scalars),
        ] {
            result.map_err(|error| EvaluationError::new(error.0))?;
        }
    } else {
        let (bits, dense, scalars) = packed_tensors(raster, size, bits, halves, scalars)?;
        io.binding.bind_input("bits", &bits).map_err(ort)?;
        io.binding.bind_input("dense", &dense).map_err(ort)?;
        io.binding.bind_input("scalars", &scalars).map_err(ort)?;
        // MemoryInfo isn't Send, so it can't be kept in the service.
        let device_memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            io.device,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(ort)?;
        io.binding
            .bind_output_to_device("policy_logits", &device_memory)
            .map_err(ort)?;
        io.binding
            .bind_output_to_device("values", &device_memory)
            .map_err(ort)?;
    }
    let mut outputs = session
        .run_binding(&io.binding)
        .map_err(|error| evaluation_error("run ONNX inference", error))?;
    let mut host = match io.pool.1.try_recv() {
        Ok(host) => host,
        Err(_) => HostOutputs::new(io.device, maximum_batch, policy_size)?,
    };
    for (name, target) in [
        ("policy_logits", &mut host.policy[..size * policy_size]),
        ("values", &mut host.values[..size]),
    ] {
        let tensor = outputs
            .remove(name)
            .ok_or_else(|| EvaluationError::new(format!("ONNX output {name} is missing")))?
            .downcast::<DynTensorValueType>()
            .map_err(|error| evaluation_error("read ONNX output", error))?;
        // A graph runs the full batch; otherwise the run is exactly this batch.
        let rows = if io.graph.is_some() {
            maximum_batch
        } else {
            size
        };
        let shape = tensor.shape();
        if shape.first().copied() != Some(rows as i64) {
            return Err(EvaluationError::new(format!(
                "ONNX output {name} has shape {shape}, expected {rows} rows"
            )));
        }
        vgo_pinned::copy_to_host(&io.stream, tensor, target)
            .map_err(|error| EvaluationError::new(format!("{name}: {error}")))?;
    }
    Ok(RawOutputs::Host {
        host,
        size,
        policy_size,
        back: io.pool.0.clone(),
    })
}

/// One batch's packed inputs in locked memory, filled by the stager thread and
/// returned to its pool once the lane has run them.
struct StagedInputs {
    // Declared first so the locks are released before the memory is freed.
    _locks: [vgo_pinned::Lock; 3],
    bits: Vec<u8>,
    halves: Vec<f16>,
    scalars: Vec<f16>,
    size: usize,
    back: mpsc::Sender<StagedInputs>,
}

impl StagedInputs {
    fn new(
        device: i32,
        raster: RasterConfig,
        maximum_batch: usize,
        back: mpsc::Sender<Self>,
    ) -> Result<Self, EvaluationError> {
        let (bits_size, halves_size, scalars_size) = staging_sizes(raster, maximum_batch);
        let mut bits = Vec::with_capacity(bits_size);
        let mut halves = Vec::with_capacity(halves_size);
        let mut scalars = Vec::with_capacity(scalars_size);
        let lock = |error: vgo_pinned::Error| EvaluationError::new(error.0);
        let locks = [
            vgo_pinned::Lock::new(device, &mut bits).map_err(lock)?,
            vgo_pinned::Lock::new(device, &mut halves).map_err(lock)?,
            vgo_pinned::Lock::new(device, &mut scalars).map_err(lock)?,
        ];
        Ok(Self {
            _locks: locks,
            bits,
            halves,
            scalars,
            size: 0,
            back,
        })
    }
}

/// Set `VGO_STAGED_PACKING=0` to pack on the lane instead, for comparison.
fn staged_packing_wanted() -> bool {
    std::env::var("VGO_STAGED_PACKING").map_or(true, |value| value != "0")
}

impl BatchService for OnnxBatchService {
    fn fatal_native_failure(&self) -> bool {
        true
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn contract(&self) -> BatchContract {
        BatchContract {
            raster: self.raster,
            policy: self.policy,
            maximum_batch: self.maximum_batch,
            input_layout: self.input_layout,
        }
    }

    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError> {
        let raw = self.run_batch(batch)?;
        let started = Instant::now();
        let ids: Vec<u64> = batch.iter().map(InferenceInput::id).collect();
        let result = materialize(raw, &ids, self.policy.pixels() + 1);
        self.last_stages.output_materialization_nanoseconds = started.elapsed().as_nanos() as u64;
        result
    }

    fn infer_deferred(&mut self, batch: &[InferenceInput]) -> Result<FinishBatch, EvaluationError> {
        let raw = self.run_batch(batch)?;
        let ids: Vec<u64> = batch.iter().map(InferenceInput::id).collect();
        let policy_size = self.policy.pixels() + 1;
        Ok(Box::new(move || materialize(raw, &ids, policy_size)))
    }

    fn stager(&mut self) -> Option<Stager> {
        let device = self.pinned.as_ref()?.device;
        if self.input_layout != InputLayout::Packed || !staged_packing_wanted() {
            return None;
        }
        let (raster, maximum_batch) = (self.raster, self.maximum_batch);
        let (back, pool) = mpsc::channel::<StagedInputs>();
        Some(Box::new(move |batch: &[InferenceInput]| {
            if batch.is_empty() || batch.len() > maximum_batch {
                return Err(EvaluationError::new(format!(
                    "ONNX batch size {} is outside supported range 1..{maximum_batch}",
                    batch.len()
                )));
            }
            if batch.iter().any(|input| input.config() != raster) {
                return Err(EvaluationError::new("ONNX batch raster shape mismatch"));
            }
            let mut staged = match pool.try_recv() {
                Ok(staged) => staged,
                Err(_) => StagedInputs::new(device, raster, maximum_batch, back.clone())?,
            };
            pack(
                batch,
                &mut staged.bits,
                &mut staged.halves,
                &mut staged.scalars,
            )?;
            staged.size = batch.len();
            Ok(Box::new(staged) as StagedBatch)
        }))
    }

    fn infer_staged(
        &mut self,
        staged: StagedBatch,
        ids: Vec<u64>,
    ) -> Result<FinishBatch, EvaluationError> {
        self.last_stages = InferenceStageMetrics::default();
        let staged = staged
            .downcast::<StagedInputs>()
            .map_err(|_| EvaluationError::new("staged batch is not from this service"))?;
        let io = self
            .pinned
            .as_mut()
            .ok_or_else(|| EvaluationError::new("staged batch without pinned buffers"))?;
        if staged.size != ids.len() {
            return Err(EvaluationError::new("staged batch does not match its ids"));
        }
        let policy_size = self.policy.pixels() + 1;
        let started = Instant::now();
        let raw = run_pinned(
            &mut self.session,
            io,
            self.raster,
            (&staged.bits, &staged.halves, &staged.scalars),
            staged.size,
            self.maximum_batch,
            policy_size,
        );
        self.last_stages.session_run_nanoseconds = started.elapsed().as_nanos() as u64;
        // The run has finished with the inputs; the stager may refill them.
        let back = staged.back.clone();
        let _ = back.send(*staged);
        let raw = raw?;
        Ok(Box::new(move || materialize(raw, &ids, policy_size)))
    }

    fn last_inference_stages(&self) -> InferenceStageMetrics {
        self.last_stages
    }
}

/// A batch's outputs before they are split per position: the session's own
/// tensors, or a pooled locked buffer that goes back to its lane once read.
enum RawOutputs {
    Session {
        policy: DynValue,
        values: DynValue,
    },
    Host {
        host: HostOutputs,
        size: usize,
        policy_size: usize,
        back: mpsc::Sender<HostOutputs>,
    },
}

/// Split a batch's outputs into one result per position.
fn materialize(
    raw: RawOutputs,
    ids: &[u64],
    policy_size: usize,
) -> Result<Vec<InferenceOutput>, EvaluationError> {
    let split = |policies: &[f32], values: &[f32]| {
        ids.iter()
            .enumerate()
            .map(|(index, &id)| {
                let start = index * policy_size;
                InferenceOutput::new(
                    id,
                    f64::from(values[index]),
                    policies[start..start + policy_size].to_vec(),
                )
            })
            .collect()
    };
    match raw {
        RawOutputs::Session { policy, values } => {
            let (policy_shape, policies) = policy
                .try_extract_tensor::<f32>()
                .map_err(|error| evaluation_error("extract ONNX policy output", error))?;
            let (value_shape, values) = values
                .try_extract_tensor::<f32>()
                .map_err(|error| evaluation_error("extract ONNX value output", error))?;
            if **policy_shape != [ids.len() as i64, policy_size as i64]
                || **value_shape != [ids.len() as i64]
            {
                return Err(EvaluationError::new(format!(
                    "ONNX output shape mismatch: policy={policy_shape}, value={value_shape}"
                )));
            }
            split(policies, values)
        }
        RawOutputs::Host {
            host,
            size,
            policy_size: stored,
            back,
        } => {
            if size != ids.len() || stored != policy_size {
                return Err(EvaluationError::new(
                    "pinned ONNX outputs do not match the batch",
                ));
            }
            let result = split(&host.policy[..size * policy_size], &host.values[..size]);
            // The lane may have stopped; then the buffer is simply freed.
            let _ = back.send(host);
            result
        }
    }
}

fn cuda_provider(device_id: i32) -> ort::ep::ExecutionProviderDispatch {
    // NOTE: with_fuse_conv_bias is intentionally disabled. On this Blackwell
    // (sm_120) onnxruntime build the fused conv+bias kernel corrupts state
    // across Run() calls — the first inference is correct, then outputs compound
    // every subsequent call (maxabs 5 -> 103 -> 455 -> ...) until they overflow
    // to NaN, which surfaced as `invalid inference value` in the arena. The
    // other options are safe.
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

/// Whether the model file declares the packed input contract.
///
/// Read from the file rather than from `session.metadata()`, because TensorRT
/// wants a shape profile for every dynamic input *before* the session exists,
/// and the profile has to name the right ones. Loading twice to find out would
/// cost an engine build.
///
/// `onnx.helper.set_model_props` appends, so the properties land in the last
/// bytes of the file; a tail read finds them without parsing protobuf. A model
/// with no such property is dense, which is every model exported before the
/// packed contract existed.
fn file_declares_packed_input(path: &Path) -> bool {
    const TAIL: u64 = 64 * 1024;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(length) = file.metadata().map(|data| data.len()) else {
        return false;
    };
    let start = length.saturating_sub(TAIL);
    if file.seek(std::io::SeekFrom::Start(start)).is_err() {
        return false;
    }
    let mut tail = Vec::new();
    if file.read_to_end(&mut tail).is_err() {
        return false;
    }
    let key = b"vgo.input_layout";
    let Some(at) = tail.windows(key.len()).position(|window| window == key) else {
        return false;
    };
    // The value follows the key within a few bytes of protobuf framing.
    let window = &tail[at + key.len()..(at + key.len() + 24).min(tail.len())];
    window.windows(6).any(|candidate| candidate == b"packed")
}

fn profile_shapes(config: &OnnxServiceConfig) -> ProfileShapes {
    let packed = file_declares_packed_input(&config.model)
        .then(|| packed_layout_for(config.raster.kind))
        .flatten();
    let shape = |batch| match packed {
        None => format!(
            "states:{batch}x{}x{}x{}",
            config.raster.channels(),
            config.raster.height,
            config.raster.width
        ),
        Some(layout) => format!(
            "bits:{batch}x{}x{},dense:{batch}x{}x{}x{},scalars:{batch}x{}",
            layout.binary.len(),
            bit_plane_bytes(config.raster.pixels()),
            layout.continuous.len(),
            config.raster.height,
            config.raster.width,
            layout.scalar.len(),
        ),
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
    let dense = input_names == ["states"];
    let packed = input_names == ["bits", "dense", "scalars"];
    if !(dense || packed) || output_names != ["policy_logits", "values"] {
        return Err(EvaluationError::new(format!(
            "ONNX input/output contract mismatch: inputs={input_names:?}, outputs={output_names:?}"
        )));
    }
    Ok(())
}

/// Which input contract the model declares.
///
/// Absent means dense: every model exported before `--packed-input` existed
/// omits the property, and those all take a single float32 `states` tensor.
fn model_input_layout(session: &Session) -> InputLayout {
    let Ok(metadata) = session.metadata() else {
        return InputLayout::Dense;
    };
    match metadata.custom("vgo.input_layout").as_deref() {
        Some("packed") => InputLayout::Packed,
        _ => InputLayout::Dense,
    }
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

#[cfg(test)]
mod packed_parity_tests {
    use std::path::PathBuf;

    use vgo_core::{Color, Position, Stone};
    use vgo_raster::{RasterConfig, RasterKind};

    use super::*;
    use crate::{BatchService, InferenceInput};

    /// The Rust packer and the graph's expansion have to agree, and nothing
    /// else checks that they do.
    ///
    /// Each side is tested against its own reference elsewhere: the packed
    /// rasterizer against the dense one in `vgo-raster`, the graph against the
    /// dense export in `test_packed_input.py`. Neither catches a disagreement
    /// *between* the two -- a swapped bit order, a binary plane in the wrong
    /// slot, komi and radius transposed. None of those fail. They produce a
    /// model that quietly reads a different board than the one it was handed.
    ///
    /// Ignored because it needs two exports of one checkpoint and a GPU. Build
    /// them with:
    ///
    /// ```text
    /// python -m vgo_training.export_onnx --checkpoint C --output dense.onnx
    /// python -m vgo_training.export_onnx --checkpoint C --output packed.onnx --packed-input
    /// VGO_PARITY_DIR=<dir> cargo test -p vgo-inference packed_parity -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn the_packed_lane_agrees_with_the_dense_one() {
        let Ok(directory) = std::env::var("VGO_PARITY_DIR") else {
            panic!("set VGO_PARITY_DIR to a directory holding dense.onnx and packed.onnx");
        };
        let directory = PathBuf::from(directory);
        let raster = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let policy = Some(RasterConfig::square(128));

        let service = |name: &str| {
            OnnxBatchService::load(&OnnxServiceConfig {
                model: directory.join(name),
                raster,
                policy,
                maximum_batch: 4,
                // TensorRT by request: the expansion ops are the part most
                // likely to differ by provider, and TensorRT is what runs in
                // production. CPU is the default because it needs no device.
                provider: if std::env::var("VGO_PARITY_TENSORRT").is_ok() {
                    OnnxProvider::TensorRt
                } else {
                    OnnxProvider::Cpu
                },
                device_id: 0,
                fp16: std::env::var("VGO_PARITY_TENSORRT").is_ok(),
                cache_directory: directory.join("cache"),
            })
            .unwrap_or_else(|error| panic!("load {name}: {error}"))
        };
        let mut dense = service("dense.onnx");
        let mut packed = service("packed.onnx");
        assert_eq!(dense.input_layout(), InputLayout::Dense);
        assert_eq!(packed.input_layout(), InputLayout::Packed);

        let radius = 1.0 / 18.0;
        let step = 2.5 * radius;
        let mut stones = Vec::new();
        let mut index = 0;
        'outer: for row in 0..8 {
            for column in 0..8 {
                let x = 0.08 + step * f64::from(column);
                let y = 0.08 + step * f64::from(row);
                if x > 0.95 || y > 0.95 {
                    continue;
                }
                stones.push(Stone::new(
                    x,
                    y,
                    if index % 2 == 0 {
                        Color::Black
                    } else {
                        Color::White
                    },
                ));
                index += 1;
                if index == 24 {
                    break 'outer;
                }
            }
        }
        let position = Position::new(radius, stones, Color::White).with_komi(0.104);

        let dense_out = dense
            .infer(&[InferenceInput::new(
                1,
                vgo_raster::rasterize(&position, raster),
            )])
            .expect("dense inference");
        let packed_out = packed
            .infer(&[InferenceInput::packed(
                1,
                vgo_raster::packed::rasterize_packed(&position, raster),
            )])
            .expect("packed inference");

        let (_, dense_value, dense_policy) = dense_out[0].clone().into_parts();
        let (_, packed_value, packed_policy) = packed_out[0].clone().into_parts();
        assert_eq!(dense_policy.len(), packed_policy.len());

        let worst = dense_policy
            .iter()
            .zip(&packed_policy)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "  value {dense_value:.6} vs {packed_value:.6} (delta {:.3e})\n  policy worst delta {worst:.3e}",
            (dense_value - packed_value).abs()
        );
        // fp16 staging is exact for this data -- the states are fp16 on disk
        // already -- so the two paths should agree to floating-point noise from
        // nothing more than a different graph shape.
        assert!(worst < 1e-4, "policy diverged by {worst}");
        assert!(
            (dense_value - packed_value).abs() < 1e-4,
            "value diverged by {}",
            (dense_value - packed_value).abs()
        );
    }
}
