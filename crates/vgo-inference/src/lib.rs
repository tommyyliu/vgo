#![forbid(unsafe_code)]

mod broker;
mod deadline;
mod evaluator;
mod executor;
mod metrics;
mod onnx;
mod protocol;

pub use evaluator::BatchedEvaluatorPool;
pub use executor::{CompletedBatch, InferenceBatch, ThreadedBatchExecutorPool};
pub use metrics::BrokerMetrics;
pub use onnx::{OnnxBatchService, OnnxProvider, OnnxServiceConfig};
pub use protocol::{
    InferenceInput, InferenceOutput, encode_request_frame, read_response_frame,
    read_response_frame_with_policy,
};

use std::time::Duration;

use vgo_core::Position;
use vgo_raster::{RasterConfig, packed::rasterize_packed, rasterize};
use vgo_search::EvaluationError;

fn deadline_from_env(name: &str, default_seconds: u64) -> Result<Duration, EvaluationError> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .ok()
            .filter(|seconds| *seconds > 0)
            .map(Duration::from_secs)
            .ok_or_else(|| {
                EvaluationError::new(format!("{name} must be a positive integer in seconds"))
            }),
        Err(std::env::VarError::NotPresent) => Ok(Duration::from_secs(default_seconds)),
        Err(error) => Err(EvaluationError::new(format!("{name}: {error}"))),
    }
}

/// Renders one position into whatever form the backend asked for.
///
/// Both request producers go through here rather than each choosing, because a
/// mismatch is invisible until inference: a graph fed the wrong layout does not
/// fail, it computes something else. The backend states the layout in its
/// contract and this is the only place that reads it.
fn encode_request(id: u64, position: &Position, contract: BatchContract) -> InferenceInput {
    match contract.input_layout {
        InputLayout::Dense => InferenceInput::new(id, rasterize(position, contract.raster)),
        InputLayout::Packed => {
            InferenceInput::packed(id, rasterize_packed(position, contract.raster))
        }
    }
}

/// Which input contract a model was exported with.
///
/// A packed model takes the raster compressed -- binary planes as bits,
/// constant planes as one value each, the rest at fp16 -- and expands it in the
/// graph. At 256x256 that stages 152 KB per position against 1792 KB, and the
/// inference lanes were measured at 87% CPU while actors idled at 39%.
///
/// Read from the model rather than configured, so a binary and a model that
/// disagree fail at load with a name mismatch instead of feeding the wrong
/// bytes into a graph that will happily consume them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InputLayout {
    #[default]
    Dense,
    Packed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchContract {
    pub raster: RasterConfig,
    /// Placement grid the policy head emits. Equal to `raster` unless the model
    /// was exported with a decoupled (coarser) policy grid.
    pub policy: RasterConfig,
    pub maximum_batch: usize,
    /// Which form the backend wants positions in. Set by the model, so every
    /// producer of requests has to ask rather than assume.
    pub input_layout: InputLayout,
}

/// Time spent inside the host-visible stages of one backend call.
///
/// `session_run_nanoseconds` still includes runtime setup and any synchronous
/// host/device transfers performed by ONNX Runtime. Splitting those transfers
/// further requires I/O binding and explicit device buffers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceStageMetrics {
    pub input_packing_nanoseconds: u64,
    pub session_run_nanoseconds: u64,
    pub output_materialization_nanoseconds: u64,
}

pub trait BatchService: Send {
    /// Arm a final process watchdog after a native failure that may also block
    /// exit handlers. Synthetic services can simply return their errors.
    fn fatal_native_failure(&self) -> bool {
        false
    }
    fn description(&self) -> String {
        std::any::type_name::<Self>().to_owned()
    }
    fn contract(&self) -> BatchContract;
    fn infer(&mut self, batch: &[InferenceInput]) -> Result<Vec<InferenceOutput>, EvaluationError>;

    /// Return the stage timings for the most recent `infer` call. Backends
    /// without internal instrumentation leave the call unattributed.
    fn last_inference_stages(&self) -> InferenceStageMetrics {
        InferenceStageMetrics::default()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BrokerConfig {
    pub maximum_delay: Duration,
    pub queue_capacity: usize,
}

#[cfg(test)]
mod tests;
