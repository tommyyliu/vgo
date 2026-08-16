//! CUDA settled-mask rasterizer for the self-play path.
//!
//! The kernel is compiled by NVRTC at construction, so building this crate needs
//! no nvcc and no host compiler — which matters on this box, where CUDA 13
//! rejects GCC 16. Only the CUDA driver and NVRTC shared libraries are needed at
//! run time, and both ship in the venv the pipeline already uses.
//!
//! Why CUDA rather than the wgpu backend beside it: ONNX Runtime and TensorRT
//! already hold a CUDA context, so a CUDA allocation can be handed straight to
//! `IoBinding` as the session input. A wgpu buffer would need external-memory
//! interop or a trip through host memory, which is the cost this exists to
//! remove. The wgpu path is for the browser, where there is no CUDA.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use vgo_core::{Position, legal_set_vertices};

use vgo_raster::RasterConfig;

const KERNEL: &str = include_str!("settled.cu");

/// A compiled kernel bound to one device, reusable across positions.
///
/// Construction pays for NVRTC compilation and module load, so it is done once
/// and shared. Not `Clone`: the stream is what serialises work, and handing out
/// copies would hide that.
pub struct SettledRasterizer {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

/// Everything that can go wrong before a mask exists.
#[derive(Debug)]
pub enum CudaError {
    /// No driver, no device, or NVRTC missing. Callers fall back to the CPU.
    Unavailable(String),
    /// The kernel failed to compile. A bug here, not an environment problem.
    Compilation(String),
    Launch(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(why) => write!(f, "CUDA unavailable: {why}"),
            Self::Compilation(why) => write!(f, "settled kernel failed to compile: {why}"),
            Self::Launch(why) => write!(f, "settled kernel failed to launch: {why}"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Arithmetic width the kernel runs in.
///
/// This card is a GeForce: fp64 runs at 1/64 of fp32, and the kernel is
/// arithmetic-bound, so the choice is worth roughly that factor. `Double`
/// matches the CPU exactly; `Single` must be justified by measurement, which
/// `examples/validate.rs` does by re-testing every disagreeing pixel against
/// the definition in f64 on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Single,
    Double,
}

impl SettledRasterizer {
    /// Compiles the kernel for `device` and keeps it loaded.
    pub fn new(device: usize) -> Result<Self, CudaError> {
        Self::with_precision(device, Precision::Double)
    }

    /// Compiles the kernel at a chosen arithmetic width.
    pub fn with_precision(device: usize, precision: Precision) -> Result<Self, CudaError> {
        let context = CudaContext::new(device).map_err(|e| CudaError::Unavailable(e.to_string()))?;
        // Compile for the device actually present rather than a fixed arch:
        // this box is sm_120 (Blackwell), and a mismatched image is the failure
        // that cost a day here before (cudaErrorNoKernelImageForDevice).
        let source = match precision {
            Precision::Single => format!("#define VGO_REAL float\n{KERNEL}"),
            Precision::Double => KERNEL.to_string(),
        };
        let ptx = cudarc::nvrtc::compile_ptx(source)
            .map_err(|e| CudaError::Compilation(e.to_string()))?;
        Self::from_ptx(context, ptx)
    }

    fn from_ptx(context: Arc<CudaContext>, ptx: Ptx) -> Result<Self, CudaError> {
        let module = context
            .load_module(ptx)
            .map_err(|e| CudaError::Compilation(e.to_string()))?;
        let kernel = module
            .load_function("settled_mask")
            .map_err(|e| CudaError::Compilation(e.to_string()))?;
        Ok(Self { stream: context.default_stream(), kernel })
    }

    /// Settled masks for a batch of positions, concatenated.
    ///
    /// One launch for the whole batch. Per-position launches spent more on
    /// overhead than on work at realistic stone counts -- 14 stones measured no
    /// faster than the CPU -- and the search broker already groups positions,
    /// so batching is how this is actually used.
    ///
    /// Every position must share `radius`, which holds within a run: it is part
    /// of run identity.
    pub fn masks(
        &self,
        positions: &[&Position],
        config: RasterConfig,
    ) -> Result<Vec<Vec<bool>>, CudaError> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        let pixels = config.pixels();
        let radius = positions[0].radius();
        if let Some(odd) = positions.iter().find(|p| p.radius() != radius) {
            return Err(CudaError::Launch(format!(
                "batch mixes radii {radius} and {}; radius is part of run identity",
                odd.radius()
            )));
        }

        // Concatenate with offsets. Vertex extraction is per-position geometry
        // over the stone list, not per-pixel work, so it stays on the host.
        let mut stones: Vec<f64> = Vec::new();
        let mut stone_offsets: Vec<i32> = Vec::with_capacity(positions.len());
        let mut stone_counts: Vec<i32> = Vec::with_capacity(positions.len());
        let mut vertices: Vec<f64> = Vec::new();
        let mut vertex_offsets: Vec<i32> = Vec::with_capacity(positions.len());
        let mut vertex_counts: Vec<i32> = Vec::with_capacity(positions.len());
        for position in positions {
            stone_offsets.push((stones.len() / 2) as i32);
            stone_counts.push(position.stones().len() as i32);
            for stone in position.stones() {
                stones.push(stone.x);
                stones.push(stone.y);
            }
            let found = legal_set_vertices(position);
            vertex_offsets.push((vertices.len() / 2) as i32);
            vertex_counts.push(found.len() as i32);
            for vertex in &found {
                vertices.push(vertex.x);
                vertices.push(vertex.y);
            }
        }
        // A zero-length allocation has no valid pointer to offset from, and the
        // kernel offsets before reading its count.
        if stones.is_empty() {
            stones.extend_from_slice(&[0.0, 0.0]);
        }
        if vertices.is_empty() {
            vertices.extend_from_slice(&[0.0, 0.0]);
        }

        let copy = |values: &[f64]| {
            self.stream.clone_htod(values).map_err(|e| CudaError::Launch(e.to_string()))
        };
        let copy_i32 = |values: &[i32]| {
            self.stream.clone_htod(values).map_err(|e| CudaError::Launch(e.to_string()))
        };
        let stone_buffer = copy(&stones)?;
        let vertex_buffer = copy(&vertices)?;
        let stone_offset_buffer = copy_i32(&stone_offsets)?;
        let stone_count_buffer = copy_i32(&stone_counts)?;
        let vertex_offset_buffer = copy_i32(&vertex_offsets)?;
        let vertex_count_buffer = copy_i32(&vertex_counts)?;
        let mut out = self
            .stream
            .alloc_zeros::<u8>(pixels * positions.len())
            .map_err(|e| CudaError::Launch(e.to_string()))?;

        let block = (16u32, 16u32, 1u32);
        let grid = (
            (config.width as u32).div_ceil(block.0),
            (config.height as u32).div_ceil(block.1),
            positions.len() as u32,
        );
        let width = config.width as i32;
        let height = config.height as i32;
        let mut builder = self.stream.launch_builder(&self.kernel);
        builder
            .arg(&stone_buffer)
            .arg(&stone_offset_buffer)
            .arg(&stone_count_buffer)
            .arg(&vertex_buffer)
            .arg(&vertex_offset_buffer)
            .arg(&vertex_count_buffer)
            .arg(&radius)
            .arg(&width)
            .arg(&height)
            .arg(&mut out);
        unsafe {
            builder
                .launch(LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 })
                .map_err(|e| CudaError::Launch(e.to_string()))?;
        }

        let bytes = self
            .stream
            .clone_dtoh(&out)
            .map_err(|e| CudaError::Launch(e.to_string()))?;
        Ok(bytes
            .chunks_exact(pixels)
            .map(|chunk| chunk.iter().map(|b| *b != 0).collect())
            .collect())
    }

    /// The settled mask for one position. Delegates to [`Self::masks`] so
    /// there is only one code path; prefer the batch form in hot loops.
    pub fn mask(&self, position: &Position, config: RasterConfig) -> Result<Vec<bool>, CudaError> {
        Ok(self.masks(&[position], config)?.pop().unwrap_or_default())
    }

}
