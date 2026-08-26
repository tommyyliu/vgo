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

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
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
        SettledKernel::from_ptx(context, ptx)?.rasterizer()
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

/// The compiled kernel, shared by every thread that rasterizes.
///
/// Compile once and call [`SettledKernel::rasterizer`] per actor thread. Each
/// rasterizer gets its own CUDA stream, which is the whole point: the earlier
/// measurement that shelved this crate found that centralising rasterization at
/// the broker to get batching cost 20x throughput, because the broker is one
/// thread. Per-thread streams were named there as the only shape that could
/// win, and `default_stream` -- what this used to hand out -- serialises every
/// caller onto one queue, which is the same bottleneck wearing a different hat.
///
/// The context comes from `primary_ctx::retain`, so it is the same context ONNX
/// Runtime and TensorRT already hold rather than a second one competing with
/// them. Sharing the module means NVRTC compiles once instead of once per
/// thread, which at 32 actors is the difference between a startup pause and a
/// noticeable one.
pub struct SettledKernel {
    context: Arc<CudaContext>,
    module: Arc<cudarc::driver::CudaModule>,
}

impl SettledKernel {
    /// Compiles for the device actually present. See [`SettledRasterizer::with_precision`].
    pub fn compile(device: usize, precision: Precision) -> Result<Self, CudaError> {
        let context = CudaContext::new(device).map_err(|e| CudaError::Unavailable(e.to_string()))?;
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
        Ok(Self { context, module })
    }

    /// One rasterizer, with its own stream. Call once per thread that will use
    /// it; sharing one across threads puts them back on a single queue.
    pub fn rasterizer(&self) -> Result<SettledRasterizer, CudaError> {
        let stream = self
            .context
            .new_stream()
            .map_err(|e| CudaError::Unavailable(e.to_string()))?;
        let kernel = self
            .module
            .load_function("settled_mask")
            .map_err(|e| CudaError::Compilation(e.to_string()))?;
        Ok(SettledRasterizer { stream, kernel })
    }
}

#[cfg(test)]
mod tests {
    use vgo_core::{Color, Position, Stone};
    use vgo_raster::{RasterKind, settled_for_raster};

    use super::*;

    /// A lattice coarser than a diameter, so the position is playable.
    pub(super) fn lattice(count: usize, radius: f64) -> Position {
        let step = 2.5 * radius;
        let mut stones = Vec::new();
        let mut index = 0;
        'outer: for row in 0..12 {
            for column in 0..12 {
                let x = 0.08 + step * f64::from(column);
                let y = 0.08 + step * f64::from(row);
                if x > 0.95 || y > 0.95 {
                    continue;
                }
                stones.push(Stone::new(
                    x,
                    y,
                    if index % 2 == 0 { Color::Black } else { Color::White },
                ));
                index += 1;
                if index == count {
                    break 'outer;
                }
            }
        }
        Position::new(radius, stones, Color::White).with_komi(0.104)
    }

    /// The kernel has to agree with the CPU path, or generation would feed the
    /// model something training never renders.
    ///
    /// Not bit-exact by design: the CPU path walks a contour at 1/128 tolerance
    /// and the kernel evaluates `min_s ||x-s|| <= dist(x,L)` directly, so the
    /// kernel is *more* correct and disagreement is expected on the boundary
    /// pixels between them. What matters is that it stays a boundary effect
    /// rather than a different answer, so this bounds it rather than demanding
    /// equality.
    ///
    /// Ignored: needs a CUDA device.
    #[test]
    #[ignore]
    fn the_kernel_agrees_with_the_cpu_path() {
        let radius = 1.0 / 18.0;
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let kernel = match SettledKernel::compile(0, Precision::Single) {
            Ok(kernel) => kernel,
            Err(error) => panic!("compile: {error}"),
        };
        let rasterizer = kernel.rasterizer().expect("stream");

        for count in [8usize, 28, 52] {
            let position = lattice(count, radius);
            let cpu = settled_for_raster(&position, config);
            let gpu = rasterizer.mask(&position, config).expect("mask");
            assert_eq!(cpu.len(), gpu.len());
            let differing = cpu
                .iter()
                .zip(&gpu)
                .filter(|(left, right)| left != right)
                .count();
            let fraction = differing as f64 / cpu.len() as f64;
            println!(
                "  {count:>2} stones: {differing:>5} of {} pixels differ ({:.4}%)",
                cpu.len(),
                fraction * 100.0
            );
            assert!(
                fraction < 0.01,
                "{count} stones: {differing} pixels differ, more than a boundary effect"
            );
        }
    }

    /// Per-thread streams are the shape the earlier measurement said was the
    /// only one that could beat the CPU. This checks they actually run
    /// concurrently rather than serialising, which is what sharing
    /// `default_stream` did.
    ///
    /// Ignored: needs a CUDA device, and the timing is meaningless while
    /// anything else is using it.
    #[test]
    #[ignore]
    fn per_thread_streams_beat_one_stream() {
        use std::time::Instant;

        let radius = 1.0 / 18.0;
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let position = lattice(28, radius);
        let kernel = SettledKernel::compile(0, Precision::Single).expect("compile");
        let rounds = 40;

        for threads in [1usize, 2, 4, 8, 16, 32] {
            let rasterizers: Vec<_> = (0..threads)
                .map(|_| kernel.rasterizer().expect("stream"))
                .collect();
            // Warm every stream before timing.
            for rasterizer in &rasterizers {
                rasterizer.mask(&position, config).expect("warm");
            }
            let started = Instant::now();
            std::thread::scope(|scope| {
                for rasterizer in &rasterizers {
                    let position = &position;
                    scope.spawn(move || {
                        for _ in 0..rounds {
                            rasterizer.mask(position, config).expect("mask");
                        }
                    });
                }
            });
            let elapsed = started.elapsed().as_secs_f64();
            let total = (threads * rounds) as f64;
            println!(
                "  {threads:>2} threads: {:>8.0} masks/s  ({:.3} ms each)",
                total / elapsed,
                elapsed * 1000.0 / total
            );
        }
    }
}

#[cfg(test)]
mod batch_tests {
    use std::time::Instant;

    use vgo_raster::RasterKind;

    use super::tests::lattice;
    use super::*;

    /// How much of a single-position launch is overhead.
    ///
    /// The shelving measurement recorded 0.072 ms per position at f32, but that
    /// was one launch for a whole batch. Per-position launches were noted there
    /// as "more on overhead than on work". This puts numbers on the gap,
    /// because it decides the shape of any integration: an actor evaluating
    /// four leaves at a time can only batch four.
    ///
    /// Ignored: needs a CUDA device, and an idle one.
    #[test]
    #[ignore]
    fn batching_amortises_the_launch() {
        let radius = 1.0 / 18.0;
        let config = RasterConfig::square_of(256, RasterKind::CompactRadius);
        let position = lattice(28, radius);
        let kernel = SettledKernel::compile(0, Precision::Single).expect("compile");
        let rasterizer = kernel.rasterizer().expect("stream");

        for batch in [1usize, 2, 4, 8, 16, 32, 64] {
            let positions: Vec<&Position> = (0..batch).map(|_| &position).collect();
            let rounds = (256 / batch).max(4);
            rasterizer.masks(&positions, config).expect("warm");
            let started = Instant::now();
            for _ in 0..rounds {
                rasterizer.masks(&positions, config).expect("masks");
            }
            let elapsed = started.elapsed().as_secs_f64();
            let total = (batch * rounds) as f64;
            println!(
                "  batch {batch:>3}: {:>8.0} masks/s  ({:.3} ms per position)",
                total / elapsed,
                elapsed * 1000.0 / total
            );
        }
    }
}
