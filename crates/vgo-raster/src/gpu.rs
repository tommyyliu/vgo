//! GPU evaluation of the settled-region mask.
//!
//! The CPU path (`settled_mask` in `lib.rs`) solves each stone's settled
//! boundary as a contour and scanline-fills it, which costs `O(n²)` per-stone
//! setup and is 33% of self-play CPU. The direct per-pixel formulation is 42x
//! slower on a CPU but is embarrassingly parallel: 16,384 independent pixels,
//! each a bounded reduction over `n + |V|` candidates. `settled.wgsl` runs that
//! formulation one invocation per pixel. See `docs/SETTLED_REGION_PROBLEM.md`.
//!
//! This module is behind the `gpu` feature because it pulls in `wgpu` (and thus
//! Vulkan), which the WASM/CPU-only builds do not need.

use std::sync::LazyLock;

use vgo_core::{Position, legal_set_vertices};
use wgpu::util::DeviceExt;

use crate::RasterConfig;

/// Mirrors `Params` in `settled.wgsl`, byte-for-byte.
///
/// WGSL uniform layout: `radius` sits at offset 16 (after the four `u32`
/// fields), and the trailing `vec3<f32>` rounds the struct to 48 bytes (a
/// multiple of 16). Four `u32` (16) + one `f32` (4) + 7 `f32` of padding (28)
/// = 48, matching the byte size `naga` reports for the shader struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    stone_count: u32,
    vertex_count: u32,
    radius: f32,
    _pad: [f32; 7],
}
/// A device, queue, and the compiled pipeline, created once and shared across
/// every call. Device creation is the expensive part; the per-call work is just
/// buffer upload, dispatch, and readback.
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static CONTEXT: LazyLock<Result<Context, String>> = LazyLock::new(build_context);

fn context() -> Result<&'static Context, String> {
    CONTEXT.as_ref().map_err(|e| e.clone())
}

fn build_context() -> Result<Context, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no adapter: {e}"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vgo-raster settled"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("no device: {e}"))?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("settled.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("settled.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("settled bind group"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("settled pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("settled pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("settled_mask"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(Context {
        device,
        queue,
        pipeline,
        bind_group_layout,
    })
}

/// The settled channel as a mask, evaluated on the GPU.
///
/// Returns `None` when no GPU is available or the shader cannot run, so callers
/// can fall back to [`crate::settled_mask`]. The result is computed in f32 and
/// may differ from the f64 CPU path on pixels within f32 epsilon of the settled
/// boundary; see the validation test in `lib.rs`.
#[must_use]
pub fn settled_mask_gpu(position: &Position, config: RasterConfig) -> Option<Vec<bool>> {
    let context = context().ok()?;
    let pixels = config.pixels();
    let stones = position.stones();
    let vertices = legal_set_vertices(position);

    let stone_data: Vec<[f32; 2]> = stones
        .iter()
        .map(|s| [s.x as f32, s.y as f32])
        .collect();
    let vertex_data: Vec<[f32; 2]> = vertices
        .iter()
        .map(|p| [p.x as f32, p.y as f32])
        .collect();

    let params = Params {
        width: config.width as u32,
        height: config.height as u32,
        stone_count: stones.len() as u32,
        vertex_count: vertices.len() as u32,
        radius: position.radius() as f32,
        _pad: [0.0; 7],
    };

    let params_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let stone_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stones"),
            contents: bytemuck::cast_slice(&stone_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let vertex_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // The compute shader writes this; it is never mapped. A second staging
    // buffer receives the copy and is the one mapped for readback: the Vulkan
    // backend (without `MAPPABLE_PRIMARY_BUFFERS`) refuses to map a storage
    // buffer.
    let settled_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("settled"),
        size: (pixels * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("settled staging"),
        size: (pixels * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("settled bind group"),
            layout: &context.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stone_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: settled_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("settled encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("settled pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&context.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            config.width.div_ceil(8) as u32,
            config.height.div_ceil(8) as u32,
            1,
        );
    }
    // Copy the shader's result into the host-readable staging buffer, then
    // submit the whole command stream once.
    encoder.copy_buffer_to_buffer(
        &settled_buffer,
        0,
        &staging_buffer,
        0,
        (pixels * std::mem::size_of::<u32>()) as u64,
    );
    context.queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| format!("poll: {e}"))
        .ok()?;

    let view = slice.get_mapped_range().map_err(|e| format!("map: {e}")).ok()?;
    let mut settled = Vec::with_capacity(pixels);
    for chunk in view.chunks_exact(std::mem::size_of::<u32>()) {
        let value = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        settled.push(value != 0);
    }
    drop(view);
    staging_buffer.unmap();

    Some(settled)
}
