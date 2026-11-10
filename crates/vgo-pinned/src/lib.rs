//! Page-locked host memory for copying inference inputs and outputs by DMA.
//!
//! From ordinary memory the CUDA driver stages each copy through a buffer of its own
//! and blocks the calling thread while the CPU does that second copy. In a live
//! generator, with the actors using the memory bus, a lane's 2 MB policy copy ran at
//! 10 GB/s and left the GPU idle a quarter of the time (nsys, 2026-11-09). Locked
//! memory is copied by the GPU's copy engine, without the CPU.
//!
//! The driver calls are unsafe; this crate keeps them behind checks so callers stay
//! in safe code.

use std::ffi::c_void;

use cudarc::driver::sys;
use ort::{
    memory::AllocationDevice,
    value::{DynTensor, Value},
};

/// A driver failure, with what was being attempted.
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn check(what: &str, result: sys::CUresult) -> Result<(), Error> {
    result
        .result()
        .map_err(|error| Error(format!("{what}: {error}")))
}

thread_local! {
    static CURRENT: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

/// Make `device`'s primary context, the one ONNX Runtime uses, current on this
/// thread. Driver calls need it; this is a no-op after the first call per thread.
pub fn use_device(device: i32) -> Result<(), Error> {
    if CURRENT.with(std::cell::Cell::get) == Some(device) {
        return Ok(());
    }
    // SAFETY: cuInit is idempotent, cuDeviceGet validates the ordinal, and the
    // retained primary context outlives the process's use of it.
    unsafe {
        check("cuInit", sys::cuInit(0))?;
        let mut handle = 0;
        check("cuDeviceGet", sys::cuDeviceGet(&mut handle, device))?;
        let mut context = std::ptr::null_mut();
        check(
            "retain primary context",
            sys::cuDevicePrimaryCtxRetain(&mut context, handle),
        )?;
        check("set current context", sys::cuCtxSetCurrent(context))?;
    }
    CURRENT.with(|current| current.set(Some(device)));
    Ok(())
}

/// A page lock on a vector's whole allocation, released on drop.
///
/// The vector must not reallocate while locked, or the lock covers freed memory:
/// harmless to Rust, but the copies lose their speed. Callers size it once.
pub struct Lock {
    address: usize,
}

impl Lock {
    /// Lock `memory`'s allocation, including spare capacity, on `device`.
    pub fn new<T>(device: i32, memory: &mut Vec<T>) -> Result<Self, Error> {
        use_device(device)?;
        let bytes = memory.capacity() * size_of::<T>();
        if bytes == 0 {
            return Err(Error("nothing to lock".into()));
        }
        let pointer = memory.as_mut_ptr().cast::<c_void>();
        // SAFETY: the range is exactly the vector's live allocation. PORTABLE makes
        // the lock valid in every context, so any lane thread may copy into it.
        unsafe {
            check(
                "lock host memory",
                sys::cuMemHostRegister_v2(pointer, bytes, sys::CU_MEMHOSTREGISTER_PORTABLE),
            )?;
        }
        Ok(Self {
            address: pointer as usize,
        })
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        // SAFETY: registered in `new`. A failure leaves the pages locked, which the
        // allocator that frees them doesn't notice.
        unsafe {
            let _ = sys::cuMemHostUnregister(self.address as *mut c_void);
        }
    }
}

/// A lane's own stream for copies.
///
/// A plain `cuMemcpy` runs on the legacy default stream, which waits for every other
/// stream's work and makes theirs wait for it. With two lanes each copy then waited
/// on the other lane's kernels, the lanes could not overlap, and the GPU sat fully
/// idle a quarter of the time (nsys, 2026-11-09: every copy on stream 7). A
/// non-blocking stream orders the copies only against themselves; ONNX Runtime
/// synchronizes its own streams before a run returns, and each copy here waits
/// for itself to finish.
pub struct CopyStream {
    device: i32,
    stream: usize,
}

impl CopyStream {
    pub fn new(device: i32) -> Result<Self, Error> {
        use_device(device)?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: creates a stream in the current (primary) context.
        unsafe {
            check(
                "create copy stream",
                sys::cuStreamCreate(
                    &mut stream,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                ),
            )?;
        }
        Ok(Self {
            device,
            stream: stream as usize,
        })
    }

    fn handle(&self) -> sys::CUstream {
        self.stream as sys::CUstream
    }

    fn finish(&self, what: &str) -> Result<(), Error> {
        // SAFETY: the stream is live until drop.
        unsafe { check(what, sys::cuStreamSynchronize(self.handle())) }
    }
}

impl Drop for CopyStream {
    fn drop(&mut self) {
        // SAFETY: created in `new`; any copy on it was waited for.
        unsafe {
            let _ = sys::cuStreamDestroy_v2(self.handle());
        }
    }
}

/// Copy the first `target.len()` values of a device tensor of `f32` into `target`,
/// which should be locked memory. Checks that the tensor lives on a CUDA device and
/// holds at least that many values, so the copy can't read or write out of bounds.
pub fn copy_to_host(
    stream: &CopyStream,
    tensor: Value<ort::value::DynTensorValueType>,
    target: &mut [f32],
) -> Result<(), Error> {
    let tensor: DynTensor = tensor;
    let info = tensor.memory_info();
    if info.allocation_device() != AllocationDevice::CUDA {
        return Err(Error(format!(
            "tensor is on {:?}, not a CUDA device",
            info.allocation_device()
        )));
    }
    if *tensor.data_type() != ort::value::TensorElementType::Float32 {
        return Err(Error(format!(
            "tensor holds {}, not f32",
            tensor.data_type()
        )));
    }
    let count = tensor.shape().num_elements();
    if count < target.len() {
        return Err(Error(format!(
            "tensor holds {count} values, target wants {}",
            target.len()
        )));
    }
    use_device(stream.device)?;
    // SAFETY: the source is a live CUDA allocation of `count` f32 values, checked
    // above, and the target is a mutable slice of at most that many. The session's
    // run has finished and synchronized, so the device data is complete.
    unsafe {
        check(
            "copy outputs to host",
            sys::cuMemcpyDtoHAsync_v2(
                target.as_mut_ptr().cast(),
                tensor.data_ptr() as sys::CUdeviceptr,
                size_of_val(target),
                stream.handle(),
            ),
        )?;
    }
    // The target is borrowed only for this call, so wait for the copy here.
    stream.finish("finish copy to host")
}

/// Copy `source` into the start of a device tensor of the same element type,
/// which must hold at least `source.len()` values. `source` should be locked memory.
pub fn copy_to_device<T: ort::value::PrimitiveTensorElementType + Copy>(
    stream: &CopyStream,
    source: &[T],
    tensor: &mut DynTensor,
) -> Result<(), Error> {
    let info = tensor.memory_info();
    if info.allocation_device() != AllocationDevice::CUDA {
        return Err(Error(format!(
            "tensor is on {:?}, not a CUDA device",
            info.allocation_device()
        )));
    }
    if *tensor.data_type() != T::into_tensor_element_type() {
        return Err(Error(format!(
            "tensor holds {}, source {}",
            tensor.data_type(),
            T::into_tensor_element_type()
        )));
    }
    let count = tensor.shape().num_elements();
    if source.len() > count {
        return Err(Error(format!(
            "source has {} values, tensor room for {count}",
            source.len()
        )));
    }
    use_device(stream.device)?;
    // SAFETY: the target is a live CUDA allocation of `count` values of `T`, checked
    // above, and the copy writes `source.len() <= count` of them. The copy is
    // synchronous, so it is complete before the next run reads the tensor.
    unsafe {
        check(
            "copy inputs to device",
            sys::cuMemcpyHtoDAsync_v2(
                tensor.data_ptr_mut() as sys::CUdeviceptr,
                source.as_ptr().cast(),
                size_of_val(source),
                stream.handle(),
            ),
        )?;
    }
    // The source is borrowed only for this call, so wait for the copy here.
    stream.finish("finish copy to device")
}
