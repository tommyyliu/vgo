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

/// Copy a device tensor of `f32` into `target`, which should be locked memory.
/// Checks that the tensor lives on a CUDA device and holds exactly `target.len()`
/// values, so the copy can't read or write out of bounds.
pub fn copy_to_host(
    device: i32,
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
    if count != target.len() {
        return Err(Error(format!(
            "tensor holds {count} values, target {}",
            target.len()
        )));
    }
    use_device(device)?;
    // SAFETY: the source is a live CUDA allocation of `count` f32 values, checked
    // above, and the target is a mutable slice of the same length. The session's
    // run has finished and synchronized, so the device data is complete.
    unsafe {
        check(
            "copy outputs to host",
            sys::cuMemcpyDtoH_v2(
                target.as_mut_ptr().cast(),
                tensor.data_ptr() as sys::CUdeviceptr,
                size_of_val(target),
            ),
        )
    }
}
