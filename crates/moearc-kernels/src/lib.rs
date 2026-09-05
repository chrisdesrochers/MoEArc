//! The GPU seam: the one place MoEArc talks to a device.
//!
//! Everything above this crate is device-independent Rust. Everything below is SYCL C++ behind
//! a plain C ABI. Keeping that boundary narrow is what lets the engine be tested on any
//! machine, and what lets the kernels be compiled on ours and shipped as an artifact rather
//! than built on the user's.

mod ffi;

use std::ffi::CStr;

/// A GPU context: a SYCL queue and the device it targets.
pub struct Context {
    raw: *mut ffi::MoearcCtx,
}

/// Why a device operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// No usable GPU, or the SYCL runtime could not create a queue.
    NoDevice,
    /// A device allocation returned null.
    OutOfMemory { bytes: usize },
    /// A copy or kernel launch failed.
    Failed(&'static str),
    /// `slot_bytes` must be a multiple of 4: the gather kernel moves 32-bit words.
    Misaligned { slot_bytes: usize },
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(
                f,
                "no usable GPU device — the SYCL runtime could not create a queue. Run \
                 `moearc` with no arguments for a full device report."
            ),
            Self::OutOfMemory { bytes } => {
                write!(f, "device allocation of {bytes} bytes failed")
            }
            Self::Failed(what) => write!(f, "{what} failed on the device"),
            Self::Misaligned { slot_bytes } => write!(
                f,
                "slot size {slot_bytes} is not a multiple of 4; the gather kernel moves 32-bit \
                 words"
            ),
        }
    }
}

impl std::error::Error for KernelError {}

/// An owned device buffer.
pub struct DeviceBuffer<'a> {
    ptr: *mut std::os::raw::c_void,
    len: usize,
    ctx: &'a Context,
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        unsafe { ffi::moearc_free_device(self.ctx.raw, self.ptr) }
    }
}

impl DeviceBuffer<'_> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Context {
    /// Open a queue on the default GPU.
    pub fn new() -> Result<Self, KernelError> {
        let raw = unsafe { ffi::moearc_ctx_create() };
        if raw.is_null() { Err(KernelError::NoDevice) } else { Ok(Self { raw }) }
    }

    /// The device's reported name.
    pub fn device_name(&self) -> Result<String, KernelError> {
        let mut buf = [0i8; 256];
        let rc = unsafe {
            ffi::moearc_device_name(self.raw, buf.as_mut_ptr(), buf.len() as ffi::c_ulong)
        };
        if rc != 0 {
            return Err(KernelError::Failed("querying the device name"));
        }
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(s.to_string_lossy().into_owned())
    }

    /// Allocate `bytes` of device memory.
    ///
    /// 🔴 Success here is **not** evidence the memory exists. Measured on an Arc B580:
    /// `malloc_device` returned valid pointers for 38 GiB on an 11.33 GiB card, because pages
    /// are not committed until written. A cache must therefore compute its budget in advance
    /// rather than allocating until failure — see `docs/calibration.md`.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer<'_>, KernelError> {
        let ptr = unsafe { ffi::moearc_alloc_device(self.raw, bytes as ffi::c_ulong) };
        if ptr.is_null() {
            return Err(KernelError::OutOfMemory { bytes });
        }
        Ok(DeviceBuffer { ptr, len: bytes, ctx: self })
    }

    /// Copy host bytes to a device buffer.
    pub fn upload(&self, dst: &DeviceBuffer<'_>, src: &[u8]) -> Result<(), KernelError> {
        let n = src.len().min(dst.len);
        let rc = unsafe {
            ffi::moearc_copy_h2d(self.raw, dst.ptr, src.as_ptr().cast(), n as ffi::c_ulong)
        };
        if rc == 0 { Ok(()) } else { Err(KernelError::Failed("host-to-device copy")) }
    }

    /// Copy device bytes back to the host.
    pub fn download(&self, dst: &mut [u8], src: &DeviceBuffer<'_>) -> Result<(), KernelError> {
        let n = dst.len().min(src.len);
        let rc = unsafe {
            ffi::moearc_copy_d2h(self.raw, dst.as_mut_ptr().cast(), src.ptr, n as ffi::c_ulong)
        };
        if rc == 0 { Ok(()) } else { Err(KernelError::Failed("device-to-host copy")) }
    }

    /// Gather expert slots from a resident pool into a packed buffer, on the device.
    ///
    /// This is the residency cache's hot path. The router names a handful of experts per block
    /// and their weights must reach the matmul contiguously; doing that as one device-side
    /// gather rather than a copy per expert is the difference between one launch and 320 per
    /// token.
    pub fn gather_experts(
        &self,
        dst: &DeviceBuffer<'_>,
        pool: &DeviceBuffer<'_>,
        indices: &[u32],
        slot_bytes: usize,
    ) -> Result<(), KernelError> {
        if !slot_bytes.is_multiple_of(4) {
            return Err(KernelError::Misaligned { slot_bytes });
        }
        let rc = unsafe {
            ffi::moearc_gather_experts(
                self.raw,
                dst.ptr,
                pool.ptr,
                indices.as_ptr(),
                indices.len() as std::os::raw::c_uint,
                slot_bytes as ffi::c_ulong,
            )
        };
        if rc == 0 { Ok(()) } else { Err(KernelError::Failed("expert gather")) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::moearc_ctx_destroy(self.raw) }
    }
}

// `Context` holds a raw pointer, so it is automatically neither `Send` nor `Sync` — which is
// what we want, since the SYCL queue behind it is not shared across threads by this API. An
// explicit `impl !Sync` would say the same thing and requires a nightly compiler.
