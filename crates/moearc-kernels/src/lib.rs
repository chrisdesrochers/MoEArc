//! The GPU seam: the one place MoEArc talks to a device.
//!
//! Everything above this crate is device-independent Rust. Everything below is SYCL C++ behind
//! a plain C ABI. Keeping that boundary narrow is what lets the engine be tested on any
//! machine, and what lets the kernels be compiled on ours and shipped as an artifact rather
//! than built on the user's.
//!
//! # What is here
//!
//! The kernels a decode step needs, in the order a forward pass uses them: block-quantised
//! weight expansion ([`Context::dequant`]), matrix-vector against those weights without
//! expanding them first ([`Context::matvec_q`]), [`Context::rmsnorm`], [`Context::rope`],
//! [`Context::softmax`], [`Context::swiglu`], and the MoE router ([`Context::topk_router`]).
//!
//! Every one of them has a CPU twin in [`reference`], and every one is asserted against that
//! twin on real hardware in `tests/`. The dequantisers are additionally checked against
//! llama.cpp's own output for real tensors out of a real model — see `tests/gguf_crosscheck.rs`
//! and `tools/ggml_dequant_dump.c`.
//!
//! # Shapes
//!
//! Everything is row-major with the last axis contiguous, which is how GGUF stores weights and
//! how ggml numbers its dimensions (`ne[0]` is the fastest-varying axis). A quantised weight
//! matrix is `n_rows` rows of `n_cols` elements, each row an independent run of
//! `n_cols / ty.block_elems()` blocks — 256 elements per block for the K-quants, 32 for Q8_0.
//!
//! # Performance
//!
//! Not yet a goal. The kernels are written for auditability: the dequantisation formula lives
//! in one function that both the expansion and the matvec call, at the cost of recomputing
//! per-block constants per element. Nothing here has been tuned, and no throughput claim is
//! made anywhere in this crate.

mod ffi;
pub mod reference;

use std::ffi::CStr;
use std::os::raw::c_int;

pub use reference::{QK_K, QK8_0, QuantType, RopeKind};

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
    /// A device buffer is too small for the shape it was asked to hold.
    ///
    /// Checked here rather than in the kernel: a kernel that overruns a USM allocation
    /// corrupts whatever is next in the device heap and reports success.
    BufferTooSmall { what: &'static str, need: usize, have: usize },
    /// The kernel rejected an argument — an unsupported quantisation type, a row length that
    /// is not a whole number of blocks, a `k` above the router's ceiling.
    BadArgument(&'static str),
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
            Self::BufferTooSmall { what, need, have } => write!(
                f,
                "the {what} buffer holds {have} bytes but the requested shape needs {need}"
            ),
            Self::BadArgument(what) => write!(f, "{what} was rejected by the kernel"),
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

    fn require(&self, what: &'static str, need: usize) -> Result<(), KernelError> {
        if self.len < need {
            return Err(KernelError::BufferTooSmall { what, need, have: self.len });
        }
        Ok(())
    }
}

/// Turn a C return code into a `Result`. `-2` is an argument the kernel refused; anything else
/// non-zero is a device failure.
fn check(rc: c_int, what: &'static str) -> Result<(), KernelError> {
    match rc {
        0 => Ok(()),
        -2 => Err(KernelError::BadArgument(what)),
        _ => Err(KernelError::Failed(what)),
    }
}

/// Reinterpret a slice of plain scalars as bytes.
///
/// Sound for `f32`/`u32`/`i32`: all are `Copy`, have no padding, and every bit pattern of the
/// underlying bytes is a valid value of the byte type.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    let n = std::mem::size_of_val(v);
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n) }
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

    /// Allocate room for `n` scalars of type `T`.
    pub fn alloc_n<T: Copy>(&self, n: usize) -> Result<DeviceBuffer<'_>, KernelError> {
        self.alloc(n * std::mem::size_of::<T>())
    }

    /// Copy a slice of scalars to the device.
    pub fn upload_slice<T: Copy>(
        &self,
        dst: &DeviceBuffer<'_>,
        src: &[T],
    ) -> Result<(), KernelError> {
        dst.require("upload destination", std::mem::size_of_val(src))?;
        self.upload(dst, as_bytes(src))
    }

    /// Copy a slice of scalars back from the device.
    pub fn download_slice<T: Copy>(
        &self,
        dst: &mut [T],
        src: &DeviceBuffer<'_>,
    ) -> Result<(), KernelError> {
        src.require("download source", std::mem::size_of_val(dst))?;
        self.download(as_bytes_mut(dst), src)
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
        if slot_bytes % 4 != 0 {
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

    /// Expand `nblocks` blocks into `nblocks * ty.block_elems()` f32 on the device.
    ///
    /// The inverse of what a quantiser did, and the operation everything else is checked
    /// against — [`reference::dequant`] is the CPU twin, and llama.cpp's own `to_float` is the
    /// third opinion the cross-check test consults.
    pub fn dequant(
        &self,
        ty: QuantType,
        dst: &DeviceBuffer<'_>,
        src: &DeviceBuffer<'_>,
        nblocks: usize,
    ) -> Result<(), KernelError> {
        src.require("quantised source", nblocks * ty.block_bytes())?;
        dst.require("dequantised destination", nblocks * ty.block_elems() * 4)?;
        let rc = unsafe {
            ffi::moearc_dequant(
                self.raw,
                ty.type_id(),
                dst.ptr.cast(),
                src.ptr,
                nblocks as ffi::c_ulong,
            )
        };
        check(rc, "dequantisation")
    }

    /// `out[row] = sum_col W[row][col] * x[col]` against block-quantised weights.
    ///
    /// The weights are consumed in place — they are never expanded to f32 anywhere. `n_cols`
    /// must be a whole number of blocks, which is what GGUF guarantees for a quantised tensor.
    ///
    /// 🔴 The reduction is a tree over 32 lanes, so the summation order is not a CPU dot
    /// product's and the results are not bit-identical to one. See the tolerance discussion in
    /// `tests/kernels_gpu.rs`.
    pub fn matvec_q(
        &self,
        ty: QuantType,
        out: &DeviceBuffer<'_>,
        w: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), KernelError> {
        if n_cols % ty.block_elems() != 0 {
            return Err(KernelError::BadArgument("n_cols is not a whole number of blocks"));
        }
        w.require("weight matrix", n_rows * (n_cols / ty.block_elems()) * ty.block_bytes())?;
        x.require("activation vector", n_cols * 4)?;
        out.require("matvec output", n_rows * 4)?;
        let rc = unsafe {
            ffi::moearc_matvec_q(
                self.raw,
                ty.type_id(),
                out.ptr.cast(),
                w.ptr,
                x.ptr.cast(),
                n_rows as ffi::c_ulong,
                n_cols as ffi::c_ulong,
            )
        };
        check(rc, "quantised matvec")
    }

    /// The same product against unquantised f32 weights.
    pub fn matvec_f32(
        &self,
        out: &DeviceBuffer<'_>,
        w: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), KernelError> {
        w.require("weight matrix", n_rows * n_cols * 4)?;
        x.require("activation vector", n_cols * 4)?;
        out.require("matvec output", n_rows * 4)?;
        let rc = unsafe {
            ffi::moearc_matvec_f32(
                self.raw,
                out.ptr.cast(),
                w.ptr.cast(),
                x.ptr.cast(),
                n_rows as ffi::c_ulong,
                n_cols as ffi::c_ulong,
            )
        };
        check(rc, "f32 matvec")
    }

    /// RMSNorm over the last axis, optionally scaled by a per-column weight.
    ///
    /// 🔴 The sum of squares is accumulated in f32. ggml accumulates it in double. The
    /// difference is real and is bounded by the test rather than assumed away; fp64 on Arc is
    /// emulated where it exists at all, so matching ggml here is not an option.
    pub fn rmsnorm(
        &self,
        out: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        weight: Option<&DeviceBuffer<'_>>,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
    ) -> Result<(), KernelError> {
        x.require("rmsnorm input", n_rows * n_cols * 4)?;
        out.require("rmsnorm output", n_rows * n_cols * 4)?;
        if let Some(w) = weight {
            w.require("rmsnorm weight", n_cols * 4)?;
        }
        let wptr = weight.map_or(std::ptr::null(), |w| w.ptr.cast());
        let rc = unsafe {
            ffi::moearc_rmsnorm(
                self.raw,
                out.ptr.cast(),
                x.ptr.cast(),
                wptr,
                n_rows as ffi::c_ulong,
                n_cols as ffi::c_ulong,
                eps,
            )
        };
        check(rc, "rmsnorm")
    }

    /// SiLU: `x / (1 + exp(-x))`, elementwise.
    pub fn silu(
        &self,
        out: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        x.require("silu input", n * 4)?;
        out.require("silu output", n * 4)?;
        let rc =
            unsafe { ffi::moearc_silu(self.raw, out.ptr.cast(), x.ptr.cast(), n as ffi::c_ulong) };
        check(rc, "silu")
    }

    /// SwiGLU: `silu(gate) * up`, elementwise — the gated FFN activation, fused.
    pub fn swiglu(
        &self,
        out: &DeviceBuffer<'_>,
        gate: &DeviceBuffer<'_>,
        up: &DeviceBuffer<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        gate.require("swiglu gate", n * 4)?;
        up.require("swiglu up", n * 4)?;
        out.require("swiglu output", n * 4)?;
        let rc = unsafe {
            ffi::moearc_swiglu(
                self.raw,
                out.ptr.cast(),
                gate.ptr.cast(),
                up.ptr.cast(),
                n as ffi::c_ulong,
            )
        };
        check(rc, "swiglu")
    }

    /// Row-wise softmax, max-subtracted.
    pub fn softmax(
        &self,
        out: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), KernelError> {
        x.require("softmax input", n_rows * n_cols * 4)?;
        out.require("softmax output", n_rows * n_cols * 4)?;
        let rc = unsafe {
            ffi::moearc_softmax(
                self.raw,
                out.ptr.cast(),
                x.ptr.cast(),
                n_rows as ffi::c_ulong,
                n_cols as ffi::c_ulong,
            )
        };
        check(rc, "softmax")
    }

    /// Rotary position embedding over `[n_tokens][n_heads][head_dim]`, head_dim contiguous.
    ///
    /// `n_dims` is how much of each head rotates; channels at or above it are copied through,
    /// which is what ggml does. [`RopeKind`] picks which channels pair up, and the two
    /// conventions are not interchangeable — the wrong one yields a model that is fluent and
    /// wrong rather than one that errors.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        dst: &DeviceBuffer<'_>,
        src: &DeviceBuffer<'_>,
        pos: &DeviceBuffer<'_>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_dims: usize,
        freq_base: f32,
        kind: RopeKind,
    ) -> Result<(), KernelError> {
        let n = n_tokens * n_heads * head_dim;
        src.require("rope input", n * 4)?;
        dst.require("rope output", n * 4)?;
        pos.require("rope positions", n_tokens * 4)?;
        let rc = unsafe {
            ffi::moearc_rope(
                self.raw,
                dst.ptr.cast(),
                src.ptr.cast(),
                pos.ptr.cast(),
                n_tokens as ffi::c_ulong,
                n_heads as ffi::c_ulong,
                head_dim as ffi::c_ulong,
                n_dims as ffi::c_ulong,
                freq_base,
                match kind {
                    RopeKind::Normal => 0,
                    RopeKind::Neox => 1,
                },
            )
        };
        check(rc, "rope")
    }

    /// Top-k expert selection from router logits.
    ///
    /// Writes `n_tokens * k` expert indices and `n_tokens * k` weights. Semantics follow
    /// llama.cpp's `build_moe_ffn` for the softmax-gated case: softmax over all experts, take
    /// the k largest, and with `normalize` divide by their sum. Ties break toward the lower
    /// expert index, so the same logits always name the same experts — which is what makes a
    /// residency cache's hit rate a property of the model rather than of the scheduler.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_router(
        &self,
        idx: &DeviceBuffer<'_>,
        weights: &DeviceBuffer<'_>,
        logits: &DeviceBuffer<'_>,
        n_tokens: usize,
        n_expert: usize,
        k: usize,
        normalize: bool,
    ) -> Result<(), KernelError> {
        logits.require("router logits", n_tokens * n_expert * 4)?;
        idx.require("router indices", n_tokens * k * 4)?;
        weights.require("router weights", n_tokens * k * 4)?;
        let rc = unsafe {
            ffi::moearc_topk_router(
                self.raw,
                idx.ptr.cast(),
                weights.ptr.cast(),
                logits.ptr.cast(),
                n_tokens as ffi::c_ulong,
                n_expert as ffi::c_ulong,
                k as std::os::raw::c_uint,
                i32::from(normalize),
            )
        };
        check(rc, "top-k router")
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
