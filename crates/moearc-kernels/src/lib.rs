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
//! One kernel has been tuned, because a decode step spent 92% of its time in it: `matvec_q`.
//! Three things were wrong with the first version, and all three were measured rather than
//! guessed — see `Context::matvec_q` and the `unit_acc` / `matvec_q_submit` notes in
//! `kernels.cpp`. The rest of this crate is still written for auditability first: `dequant`,
//! `embed_rows` and the reference twins recompute per-block constants per element, which is
//! redundant work in exchange for the element formula living in exactly one place.
//!
//! What is *not* claimed anywhere here is a throughput number. `examples/launch_overhead.rs`
//! measures the one thing this crate can honestly report on its own — what a submission costs
//! — and the engine's `olmoe_profile` example measures the rest.

mod ffi;
pub mod reference;

use std::ffi::CStr;
use std::os::raw::c_int;

pub use reference::{KvType, QK_K, QK8_0, QuantType, RopeKind};

/// A GPU context: a SYCL queue and the device it targets.
///
/// # Submission, not completion
///
/// 🔴 The kernel methods below **submit work and return**. `Ok(())` means the device accepted
/// the launch, not that it ran. The queue is **in-order**, so a kernel cannot start before
/// everything submitted ahead of it has finished and no caller has to express a dependency —
/// but a host that wants to *read* a result, or to reuse a host buffer a copy is sourcing
/// from, must first reach a synchronisation point. There are two, and they are the same two
/// that report a failed kernel: [`Context::sync`] and [`Context::download`] (with the
/// `_slice` wrapper over it).
///
/// [`Context::upload`] also waits, which is what makes `ctx.upload_slice(&buf, &[x])` on a
/// temporary safe to write. That is a deliberate cost: an upload happens a couple of times per
/// token, and a copy that did not wait would hand every caller a lifetime problem the type
/// system is not expressing.
pub struct Context {
    raw: *mut ffi::MoearcCtx,
    /// Wait after every kernel, so a host-side profile attributes device time to the call that
    /// caused it. Off unless `MOEARC_SYNC_EACH=1`; see [`Context::new`].
    sync_each: bool,
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
    ///
    /// `MOEARC_SYNC_EACH=1` makes every kernel wait for the device before returning. That is
    /// the old, slow behaviour, kept as a measurement tool: with an asynchronous queue a
    /// host-side timer around a launch measures the *submission*, and all the device time
    /// piles up at whichever call happens to synchronise next. Setting this hands the time
    /// back to the call that caused it — at a cost that is the whole reason the flag is a flag.
    pub fn new() -> Result<Self, KernelError> {
        let raw = unsafe { ffi::moearc_ctx_create() };
        if raw.is_null() {
            return Err(KernelError::NoDevice);
        }
        let sync_each = std::env::var("MOEARC_SYNC_EACH").ok().as_deref() == Some("1");
        Ok(Self { raw, sync_each })
    }

    /// Turn a kernel's return code into a `Result`, and wait for it if asked to.
    fn finish(&self, rc: c_int, what: &'static str) -> Result<(), KernelError> {
        check(rc, what)?;
        if self.sync_each { self.sync() } else { Ok(()) }
    }

    /// Block until every kernel submitted so far has finished.
    ///
    /// The kernel wrappers below submit and return; they do not wait. That is what lets a
    /// forward pass hand the device a run of dependent work instead of a launch, a stall, a
    /// launch. The queue is in-order, so ordering is guaranteed without this — what this adds
    /// is *completion*, which the host needs before it reads a result or reuses a host buffer
    /// a copy is still sourcing from.
    ///
    /// 🔴 It is also where a failed kernel is reported. A wrapper returning `Ok` means the
    /// work was accepted, not that it succeeded; the verdict arrives here or at the next
    /// [`Context::download`], which waits for the same reason.
    pub fn sync(&self) -> Result<(), KernelError> {
        check(unsafe { ffi::moearc_sync(self.raw) }, "queue synchronisation")
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
        self.finish(rc, "dequantisation")
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
        self.finish(rc, "quantised matvec")
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
        self.finish(rc, "f32 matvec")
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
        self.finish(rc, "rmsnorm")
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
        self.finish(rc, "silu")
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
        self.finish(rc, "swiglu")
    }

    /// Row-wise softmax, max-subtracted, unmasked and unscaled.
    pub fn softmax(
        &self,
        out: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), KernelError> {
        self.softmax_ext(out, x, None, n_rows, n_cols, 1.0)
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
        self.finish(rc, "rope")
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
        self.finish(rc, "top-k router")
    }

    /// Row-wise softmax of `x * scale + mask` — `ggml_soft_max_ext` with no ALiBi.
    ///
    /// `mask` is additive and holds `f32::NEG_INFINITY` where a key must not be seen; that is
    /// how a causal mask is expressed, and [`reference::causal_mask`] builds one. Passing
    /// `None` and `scale = 1.0` gives the plain row softmax the router uses — see
    /// [`Context::softmax`].
    ///
    /// A fully-masked row comes back as zeros rather than NaN. That matters more than it
    /// sounds: one NaN row would propagate through every later matmul and destroy the whole
    /// batch, not just the row that was empty.
    pub fn softmax_ext(
        &self,
        out: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        mask: Option<&DeviceBuffer<'_>>,
        n_rows: usize,
        n_cols: usize,
        scale: f32,
    ) -> Result<(), KernelError> {
        x.require("softmax input", n_rows * n_cols * 4)?;
        out.require("softmax output", n_rows * n_cols * 4)?;
        if let Some(m) = mask {
            m.require("softmax mask", n_rows * n_cols * 4)?;
        }
        let rc = unsafe {
            ffi::moearc_softmax(
                self.raw,
                out.ptr.cast(),
                x.ptr.cast(),
                mask.map_or(std::ptr::null(), |m| m.ptr.cast()),
                n_rows as ffi::c_ulong,
                n_cols as ffi::c_ulong,
                scale,
            )
        };
        self.finish(rc, "softmax")
    }

    /// `out[i] = a[i] + b[i]` — the residual add.
    pub fn add(
        &self,
        out: &DeviceBuffer<'_>,
        a: &DeviceBuffer<'_>,
        b: &DeviceBuffer<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        a.require("add lhs", n * 4)?;
        b.require("add rhs", n * 4)?;
        out.require("add output", n * 4)?;
        let rc = unsafe {
            ffi::moearc_add(self.raw, out.ptr.cast(), a.ptr.cast(), b.ptr.cast(), n as ffi::c_ulong)
        };
        self.finish(rc, "add")
    }

    /// `out[i] = a[i] * b[i]`.
    pub fn mul(
        &self,
        out: &DeviceBuffer<'_>,
        a: &DeviceBuffer<'_>,
        b: &DeviceBuffer<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        a.require("mul lhs", n * 4)?;
        b.require("mul rhs", n * 4)?;
        out.require("mul output", n * 4)?;
        let rc = unsafe {
            ffi::moearc_mul(self.raw, out.ptr.cast(), a.ptr.cast(), b.ptr.cast(), n as ffi::c_ulong)
        };
        self.finish(rc, "mul")
    }

    /// Fill `n` f32 with zeros.
    ///
    /// The accumulator [`Context::axpy`] folds into has to start empty. Uploading a host vector
    /// of zeros would do it, and did — but a copy synchronises and a kernel does not, so on a
    /// queue everything else runs ahead of, the upload was the only thing stopping.
    pub fn zero(&self, dst: &DeviceBuffer<'_>, n: usize) -> Result<(), KernelError> {
        dst.require("zero-fill target", n * 4)?;
        let rc = unsafe { ffi::moearc_zero(self.raw, dst.ptr.cast(), n as ffi::c_ulong) };
        self.finish(rc, "zero fill")
    }

    /// `out[i] += alpha * x[i]`, in place — the MoE combine.
    ///
    /// Each active expert's output is folded into one running total with its router weight.
    /// Doing that as a scale followed by an add would write a full intermediate vector per
    /// expert, `k` times per token per layer, for no reason.
    pub fn axpy(
        &self,
        out: &DeviceBuffer<'_>,
        x: &DeviceBuffer<'_>,
        alpha: f32,
        n: usize,
    ) -> Result<(), KernelError> {
        x.require("axpy input", n * 4)?;
        out.require("axpy accumulator", n * 4)?;
        let rc = unsafe {
            ffi::moearc_axpy(self.raw, out.ptr.cast(), x.ptr.cast(), alpha, n as ffi::c_ulong)
        };
        self.finish(rc, "axpy")
    }

    /// Convert `n` f32 to f16 on the device, round-to-nearest-even.
    ///
    /// The read side needs no counterpart: f16 is a [`QuantType`], so [`Context::dequant`]
    /// already expands one and [`Context::matvec_q`] already consumes one.
    pub fn quantize_f16(
        &self,
        dst: &DeviceBuffer<'_>,
        src: &DeviceBuffer<'_>,
        n: usize,
    ) -> Result<(), KernelError> {
        src.require("f16 conversion input", n * 4)?;
        dst.require("f16 conversion output", n * 2)?;
        let rc = unsafe {
            ffi::moearc_quantize_f16(self.raw, dst.ptr, src.ptr.cast(), n as ffi::c_ulong)
        };
        self.finish(rc, "f16 conversion")
    }

    /// Gather token rows out of an embedding table and expand them to f32.
    ///
    /// 🔴 `ty` comes from the GGUF tensor header, never from an assumption. The table is Q4_K in
    /// OLMoE and Q8_0 in the Qwen3.6 file; hard-coding either would silently misread the other.
    ///
    /// `token_ids` is a device buffer of `u32`. Out-of-range ids are **not** checked — the
    /// kernel would read outside the table. The caller holds the vocabulary size and is the only
    /// party that can check cheaply.
    pub fn embed_rows(
        &self,
        ty: QuantType,
        out: &DeviceBuffer<'_>,
        table: &DeviceBuffer<'_>,
        token_ids: &DeviceBuffer<'_>,
        n_tokens: usize,
        n_embd: usize,
    ) -> Result<(), KernelError> {
        if n_embd % ty.block_elems() != 0 {
            return Err(KernelError::BadArgument("n_embd is not a whole number of blocks"));
        }
        token_ids.require("token ids", n_tokens * 4)?;
        out.require("embedding output", n_tokens * n_embd * 4)?;
        let rc = unsafe {
            ffi::moearc_embed_rows(
                self.raw,
                ty.type_id(),
                out.ptr.cast(),
                table.ptr,
                token_ids.ptr.cast(),
                n_tokens as ffi::c_ulong,
                n_embd as ffi::c_ulong,
            )
        };
        self.finish(rc, "embedding lookup")
    }

    /// Write one token's K and V into the page slot the cache allocator handed out.
    ///
    /// `page_id` and `slot` are exactly what `moearc_engine::kv::PagedKvCache::append` returns.
    /// The layout written is `[page][slot][kv_head][head_dim]`, which is what
    /// [`Context::attn_decode`] reads and what [`reference::kv_index`] computes.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        &self,
        k_pages: &DeviceBuffer<'_>,
        v_pages: &DeviceBuffer<'_>,
        k: &DeviceBuffer<'_>,
        v: &DeviceBuffer<'_>,
        page_id: u32,
        slot: u32,
        n_kv_heads: usize,
        head_dim: usize,
        page_tokens: usize,
        kv: KvType,
    ) -> Result<(), KernelError> {
        let row = n_kv_heads * head_dim;
        k.require("key vector", row * 4)?;
        v.require("value vector", row * 4)?;
        let need = (page_id as usize * page_tokens + slot as usize + 1) * row * kv.elem_bytes();
        k_pages.require("key pages", need)?;
        v_pages.require("value pages", need)?;
        let rc = unsafe {
            ffi::moearc_kv_append(
                self.raw,
                k_pages.ptr,
                v_pages.ptr,
                k.ptr.cast(),
                v.ptr.cast(),
                page_id,
                slot as std::os::raw::c_uint,
                n_kv_heads as ffi::c_ulong,
                head_dim as ffi::c_ulong,
                page_tokens as ffi::c_ulong,
                kv.type_id(),
            )
        };
        self.finish(rc, "KV append")
    }

    /// Single-query attention over the paged KV cache: `softmax(scale * q.K^T) . V`.
    ///
    /// 🔴 This is the **decode** shape and only that: one query token, whose own K and V have
    /// already been appended. Every cached key is then at or before the query, so causality is
    /// the loop bound `j < n_kv` and no mask tensor is involved. A multi-token prefill genuinely
    /// needs the mask, and [`Context::softmax_ext`] is where that lives — this kernel does not
    /// cover prefill and does not pretend to.
    ///
    /// Keys are reached through `block_table`, which maps a sequence's logical page index to a
    /// physical page. The pages are not contiguous and must not be assumed to be: a sequence
    /// that outlives a neighbour gets whatever the allocator has free.
    ///
    /// GQA is implemented — query head `h` reads KV head `h / (n_heads / n_kv_heads)` — but the
    /// model this targets sets `n_kv_heads == n_heads`, so only synthetic shapes have exercised
    /// the grouped case.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        out: &DeviceBuffer<'_>,
        q: &DeviceBuffer<'_>,
        k_pages: &DeviceBuffer<'_>,
        v_pages: &DeviceBuffer<'_>,
        block_table: &DeviceBuffer<'_>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv: usize,
        page_tokens: usize,
        scale: f32,
        kv: KvType,
    ) -> Result<(), KernelError> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(KernelError::BadArgument("n_heads is not a multiple of n_kv_heads"));
        }
        if page_tokens == 0 {
            return Err(KernelError::BadArgument("page_tokens must be non-zero"));
        }
        q.require("query", n_heads * head_dim * 4)?;
        out.require("attention output", n_heads * head_dim * 4)?;
        block_table.require("block table", n_kv.div_ceil(page_tokens) * 4)?;
        let rc = unsafe {
            ffi::moearc_attn_decode(
                self.raw,
                out.ptr.cast(),
                q.ptr.cast(),
                k_pages.ptr,
                v_pages.ptr,
                block_table.ptr.cast(),
                n_heads as ffi::c_ulong,
                n_kv_heads as ffi::c_ulong,
                head_dim as ffi::c_ulong,
                n_kv as ffi::c_ulong,
                page_tokens as ffi::c_ulong,
                scale,
                kv.type_id(),
            )
        };
        self.finish(rc, "paged attention")
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
