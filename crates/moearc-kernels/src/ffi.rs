//! Hand-written declarations for the SYCL kernels.
//!
//! Deliberately not bindgen. Generating these would mean parsing `<sycl/sycl.hpp>`, which would
//! require the oneAPI headers in every consumer's build — the exact dependency this crate
//! exists to keep off the user's machine. The C ABI on the other side is small and stable
//! enough to write out, and writing it out is what keeps the seam narrow.
//!
//! Every function returns `0` on success, `-1` for a device failure or a null argument, and
//! `-2` for an argument that is out of range (an unsupported quantisation type, a shape that
//! is not a whole number of blocks, a `k` above the router's ceiling).

use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};

#[repr(C)]
pub struct MoearcCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn moearc_ctx_create() -> *mut MoearcCtx;
    pub fn moearc_ctx_destroy(c: *mut MoearcCtx);
    pub fn moearc_device_name(c: *mut MoearcCtx, out: *mut c_char, cap: c_ulong) -> c_int;
    pub fn moearc_alloc_device(c: *mut MoearcCtx, bytes: c_ulong) -> *mut c_void;
    pub fn moearc_free_device(c: *mut MoearcCtx, p: *mut c_void);
    pub fn moearc_copy_h2d(
        c: *mut MoearcCtx,
        dst: *mut c_void,
        src: *const c_void,
        bytes: c_ulong,
    ) -> c_int;
    pub fn moearc_copy_d2h(
        c: *mut MoearcCtx,
        dst: *mut c_void,
        src: *const c_void,
        bytes: c_ulong,
    ) -> c_int;
    pub fn moearc_gather_experts(
        c: *mut MoearcCtx,
        dst: *mut c_void,
        pool: *const c_void,
        idx: *const c_uint,
        count: c_uint,
        slot_bytes: c_ulong,
    ) -> c_int;

    pub fn moearc_dequant(
        c: *mut MoearcCtx,
        type_id: c_uint,
        dst: *mut c_float,
        src: *const c_void,
        nblocks: c_ulong,
    ) -> c_int;
    pub fn moearc_matvec_q(
        c: *mut MoearcCtx,
        type_id: c_uint,
        out: *mut c_float,
        w: *const c_void,
        x: *const c_float,
        n_rows: c_ulong,
        n_cols: c_ulong,
    ) -> c_int;
    pub fn moearc_matvec_f32(
        c: *mut MoearcCtx,
        out: *mut c_float,
        w: *const c_float,
        x: *const c_float,
        n_rows: c_ulong,
        n_cols: c_ulong,
    ) -> c_int;
    pub fn moearc_rmsnorm(
        c: *mut MoearcCtx,
        out: *mut c_float,
        x: *const c_float,
        weight: *const c_float,
        n_rows: c_ulong,
        n_cols: c_ulong,
        eps: c_float,
    ) -> c_int;
    pub fn moearc_silu(
        c: *mut MoearcCtx,
        out: *mut c_float,
        x: *const c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_swiglu(
        c: *mut MoearcCtx,
        out: *mut c_float,
        gate: *const c_float,
        up: *const c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_softmax(
        c: *mut MoearcCtx,
        out: *mut c_float,
        x: *const c_float,
        n_rows: c_ulong,
        n_cols: c_ulong,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn moearc_rope(
        c: *mut MoearcCtx,
        dst: *mut c_float,
        src: *const c_float,
        pos: *const c_int,
        n_tokens: c_ulong,
        n_heads: c_ulong,
        head_dim: c_ulong,
        n_dims: c_ulong,
        freq_base: c_float,
        neox: c_int,
    ) -> c_int;
    pub fn moearc_topk_router(
        c: *mut MoearcCtx,
        idx: *mut c_uint,
        weights: *mut c_float,
        logits: *const c_float,
        n_tokens: c_ulong,
        n_expert: c_ulong,
        k: c_uint,
        normalize: c_int,
    ) -> c_int;
}

#[allow(non_camel_case_types)]
pub type c_ulong = std::os::raw::c_ulong;
