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
    pub fn moearc_profile_events_enabled(c: *mut MoearcCtx) -> c_int;
    pub fn moearc_profile_events_reset(c: *mut MoearcCtx) -> c_int;
    pub fn moearc_profile_events_report(
        c: *mut MoearcCtx,
        out: *mut std::os::raw::c_char,
        cap: c_ulong,
    ) -> c_int;
    pub fn moearc_sync(c: *mut MoearcCtx) -> c_int;
    pub fn moearc_zero(c: *mut MoearcCtx, dst: *mut c_float, n: c_ulong) -> c_int;
    pub fn moearc_device_name(c: *mut MoearcCtx, out: *mut c_char, cap: c_ulong) -> c_int;
    pub fn moearc_alloc_device(c: *mut MoearcCtx, bytes: c_ulong) -> *mut c_void;
    pub fn moearc_free_device(c: *mut MoearcCtx, p: *mut c_void);
    pub fn moearc_copy_h2d(
        c: *mut MoearcCtx,
        dst: *mut c_void,
        src: *const c_void,
        bytes: c_ulong,
    ) -> c_int;
    pub fn moearc_copy_h2d_async(
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
    #[allow(clippy::too_many_arguments)]
    pub fn moearc_matvec_q_batched(
        c: *mut MoearcCtx,
        type_id: c_uint,
        out: *mut c_float,
        w: *const *const c_void,
        n_mat: c_uint,
        x: *const c_float,
        x_stride: c_ulong,
        n_rows: c_ulong,
        n_cols: c_ulong,
    ) -> c_int;
    pub fn moearc_moe_combine(
        c: *mut MoearcCtx,
        out: *mut c_float,
        parts: *const c_float,
        weights: *const c_float,
        n_mat: c_uint,
        n: c_ulong,
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
    pub fn moearc_swiglu_halves(
        c: *mut MoearcCtx,
        out: *mut c_float,
        gu: *const c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_softmax(
        c: *mut MoearcCtx,
        out: *mut c_float,
        x: *const c_float,
        mask: *const c_float,
        n_rows: c_ulong,
        n_cols: c_ulong,
        scale: c_float,
    ) -> c_int;
    pub fn moearc_add(
        c: *mut MoearcCtx,
        out: *mut c_float,
        a: *const c_float,
        b: *const c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_mul(
        c: *mut MoearcCtx,
        out: *mut c_float,
        a: *const c_float,
        b: *const c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_axpy(
        c: *mut MoearcCtx,
        out: *mut c_float,
        x: *const c_float,
        alpha: c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_quantize_f16(
        c: *mut MoearcCtx,
        dst: *mut c_void,
        src: *const c_float,
        n: c_ulong,
    ) -> c_int;
    pub fn moearc_embed_rows(
        c: *mut MoearcCtx,
        type_id: c_uint,
        out: *mut c_float,
        table: *const c_void,
        token_ids: *const c_uint,
        n_tokens: c_ulong,
        n_embd: c_ulong,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn moearc_kv_append(
        c: *mut MoearcCtx,
        k_pages: *mut c_void,
        v_pages: *mut c_void,
        k: *const c_float,
        v: *const c_float,
        page_id: c_uint,
        slot: c_uint,
        n_kv_heads: c_ulong,
        head_dim: c_ulong,
        page_tokens: c_ulong,
        kv_type: c_uint,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn moearc_attn_decode(
        c: *mut MoearcCtx,
        out: *mut c_float,
        q: *const c_float,
        k_pages: *const c_void,
        v_pages: *const c_void,
        block_table: *const c_uint,
        n_heads: c_ulong,
        n_kv_heads: c_ulong,
        head_dim: c_ulong,
        n_kv: c_ulong,
        page_tokens: c_ulong,
        scale: c_float,
        kv_type: c_uint,
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
