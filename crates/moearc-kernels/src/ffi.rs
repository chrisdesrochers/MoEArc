//! Hand-written declarations for the SYCL kernels.
//!
//! Deliberately not bindgen. Generating these would mean parsing `<sycl/sycl.hpp>`, which would
//! require the oneAPI headers in every consumer's build — the exact dependency this crate
//! exists to keep off the user's machine. The C ABI on the other side is small and stable
//! enough to write out, and writing it out is what keeps the seam narrow.

use std::os::raw::{c_char, c_int, c_uint, c_void};

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
}

#[allow(non_camel_case_types)]
pub type c_ulong = std::os::raw::c_ulong;
