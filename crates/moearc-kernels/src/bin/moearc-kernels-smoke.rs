//! A binary whose only job is to be started.
//!
//! It exists because of a gap `docs/packaging.md` records: `libmoearc_kernels.so` is linked
//! dynamically, and for a long time nothing in the suite ever *ran* a program that had it in
//! `DT_NEEDED`. This crate's own tests did — and they are also the one target class that
//! inherits the build script's rpath, so they could not see the problem. 309 tests passed
//! while `moearc-server` died in the dynamic loader before reaching `main`.
//!
//! So: call one real symbol through the C ABI, print a marker, exit 0. Calling something is
//! load-bearing — the linker drops an unused `DT_NEEDED` under `--as-needed`, and a binary
//! that does not use the library is not evidence about a binary that does.
//!
//! **A GPU is not required.** `moearc_ctx_create` returns null rather than throwing when there
//! is no device, so this reports the absence and still exits 0. The question it answers is
//! "did the process start", not "is there hardware"; `tests/gpu.rs` is where hardware is
//! asserted.

fn main() {
    let device = match moearc_kernels::Context::new() {
        Ok(ctx) => ctx.device_name().unwrap_or_else(|e| format!("<unnamed: {e}>")),
        Err(e) => format!("<none: {e}>"),
    };
    println!("moearc-kernels-smoke: ok device={device}");
}
