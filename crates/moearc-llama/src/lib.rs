//! `moearc-llama` — MoEArc's binding to llama.cpp.
//!
//! MoEArc's architecture is:
//!
//! ```text
//! Rust             installer · hardware detection · model catalog · TUI · server · tuning
//! llama.cpp+SYCL   the engine
//! ```
//!
//! This crate is the seam. Nothing above it links llama.cpp directly, and nothing
//! below it knows about MoEArc.
//!
//! # Layout
//!
//! - [`params`] — the tuning vocabulary. **Pure Rust, always compiled.** A plan can
//!   be built, tested and serialised on a laptop with no GPU and no llama.cpp.
//! - `runtime` — the FFI. Behind the **`runtime`** feature, because requiring a
//!   built llama.cpp to run `cargo test` would make the workspace untestable
//!   anywhere but the reference machine.
//!
//! # Why not `llama-cpp-2`
//!
//! The obvious candidate is `utilityai/llama-cpp-rs`. It was evaluated and
//! rejected on two independent grounds, either of which is disqualifying:
//!
//! 1. **It has no SYCL backend.** Its `llama-cpp-sys-2` feature list is
//!    `common, cuda, cuda-no-vmm, metal, dynamic-link, vulkan, opencl, mkl, openmp,
//!    static-openmp, rocm, static-stdcxx, shared-stdcxx, system-ggml,
//!    system-ggml-static, mtmd, dynamic-backends`. There is no `sycl`, `oneapi` or
//!    `intel` feature; `mkl` is a host BLAS library, not GPU offload. Its
//!    `build.rs` has no branch that sets `GGML_SYCL=ON` and never arranges the
//!    `icpx`/`setvars.sh` toolchain that llama.cpp's SYCL CMake requires. Its issue
//!    tracker has zero hits for "SYCL" and zero for "oneAPI". Intel Arc is
//!    mentioned once, about the *Vulkan* backend — and the Vulkan build on this
//!    box is measured at 4.8x slower than SYCL.
//!
//! 2. **It cannot be pointed at our llama.cpp.** It builds a vendored git submodule
//!    through the `cmake` crate, from `$CARGO_MANIFEST_DIR/llama.cpp`, with no
//!    `LLAMA_CPP_LIB`-style escape hatch. Its submodule currently pins llama.cpp at
//!    `e79e4bf66` (2026-08-13); MoEArc's reference build pins `e107984bc`. MoEArc's
//!    benchmark protocol requires the engine and the baseline to be the identical
//!    commit — a project that has already retracted results for measuring the wrong
//!    thing cannot adopt a dependency that guarantees two llama.cpps in one tree.
//!
//! Adopting it would mean forking it to add a SYCL feature *and* forking it again
//! to unpin the submodule. What remains after that is a build script, which is the
//! part this crate replaces in ~180 lines.
//!
//! Its API design was nonetheless informative, and it is worth recording that it
//! solves the `-ncmoe` problem the same way: `add_cpu_moe_override` /
//! `add_cpu_buft_override`, not a scalar field. See [`params::ModelParams::n_cpu_moe`].
//!
//! # Why not drive `llama-server` as a subprocess
//!
//! A legitimate option, and the one to fall back to if the FFI proves brittle. It
//! was not chosen because three things MoEArc needs are awkward or impossible
//! across a process boundary: reading `llama_perf_context` for per-run timings
//! without parsing stderr; changing thread counts on a live context
//! ([`runtime::Context::set_n_threads`]) during a tuning sweep; and giving the TUI
//! model-load progress and a real error string instead of an exit code. It also
//! reintroduces a process to supervise, a port to allocate, and a startup race to
//! poll — for a product whose pitch is "paste a URL and it works".
//!
//! The subprocess path stays available and is not mutually exclusive: `packaging/`
//! could ship `llama-server` alongside, and the OpenAI-compatible server could
//! proxy rather than embed. That decision is separable from this one.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod params;

#[cfg(feature = "runtime")]
mod ffi;

#[cfg(feature = "runtime")]
pub mod runtime;

#[cfg(feature = "runtime")]
pub use runtime::{
    Context, Error, Generation, LlamaDefaults, Model, Perf, Sampler, generate, init,
    llama_defaults, system_info,
};

pub use params::{ContextParams, FlashAttn, KvType, LoadMode, ModelParams, SplitMode};
