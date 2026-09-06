//! MoEArc engine: the device-independent half of the inference runtime.
//!
//! Everything in this crate is pure Rust with no GPU dependency — memory planning, expert
//! residency, scheduling and routing policy. Device work lives behind the kernel seam in
//! `moearc-kernels`. Keeping that split strict is what makes this half unit-testable on any
//! machine, including CI without an Arc card — and it is why the central claim of the project
//! (that dynamic residency beats a static split) can be tested before a kernel exists.

pub mod cache;
pub mod host_budget;
pub mod kv;
pub mod memory;
pub mod profile;
pub mod residency;
pub mod runtime;

// The forward pass. Behind a feature because it is the one part of this crate that is not
// device-independent: `moearc-kernels` compiles SYCL with Intel's DPC++ at build time, so
// depending on it unconditionally would put an oneAPI toolchain in the way of building the
// scheduler and the memory planner — which are testable on any machine and are meant to stay
// that way. Build with `--features gpu` to get `Session`.
#[cfg(feature = "gpu")]
pub mod host_experts;
#[cfg(feature = "gpu")]
pub mod moe;
#[cfg(feature = "gpu")]
pub mod session;
