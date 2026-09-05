//! MoEArc engine: the device-independent half of the inference runtime.
//!
//! Everything in this crate is pure Rust with no GPU dependency — memory planning, expert
//! residency, scheduling and routing policy. Device work lives behind the kernel seam in
//! `moearc-kernels`. Keeping that split strict is what makes this half unit-testable on any
//! machine, including CI without an Arc card — and it is why the central claim of the project
//! (that dynamic residency beats a static split) can be tested before a kernel exists.

pub mod cache;
pub mod kv;
pub mod memory;
pub mod residency;
pub mod runtime;
