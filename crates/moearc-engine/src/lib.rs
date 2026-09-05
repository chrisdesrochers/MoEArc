//! MoEArc engine: the device-independent half of the inference runtime.
//!
//! Everything in this crate is pure Rust with no GPU dependency — memory planning, scheduling
//! and routing policy. Device work lives behind the kernel seam in `moearc-kernels`. Keeping
//! that split strict is what makes this half unit-testable on any machine, including CI
//! without an Arc card.

pub mod memory;
