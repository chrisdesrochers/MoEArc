//! MoEArc engine: the device-independent half of the inference runtime.
//!
//! Everything in this crate is pure Rust with no GPU dependency — scheduling, memory
//! planning, paging and routing policy. Device work lives behind the kernel seam in
//! `moearc-kernels`. Keeping the split strict is what makes this half unit-testable on any
//! machine, including CI without an Arc card.

pub mod cache_budget;
