//! Tuning profiles — the reason to install MoEArc rather than to build llama.cpp yourself.
//!
//! The engine underneath is llama.cpp on SYCL. What this project adds is the answer to
//! *"what should I actually pass it?"*, and the two flags that answer costs the most are:
//!
//! * **`-t`.** `llama-bench` defaults to **4 threads**. On this project's 20-core box that
//!   default cost **2.1×** — 13.6 tok/s against 28.5 on a 59 GiB model. Nobody guesses their
//!   way there, and nothing warns them.
//! * **`-ncmoe`.** It has a **model-specific floor**, below which llama.cpp aborts with
//!   `OUT_OF_DEVICE_MEMORY` after loading tens of gigabytes. It is documented nowhere. Users
//!   find it by crashing, then back off further than they need to and lose throughput to
//!   superstition.
//!
//! # The four pieces
//!
//! | module | job |
//! | --- | --- |
//! | [`schema`] | what a profile is, and the [`Origin`](schema::Origin) every value carries |
//! | [`store`] | finding and validating `bench/tuning-profiles.json`, and coping with its absence |
//! | [`resolve`] | measured → extrapolated → derived, and the invariant that keeps those distinct |
//! | [`compare`] | is a candidate a win, or is it inside the noise? |
//! | [`coverage`] | the measured expert-coverage curve, read only for the model it describes |
//!
//! # The rule
//!
//! 🔴 **A setting with no measurement behind it must be distinguishable from one with, in
//! every place it appears.** This project has reported a guessed constant to its owner as a
//! result; the answer is not diligence, it is that a bare number cannot reach a renderer.
//! [`schema::Setting`] has no `Deref`, no `From<T>` and no constructor that omits the origin.

pub mod compare;
pub mod coverage;
pub mod resolve;
pub mod schema;
pub mod store;
