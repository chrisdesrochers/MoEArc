//! MoEArc's serving layer: an OpenAI-compatible HTTP API, tokenisation, chat templating and
//! sampling.
//!
//! `docs/ux.md` puts it plainly — *"serving is the boring part"*. Boring is the specification:
//! every client written against `api.openai.com` has to work here unchanged, which is why this
//! crate matches that schema rather than inventing a better one.
//!
//! # Shape
//!
//! ```text
//!   HTTP request
//!        │
//!        ▼
//!   openai.rs   parse + clamp; unknown fields ignored, unsupported ones refused by name
//!        │
//!        ▼
//!   chat.rs     render the MODEL'S OWN Jinja template (minijinja)      ─┐
//!        │                                                              │ no engine
//!        ▼                                                              │ dependency
//!   tokenize.rs encode with the model's tokeniser (HF `tokenizers`)     │ anywhere in
//!        │                                                              │ this crate
//!        ▼                                                              │
//!   generate.rs ── trait Generator ──►  EchoGenerator today            ─┘
//!        │                              moearc-engine tomorrow
//!        ▼
//!   sampling.rs temperature / top-k / top-p / repetition penalty / seed
//!        │
//!        ▼
//!   routes.rs   incremental detokenisation, stop strings, SSE framing
//! ```
//!
//! # 🔴 Where the engine plugs in
//!
//! This crate does **not** depend on `moearc-engine`, `moearc-kernels` or `moearc-model`, and
//! must not start to. The entire coupling is [`generate::Generator`], a synchronous trait over
//! `&[u32] -> tokens`. Everything else here — routing, templating, sampling, SSE, stop
//! sequences, usage accounting — is written and tested against
//! [`generate::EchoGenerator`], a stub that echoes the prompt at `temperature = 0`.
//!
//! Swapping in the real engine is one expression, in `src/bin/moearc-server.rs`:
//!
//! ```ignore
//! // today
//! let generator: SharedGenerator = Arc::new(EchoGenerator::new(tokenizer.vocab_size()));
//! // tomorrow
//! let generator: SharedGenerator = Arc::new(moearc_engine::Session::load(&model_path)?);
//! ```
//!
//! plus one `impl Generator` on the engine side and one line in this crate's `Cargo.toml`.
//! No handler, template, encoder or test names a concrete generator type. The contract the
//! implementation must honour — blocking, ordered, cancellable — is documented on the trait.
//!
//! # Security
//!
//! The server binds `127.0.0.1` by default and there is no flag that silently widens that: an
//! inference server is an unauthenticated arbitrary-compute endpoint, and this fleet is on a
//! real network.

pub mod chat;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod openai;
pub mod routes;
pub mod sampling;
pub mod state;
pub mod testing;
pub mod tokenize;

pub use generate::{EchoGenerator, GenerationStats, Generator, SharedGenerator, StopReason};
pub use routes::{bind, router, serve};
pub use sampling::SamplingParams;
pub use state::{AppState, ServerState};
