//! The bridge from this crate's [`Generator`] contract to llama.cpp.
//!
//! This is the whole integration. It lives here rather than in `moearc-llama` because
//! `Generator` is defined here: implementing it there would make the engine seam depend on
//! the server, inverting the dependency the rest of this crate is careful to avoid.
//!
//! Behind the `engine` feature, because `moearc-llama/runtime` needs a built llama.cpp to
//! link against. Without the feature this crate still builds, tests and serves anywhere —
//! which is the property that let the whole serving path be written and proven before a
//! single token existed.
//!
//! # Why a dedicated thread rather than a `Mutex<Context>`
//!
//! [`Generator::generate`] takes `&self` and one `EngineGenerator` is shared across every
//! in-flight request, so the mutable `llama_context` has to be reached through *something*.
//! A `Mutex` would be fewer lines. It is not what this does, for three reasons:
//!
//! * 🔴 **A `llama_context` is not `Send` and nobody has established that it is.** Under a
//!   mutex, consecutive decodes of the same sequence would run on whichever of tokio's
//!   blocking threads happened to win the lock. llama.cpp's own server keeps a context on
//!   one thread; ggml's CPU threadpool and the SYCL queues are created there and live for
//!   the context's lifetime. Asserting `unsafe impl Send` to save this file would be a
//!   claim about somebody else's threading model made without evidence, and the failure it
//!   buys is the kind that shows up as wrong tokens rather than as a crash.
//! * **`Context<'m>` borrows its `Model`.** Held together in a struct that is a
//!   self-referential type, which is normally answered with `Box::leak` or a transmute.
//!   Owning both on a thread's own stack answers it with neither: the model outlives the
//!   context because it is declared before it.
//! * It is the shape the retired engine used, so the serving contract below maps onto it
//!   clause for clause instead of being re-derived.
//!
//! The cost is one `Vec<f32>` of logits per token crossing a channel. That is the same
//! trade the retired `Session` made, and it is bounded: 201k f32 is 800 KB, against a
//! decode step that moves gigabytes.
//!
//! # The device
//!
//! ⚠️ Device selection is `main_gpu` — an *index* into what llama.cpp enumerated, and on a
//! box with an integrated GPU index 0 is not reliably the discrete card. Pin it with
//! `ONEAPI_DEVICE_SELECTOR=level_zero:0` before starting this process. `moearc serve` does
//! this by name through `moearc-device` and confirms it against the child's own startup
//! log; this binary is the plumbing-level entry point and does not.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{Context as _, anyhow, bail};
use moearc_llama::{Context, ContextParams, Model, ModelParams};

use crate::generate::{GenerationStats, Generator, SharedGenerator, drive};
use crate::sampling::{Rng, SamplingParams};

// ---------------------------------------------------------------------------------------
// The engine thread's protocol
// ---------------------------------------------------------------------------------------

enum Request {
    /// Drop the KV cache. Every generation starts with one.
    Reset,
    /// Feed these tokens and return the logits that follow the last of them.
    Decode(Vec<i32>),
    Shutdown,
}

enum Reply {
    Done,
    Logits(Vec<f32>),
    /// llama.cpp said no. Carried as a string because its error type does not cross the
    /// channel and the caller only ever renders it.
    Failed(String),
}

/// What the thread reports once the model is up, or why it is not.
struct Ready {
    vocab_size: usize,
    n_ctx: usize,
}

/// The request-side end of the engine thread. Held behind a mutex so that two concurrent
/// requests cannot interleave their commands on one sequence.
struct Link {
    tx: Sender<Request>,
    rx: Receiver<Reply>,
}

impl Link {
    fn call(&self, req: Request) -> anyhow::Result<Reply> {
        self.tx.send(req).map_err(|_| anyhow!("the engine thread is gone"))?;
        self.rx.recv().map_err(|_| anyhow!("the engine thread stopped without replying"))
    }

    fn reset(&self) -> anyhow::Result<()> {
        match self.call(Request::Reset)? {
            Reply::Done => Ok(()),
            Reply::Failed(e) => bail!("resetting the context: {e}"),
            Reply::Logits(_) => bail!("engine protocol: logits in answer to a reset"),
        }
    }

    fn decode(&self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let tokens: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        match self.call(Request::Decode(tokens))? {
            Reply::Logits(l) => Ok(l),
            Reply::Failed(e) => bail!("decoding: {e}"),
            Reply::Done => bail!("engine protocol: no logits in answer to a decode"),
        }
    }
}

// ---------------------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------------------

/// A [`Generator`] backed by real model weights on a real device, via llama.cpp.
pub struct EngineGenerator {
    link: Mutex<Link>,
    vocab_size: usize,
    /// The context length llama.cpp actually gave us, which is not always what was asked
    /// for — it clamps to the model's trained context.
    n_ctx: usize,
    thread: Option<JoinHandle<()>>,
}

impl EngineGenerator {
    /// Load a model and open a context with MoEArc's defaults.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::load_with(path, ModelParams::default(), ContextParams::default())
    }

    /// Load with an explicit plan.
    ///
    /// The tuning layer produces exactly these two structs, so a plan that `moearc info`
    /// printed can be handed here unchanged rather than re-derived from flags.
    pub fn load_with(
        path: &Path,
        model_params: ModelParams,
        ctx_params: ContextParams,
    ) -> anyhow::Result<Self> {
        let path: PathBuf = path.to_path_buf();
        let (req_tx, req_rx) = channel::<Request>();
        let (rep_tx, rep_rx) = channel::<Reply>();
        let (ready_tx, ready_rx) = channel::<Result<Ready, String>>();

        let display = path.display().to_string();
        let thread = std::thread::Builder::new()
            .name("moearc-engine".into())
            .spawn(move || {
                engine_thread(&path, &model_params, &ctx_params, &ready_tx, &req_rx, &rep_tx)
            })
            .with_context(|| "spawning the engine thread")?;

        // The thread owns the model, so a load failure has to come back over this channel.
        // A `RecvError` here means it panicked before reporting, which `join` can explain
        // and a bare "channel closed" cannot.
        let ready = match ready_rx.recv() {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                let _ = thread.join();
                bail!("loading {display}: {e}");
            }
            Err(_) => {
                let _ = thread.join();
                bail!("the engine thread died while loading {display}");
            }
        };

        Ok(Self {
            link: Mutex::new(Link { tx: req_tx, rx: rep_rx }),
            vocab_size: ready.vocab_size,
            n_ctx: ready.n_ctx,
            thread: Some(thread),
        })
    }

    /// Load and wrap for [`crate::state::ServerState`].
    pub fn shared(path: &Path) -> anyhow::Result<SharedGenerator> {
        Ok(Arc::new(Self::load(path)?))
    }
}

impl Generator for EngineGenerator {
    fn generate(
        &self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<GenerationStats> {
        if prompt_tokens.len() >= self.n_ctx {
            bail!(
                "prompt is {} tokens and the context holds {} — there is no room to generate. \
                 Reload with a larger --ctx, or send less.",
                prompt_tokens.len(),
                self.n_ctx
            );
        }

        // Seeding matches EchoGenerator exactly: no seed means "vary between requests", and
        // a seed makes the whole completion reproducible. Diverging here would make the stub
        // and the real engine behave differently for the same request, which is precisely the
        // difference a stub exists to avoid.
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        });
        let mut rng = Rng::seed_from_u64(seed);

        // One request at a time. The lock is held across the whole generation on purpose:
        // there is one KV cache, and interleaving two sequences into it is not a slowdown,
        // it is wrong output.
        let link = self.link.lock().map_err(|_| anyhow!("the engine thread panicked"))?;
        link.reset()?;

        // The sampler stays this crate's, and the loop is `generate::drive` — the same one
        // the contract is written against — so there is one sampler and one token loop in
        // the system rather than two that drift.
        drive(prompt_tokens, params, &mut rng, &mut |tokens| link.decode(tokens), on_token)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    // Deliberately left at the default `false`: this one is not a stub, and `/health` and the
    // startup banner both read it.
}

impl Drop for EngineGenerator {
    /// Ask the thread to stop and wait for it, so the model is unloaded and the device
    /// memory released before this returns. Without the join, dropping a generator would
    /// leave a context — and on a large model, tens of gigabytes of VRAM — alive for an
    /// unbounded time.
    fn drop(&mut self) {
        if let Ok(link) = self.link.lock() {
            let _ = link.tx.send(Request::Shutdown);
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ---------------------------------------------------------------------------------------
// The engine thread
// ---------------------------------------------------------------------------------------

/// Owns the model and the context for their whole lives, and never lets either escape.
fn engine_thread(
    path: &Path,
    model_params: &ModelParams,
    ctx_params: &ContextParams,
    ready: &Sender<Result<Ready, String>>,
    rx: &Receiver<Request>,
    tx: &Sender<Reply>,
) {
    let model = match Model::load(path, model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let mut ctx = match Context::new(&model, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };

    let vocab_size = usize::try_from(model.vocab_len()).unwrap_or(0);
    if vocab_size == 0 {
        let _ = ready.send(Err("the model reports a zero-length vocabulary".into()));
        return;
    }
    let n_ctx = ctx.n_ctx() as usize;
    if ready.send(Ok(Ready { vocab_size, n_ctx })).is_err() {
        return;
    }

    // `n_batch` is the largest batch `llama_decode` accepts; a longer prompt has to be fed
    // in pieces. Only the last piece's logits are ever read, and those are the ones asked
    // for after the loop.
    let chunk = ctx_params.n_batch.max(1) as usize;

    while let Ok(req) = rx.recv() {
        let reply = match req {
            Request::Shutdown => break,
            Request::Reset => {
                ctx.reset();
                Reply::Done
            }
            Request::Decode(tokens) => decode(&mut ctx, &tokens, chunk),
        };
        if tx.send(reply).is_err() {
            break;
        }
    }
}

fn decode(ctx: &mut Context<'_>, tokens: &[i32], chunk: usize) -> Reply {
    for part in tokens.chunks(chunk) {
        if let Err(e) = ctx.decode(part) {
            return Reply::Failed(e.to_string());
        }
    }
    match ctx.logits(-1) {
        Some(l) => Reply::Logits(l.to_vec()),
        None => Reply::Failed("llama.cpp computed no logits for the last token".into()),
    }
}
