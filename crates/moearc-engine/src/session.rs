//! A loaded model you can call from anywhere: `Session`.
//!
//! # Why a thread
//!
//! `moearc_kernels::Context` holds a raw SYCL queue, so it is neither `Send` nor `Sync`, and
//! every `DeviceBuffer` borrows from it. A struct owning both would be self-referential *and*
//! unshareable — and the serving contract this has to satisfy needs the opposite: one value
//! behind an `Arc`, `Send + Sync`, called from whatever blocking thread the runtime picked.
//!
//! So the device lives on one thread of its own, where the borrows are ordinary stack locals
//! and no lifetime has to be laundered. `Session` is the mailbox in front of it. That also makes
//! the single-sequence limitation explicit rather than accidental: requests serialise on one
//! mutex because there is one KV cache, not because of a lock that could be relaxed later.
//!
//! # The serving contract
//!
//! 🔴 `moearc-server`'s `Generator` trait is deliberately **not** implemented here, and the
//! direction of the dependency is the reason: that crate is written to have no dependency on
//! the engine, so implementing its trait would mean this crate depending on the server. What is
//! provided instead is the same shape, so the `impl` is a forwarding block wherever the two are
//! wired together:
//!
//! - [`Session::generate_with`] is **blocking**, takes `&self`, and never assumes a reactor.
//! - `on_token` is called **once per accepted token, in order**, and a `false` return stops
//!   generation promptly and returns the stats so far.
//! - `stop_tokens` and `max_tokens` are enforced here; stop *strings* are the caller's, through
//!   `on_token`, because they are a property of decoded text.
//! - Sampling is a caller-supplied closure, so the server can pass its own `sampling::sample`
//!   and keep one sampler in the system. [`Session::generate`] is the greedy special case.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use moearc_kernels::Context;
use moearc_model::tensors::MappedModel;

use crate::olmoe::{Config, EngineError, Model, Tap};

/// What a loaded model reports about itself.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub config: Config,
    /// The device's reported name, e.g. `Intel(R) Arc(TM) B580 Graphics`.
    pub device: String,
    /// Context length this session was built for.
    pub n_ctx: usize,
    /// Weight bytes copied to the card, summed from the uploads themselves.
    pub bytes_uploaded: u64,
}

/// Why a generation ended. Mirrors `moearc_server::generate::StopReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopReason {
    /// The model emitted an id in `stop_tokens`.
    #[default]
    EndOfTurn,
    /// `max_tokens` was reached.
    Length,
    /// `on_token` returned `false`.
    Cancelled,
}

/// What a completed generation cost. Mirrors `moearc_server::generate::GenerationStats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub stop_reason: StopReason,
}

/// The stop conditions the engine enforces.
#[derive(Debug, Clone, Default)]
pub struct StopConditions {
    pub max_tokens: usize,
    /// Ids that end the turn. They are **not** emitted through `on_token`.
    pub stop_tokens: Vec<u32>,
}

enum Command {
    Decode { token: u32, tap: bool },
    Reset,
    Shutdown,
}

enum Reply {
    Ready(Box<SessionInfo>),
    Logits(Vec<f32>, Option<Tap>),
    Done,
    Failed(String),
}

struct Link {
    tx: mpsc::Sender<Command>,
    rx: mpsc::Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
}

/// The whole thread arrangement above exists to make this true; a change that quietly
/// reintroduces a device handle into `Session` should fail to compile rather than fail to
/// serve.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Session>();
};

/// A model loaded on the device, callable from any thread.
pub struct Session {
    link: Mutex<Link>,
    info: Arc<SessionInfo>,
}

impl Session {
    /// Load a GGUF and upload it to the default GPU.
    ///
    /// The context length defaults to the model's trained maximum. Pass
    /// [`Session::load_with_context`] to use less, which is the only knob that changes how much
    /// device memory the KV cache takes.
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        Self::load_with_context(path, None)
    }

    /// [`Session::load`] with an explicit context length.
    pub fn load_with_context(path: &Path, n_ctx: Option<usize>) -> Result<Self, EngineError> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let (rep_tx, rep_rx) = mpsc::channel::<Reply>();
        let owned: PathBuf = path.to_path_buf();

        let thread = std::thread::Builder::new()
            .name("moearc-device".into())
            .spawn(move || worker(&owned, n_ctx, &cmd_rx, &rep_tx))
            .map_err(|e| {
                EngineError::Unsupported(format!("could not start the device thread: {e}"))
            })?;

        let info = match rep_rx.recv() {
            Ok(Reply::Ready(info)) => *info,
            Ok(Reply::Failed(m)) => return Err(EngineError::Unsupported(m)),
            _ => {
                return Err(EngineError::Unsupported(
                    "the device thread stopped before reporting a model".to_string(),
                ));
            }
        };

        Ok(Self {
            link: Mutex::new(Link { tx: cmd_tx, rx: rep_rx, thread: Some(thread) }),
            info: Arc::new(info),
        })
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn config(&self) -> &Config {
        &self.info.config
    }

    /// Vocabulary size the logits span. `Generator::vocab_size`.
    pub fn vocab_size(&self) -> usize {
        self.info.config.n_vocab
    }

    /// `Generator::name`.
    pub fn name(&self) -> &'static str {
        "moearc"
    }

    /// Run a whole sequence from an empty cache and return the final token's logits.
    ///
    /// This is the shape a comparison against another implementation needs: one prompt in, one
    /// logit vector out, no sampling in the way.
    pub fn logits(&self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let (logits, _) = self.run(tokens, false)?;
        Ok(logits)
    }

    /// [`Session::logits`], also capturing every block's residual stream.
    ///
    /// The [`Tap`] is from the **last** token only; the earlier ones still run, because their
    /// KV entries are what the last token attends to.
    pub fn logits_tapped(&self, tokens: &[u32]) -> Result<(Vec<f32>, Tap), EngineError> {
        let (logits, tap) = self.run(tokens, true)?;
        let tap = tap.ok_or_else(|| {
            EngineError::Unsupported("the device thread returned no tap".to_string())
        })?;
        Ok((logits, tap))
    }

    fn run(&self, tokens: &[u32], tap: bool) -> Result<(Vec<f32>, Option<Tap>), EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::Unsupported("a prompt needs at least one token".to_string()));
        }
        let link = self.link.lock().expect("the device thread panicked");
        send(&link, Command::Reset)?;
        expect_done(&link)?;
        let mut last = None;
        for (i, t) in tokens.iter().enumerate() {
            let want_tap = tap && i + 1 == tokens.len();
            send(&link, Command::Decode { token: *t, tap: want_tap })?;
            last = Some(expect_logits(&link)?);
        }
        Ok(last.expect("the prompt is non-empty"))
    }

    /// Greedy generation: the acceptance case, and the one a comparison against another
    /// implementation can be exact about.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        stop: &StopConditions,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerationStats, EngineError> {
        self.generate_with(prompt_tokens, stop, &mut |logits, _| argmax(logits), on_token)
    }

    /// Generate until a stop condition fires, calling `sample` for each step.
    ///
    /// `sample` receives the logits and the tokens produced so far (prompt included, so a
    /// repetition penalty can see the whole history) and returns the chosen id. That is where
    /// `moearc-server`'s sampler plugs in.
    pub fn generate_with(
        &self,
        prompt_tokens: &[u32],
        stop: &StopConditions,
        sample: &mut dyn FnMut(&[f32], &[u32]) -> u32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerationStats, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Unsupported("a prompt needs at least one token".to_string()));
        }
        let link = self.link.lock().expect("the device thread panicked");
        send(&link, Command::Reset)?;
        expect_done(&link)?;

        let mut history: Vec<u32> = prompt_tokens.to_vec();
        let mut logits = Vec::new();
        for t in prompt_tokens {
            send(&link, Command::Decode { token: *t, tap: false })?;
            logits = expect_logits(&link)?.0;
        }

        let mut stats = GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: 0,
            stop_reason: StopReason::Length,
        };
        if stop.max_tokens == 0 {
            return Ok(stats);
        }

        loop {
            let token = sample(&logits, &history);
            if stop.stop_tokens.contains(&token) {
                stats.stop_reason = StopReason::EndOfTurn;
                return Ok(stats);
            }
            history.push(token);
            stats.completion_tokens += 1;
            if !on_token(token) {
                stats.stop_reason = StopReason::Cancelled;
                return Ok(stats);
            }
            if stats.completion_tokens >= stop.max_tokens {
                stats.stop_reason = StopReason::Length;
                return Ok(stats);
            }
            // Only now is the accepted token fed back in — a token that was never emitted must
            // not reach the KV cache, or a cancelled request would leave the sequence one token
            // ahead of what the client saw.
            send(&link, Command::Decode { token, tap: false })?;
            logits = expect_logits(&link)?.0;
        }
    }
}

/// Greedy choice, first index winning a tie.
///
/// The tie rule is not incidental: `llama_sampler_greedy_apply` in llama.cpp keeps its running
/// best on a strict `>`, so it also keeps the lowest index. A `>=` here would pick the highest
/// and the two would disagree on any exact tie.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

fn send(link: &Link, cmd: Command) -> Result<(), EngineError> {
    link.tx.send(cmd).map_err(|_| EngineError::Unsupported("the device thread is gone".to_string()))
}

fn recv(link: &Link) -> Result<Reply, EngineError> {
    link.rx.recv().map_err(|_| EngineError::Unsupported("the device thread is gone".to_string()))
}

fn expect_done(link: &Link) -> Result<(), EngineError> {
    match recv(link)? {
        Reply::Done => Ok(()),
        Reply::Failed(m) => Err(EngineError::Unsupported(m)),
        _ => Err(EngineError::Unsupported("unexpected reply from the device".to_string())),
    }
}

fn expect_logits(link: &Link) -> Result<(Vec<f32>, Option<Tap>), EngineError> {
    match recv(link)? {
        Reply::Logits(l, t) => Ok((l, t)),
        Reply::Failed(m) => Err(EngineError::Unsupported(m)),
        _ => Err(EngineError::Unsupported("unexpected reply from the device".to_string())),
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Ok(mut link) = self.link.lock() {
            let _ = link.tx.send(Command::Shutdown);
            if let Some(t) = link.thread.take() {
                let _ = t.join();
            }
        }
    }
}

/// The device thread. Everything it owns is a stack local, so the borrows are ordinary.
fn worker(
    path: &Path,
    n_ctx: Option<usize>,
    rx: &mpsc::Receiver<Command>,
    tx: &mpsc::Sender<Reply>,
) {
    let started = || -> Result<(MappedModel, Context), EngineError> {
        Ok((MappedModel::open(path)?, Context::new()?))
    };
    let (mapped, ctx) = match started() {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(Reply::Failed(e.to_string()));
            return;
        }
    };

    let n_ctx = match n_ctx {
        Some(n) => n,
        None => match Config::from_model(&mapped) {
            Ok(c) => c.n_ctx_train,
            Err(e) => {
                let _ = tx.send(Reply::Failed(e.to_string()));
                return;
            }
        },
    };

    let mut model = match Model::new(&ctx, &mapped, n_ctx) {
        Ok(m) => m,
        Err(e) => {
            let _ = tx.send(Reply::Failed(e.to_string()));
            return;
        }
    };

    let info = SessionInfo {
        config: model.cfg().clone(),
        device: ctx.device_name().unwrap_or_else(|_| "unknown device".to_string()),
        n_ctx,
        bytes_uploaded: model.weights.bytes_uploaded,
    };
    if tx.send(Reply::Ready(Box::new(info))).is_err() {
        return;
    }

    while let Ok(cmd) = rx.recv() {
        let reply = match cmd {
            Command::Shutdown => return,
            Command::Reset => match model.reset() {
                Ok(()) => Reply::Done,
                Err(e) => Reply::Failed(e.to_string()),
            },
            Command::Decode { token, tap } => {
                model.state.tap = tap.then(Tap::default);
                match model.decode(token) {
                    // `to_vec` ends the borrow of `model`, which the tap below needs back.
                    Ok(l) => {
                        let logits = l.to_vec();
                        Reply::Logits(logits, model.state.tap.take())
                    }
                    Err(e) => Reply::Failed(e.to_string()),
                }
            }
        };
        if tx.send(reply).is_err() {
            return;
        }
    }
}
