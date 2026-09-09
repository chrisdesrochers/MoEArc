//! Safe wrappers over the shim.
//!
//! Available under the `runtime` feature, which requires a built llama.cpp.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Once;

use crate::ffi;
use crate::params::{ContextParams, ModelParams};

/// An error from llama.cpp, carrying whatever it wrote to its log.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load model {path}: {reason}")]
    Load { path: String, reason: String },

    #[error("failed to create context: {reason}")]
    Context { reason: String },

    #[error("llama_decode returned {code} ({meaning})")]
    Decode { code: i32, meaning: &'static str },

    #[error("tokenization failed: buffer of {capacity} tokens was too small, need {needed}")]
    Tokenize { capacity: i32, needed: i32 },

    #[error("path contains an interior NUL byte and cannot cross the C boundary")]
    BadPath,
}

/// `llama_decode`'s documented return codes. Kept as text so a failure explains
/// itself instead of surfacing a bare integer.
fn decode_meaning(code: i32) -> &'static str {
    match code {
        1 => "no KV slot for the batch — reduce batch size or raise n_ctx",
        2 => "aborted; processed ubatches remain in the context's memory",
        -1 => "invalid input batch",
        c if c < -1 => "fatal error; processed ubatches remain in the context's memory",
        _ => "unknown",
    }
}

fn last_error() -> String {
    let mut buf = vec![0u8; 8192];
    // SAFETY: buf is a valid writable allocation of the length passed.
    let n = unsafe { ffi::mla_last_error(buf.as_mut_ptr().cast(), buf.len()) };
    if n <= 0 {
        return "llama.cpp reported no diagnostic".to_string();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim_end().to_string()
}

static INIT: Once = Once::new();

/// Initialise ggml's backend registry. Idempotent; every entry point calls it.
///
/// `verbose` mirrors llama.cpp's own stderr logging. It is off by default because
/// a load prints roughly a hundred lines, but the log is always *captured* either
/// way so that a failure can report its cause.
pub fn init(verbose: bool) {
    INIT.call_once(|| {
        // GGML_LOG_LEVEL_INFO == 2 (ggml.h). Only consulted when verbose.
        // SAFETY: called exactly once, before any other llama.cpp entry point.
        unsafe { ffi::mla_backend_init(i32::from(verbose), 2) };
    });
}

/// llama.cpp's own description of the CPU features and backends it compiled with.
#[must_use]
pub fn system_info() -> String {
    init(false);
    // SAFETY: returns a static NUL-terminated string owned by llama.cpp.
    unsafe { CStr::from_ptr(ffi::mla_system_info()) }.to_string_lossy().into_owned()
}

/// Release ggml's backend registry.
///
/// Optional: the process exiting does the same thing. Exposed for a host that
/// embeds MoEArc and wants a clean teardown. Calling it invalidates every
/// [`Model`] and [`Context`], so it must come last.
///
/// # Safety
/// No `Model`, `Context` or `Sampler` may be alive when this is called.
pub unsafe fn shutdown() {
    unsafe { ffi::mla_backend_free() };
}

/// llama.cpp's own default parameter values, read from the library rather than
/// copied into Rust.
///
/// This exists for two reasons. The tuning layer can show what a plan is actually
/// overriding, and — more usefully — the values can be asserted against, so that
/// a llama.cpp upgrade which silently changes a default is caught by a test
/// instead of by a confusing benchmark result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaDefaults {
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub flash_attn_type: i32,
    pub type_k: i32,
    pub type_v: i32,
    pub offload_kqv: bool,
}

/// Read llama.cpp's compiled-in defaults.
#[must_use]
pub fn llama_defaults() -> LlamaDefaults {
    init(false);
    let mut m = ffi::MlaModelParams::default();
    let mut c = ffi::MlaCtxParams::default();
    // SAFETY: both out-params are valid, correctly typed and exclusively borrowed.
    unsafe {
        ffi::mla_model_default_params(&mut m);
        ffi::mla_ctx_default_params(&mut c);
    }
    LlamaDefaults {
        n_gpu_layers: m.n_gpu_layers,
        main_gpu: m.main_gpu,
        n_ctx: c.n_ctx,
        n_batch: c.n_batch,
        n_ubatch: c.n_ubatch,
        n_threads: c.n_threads,
        n_threads_batch: c.n_threads_batch,
        flash_attn_type: c.flash_attn_type,
        type_k: c.type_k,
        type_v: c.type_v,
        offload_kqv: c.offload_kqv != 0,
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A loaded model. Weights are shared; contexts borrow from it.
pub struct Model {
    raw: NonNull<ffi::LlamaModel>,
}

// SAFETY: llama_model is immutable after load and llama.cpp supports using one
// model from several threads (that is how llama-server shares weights across
// slots). The only mutating operation is Drop, which takes ownership.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Load a GGUF from disk.
    pub fn load(path: impl AsRef<Path>, p: &ModelParams) -> Result<Self, Error> {
        init(false);
        let path = path.as_ref();
        let cpath =
            CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| Error::BadPath)?;

        let raw_params = ffi::MlaModelParams {
            n_gpu_layers: p.n_gpu_layers,
            main_gpu: p.main_gpu,
            n_cpu_moe: p.n_cpu_moe,
            split_mode: p.split_mode as i32,
            load_mode: p.load_mode as i32,
            use_extra_bufts: u8::from(p.use_extra_bufts),
            no_host: u8::from(p.no_host),
            check_tensors: u8::from(p.check_tensors),
            vocab_only: u8::from(p.vocab_only),
        };

        // SAFETY: both pointers are valid for the duration of the call; the shim
        // does not retain either.
        unsafe { ffi::mla_clear_error() };
        let raw = unsafe { ffi::mla_model_load(cpath.as_ptr(), &raw_params) };

        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| Error::Load { path: path.display().to_string(), reason: last_error() })
    }

    /// Number of blocks. ⚠️ On a hybrid model this counts *all* blocks, including
    /// recurrent ones that hold no KV cache; do not use it as a KV multiplier.
    #[must_use]
    pub fn n_layer(&self) -> i32 {
        unsafe { ffi::mla_model_n_layer(self.raw.as_ptr()) }
    }

    #[must_use]
    pub fn n_embd(&self) -> i32 {
        unsafe { ffi::mla_model_n_embd(self.raw.as_ptr()) }
    }

    #[must_use]
    pub fn n_head_kv(&self) -> i32 {
        unsafe { ffi::mla_model_n_head_kv(self.raw.as_ptr()) }
    }

    /// The context length the model was trained at.
    #[must_use]
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { ffi::mla_model_n_ctx_train(self.raw.as_ptr()) }
    }

    /// Total size of all tensors, in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        unsafe { ffi::mla_model_size(self.raw.as_ptr()) }
    }

    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { ffi::mla_model_n_params(self.raw.as_ptr()) }
    }

    #[must_use]
    pub fn vocab_len(&self) -> i32 {
        unsafe { ffi::mla_vocab_n_tokens(self.raw.as_ptr()) }
    }

    /// llama.cpp's own one-line description, e.g. `olmoe 7B Q4_K - Medium`.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut buf = vec![0u8; 256];
        let n =
            unsafe { ffi::mla_model_desc(self.raw.as_ptr(), buf.as_mut_ptr().cast(), buf.len()) };
        if n <= 0 {
            return String::new();
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }

    /// Read a GGUF metadata string, e.g. `olmoe.expert_count`.
    ///
    /// Returns `None` when the key is absent, which is the normal answer for a
    /// dense model asked about experts.
    #[must_use]
    pub fn meta(&self, key: &str) -> Option<String> {
        let ckey = CString::new(key).ok()?;
        let mut buf = vec![0u8; 512];
        let n = unsafe {
            ffi::mla_model_meta_val_str(
                self.raw.as_ptr(),
                ckey.as_ptr(),
                buf.as_mut_ptr().cast(),
                buf.len(),
            )
        };
        if n < 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    /// Convenience over [`Model::meta`]: the number of experts per block, or
    /// `None` for a dense model. Tries the architecture-qualified key first.
    #[must_use]
    pub fn n_expert(&self) -> Option<u32> {
        let arch = self.meta("general.architecture")?;
        self.meta(&format!("{arch}.expert_count"))?.trim().parse().ok()
    }

    /// The number of experts routed to per token, or `None` for a dense model.
    #[must_use]
    pub fn n_expert_used(&self) -> Option<u32> {
        let arch = self.meta("general.architecture")?;
        self.meta(&format!("{arch}.expert_used_count"))?.trim().parse().ok()
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>, Error> {
        // A token is at worst one byte of input, plus room for added specials.
        let cap = i32::try_from(text.len() + 8).unwrap_or(i32::MAX);
        let mut out = vec![0i32; cap as usize];
        // SAFETY: `out` has `cap` writable i32s; `text` is passed with its length
        // and need not be NUL-terminated.
        let n = unsafe {
            ffi::mla_tokenize(
                self.raw.as_ptr(),
                text.as_ptr().cast(),
                i32::try_from(text.len()).unwrap_or(i32::MAX),
                out.as_mut_ptr(),
                cap,
                i32::from(add_special),
                1, // parse_special: honour chat-template markers
            )
        };
        if n < 0 {
            return Err(Error::Tokenize { capacity: cap, needed: -n });
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Render one token. Pieces are byte sequences and a single token need not be
    /// valid UTF-8 on its own, so this returns bytes; callers accumulate and
    /// decode. Returning a `String` here would corrupt any multi-byte character
    /// split across two tokens.
    #[must_use]
    pub fn token_bytes(&self, token: i32, special: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 128];
        let n = unsafe {
            ffi::mla_token_to_piece(
                self.raw.as_ptr(),
                token,
                buf.as_mut_ptr().cast(),
                i32::try_from(buf.len()).unwrap_or(i32::MAX),
                i32::from(special),
            )
        };
        if n < 0 {
            // llama.cpp signals "buffer too small" by returning -needed.
            buf.resize((-n) as usize, 0);
            let n2 = unsafe {
                ffi::mla_token_to_piece(
                    self.raw.as_ptr(),
                    token,
                    buf.as_mut_ptr().cast(),
                    i32::try_from(buf.len()).unwrap_or(i32::MAX),
                    i32::from(special),
                )
            };
            buf.truncate(n2.max(0) as usize);
            return buf;
        }
        buf.truncate(n as usize);
        buf
    }

    /// True if this token ends generation.
    #[must_use]
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { ffi::mla_vocab_is_eog(self.raw.as_ptr(), token) != 0 }
    }

    #[must_use]
    pub fn bos(&self) -> i32 {
        unsafe { ffi::mla_vocab_bos(self.raw.as_ptr()) }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::mla_model_free(self.raw.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// llama.cpp's decode timers, in the shape the tuning layer wants them.
#[derive(Debug, Clone, Copy, Default)]
pub struct Perf {
    pub t_load_ms: f64,
    pub t_prompt_ms: f64,
    pub t_decode_ms: f64,
    pub n_prompt: i32,
    pub n_decode: i32,
    pub n_reused: i32,
}

impl Perf {
    /// Decode throughput in tokens/second, or `None` if nothing was decoded.
    ///
    /// Returns `None` rather than 0.0 on an empty run: a throughput number with
    /// no tokens behind it is exactly the kind of figure this project has had to
    /// retract, and it should be impossible to print one by accident.
    #[must_use]
    pub fn decode_tok_per_s(&self) -> Option<f64> {
        (self.n_decode > 0 && self.t_decode_ms > 0.0)
            .then(|| f64::from(self.n_decode) / (self.t_decode_ms / 1000.0))
    }

    /// Prompt-processing throughput in tokens/second, or `None`.
    #[must_use]
    pub fn prompt_tok_per_s(&self) -> Option<f64> {
        (self.n_prompt > 0 && self.t_prompt_ms > 0.0)
            .then(|| f64::from(self.n_prompt) / (self.t_prompt_ms / 1000.0))
    }
}

/// An inference context: KV cache, compute buffers, thread pool.
pub struct Context<'m> {
    raw: NonNull<ffi::LlamaContext>,
    model: &'m Model,
}

impl<'m> Context<'m> {
    pub fn new(model: &'m Model, p: &ContextParams) -> Result<Self, Error> {
        let raw_params = ffi::MlaCtxParams {
            n_ctx: p.n_ctx,
            n_batch: p.n_batch,
            n_ubatch: p.n_ubatch,
            n_seq_max: p.n_seq_max,
            n_threads: p.n_threads,
            n_threads_batch: p.n_threads_batch,
            flash_attn_type: p.flash_attn as i32,
            type_k: p.type_k as i32,
            type_v: p.type_v as i32,
            offload_kqv: u8::from(p.offload_kqv),
            op_offload: u8::from(p.op_offload),
            swa_full: u8::from(p.swa_full),
            kv_unified: u8::from(p.kv_unified),
            no_perf: u8::from(p.no_perf),
        };

        unsafe { ffi::mla_clear_error() };
        let raw = unsafe { ffi::mla_ctx_create(model.raw.as_ptr(), &raw_params) };

        NonNull::new(raw)
            .map(|raw| Self { raw, model })
            .ok_or_else(|| Error::Context { reason: last_error() })
    }

    #[must_use]
    pub fn model(&self) -> &'m Model {
        self.model
    }

    /// The context length llama.cpp actually gave us, which can differ from what
    /// was requested (it is clamped to the model's trained context unless the
    /// caller overrides RoPE scaling).
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { ffi::mla_ctx_n_ctx(self.raw.as_ptr()) }
    }

    /// Change thread counts without rebuilding the context — the cheap knob for a
    /// thread sweep.
    pub fn set_n_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        unsafe { ffi::mla_ctx_set_n_threads(self.raw.as_ptr(), n_threads, n_threads_batch) };
    }

    /// Run a forward pass over `tokens`, appending them to the KV cache.
    pub fn decode(&mut self, tokens: &[i32]) -> Result<(), Error> {
        if tokens.is_empty() {
            return Ok(());
        }
        let code = unsafe {
            ffi::mla_decode(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                i32::try_from(tokens.len()).unwrap_or(i32::MAX),
            )
        };
        if code == 0 { Ok(()) } else { Err(Error::Decode { code, meaning: decode_meaning(code) }) }
    }

    /// The logit row for output `idx`; `-1` is the last token of the last decode.
    ///
    /// Borrowed straight out of llama.cpp's output buffer -- no copy, which for a
    /// 201k-token vocabulary is 800 KB saved per generated token. That buffer is
    /// overwritten by the next [`Context::decode`], and the borrow checker is what
    /// enforces it: this takes `&self`, `decode` takes `&mut self`, so a slice
    /// handed out here cannot still be alive across a decode.
    ///
    /// `None` when llama.cpp computed no logits for that index.
    #[must_use]
    pub fn logits(&self, idx: i32) -> Option<&[f32]> {
        let n = usize::try_from(self.model.vocab_len()).ok()?;
        if n == 0 {
            return None;
        }
        // SAFETY: the shim returns either NULL or a pointer to `n_vocab`
        // contiguous f32 owned by this context. The lifetime is tied to `&self`,
        // and `decode` -- the only thing that invalidates it -- needs `&mut self`.
        let p = unsafe { ffi::mla_get_logits_ith(self.raw.as_ptr(), idx) };
        (!p.is_null()).then(|| unsafe { std::slice::from_raw_parts(p, n) })
    }

    /// Drop the KV cache so the context can be reused for a fresh sequence.
    pub fn reset(&mut self) {
        unsafe { ffi::mla_memory_clear(self.raw.as_ptr(), 1) };
    }

    #[must_use]
    pub fn perf(&self) -> Perf {
        let mut p = ffi::MlaPerf::default();
        unsafe { ffi::mla_ctx_perf(self.raw.as_ptr(), &mut p) };
        Perf {
            t_load_ms: p.t_load_ms,
            t_prompt_ms: p.t_p_eval_ms,
            t_decode_ms: p.t_eval_ms,
            n_prompt: p.n_p_eval,
            n_decode: p.n_eval,
            n_reused: p.n_reused,
        }
    }

    pub fn perf_reset(&mut self) {
        unsafe { ffi::mla_ctx_perf_reset(self.raw.as_ptr()) };
    }
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        unsafe { ffi::mla_ctx_free(self.raw.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// A sampler chain.
pub struct Sampler {
    raw: NonNull<ffi::LlamaSampler>,
}

impl Sampler {
    /// Greedy: always the argmax token. The right choice for a correctness check,
    /// because it makes a run reproducible and comparable token-for-token against
    /// another engine.
    #[must_use]
    pub fn greedy() -> Self {
        init(false);
        // SAFETY: the shim always returns a valid chain.
        let raw = unsafe { ffi::mla_sampler_greedy() };
        Self { raw: NonNull::new(raw).expect("sampler chain init returned null") }
    }

    /// top-k → top-p → temperature → distribution, seeded.
    #[must_use]
    pub fn dist(seed: u32, top_k: i32, top_p: f32, temp: f32) -> Self {
        init(false);
        let raw = unsafe { ffi::mla_sampler_dist(seed, top_k, top_p, temp) };
        Self { raw: NonNull::new(raw).expect("sampler chain init returned null") }
    }

    /// Sample from the logits of output `idx` (-1 = the last one).
    pub fn sample(&mut self, ctx: &mut Context<'_>, idx: i32) -> i32 {
        unsafe { ffi::mla_sampler_sample(self.raw.as_ptr(), ctx.raw.as_ptr(), idx) }
    }

    /// Tell stateful samplers (penalties, grammars) that a token was committed.
    pub fn accept(&mut self, token: i32) {
        unsafe { ffi::mla_sampler_accept(self.raw.as_ptr(), token) };
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { ffi::mla_sampler_free(self.raw.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// The result of a generation run.
#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub tokens: Vec<i32>,
    /// True if generation stopped on an end-of-generation token rather than on
    /// the token budget.
    pub hit_eog: bool,
    pub perf: Perf,
}

/// Prompt, then decode greedily until `max_tokens` or an end-of-generation token.
///
/// This is the minimal proof-of-life loop, not the server's sampler. It exists so
/// that "Rust drives llama.cpp on the Arc" is a thing that can be asserted.
pub fn generate(
    ctx: &mut Context<'_>,
    sampler: &mut Sampler,
    prompt: &str,
    max_tokens: usize,
) -> Result<Generation, Error> {
    let model = ctx.model();
    let prompt_tokens = model.tokenize(prompt, true)?;

    ctx.decode(&prompt_tokens)?;

    let mut out_bytes: Vec<u8> = Vec::new();
    let mut tokens: Vec<i32> = Vec::with_capacity(max_tokens);
    let mut hit_eog = false;

    for _ in 0..max_tokens {
        let tok = sampler.sample(ctx, -1);
        if model.is_eog(tok) {
            hit_eog = true;
            break;
        }
        sampler.accept(tok);
        tokens.push(tok);
        out_bytes.extend_from_slice(&model.token_bytes(tok, false));
        ctx.decode(&[tok])?;
    }

    Ok(Generation {
        // A token boundary can split a UTF-8 sequence, so decode the accumulated
        // bytes once at the end rather than per token.
        text: String::from_utf8_lossy(&out_bytes).into_owned(),
        tokens,
        hit_eog,
        perf: ctx.perf(),
    })
}
