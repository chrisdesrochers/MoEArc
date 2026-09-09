//! The seam between the HTTP layer and the inference engine.
//!
//! # The integration point
//!
//! Everything above this module — routing, chat templating, SSE framing, stop sequences,
//! usage accounting — talks to [`Generator`] and nothing else. There is no dependency here on
//! any engine, by design: the serving layer had to be finishable and testable while the engine
//! was still being written, and a server wired directly to a half-built engine cannot be
//! tested at all. That property is what made swapping the engine underneath it a change to one
//! file.
//!
//! **Swapping the stub for the real engine is one line.** In
//! [`crate::state::ServerState::new`], the `Arc<dyn Generator>` handed in is
//! [`EchoGenerator`]; replacing it with the engine's implementation is the entire change.
//! Nothing else in this crate names a concrete generator. The engine side then owes exactly
//! one `impl`:
//!
//! ```ignore
//! impl Generator for EngineGenerator {
//!     fn generate(&self, prompt_tokens: &[u32], params: &SamplingParams,
//!                 on_token: &mut dyn FnMut(u32) -> bool) -> Result<GenerationStats> { .. }
//! }
//! ```
//!
//! It should not write the token loop itself. [`drive`] is that loop, and the clauses below
//! are what it implements.
//!
//! # Contract the engine must honour
//!
//! - **Blocking is expected.** `generate` runs on a blocking thread ([`tokio::task::spawn_blocking`]),
//!   so it may sit on a queue or a device fence. It must not be `async`, and it must not
//!   assume a reactor.
//! - **`on_token` is called once per accepted token, in order,** and its `bool` return is
//!   *continue*. Returning `false` means the consumer is gone (client disconnected) or a stop
//!   condition fired upstream; the engine must stop promptly and return the stats it has.
//!   Ignoring it leaks a full generation per abandoned request.
//! - **Stop conditions are shared.** `params.stop_tokens`/`max_tokens` are the engine's to
//!   enforce; *stop strings* are enforced by the caller through `on_token`, because they are a
//!   property of decoded text and the engine has no detokeniser.
//! - **Sampling belongs to the engine,** using [`crate::sampling`] so behaviour is identical
//!   to what the tests here pin down. The engine produces logits; it should not grow a second
//!   sampler.

use std::sync::Arc;

use crate::sampling::{Rng, SamplingParams, sample};

/// What a completed generation cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Why generation ended, which becomes OpenAI's `finish_reason`.
    pub stop_reason: StopReason,
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted an id in `stop_tokens` (usually EOS).
    #[default]
    EndOfTurn,
    /// `max_tokens` was reached.
    Length,
    /// `on_token` returned `false` — a stop string matched, or the client went away.
    Cancelled,
}

impl StopReason {
    /// The OpenAI wire value. `Cancelled` reports `stop` because from the client's side a stop
    /// string matching *is* a normal stop; a disconnected client is not reading this anyway.
    pub fn as_openai(self) -> &'static str {
        match self {
            Self::EndOfTurn | Self::Cancelled => "stop",
            Self::Length => "length",
        }
    }
}

/// Produce tokens from a prompt.
///
/// `Send + Sync` because one instance is shared across every in-flight request behind an
/// `Arc`; concurrency control (batching, a queue, a device lock) is the implementation's
/// business, not the router's.
pub trait Generator: Send + Sync {
    /// Generate until a stop condition fires, calling `on_token` for each token.
    ///
    /// Returning `Err` aborts the request with a 500. A stop condition is not an error.
    fn generate(
        &self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<GenerationStats>;

    /// Vocabulary size the generator produces logits over. Used only to sanity-check that the
    /// tokeniser and the generator agree; a mismatch is a misconfigured deployment, and it is
    /// far better caught at startup than as garbage text.
    fn vocab_size(&self) -> usize;

    /// What to call this in the startup banner and in `/health`.
    fn name(&self) -> &'static str;

    /// Whether this produces text from real model weights.
    ///
    /// 🔴 Not cosmetic. `/health` reports it and the banner prints it, because output that
    /// looks like a measurement and is not is worse than no output — the same reason
    /// `moearc-cli`'s `Sources` carries a `stubbed` flag. Defaults to `false` so a real engine
    /// cannot accidentally inherit "this is fake".
    fn is_stub(&self) -> bool {
        false
    }
}

/// The token loop the contract above describes, written once.
///
/// A real generator differs from another only in how it turns tokens into logits, so that is
/// the only thing this asks for: `advance` feeds tokens to the model and returns the logits
/// that follow the last of them. It is called once with the whole prompt, and then **once per
/// accepted token** — never with anything else.
///
/// That last sentence is the point of the function.
///
/// 🔴 **A token that is never emitted must not reach the KV cache.** Sampling produces a
/// candidate; three things can happen next that mean the client will never see it — it is a
/// stop token, `on_token` returned `false` (a stop string matched, or the client hung up), or
/// the token budget is spent. In each case generation ends *before* `advance` is called
/// again, so the model's state stops exactly where the client's last-seen token left it.
/// Advancing first and deciding afterwards produces a sequence one token ahead of the
/// transcript, which is not an error anywhere — it is drift that shows up as wrong text some
/// distance later, in a different request, on a reused context. It has no symptom at the
/// point of the bug, which is why the rule gets one implementation and a test rather than a
/// comment in each generator.
pub fn drive(
    prompt_tokens: &[u32],
    params: &SamplingParams,
    rng: &mut Rng,
    advance: &mut dyn FnMut(&[u32]) -> anyhow::Result<Vec<f32>>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> anyhow::Result<GenerationStats> {
    if prompt_tokens.is_empty() {
        anyhow::bail!("a prompt needs at least one token");
    }

    let mut history: Vec<u32> = prompt_tokens.to_vec();
    let mut stats = GenerationStats {
        prompt_tokens: prompt_tokens.len(),
        completion_tokens: 0,
        stop_reason: StopReason::Length,
    };

    // The prompt is processed even when no tokens are wanted. `max_tokens = 0` is a legitimate
    // request — it is how a client asks what a prompt costs — and reporting `prompt_tokens` for
    // a prompt that was never run would be a usage figure with nothing behind it.
    let mut logits = advance(prompt_tokens)?;
    if params.max_tokens == 0 {
        return Ok(stats);
    }

    loop {
        let token = sample(&logits, &history, params, rng);

        if params.stop_tokens.contains(&token) {
            stats.stop_reason = StopReason::EndOfTurn;
            return Ok(stats);
        }
        history.push(token);
        stats.completion_tokens += 1;
        if !on_token(token) {
            stats.stop_reason = StopReason::Cancelled;
            return Ok(stats);
        }
        if stats.completion_tokens >= params.max_tokens {
            stats.stop_reason = StopReason::Length;
            return Ok(stats);
        }

        // Only here, and only for a token the caller has already seen.
        logits = advance(&[token])?;
    }
}

/// A [`Generator`] with no model behind it, for exercising the whole serving path today.
///
/// It is not a random token source. At each step it synthesises a logits vector peaked on the
/// prompt token at that position, so:
///
/// - **at `temperature = 0` it echoes the prompt back exactly** — a completion you can assert
///   on, byte for byte, without a model;
/// - at `temperature > 0` the surrounding noise is real enough that top-k, top-p and the seed
///   visibly change the output, so the sampler is genuinely in the request path rather than
///   bypassed by a stub that returns constants.
///
/// The noise is a hash of `(position, token id)`, so it is fixed for a given prompt — the
/// generator contributes no entropy of its own and the seed alone decides.
pub struct EchoGenerator {
    vocab_size: usize,
    /// Logit given to the intended token. Large enough that greedy always picks it, small
    /// enough that a high temperature can still pick something else.
    peak: f32,
}

impl EchoGenerator {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size, peak: 8.0 }
    }

    /// Deterministic noise in roughly `-1.0..1.0`, from a 64-bit integer hash.
    fn noise(pos: usize, id: usize) -> f32 {
        let mut h = (pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (id as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        ((h >> 40) as f32 / 8_388_608.0) - 1.0
    }

    /// The logits for step `pos`, peaked on `intended`.
    fn logits_at(&self, pos: usize, intended: Option<u32>) -> Vec<f32> {
        let mut logits: Vec<f32> = (0..self.vocab_size).map(|id| Self::noise(pos, id)).collect();
        if let Some(id) = intended
            && let Some(l) = logits.get_mut(id as usize)
        {
            *l = self.peak;
        }
        logits
    }
}

impl Generator for EchoGenerator {
    fn generate(
        &self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<GenerationStats> {
        // No seed means "vary between requests"; the stub still has to be a *generator*, so it
        // takes entropy from the clock exactly as a real one would.
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        });
        let mut rng = Rng::seed_from_u64(seed);

        let mut history: Vec<u32> = prompt_tokens.to_vec();
        let mut emitted = 0usize;
        let mut stop_reason = StopReason::Length;

        for pos in 0..params.max_tokens {
            // Echo the prompt, then fall off the end and let the noise decide — which is what
            // makes "generated past the prompt" visible rather than silently repeating.
            let intended = prompt_tokens.get(pos).copied();
            let logits = self.logits_at(pos, intended);
            let token = sample(&logits, &history, params, &mut rng);

            if params.stop_tokens.contains(&token) {
                stop_reason = StopReason::EndOfTurn;
                break;
            }
            history.push(token);
            emitted += 1;
            if !on_token(token) {
                stop_reason = StopReason::Cancelled;
                break;
            }
            if pos + 1 == params.max_tokens {
                stop_reason = StopReason::Length;
            }
        }

        Ok(GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: emitted,
            stop_reason,
        })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn name(&self) -> &'static str {
        "echo"
    }

    fn is_stub(&self) -> bool {
        true
    }
}

/// Convenience alias — the router holds one of these and never a concrete type.
pub type SharedGenerator = Arc<dyn Generator>;

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(
        g: &dyn Generator,
        prompt: &[u32],
        params: &SamplingParams,
    ) -> (Vec<u32>, GenerationStats) {
        let mut out = Vec::new();
        let stats = g
            .generate(prompt, params, &mut |t| {
                out.push(t);
                true
            })
            .unwrap();
        (out, stats)
    }

    /// A model whose next token is whatever the script says, which records every token it
    /// was ever fed. What it is fed *is* the KV cache, so the recording is the assertion.
    struct Scripted {
        script: Vec<u32>,
        step: usize,
        fed: Vec<Vec<u32>>,
    }

    impl Scripted {
        fn new(script: &[u32]) -> Self {
            Self { script: script.to_vec(), step: 0, fed: Vec::new() }
        }

        fn advance(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
            self.fed.push(tokens.to_vec());
            let mut logits = vec![0.0f32; 16];
            if let Some(&t) = self.script.get(self.step) {
                logits[t as usize] = 10.0;
            }
            self.step += 1;
            Ok(logits)
        }
    }

    fn greedy(max_tokens: usize, stop_tokens: Vec<u32>) -> SamplingParams {
        SamplingParams { temperature: 0.0, max_tokens, stop_tokens, ..Default::default() }
    }

    /// Run `drive` against a scripted model, returning what was emitted, what reached the
    /// model, and the stats.
    fn run(
        script: &[u32],
        prompt: &[u32],
        params: &SamplingParams,
        mut keep_going: impl FnMut(usize) -> bool,
    ) -> (Vec<u32>, Vec<Vec<u32>>, GenerationStats) {
        let mut model = Scripted::new(script);
        let mut rng = Rng::seed_from_u64(0);
        let mut out = Vec::new();
        let stats = drive(prompt, params, &mut rng, &mut |t| model.advance(t), &mut |t| {
            out.push(t);
            keep_going(out.len())
        })
        .unwrap();
        (out, model.fed, stats)
    }

    /// 🔴 The load-bearing clause. The stop token is sampled, is not emitted, and must not
    /// have been fed to the model either.
    #[test]
    fn a_stop_token_never_reaches_the_kv_cache() {
        let params = greedy(10, vec![3]);
        let (out, fed, stats) = run(&[1, 2, 3, 4], &[7, 8, 9], &params, |_| true);
        assert_eq!(out, vec![1, 2]);
        assert_eq!(fed, vec![vec![7, 8, 9], vec![1], vec![2]]);
        assert!(!fed.iter().any(|f| f.contains(&3)), "the stop token was fed to the model");
        assert_eq!(stats.stop_reason, StopReason::EndOfTurn);
        assert_eq!(stats.completion_tokens, 2);
    }

    /// The same clause on the cancellation path, and it is the subtler half: token 2 *was*
    /// emitted, the callback then said stop, and it must still not have been fed back.
    #[test]
    fn a_cancelled_token_is_emitted_but_never_fed_back() {
        let params = greedy(10, vec![]);
        let (out, fed, stats) = run(&[1, 2, 3, 4], &[7], &params, |n| n < 2);
        assert_eq!(out, vec![1, 2]);
        assert_eq!(fed, vec![vec![7], vec![1]]);
        assert_eq!(stats.stop_reason, StopReason::Cancelled);
        assert_eq!(stats.completion_tokens, 2);
    }

    /// And on the budget path: the last token of a completion is emitted, never fed.
    #[test]
    fn the_final_token_of_a_full_budget_is_not_fed_back() {
        let params = greedy(2, vec![]);
        let (out, fed, stats) = run(&[1, 2, 3, 4], &[7], &params, |_| true);
        assert_eq!(out, vec![1, 2]);
        assert_eq!(fed, vec![vec![7], vec![1]]);
        assert_eq!(stats.stop_reason, StopReason::Length);
    }

    #[test]
    fn each_accepted_token_is_fed_exactly_once_in_order() {
        let params = greedy(4, vec![]);
        let (out, fed, _) = run(&[1, 2, 3, 4, 5], &[7, 8], &params, |_| true);
        assert_eq!(out, vec![1, 2, 3, 4]);
        assert_eq!(fed, vec![vec![7, 8], vec![1], vec![2], vec![3]]);
    }

    /// `max_tokens = 0` still runs the prompt: the usage figure has to describe work that
    /// actually happened.
    #[test]
    fn a_zero_budget_still_processes_the_prompt() {
        let params = greedy(0, vec![]);
        let (out, fed, stats) = run(&[1, 2], &[7, 8, 9], &params, |_| true);
        assert!(out.is_empty());
        assert_eq!(fed, vec![vec![7, 8, 9]]);
        assert_eq!(stats.prompt_tokens, 3);
        assert_eq!(stats.completion_tokens, 0);
    }

    #[test]
    fn an_empty_prompt_is_refused_before_the_model_is_touched() {
        let mut model = Scripted::new(&[1]);
        let mut rng = Rng::seed_from_u64(0);
        let params = greedy(4, vec![]);
        let err = drive(&[], &params, &mut rng, &mut |t| model.advance(t), &mut |_| true)
            .expect_err("an empty prompt must not be generated from");
        assert!(err.to_string().contains("at least one token"), "{err}");
        assert!(model.fed.is_empty());
    }

    #[test]
    fn greedy_echoes_the_prompt_exactly() {
        let g = EchoGenerator::new(512);
        let prompt = [11u32, 22, 33, 44];
        let params = SamplingParams { temperature: 0.0, max_tokens: 4, ..Default::default() };
        let (out, stats) = collect(&g, &prompt, &params);
        assert_eq!(out, prompt);
        assert_eq!(stats.prompt_tokens, 4);
        assert_eq!(stats.completion_tokens, 4);
        assert_eq!(stats.stop_reason, StopReason::Length);
    }

    #[test]
    fn seeded_sampling_replays() {
        let g = EchoGenerator::new(512);
        let prompt = [5u32, 6, 7];
        let params = SamplingParams {
            temperature: 1.2,
            seed: Some(2026),
            max_tokens: 24,
            ..Default::default()
        };
        assert_eq!(collect(&g, &prompt, &params).0, collect(&g, &prompt, &params).0);
        let other = SamplingParams { seed: Some(2027), ..params.clone() };
        assert_ne!(collect(&g, &prompt, &params).0, collect(&g, &prompt, &other).0);
    }

    #[test]
    fn stop_token_ends_the_turn_and_is_not_emitted() {
        let g = EchoGenerator::new(512);
        let prompt = [1u32, 2, 3, 4, 5];
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            stop_tokens: vec![3],
            ..Default::default()
        };
        let (out, stats) = collect(&g, &prompt, &params);
        assert_eq!(out, vec![1, 2]);
        assert_eq!(stats.stop_reason, StopReason::EndOfTurn);
        assert_eq!(stats.completion_tokens, 2);
    }

    #[test]
    fn a_false_callback_stops_generation() {
        let g = EchoGenerator::new(512);
        let prompt = [9u32; 100];
        let params = SamplingParams { temperature: 0.0, max_tokens: 100, ..Default::default() };
        let mut seen = 0;
        let stats = g
            .generate(&prompt, &params, &mut |_| {
                seen += 1;
                seen < 3
            })
            .unwrap();
        assert_eq!(seen, 3);
        assert_eq!(stats.stop_reason, StopReason::Cancelled);
    }

    #[test]
    fn top_k_constrains_the_stub_too() {
        // The stub feeds the real sampler, so a restriction proven in `sampling` must also
        // hold end-to-end here. With top_k = 1 the noise cannot win over the peak.
        let g = EchoGenerator::new(256);
        let prompt = [77u32, 78, 79];
        let params = SamplingParams {
            temperature: 5.0,
            top_k: 1,
            seed: Some(1),
            max_tokens: 3,
            ..Default::default()
        };
        assert_eq!(collect(&g, &prompt, &params).0, prompt);
    }
}
