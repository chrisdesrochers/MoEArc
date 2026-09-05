//! The seam between the HTTP layer and the inference engine.
//!
//! # The integration point
//!
//! Everything above this module — routing, chat templating, SSE framing, stop sequences,
//! usage accounting — talks to [`Generator`] and nothing else. There is no dependency here on
//! `moearc-engine`, `moearc-kernels` or `moearc-model`, by design: the serving layer had to be
//! finishable and testable while those three were still being written, and a server wired
//! directly to a half-built engine cannot be tested at all.
//!
//! **Swapping the stub for the real engine is one line.** In
//! [`crate::state::ServerState::new`], the `Arc<dyn Generator>` handed in is
//! [`EchoGenerator`]; replacing it with the engine's implementation is the entire change.
//! Nothing else in this crate names a concrete generator. The engine side then owes exactly
//! one `impl`:
//!
//! ```ignore
//! impl Generator for moearc_engine::Session {
//!     fn generate(&self, prompt_tokens: &[u32], params: &SamplingParams,
//!                 on_token: &mut dyn FnMut(u32) -> bool) -> Result<GenerationStats> { .. }
//! }
//! ```
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
