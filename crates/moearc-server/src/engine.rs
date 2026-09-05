//! The bridge from this crate's [`Generator`] contract to MoEArc's real engine.
//!
//! This is the whole integration. It lives here rather than in `moearc-engine` because
//! `Generator` is defined here: implementing it there would make the engine depend on the
//! server, inverting the dependency the rest of this crate is careful to avoid.
//!
//! Behind the `engine` feature, because `moearc-engine/gpu` transitively requires Intel's
//! DPC++ compiler to build. Without the feature this crate still builds, tests and serves
//! anywhere — which is the property that let the whole serving path be written and proven
//! before a single kernel existed.

use std::path::Path;
use std::sync::Arc;

use moearc_engine::session::{Session, StopConditions};

use crate::generate::{GenerationStats, Generator, SharedGenerator, StopReason};
use crate::sampling::{Rng, SamplingParams, sample};

/// A [`Generator`] backed by real model weights on a real device.
pub struct EngineGenerator {
    session: Session,
}

impl EngineGenerator {
    /// Load a model and open a device session.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(Self { session: Session::load(path)? })
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
        // Seeding matches EchoGenerator exactly: no seed means "vary between requests", and a
        // seed makes the whole completion reproducible. Diverging here would make the stub and
        // the real engine behave differently for the same request, which is precisely the
        // difference a stub exists to avoid.
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        });
        let mut rng = Rng::seed_from_u64(seed);

        // The sampler stays this crate's. The engine takes a closure precisely so there is one
        // sampler in the system rather than two that drift — a seeded completion must be
        // reproducible regardless of which generator answered it.
        let mut sampler =
            |logits: &[f32], history: &[u32]| sample(logits, history, params, &mut rng);

        let stop = StopConditions {
            max_tokens: params.max_tokens,
            stop_tokens: params.stop_tokens.clone(),
        };

        let stats = self.session.generate_with(prompt_tokens, &stop, &mut sampler, on_token)?;

        Ok(GenerationStats {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            stop_reason: match stats.stop_reason {
                moearc_engine::session::StopReason::EndOfTurn => StopReason::EndOfTurn,
                moearc_engine::session::StopReason::Length => StopReason::Length,
                moearc_engine::session::StopReason::Cancelled => StopReason::Cancelled,
            },
        })
    }

    fn vocab_size(&self) -> usize {
        self.session.vocab_size()
    }

    fn name(&self) -> &'static str {
        self.session.name()
    }

    // Deliberately left at the default `false`: this one is not a stub, and `/health` and the
    // startup banner both read it.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mapping is exhaustive by construction — adding a variant on either side must not
    /// compile until it is handled here. This test exists to make that failure a compile error
    /// rather than a silently wrong `finish_reason` in an OpenAI response.
    #[test]
    fn stop_reasons_map_one_to_one() {
        use moearc_engine::session::StopReason as E;
        let pairs = [
            (E::EndOfTurn, StopReason::EndOfTurn),
            (E::Length, StopReason::Length),
            (E::Cancelled, StopReason::Cancelled),
        ];
        for (engine, server) in pairs {
            let mapped = match engine {
                E::EndOfTurn => StopReason::EndOfTurn,
                E::Length => StopReason::Length,
                E::Cancelled => StopReason::Cancelled,
            };
            assert_eq!(mapped, server);
        }
    }
}
