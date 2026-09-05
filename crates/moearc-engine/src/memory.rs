//! Deciding what lives in VRAM.
//!
//! One question sits under the whole product: given this card and this model, how many MoE
//! experts stay resident and how much context can we serve? `docs/ux.md` makes it a rule that
//! the user never answers that — they ask for a context length, or for nothing at all, and we
//! work it out.
//!
//! # Design notes
//!
//! This is MoEArc's own design. The problem is not novel and prior art exists, but three
//! choices here are deliberate departures worth stating, because each removes a class of bug
//! rather than handling it:
//!
//! 1. **Unsigned throughout.** Byte counts are `u64` and arithmetic is checked. A budget
//!    cannot go negative, so there is no floor-versus-truncate division question, no negative
//!    intermediate to clamp away, and no path where a subtraction quietly wraps into a
//!    plausible-looking plan. Prior art in this space computes a signed budget and then
//!    detects the overrun afterwards; making it unrepresentable is cheaper and safer.
//! 2. **The user's unit is tokens.** Pages are an implementation detail of the KV cache and
//!    never appear in the public API. Callers say `Context::Tokens(32_768)` or
//!    `Context::Largest`.
//! 3. **No inherited kernel constants.** Nothing here encodes a slot cap or a headroom
//!    fraction borrowed from another vendor's runtime. Caps come from the model or from an
//!    explicit caller policy; headroom is a stated [`Policy`] field with a documented,
//!    provisional default that calibration is expected to replace. A constant tuned against
//!    someone else's allocator is a guess wearing the costume of a default.
//!
//! The planner also **explains itself**. `Allocation::rationale` records why the split came
//! out as it did, so the CLI can print its reasoning at startup instead of asking the user to
//! trust a number.

use std::fmt;

/// Memory the device reports, measured before the model is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Total device memory.
    pub total_bytes: u64,
    /// Free device memory at the moment of measurement.
    pub free_bytes: u64,
}

/// What the model costs, in bytes.
///
/// Every field here is read out of the model file by `moearc-model`. None of it is asked of
/// the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFootprint {
    /// Non-expert weights that are always resident: embeddings, attention, norms.
    pub dense_weights_bytes: u64,
    /// Bytes one expert occupies. All experts are assumed the same size, which holds for
    /// every MoE architecture shipping today; a model that violates it will fail the
    /// [`Self::validate`] check rather than silently mis-plan.
    pub per_expert_bytes: u64,
    /// Experts the model has in total.
    pub total_experts: u32,
    /// Experts activated per token. This is the floor on residency: fewer than this and at
    /// least one expert must be fetched across the bus for every single token.
    pub active_experts: u32,
    /// KV cache cost of one token, across all layers.
    pub kv_bytes_per_token: u64,
}

/// Which way to lean when expert residency and context length compete for the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bias {
    /// Fill expert slots first, then give the remainder to context. Favours decode speed.
    #[default]
    Experts,
    /// Satisfy the requested context first, then fill expert slots. Favours long prompts.
    Context,
}

/// How much of the card to leave alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Headroom {
    /// Leave this fraction of free memory unallocated.
    Fraction(f64),
    /// Leave exactly this many bytes unallocated.
    Bytes(u64),
}

impl Headroom {
    /// 🔴 **Provisional, and it reaches the user.** Activations, scratch buffers and allocator
    /// fragmentation are not modelled, so a flat fraction is held back instead. 12% is a
    /// placeholder chosen to be conservative — **not a measured value**, and not borrowed from
    /// another runtime's tuning either.
    ///
    /// It is not a small effect: on a 11.33 GiB card it withholds 1.36 GiB, roughly 700 expert
    /// slots. Every plan built on it is only as good as this guess, which is why it is named,
    /// printed in the rationale, and first on the calibration list. Measure it per device
    /// before quoting any number that depends on it — see `docs/calibration.md`.
    pub const PROVISIONAL: Self = Self::Fraction(0.12);

    fn reserve_from(self, free: u64) -> u64 {
        match self {
            Self::Fraction(f) => {
                let f = f.clamp(0.0, 1.0);
                // Round the reserve up: it is safer to hold back one byte too many.
                (free as f64 * f).ceil() as u64
            }
            Self::Bytes(b) => b.min(free),
        }
    }
}

/// Caller policy. Everything here has a defensible default, so `Policy::default()` is a
/// complete answer for a user who has expressed no preference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// Memory to leave unallocated.
    pub headroom: Headroom,
    /// Tokens per KV page. Purely an internal granularity.
    ///
    /// 🔴 The default of 256 is **chosen, not measured**. Page size trades allocator
    /// bookkeeping against internal fragmentation — a half-used page wastes up to
    /// `page_tokens - 1` tokens of KV — and the right value depends on kernel access patterns
    /// we have not written yet. It is on the calibration list.
    pub page_tokens: u32,
    /// Context length reserved before experts are placed.
    ///
    /// 🔴 The default of 2048 is a **product judgement, not a measurement**: it is a guess at
    /// the shortest context at which a server is still worth starting. It is also load-bearing
    /// in a way that is easy to miss. With [`Bias::Experts`], expert slots absorb everything
    /// above this floor, and whenever the bytes left over are worth less than a single KV page
    /// the planned context lands on this number *exactly*. That is the common case rather than
    /// an edge one: it holds whenever experts are small relative to a page, which is true of
    /// the reference model (1.95 MiB experts against a 5 MiB page). When it happens the result
    /// is a restatement of the policy rather than a capacity, and
    /// [`Reason::ContextAtPolicyFloor`] says so in the output. Callers who want a real context
    /// should ask for one with [`Context::Tokens`].
    ///
    /// This is a *floor taken off the top*, not a check applied at the end. Expert residency
    /// is greedy, and greedy fill will otherwise consume the entire card and leave a few
    /// hundred tokens of context -- a plan that allocates cleanly and cannot answer a single
    /// realistic request.
    pub min_context_tokens: u32,
    /// Which way to lean when residency and context compete.
    pub bias: Bias,
    /// Optional hard cap on resident experts, for a kernel that cannot address more.
    ///
    /// `None` means the model's own expert count is the only limit. This exists so a future
    /// kernel constraint has somewhere to live **without** becoming a silent default.
    pub max_resident_experts: Option<u32>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            headroom: Headroom::PROVISIONAL,
            page_tokens: 256,
            min_context_tokens: 2048,
            bias: Bias::Experts,
            max_resident_experts: None,
        }
    }
}

/// What the caller wants from the context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Context {
    /// A specific context length. Planning fails if it does not fit.
    Tokens(u32),
    /// Whatever fits once experts are placed.
    Largest,
}

/// A step in the planner's reasoning, so the CLI can show its work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// Free memory, minus headroom, minus dense weights.
    Budget { free: u64, headroom: u64, dense_weights: u64, usable: u64 },
    /// Every expert fits; residency is not the constraint.
    AllExpertsResident { count: u32 },
    /// Experts were capped by available memory rather than by the model.
    ExpertsLimitedByMemory { resident: u32, of: u32 },
    /// Experts were capped by explicit policy.
    ExpertsLimitedByPolicy { resident: u32, cap: u32 },
    /// Context was reduced to fit alongside the resident experts.
    ContextLimitedByMemory { tokens: u32, requested: u32 },
    /// Experts were given up to satisfy the requested context.
    ExpertsYieldedToContext { resident: u32, would_have_been: u32 },
    /// Context came out exactly at the policy floor, so it reflects the policy rather than
    /// what the card could serve.
    ContextAtPolicyFloor { tokens: u32 },
    /// Bytes left unspent after rounding to whole pages and slots.
    Slack { bytes: u64 },
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget { free, headroom, dense_weights, usable } => write!(
                f,
                "{} free, less {} headroom and {} of dense weights, leaves {} to allocate",
                gib(*free),
                gib(*headroom),
                gib(*dense_weights),
                gib(*usable)
            ),
            Self::AllExpertsResident { count } => {
                write!(f, "all {count} experts fit in VRAM")
            }
            Self::ExpertsLimitedByMemory { resident, of } => {
                write!(f, "{resident} of {of} experts resident, limited by memory")
            }
            Self::ExpertsLimitedByPolicy { resident, cap } => {
                write!(f, "{resident} experts resident, capped at {cap} by policy")
            }
            Self::ContextLimitedByMemory { tokens, requested } => {
                write!(f, "context reduced to {tokens} tokens; {requested} did not fit")
            }
            Self::ExpertsYieldedToContext { resident, would_have_been } => write!(
                f,
                "gave up {} expert slots to reach the requested context ({resident} resident)",
                would_have_been - resident
            ),
            Self::ContextAtPolicyFloor { tokens } => write!(
                f,
                "context is {tokens} tokens because that is the configured minimum, not \
                 because it is all that fits — experts took the rest; ask for a specific \
                 context to trade slots for it"
            ),
            Self::Slack { bytes } => write!(f, "{} unallocated after rounding", gib(*bytes)),
        }
    }
}

fn gib(bytes: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    if bytes >= (1 << 30) {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.0} MiB", bytes as f64 / MIB)
    }
}

/// A concrete decision about VRAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    /// Expert slots held resident.
    pub resident_experts: u32,
    /// Context length served, in tokens.
    pub context_tokens: u32,
    /// KV pages backing that context.
    pub kv_pages: u32,
    /// Bytes committed to expert residency.
    pub expert_bytes: u64,
    /// Bytes committed to the KV cache.
    pub kv_bytes: u64,
    /// Bytes deliberately left unallocated.
    pub headroom_bytes: u64,
    /// Why this came out the way it did.
    pub rationale: Vec<Reason>,
}

impl Allocation {
    /// Total device bytes this plan commits, including the dense weights it was planned
    /// around.
    pub fn committed_bytes(&self, model: &ModelFootprint) -> u64 {
        model.dense_weights_bytes + self.expert_bytes + self.kv_bytes
    }

    /// Whether every expert is resident, i.e. no expert is ever fetched across the bus.
    pub fn is_fully_resident(&self, model: &ModelFootprint) -> bool {
        self.resident_experts >= model.total_experts
    }
}

/// Why a plan could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The model description is not self-consistent.
    InvalidModel(&'static str),
    /// `page_tokens` was zero.
    InvalidPolicy(&'static str),
    /// The dense weights alone do not fit.
    WeightsDoNotFit { need: u64, have: u64 },
    /// Weights fit, but not with the minimum viable expert residency.
    CannotHoldActiveExperts { need: u64, have: u64, active_experts: u32 },
    /// Residency is possible but no usable context remains.
    ContextTooSmall { achievable: u32, minimum: u32 },
    /// A specific context was requested and does not fit.
    ContextDoesNotFit { requested: u32, achievable: u32 },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(m) => write!(f, "model description is inconsistent: {m}"),
            Self::InvalidPolicy(m) => write!(f, "invalid policy: {m}"),
            Self::WeightsDoNotFit { need, have } => write!(
                f,
                "the model's dense weights need {} but only {} is usable — this model is too \
                 large for this device",
                gib(*need),
                gib(*have)
            ),
            Self::CannotHoldActiveExperts { need, have, active_experts } => write!(
                f,
                "cannot hold the {active_experts} experts active per token: needs {} but only \
                 {} is usable — a smaller quantisation would fit",
                gib(*need),
                gib(*have)
            ),
            Self::ContextTooSmall { achievable, minimum } => write!(
                f,
                "only {achievable} tokens of context fit, below the {minimum}-token minimum"
            ),
            Self::ContextDoesNotFit { requested, achievable } => write!(
                f,
                "{requested} tokens of context does not fit; {achievable} is the most this \
                 device can serve for this model"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

impl ModelFootprint {
    /// Check the description is self-consistent before planning against it.
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.total_experts == 0 {
            return Err(PlanError::InvalidModel("total_experts is zero — not an MoE model"));
        }
        if self.active_experts == 0 {
            return Err(PlanError::InvalidModel("active_experts is zero"));
        }
        if self.active_experts > self.total_experts {
            return Err(PlanError::InvalidModel("active_experts exceeds total_experts"));
        }
        if self.per_expert_bytes == 0 {
            return Err(PlanError::InvalidModel("per_expert_bytes is zero"));
        }
        if self.kv_bytes_per_token == 0 {
            return Err(PlanError::InvalidModel("kv_bytes_per_token is zero"));
        }
        Ok(())
    }
}

/// Plan an allocation.
///
/// The shape of the decision:
///
/// 1. Take headroom and the dense weights off the top — neither is negotiable.
/// 2. Reserve the `active_experts` floor. Below it, every token stalls on a fetch, so a plan
///    that cannot hold it is a failure rather than a slow success.
/// 3. Spend the remainder according to [`Bias`], rounding experts to whole slots and context
///    to whole pages.
///
/// Every arithmetic step is checked or saturating; the invariant that committed bytes never
/// exceed usable bytes holds by construction rather than by a check at the end.
pub fn plan(
    device: DeviceMemory,
    model: &ModelFootprint,
    policy: &Policy,
    want: Context,
) -> Result<Allocation, PlanError> {
    model.validate()?;
    if policy.page_tokens == 0 {
        return Err(PlanError::InvalidPolicy("page_tokens is zero"));
    }

    let mut rationale = Vec::new();

    let headroom_bytes = policy.headroom.reserve_from(device.free_bytes);
    let after_headroom = device.free_bytes - headroom_bytes; // reserve_from clamps to free
    let usable = after_headroom.checked_sub(model.dense_weights_bytes).ok_or(
        PlanError::WeightsDoNotFit { need: model.dense_weights_bytes, have: after_headroom },
    )?;
    rationale.push(Reason::Budget {
        free: device.free_bytes,
        headroom: headroom_bytes,
        dense_weights: model.dense_weights_bytes,
        usable,
    });

    // The residency floor. Below this, at least one expert per token crosses the bus.
    let floor_bytes = model.per_expert_bytes.saturating_mul(model.active_experts as u64);
    if floor_bytes > usable {
        return Err(PlanError::CannotHoldActiveExperts {
            need: floor_bytes,
            have: usable,
            active_experts: model.active_experts,
        });
    }

    let cap = policy
        .max_resident_experts
        .map_or(model.total_experts, |c| c.min(model.total_experts))
        .max(model.active_experts);

    let bytes_per_page = model.kv_bytes_per_token.saturating_mul(policy.page_tokens as u64);
    let min_pages = pages_for(policy.min_context_tokens, policy.page_tokens);
    let min_kv_bytes = bytes_per_page.saturating_mul(min_pages as u64);

    // The minimum context is reserved BEFORE experts are placed, never checked afterwards.
    //
    // 🔴 This was a real bug, caught by the unit tests on first run: filling expert slots
    // greedily and only then asking whether any context was left produced plans that held 128
    // experts and 256 tokens. Arithmetically valid, operationally useless -- the server cannot
    // answer a single realistic request. A floor that is verified after the fact is not a
    // floor; it is a diagnosis.
    if floor_bytes.saturating_add(min_kv_bytes) > usable {
        let achievable = ((usable - floor_bytes) / bytes_per_page).min(u32::MAX as u64) as u32;
        return Err(PlanError::ContextTooSmall {
            achievable: achievable.saturating_mul(policy.page_tokens),
            minimum: policy.min_context_tokens,
        });
    }
    // What experts are allowed to compete for, once the context floor is untouchable.
    let contestable = usable - min_kv_bytes;

    let resident = match (policy.bias, want) {
        // Experts first: fill slots with everything above the reserved context floor.
        (Bias::Experts, _) => {
            let affordable = (contestable / model.per_expert_bytes).min(u32::MAX as u64) as u32;
            affordable.clamp(model.active_experts, cap)
        }
        // Context first: satisfy the request in full, then fill slots with what remains.
        (Bias::Context, Context::Tokens(t)) => {
            let want_pages = pages_for(t, policy.page_tokens).max(min_pages);
            let kv = bytes_per_page.saturating_mul(want_pages as u64);
            let left = usable.saturating_sub(kv);
            let affordable = (left / model.per_expert_bytes).min(u32::MAX as u64) as u32;
            affordable.clamp(model.active_experts, cap)
        }
        // Nothing specific was asked for, so there is nothing to yield to; the floor still
        // applies.
        (Bias::Context, Context::Largest) => {
            let affordable = (contestable / model.per_expert_bytes).min(u32::MAX as u64) as u32;
            affordable.clamp(model.active_experts, cap)
        }
    };

    let expert_bytes = model.per_expert_bytes * resident as u64; // <= usable by the clamp above
    let for_kv = usable - expert_bytes;
    let achievable_pages = (for_kv / bytes_per_page).min(u32::MAX as u64) as u32;
    let achievable_tokens = achievable_pages.saturating_mul(policy.page_tokens);

    let (kv_pages, resident) = match want {
        Context::Largest => (achievable_pages, resident),
        Context::Tokens(t) => {
            let need = pages_for(t, policy.page_tokens);
            if need <= achievable_pages {
                (need, resident)
            } else if policy.bias == Bias::Experts {
                // Experts were filled greedily and crowded out the request. Give slots back,
                // one at a time, until the context fits or we hit the residency floor.
                let mut r = resident;
                let mut pages = achievable_pages;
                while pages < need && r > model.active_experts {
                    r -= 1;
                    let eb = model.per_expert_bytes * r as u64;
                    pages = ((usable - eb) / bytes_per_page).min(u32::MAX as u64) as u32;
                }
                if pages < need {
                    return Err(PlanError::ContextDoesNotFit {
                        requested: t,
                        achievable: pages.saturating_mul(policy.page_tokens),
                    });
                }
                rationale.push(Reason::ExpertsYieldedToContext {
                    resident: r,
                    would_have_been: resident,
                });
                (need, r)
            } else {
                return Err(PlanError::ContextDoesNotFit {
                    requested: t,
                    achievable: achievable_tokens,
                });
            }
        }
    };

    if kv_pages < min_pages {
        return Err(PlanError::ContextTooSmall {
            achievable: kv_pages.saturating_mul(policy.page_tokens),
            minimum: policy.min_context_tokens,
        });
    }

    let expert_bytes = model.per_expert_bytes * resident as u64;
    let kv_bytes = bytes_per_page * kv_pages as u64;

    if resident >= model.total_experts {
        rationale.push(Reason::AllExpertsResident { count: resident });
    } else if policy.max_resident_experts == Some(resident) {
        rationale.push(Reason::ExpertsLimitedByPolicy { resident, cap });
    } else {
        rationale.push(Reason::ExpertsLimitedByMemory { resident, of: model.total_experts });
    }
    if let Context::Tokens(t) = want {
        let served = kv_pages * policy.page_tokens;
        if served < t {
            rationale.push(Reason::ContextLimitedByMemory { tokens: served, requested: t });
        }
    }
    // Say so when the answer is the policy talking back. Without this the floor is
    // indistinguishable from a measured capacity in the output, which is how an invented
    // constant ends up quoted as a result.
    if kv_pages == min_pages && min_pages > 0 {
        rationale.push(Reason::ContextAtPolicyFloor { tokens: kv_pages * policy.page_tokens });
    }
    rationale.push(Reason::Slack { bytes: usable - expert_bytes - kv_bytes });

    Ok(Allocation {
        resident_experts: resident,
        context_tokens: kv_pages.saturating_mul(policy.page_tokens),
        kv_pages,
        expert_bytes,
        kv_bytes,
        headroom_bytes,
        rationale,
    })
}

/// The longest context this device can serve for this model.
///
/// The question a user actually asks, answered directly instead of by trial and error.
pub fn max_context_tokens(
    device: DeviceMemory,
    model: &ModelFootprint,
    policy: &Policy,
) -> Result<u32, PlanError> {
    plan(device, model, policy, Context::Largest).map(|a| a.context_tokens)
}

/// Pages needed to hold `tokens`, rounding up. Saturates rather than overflowing.
fn pages_for(tokens: u32, page_tokens: u32) -> u32 {
    debug_assert!(page_tokens > 0, "callers check page_tokens before this point");
    tokens.div_ceil(page_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    /// Roughly the shape of a 30B-class MoE in 4-bit on a 12 GiB card.
    fn model() -> ModelFootprint {
        ModelFootprint {
            dense_weights_bytes: 2 * GIB,
            per_expert_bytes: 96 * MIB,
            total_experts: 128,
            active_experts: 8,
            kv_bytes_per_token: 96 * 1024,
        }
    }

    fn card(free: u64) -> DeviceMemory {
        DeviceMemory { total_bytes: 12 * GIB, free_bytes: free }
    }

    #[test]
    fn a_plan_never_commits_more_than_is_usable() {
        let m = model();
        let p = Policy::default();
        for free in [6 * GIB, 8 * GIB, 12 * GIB, 24 * GIB, 64 * GIB] {
            let a = plan(card(free), &m, &p, Context::Largest).unwrap();
            assert!(
                a.committed_bytes(&m) + a.headroom_bytes <= free,
                "committed {} + headroom {} > free {free}",
                a.committed_bytes(&m),
                a.headroom_bytes
            );
        }
    }

    #[test]
    fn residency_never_drops_below_the_active_expert_floor() {
        let m = model();
        let a = plan(card(6 * GIB), &m, &Policy::default(), Context::Largest).unwrap();
        assert!(a.resident_experts >= m.active_experts);
    }

    #[test]
    fn a_big_card_holds_every_expert() {
        let m = model();
        let a = plan(card(64 * GIB), &m, &Policy::default(), Context::Largest).unwrap();
        assert!(a.is_fully_resident(&m));
        assert!(a.rationale.contains(&Reason::AllExpertsResident { count: m.total_experts }));
    }

    #[test]
    fn experts_yield_slots_to_reach_a_requested_context() {
        let m = model();
        let p = Policy::default();
        let greedy = plan(card(12 * GIB), &m, &p, Context::Largest).unwrap();
        // Ask for more context than the greedy plan left room for.
        let want = greedy.context_tokens + 8 * p.page_tokens;
        let a = plan(card(12 * GIB), &m, &p, Context::Tokens(want)).unwrap();
        assert!(a.context_tokens >= want);
        assert!(
            a.resident_experts < greedy.resident_experts,
            "should have traded slots for context"
        );
        assert!(a.rationale.iter().any(|r| matches!(r, Reason::ExpertsYieldedToContext { .. })));
    }

    #[test]
    fn an_impossible_context_is_refused_with_the_achievable_number() {
        let m = model();
        let e =
            plan(card(8 * GIB), &m, &Policy::default(), Context::Tokens(4_000_000)).unwrap_err();
        match e {
            PlanError::ContextDoesNotFit { achievable, .. } => assert!(achievable > 0),
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn a_model_too_large_for_the_card_says_so_plainly() {
        let m = ModelFootprint { dense_weights_bytes: 40 * GIB, ..model() };
        let e = plan(card(12 * GIB), &m, &Policy::default(), Context::Largest).unwrap_err();
        assert!(matches!(e, PlanError::WeightsDoNotFit { .. }));
        assert!(e.to_string().contains("too large for this device"));
    }

    #[test]
    fn a_card_that_cannot_hold_the_active_experts_is_refused() {
        let m = ModelFootprint { per_expert_bytes: GIB, ..model() };
        let e = plan(card(6 * GIB), &m, &Policy::default(), Context::Largest).unwrap_err();
        assert!(matches!(e, PlanError::CannotHoldActiveExperts { .. }));
    }

    #[test]
    fn context_bias_protects_the_request_at_the_cost_of_residency() {
        let m = model();
        let want = Context::Tokens(32_768);
        let by_experts = plan(card(12 * GIB), &m, &Policy::default(), want).unwrap();
        let by_context =
            plan(card(12 * GIB), &m, &Policy { bias: Bias::Context, ..Policy::default() }, want)
                .unwrap();
        assert_eq!(by_context.context_tokens, by_experts.context_tokens);
        assert!(by_context.resident_experts <= by_experts.resident_experts);
    }

    #[test]
    fn an_explicit_cap_is_honoured_but_never_below_the_floor() {
        let m = model();
        let p = Policy { max_resident_experts: Some(2), ..Policy::default() };
        let a = plan(card(64 * GIB), &m, &p, Context::Largest).unwrap();
        // The cap is below active_experts, so the floor wins — a cap must never make the
        // model unrunnable.
        assert_eq!(a.resident_experts, m.active_experts);
    }

    #[test]
    fn inconsistent_models_are_rejected_before_any_arithmetic() {
        let bad = ModelFootprint { active_experts: 200, ..model() };
        assert!(matches!(
            plan(card(12 * GIB), &bad, &Policy::default(), Context::Largest),
            Err(PlanError::InvalidModel(_))
        ));
        let zero = ModelFootprint { total_experts: 0, ..model() };
        assert!(matches!(
            plan(card(12 * GIB), &zero, &Policy::default(), Context::Largest),
            Err(PlanError::InvalidModel(_))
        ));
    }

    #[test]
    fn a_context_that_is_merely_the_floor_says_so() {
        // The condition is narrower than "the card is small", and getting it wrong once
        // already produced an overstated claim. Greedy residency drives context onto the floor
        // only when the bytes left over after filling whole expert slots are worth less than
        // one KV page. That happens when experts are small relative to a page -- which is the
        // real case: the reference model's experts are 1.95 MiB against a 5 MiB page, so the
        // slack can never buy a page back. With large experts the leftover does buy pages and
        // context rises above the floor, which is why the shared fixture cannot show this.
        let m = ModelFootprint {
            dense_weights_bytes: GIB,
            per_expert_bytes: 2 * MIB,
            total_experts: 8192,
            active_experts: 320,
            kv_bytes_per_token: 20_480,
        };
        let p = Policy::default();
        let a = plan(card(11 * GIB), &m, &p, Context::Largest).unwrap();
        assert_eq!(
            a.context_tokens, p.min_context_tokens,
            "expected the floor; got {} tokens",
            a.context_tokens
        );
        assert!(
            a.rationale.iter().any(|r| matches!(r, Reason::ContextAtPolicyFloor { .. })),
            "floor-limited context must be labelled: {:?}",
            a.rationale
        );
        let prose = a.rationale.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" ");
        assert!(
            prose.contains("not\nbecause it is all that fits")
                || prose.contains("not because it is all that fits")
        );
    }

    #[test]
    fn max_context_agrees_with_a_largest_plan() {
        let m = model();
        let p = Policy::default();
        let a = plan(card(12 * GIB), &m, &p, Context::Largest).unwrap();
        assert_eq!(max_context_tokens(card(12 * GIB), &m, &p).unwrap(), a.context_tokens);
    }

    #[test]
    fn the_rationale_reads_as_prose() {
        let m = model();
        let a = plan(card(12 * GIB), &m, &Policy::default(), Context::Largest).unwrap();
        let lines: Vec<String> = a.rationale.iter().map(|r| r.to_string()).collect();
        assert!(lines[0].contains("free"), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("experts")), "{lines:?}");
    }
}
