//! "What will fit" — the answer `moearc` gives with no arguments.
//!
//! This is where `docs/ux.md`'s fourth rule lands: *the split is computed, not configured*.
//! The arithmetic is not done here. It belongs to [`moearc_engine::memory`], which owns the
//! rule that expert residency and context length compete for the same bytes; this module's
//! only job is to translate between that planner's vocabulary and the interface's.
//!
//! Two translations are worth naming, because they are the ones a user notices:
//!
//! * **No `--ctx` means [`Context::Largest`], not a default constant.** A tool that silently
//!   assumes 8k and reports success has answered a question the user did not ask. Asking the
//!   planner for the largest context that fits is both more useful and more honest.
//! * **The planner's rationale is carried through verbatim.** `Reason` exists in the engine
//!   specifically so the interface can show its work, and dropping it here would put the
//!   reasoning behind a debug flag — which `docs/ux.md` rules out.

use moearc_engine::memory::{
    Allocation, Context, DeviceMemory, ModelFootprint, Policy, plan as engine_plan,
};
use serde::Serialize;

use crate::source::{DeviceRow, ModelCard};

/// Whether a model runs on this card, and on what terms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Fit {
    pub model: String,
    /// The context that was asked for. `None` means "whatever fits".
    pub requested_ctx: Option<u32>,
    /// Set when `--moe-cache` overrode the computed residency.
    pub slot_override: Option<u32>,
    /// False while the engine's headroom is still [`Policy::default`]'s provisional fraction
    /// rather than a measured one. Rendered wherever a plan is shown: a provisional number
    /// presented as a measurement is the failure `docs/calibration.md` exists to prevent.
    pub calibrated: bool,
    pub outcome: FitOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum FitOutcome {
    Fits {
        resident_experts: u32,
        total_experts: u32,
        context_tokens: u32,
        kv_pages: u32,
        expert_bytes: u64,
        kv_bytes: u64,
        headroom_bytes: u64,
        /// The context this card could serve if residency were dropped to the model's active
        /// -expert floor — the other end of the tradeoff, so the number above reads as a
        /// choice rather than a limit.
        ceiling_tokens: Option<u32>,
        /// The planner's own reasoning, one line per step.
        rationale: Vec<String>,
    },
    /// No allocation satisfies the request. `reason` is the engine's message, which names the
    /// shortfall in gigabytes and, where it can, what would fix it.
    DoesNotFit { reason: String },
}

impl Fit {
    pub fn fits(&self) -> bool {
        matches!(self.outcome, FitOutcome::Fits { .. })
    }

    /// One line, for a table cell or a plain-text row.
    pub fn summary(&self) -> String {
        match &self.outcome {
            FitOutcome::Fits { resident_experts, total_experts, context_tokens, .. } => format!(
                "{resident_experts}/{total_experts} experts resident · {} ctx",
                crate::format::count(*context_tokens as i64)
            ),
            FitOutcome::DoesNotFit { .. } => "will not fit".to_string(),
        }
    }
}

fn device_memory(d: &DeviceRow) -> DeviceMemory {
    DeviceMemory { total_bytes: d.total_bytes, free_bytes: d.free_bytes }
}

fn footprint(card: &ModelCard) -> ModelFootprint {
    ModelFootprint {
        dense_weights_bytes: card.dense_weights_bytes,
        per_expert_bytes: card.per_expert_bytes,
        total_experts: card.experts_total,
        active_experts: card.experts_active,
        kv_bytes_per_token: card.kv_bytes_per_token,
    }
}

fn want(ctx: Option<u32>) -> Context {
    ctx.map_or(Context::Largest, Context::Tokens)
}

/// Plan `card` onto `device`, letting the engine choose the split.
pub fn plan(device: &DeviceRow, card: &ModelCard, ctx: Option<u32>) -> Fit {
    build(device, card, ctx, Policy::default(), None)
}

/// Plan with residency pinned by `--moe-cache`.
///
/// The escape hatch from `docs/ux.md`. It is expressed as a *policy cap* rather than as a
/// separate code path, so an override still goes through the same planner and still fails
/// with the same messages — an escape hatch with its own arithmetic is a second
/// implementation waiting to disagree with the first.
pub fn plan_with_slot_override(
    device: &DeviceRow,
    card: &ModelCard,
    ctx: Option<u32>,
    slots: u32,
) -> Fit {
    let policy = Policy { max_resident_experts: Some(slots), ..Policy::default() };
    build(device, card, ctx, policy, Some(slots))
}

fn build(
    device: &DeviceRow,
    card: &ModelCard,
    ctx: Option<u32>,
    policy: Policy,
    slot_override: Option<u32>,
) -> Fit {
    let (mem, model) = (device_memory(device), footprint(card));
    let outcome = match engine_plan(mem, &model, &policy, want(ctx)) {
        Ok(a) => fits(&a, card, ceiling_tokens(device, card, &policy)),
        Err(e) => FitOutcome::DoesNotFit { reason: e.to_string() },
    };
    Fit { model: card.id.clone(), requested_ctx: ctx, slot_override, calibrated: false, outcome }
}

fn fits(a: &Allocation, card: &ModelCard, ceiling_tokens: Option<u32>) -> FitOutcome {
    FitOutcome::Fits {
        resident_experts: a.resident_experts,
        total_experts: card.experts_total,
        context_tokens: a.context_tokens,
        kv_pages: a.kv_pages,
        expert_bytes: a.expert_bytes,
        kv_bytes: a.kv_bytes,
        headroom_bytes: a.headroom_bytes,
        ceiling_tokens,
        rationale: a.rationale.iter().map(ToString::to_string).collect(),
    }
}

/// The longest context this card could serve for this model, at minimum expert residency.
///
/// Asked of the planner rather than derived, so it cannot drift from the plan it sits beside.
/// It is the answer to "what am I giving up by keeping experts resident?", which is the
/// question the headline number provokes and would otherwise leave hanging.
fn ceiling_tokens(device: &DeviceRow, card: &ModelCard, policy: &Policy) -> Option<u32> {
    let floor = Policy { max_resident_experts: Some(card.experts_active), ..*policy };
    engine_plan(device_memory(device), &footprint(card), &floor, Context::Largest)
        .ok()
        .map(|a| a.context_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{DeviceSource, ModelCatalog, StubCatalog, StubDeviceSource};

    fn card() -> DeviceRow {
        StubDeviceSource.detect().unwrap().devices.remove(0)
    }

    fn model(id: &str) -> ModelCard {
        StubCatalog.resolve(id).unwrap()
    }

    #[test]
    fn a_large_moe_model_fits_by_keeping_only_some_experts_resident() {
        let f = plan(&card(), &model("qwen3-30b-a3b"), None);
        let FitOutcome::Fits { resident_experts, total_experts, context_tokens, .. } = f.outcome
        else {
            panic!("expected a fit, got {:?}", f.outcome);
        };
        // The whole point of the expert cache: a 30B model on a 12 GiB card, with residency
        // between the per-token floor and the full set.
        assert!(resident_experts > 8, "should hold more than the active floor");
        assert!(resident_experts < total_experts, "and fewer than all of them");
        assert!(context_tokens > 0);
    }

    #[test]
    fn asking_for_no_context_asks_the_planner_for_the_largest_rather_than_assuming_one() {
        let f = plan(&card(), &model("qwen3-30b-a3b"), None);
        assert_eq!(f.requested_ctx, None);
        assert!(f.fits());
    }

    #[test]
    fn a_long_context_is_paid_for_in_expert_slots() {
        let (d, m) = (card(), model("qwen3-30b-a3b"));
        let FitOutcome::Fits { resident_experts: greedy, context_tokens: short, .. } =
            plan(&d, &m, None).outcome
        else {
            panic!("expected a fit")
        };
        let FitOutcome::Fits { resident_experts: yielded, context_tokens: long, .. } =
            plan(&d, &m, Some(32_768)).outcome
        else {
            panic!("expected a fit at 32k")
        };
        assert!(long > short, "asking for more context should get more context");
        assert!(yielded < greedy, "and it should cost expert slots");
    }

    #[test]
    fn the_reasoning_is_carried_through_not_summarised_away() {
        let f = plan(&card(), &model("qwen3-30b-a3b"), None);
        let FitOutcome::Fits { rationale, .. } = f.outcome else { panic!("expected a fit") };
        assert!(!rationale.is_empty());
        assert!(
            rationale.iter().any(|r| r.contains("headroom")),
            "the budget line is what makes the rest legible: {rationale:?}"
        );
    }

    #[test]
    fn a_model_too_large_for_the_card_names_the_shortfall_and_a_way_out() {
        let f = plan(&card(), &model("qwen3-235b-a22b"), None);
        let FitOutcome::DoesNotFit { reason } = &f.outcome else {
            panic!("expected a miss, got {:?}", f.outcome);
        };
        assert!(reason.contains("GiB"), "the shortfall is in bytes, not 'out of memory'");
        assert!(reason.contains("quantisation"), "and it says what would fix it: {reason}");
    }

    #[test]
    fn an_impossible_context_fails_rather_than_being_quietly_trimmed() {
        // Silently serving 3k when 4M was asked for would be the worst kind of success.
        let f = plan(&card(), &model("qwen3-30b-a3b"), Some(4_000_000));
        assert!(!f.fits());
    }

    #[test]
    fn the_ceiling_shows_what_residency_costs_in_context() {
        let f = plan(&card(), &model("qwen3-30b-a3b"), None);
        let FitOutcome::Fits { context_tokens, ceiling_tokens, .. } = f.outcome else {
            panic!("expected a fit")
        };
        let ceiling = ceiling_tokens.expect("minimum residency should always plan");
        assert!(ceiling > context_tokens, "dropping to the floor must buy context");
    }

    #[test]
    fn the_slot_override_is_honoured_and_recorded() {
        let f = plan_with_slot_override(&card(), &model("qwen3-30b-a3b"), None, 16);
        assert_eq!(f.slot_override, Some(16));
        let FitOutcome::Fits { resident_experts, .. } = f.outcome else { panic!("expected a fit") };
        assert_eq!(resident_experts, 16);
    }

    #[test]
    fn every_plan_is_flagged_uncalibrated_until_the_headroom_is_measured() {
        // Guards the docs/calibration.md rule where a reviewer will see it. If headroom is
        // ever measured on Arc, this test is what has to be changed on purpose.
        assert!(!plan(&card(), &model("gpt-oss-20b"), None).calibrated);
    }
}
