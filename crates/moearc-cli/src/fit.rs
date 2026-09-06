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
    Allocation, Context, DeviceMemory, ModelFootprint, Policy, Reason, plan as engine_plan,
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
        /// Whether `context_tokens` is the policy minimum rather than what the card could
        /// serve.
        ///
        /// 🔴 The engine is explicit that these are different things, and on a real directory
        /// the floor is the common case, not an edge one: experts are small next to a KV page,
        /// so a greedy residency fill leaves exactly the reserved minimum and no more. Four
        /// rows all reading the same context then look like a constant the tool forgot to
        /// compute. They are not — but nothing on the row says which of the two numbers it is,
        /// so the panel has to.
        context_at_floor: bool,
        /// The planner's sentence for [`Reason::ExpertsYieldedToContext`], when it fired.
        ///
        /// 🔴 The one line of the rationale that is pulled out and shown beside the table
        /// rather than left in the detail pane, because it is *the* tradeoff this screen
        /// exists to make visible: how many expert slots the requested context cost. It is
        /// extracted by matching the typed `Reason`, never by searching the rendered strings,
        /// and the sentence is the engine's own — rewriting it here would give the same fact
        /// two spellings that could drift.
        ///
        /// `None` is the ordinary case: at [`Context::Largest`] nothing is yielded, because
        /// nothing specific was asked for.
        yield_note: Option<String>,
        /// The planner's own reasoning, one line per step.
        rationale: Vec<String>,
    },
    /// No allocation satisfies the request.
    ///
    /// `reason` is the full sentence — usually the engine's, which names the shortfall in
    /// gigabytes and, where it can, what would fix it. `headline` is the three words that fit
    /// in a table cell, because "will not fit" is right for a model too large for the card and
    /// wrong for one asked to do something it was never trained to do.
    DoesNotFit { headline: &'static str, reason: String },
}

impl Fit {
    pub fn fits(&self) -> bool {
        matches!(self.outcome, FitOutcome::Fits { .. })
    }

    /// Whether this row's context is the configured minimum rather than a capacity.
    pub fn context_at_floor(&self) -> bool {
        matches!(self.outcome, FitOutcome::Fits { context_at_floor: true, .. })
    }

    /// What the requested context cost in expert slots, in the planner's own words.
    pub fn yield_note(&self) -> Option<&str> {
        match &self.outcome {
            FitOutcome::Fits { yield_note, .. } => yield_note.as_deref(),
            FitOutcome::DoesNotFit { .. } => None,
        }
    }

    /// `"4,128 / 4,608"` — resident slots over the model's total, for a column under a
    /// heading that already says what the numbers are.
    ///
    /// Split out from [`Self::summary`] because the "what will fit" table now has a header row.
    /// Repeating "experts resident" and "ctx" on every line cost sixteen columns that the host
    /// tier needed, and a header says it once.
    pub fn residency_cell(&self) -> String {
        match &self.outcome {
            FitOutcome::Fits { resident_experts, total_experts, .. } => format!(
                "{} / {}",
                crate::format::count(*resident_experts as i64),
                crate::format::count(*total_experts as i64)
            ),
            FitOutcome::DoesNotFit { headline, .. } => (*headline).to_string(),
        }
    }

    /// `"2,048"` — the planned context, in tokens.
    pub fn context_cell(&self) -> String {
        match &self.outcome {
            FitOutcome::Fits { context_tokens, .. } => crate::format::count(*context_tokens as i64),
            FitOutcome::DoesNotFit { .. } => "—".to_string(),
        }
    }
}

/// Column widths for the "what will fit" table, measured from the rows that will go in it.
///
/// A constant was fine while the models were fixtures with short handles. Real handles come
/// from real filenames — `olmoe-1b-7b-0924-instruct` is 25 characters, twice the fixture it
/// replaced — and a fixed column either truncates the name or leaves half the row empty. Both
/// renderers take their widths from here so the plain output and the interface stay one table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Columns {
    pub id: usize,
    pub quant: usize,
    pub size: usize,
    /// Widest `"resident / total"` any row could hold.
    pub residency: usize,
    /// Widest context length any row could hold.
    pub ctx: usize,
}

impl Columns {
    /// Floors, so a one-model directory still produces a table rather than a ragged line.
    const MIN: Self = Self { id: 13, quant: 5, size: 9, residency: 9, ctx: 5 };

    /// The host tier column, fixed at the widest label [`Tier`] can produce.
    ///
    /// 🔴 Fixed rather than measured from the rows, which is the opposite of every other column
    /// here and is deliberate: the tier is the one cell that changes as the user moves the
    /// budget. A measured width would make the whole table reflow on every key press, so the
    /// numbers a viewer is trying to read would slide sideways underneath them. A little
    /// trailing space is the cheaper defect.
    pub const TIER: usize = 21;

    pub fn of(models: &[ModelCard]) -> Self {
        let mut c = Self::MIN;
        for m in models {
            // Characters, not bytes: a handle is a filename and filenames are not ASCII.
            c.id = c.id.max(m.id.chars().count());
            c.quant = c.quant.max(m.quant.chars().count());
            c.size = c.size.max(crate::format::bytes(m.file_bytes).chars().count());
            // Sized from the model rather than from the plan, so these two columns are also
            // stable while the budget moves: the widest residency a row can ever print is the
            // slot count against itself, and the widest context is the trained one.
            let slots = crate::format::count(m.expert_slots_total as i64).chars().count();
            c.residency = c.residency.max(slots * 2 + 3);
            c.ctx =
                c.ctx.max(crate::format::count(m.trained_context_tokens as i64).chars().count());
        }
        c
    }
}

/// The KV cache's element width.
///
/// 🔴 One variant today, and the type exists *because* of that rather than in spite of it.
/// `moearc-model` reads `kv_bytes_per_token` **at f16** out of the header, so f16 is the only
/// width any number on this screen is entitled to describe. Quantised KV (q8_0) halves it
/// again and is queued behind the sliding-window work in the engine's `moe.rs`/`kv.rs`, which
/// this crate does not own. When it lands it lands as a variant here and a scale in
/// [`footprint`], and no screen changes.
///
/// There is deliberately **no control for it on screen**. A dial that does nothing yet is a
/// promise, and a promise in a demo is indistinguishable from a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvPrecision {
    #[default]
    F16,
}

impl KvPrecision {
    /// Scale the header's f16 per-token figure to this width.
    fn per_token(self, f16_bytes: u64) -> u64 {
        match self {
            Self::F16 => f16_bytes,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
        }
    }
}

fn device_memory(d: &DeviceRow) -> DeviceMemory {
    DeviceMemory { total_bytes: d.total_bytes, free_bytes: d.free_bytes }
}

/// The planner's view of the model.
///
/// 🔴 `total_experts` here is the model's **residency slot** count, one per *(block, expert)*
/// pair, because that is what `per_expert_bytes` sizes and what the cache pages. Passing the
/// per-block expert count instead would understate a 36-block model by 36x and produce plans
/// that allocate a thirty-sixth of the memory they need.
fn footprint(card: &ModelCard) -> ModelFootprint {
    ModelFootprint {
        dense_weights_bytes: card.dense_weights_bytes,
        per_expert_bytes: card.per_expert_bytes,
        total_experts: card.expert_slots_total,
        active_experts: card.expert_slots_active,
        kv_bytes_per_token: KvPrecision::default().per_token(card.kv_bytes_per_token),
    }
}

fn want(ctx: Option<u32>) -> Context {
    ctx.map_or(Context::Largest, Context::Tokens)
}

/// The context lengths the dial steps through.
///
/// Powers of two rather than a linear step, because that is the ladder the field is written in
/// and because the length this product is aimed at — **64K** — is a rung rather than something
/// a user has to hunt for. `None` is the first stop and is not "zero": it is *largest that
/// fits*, which is a different question and the one `moearc` answers when nobody asks.
///
/// 🔴 The ladder deliberately runs past what an 11.33 GiB card can serve at f16. A dial that
/// stops where today's hardware stops would hide the tradeoff instead of showing it — and the
/// planner already refuses an impossible length with a sentence that names the achievable one,
/// which is better copy than a control that will not move.
pub const CONTEXT_LADDER: [Option<u32>; 9] = [
    None,
    Some(2_048),
    Some(4_096),
    Some(8_192),
    Some(16_384),
    Some(32_768),
    Some(65_536),
    Some(131_072),
    Some(262_144),
];

/// Where a context length sits on [`CONTEXT_LADDER`].
///
/// A value that is not a rung — `--ctx 3000` — resolves to the first rung that covers it, so
/// the dial steps from where the user is rather than jumping to the bottom.
pub fn ladder_index(ctx: Option<u32>) -> usize {
    match ctx {
        None => 0,
        Some(t) => CONTEXT_LADDER
            .iter()
            .position(|r| r.is_some_and(|rung| rung >= t))
            .unwrap_or(CONTEXT_LADDER.len() - 1),
    }
}

/// The rung `delta` steps away, clamped to the ends.
///
/// Clamped rather than wrapped: a dial that jumps from 262,144 back to "largest that fits" on
/// one more key press has no ends, and a control with no ends cannot be driven by feel.
pub fn ladder_step(ctx: Option<u32>, delta: isize) -> Option<u32> {
    let at = ladder_index(ctx) as isize;
    let next = (at + delta).clamp(0, CONTEXT_LADDER.len() as isize - 1) as usize;
    CONTEXT_LADDER[next]
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

    // Checked before the planner, because the planner cannot catch it: it reasons about bytes,
    // and the bytes are there. A 4,096-token model asked for 32,768 allocates 128 KV pages
    // quite happily and then answers from positions it has never seen. Trimming the request to
    // 4,096 and reporting success would be the worst kind of success; the request is refused
    // instead, and the refusal names the model's own limit.
    if let Some(asked) = ctx
        && card.trained_context_tokens > 0
        && asked > card.trained_context_tokens
    {
        return Fit {
            model: card.id.clone(),
            requested_ctx: ctx,
            slot_override,
            calibrated: false,
            outcome: FitOutcome::DoesNotFit {
                headline: "past its trained context",
                reason: format!(
                    "{} tokens were asked for and this model was trained for {}. The pages \
                     would allocate; the answers past that point would not mean anything.",
                    crate::format::count(asked as i64),
                    crate::format::count(card.trained_context_tokens as i64)
                ),
            },
        };
    }

    let outcome = match engine_plan(mem, &model, &policy, want(ctx)) {
        Ok(a) => {
            let (a, capped) = cap_to_trained_context(mem, &model, &policy, card, ctx, a);
            let ceiling = ceiling_tokens(device, card, &policy, a.context_tokens);
            fits(&a, card, ceiling, capped)
        }
        Err(e) => FitOutcome::DoesNotFit { headline: "will not fit", reason: e.to_string() },
    };
    Fit { model: card.id.clone(), requested_ctx: ctx, slot_override, calibrated: false, outcome }
}

/// Hold the reported context down to what the model was actually trained for.
///
/// [`Context::Largest`] asks for every KV page the card can hold, and the planner has no idea
/// what the model can use. Measured: `olmoe-1b-7b` is a **4,096-token** model, and a B580 with
/// 11.3 GiB free has room for 47,360 tokens of its KV cache. Printing 47,360 in a column
/// headed "what will fit" would be a claim about the model that the model does not make — the
/// single most quotable wrong number this screen could produce.
///
/// Only the *unrequested* case is capped. A context the user typed is still answered by the
/// planner, so `--ctx 8192` against a 4,096-token model fails there rather than being quietly
/// trimmed here — which is the same rule `docs/ux.md` applies to every other silent success.
fn cap_to_trained_context(
    mem: DeviceMemory,
    model: &ModelFootprint,
    policy: &Policy,
    card: &ModelCard,
    requested: Option<u32>,
    planned: Allocation,
) -> (Allocation, bool) {
    if requested.is_some()
        || card.trained_context_tokens == 0
        || planned.context_tokens <= card.trained_context_tokens
    {
        return (planned, false);
    }
    // Floored to a whole page, so the re-plan lands on or below the trained ceiling rather
    // than rounding back up over it.
    let page = policy.page_tokens.max(1);
    let target = (card.trained_context_tokens / page) * page;
    if target == 0 {
        return (planned, false);
    }
    match engine_plan(mem, model, policy, Context::Tokens(target)) {
        Ok(capped) => (capped, true),
        // Asking for less than already fits cannot fail; if it somehow does, keep the plan we
        // have rather than losing the row entirely.
        Err(_) => (planned, false),
    }
}

fn fits(
    a: &Allocation,
    card: &ModelCard,
    ceiling_tokens: Option<u32>,
    context_capped: bool,
) -> FitOutcome {
    let context_at_floor =
        a.rationale.iter().any(|r| matches!(r, Reason::ContextAtPolicyFloor { .. }));
    let yield_note = a
        .rationale
        .iter()
        .find(|r| matches!(r, Reason::ExpertsYieldedToContext { .. }))
        .map(ToString::to_string);
    let mut rationale: Vec<String> = a.rationale.iter().map(ToString::to_string).collect();
    if context_capped {
        // Appended by this crate, not by the engine. The engine reasons about bytes; this is
        // the one fact about the model that is not in its footprint.
        rationale.push(format!(
            "context held at {} tokens, the longest this model was trained for — the card \
             could hold more KV pages than the model can use",
            crate::format::count(card.trained_context_tokens as i64)
        ));
    }
    FitOutcome::Fits {
        resident_experts: a.resident_experts,
        total_experts: card.expert_slots_total,
        context_tokens: a.context_tokens,
        kv_pages: a.kv_pages,
        expert_bytes: a.expert_bytes,
        kv_bytes: a.kv_bytes,
        headroom_bytes: a.headroom_bytes,
        ceiling_tokens,
        context_at_floor,
        yield_note,
        rationale,
    }
}

/// The longest context this card could serve for this model, at minimum expert residency.
///
/// Asked of the planner rather than derived, so it cannot drift from the plan it sits beside.
/// It is the answer to "what am I giving up by keeping experts resident?", which is the
/// question the headline number provokes and would otherwise leave hanging.
fn ceiling_tokens(
    device: &DeviceRow,
    card: &ModelCard,
    policy: &Policy,
    planned_tokens: u32,
) -> Option<u32> {
    let floor = Policy { max_resident_experts: Some(card.expert_slots_active), ..*policy };
    let reachable = engine_plan(device_memory(device), &footprint(card), &floor, Context::Largest)
        .ok()
        .map(|a| a.context_tokens)?;
    // Same ceiling the plan itself is held to, and for the same reason.
    let reachable = if card.trained_context_tokens == 0 {
        reachable
    } else {
        reachable.min(card.trained_context_tokens)
    };
    // Once both are at the model's own limit there is no tradeoff left to describe, and
    // printing "ceiling 4,096" beside "context 4,096" invites the reader to look for a
    // difference that is not there.
    (reachable > planned_tokens).then_some(reachable)
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
        let FitOutcome::DoesNotFit { reason, .. } = &f.outcome else {
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
    fn a_context_beyond_the_model_is_refused_even_when_the_memory_is_there() {
        // The one the planner cannot catch. `mixtral-8x7b` is a 32,768-token fixture; the
        // bytes for 65,536 exist on this card, so the engine would say yes.
        let m = model("mixtral-8x7b");
        assert_eq!(m.trained_context_tokens, 32_768);
        let f = plan(&card(), &m, Some(65_536));
        let FitOutcome::DoesNotFit { headline, reason } = &f.outcome else {
            panic!("a context the model never saw is not a fit: {:?}", f.outcome)
        };
        assert_eq!(*headline, "past its trained context", "and not `will not fit`, which it does");
        assert!(reason.contains("32,768"), "the refusal names the model's limit: {reason}");
        // At its own limit it plans normally.
        assert!(plan(&card(), &m, Some(32_768)).fits());
    }

    #[test]
    fn a_context_at_the_policy_minimum_is_flagged_as_one() {
        // The engine distinguishes "this is all that fits" from "experts took everything above
        // the configured minimum". Dropping that distinction on the way to the screen would
        // turn a policy restatement into a capacity claim.
        let f = plan(&card(), &model("qwen3-30b-a3b"), None);
        let FitOutcome::Fits { context_tokens, context_at_floor, .. } = f.outcome else {
            panic!("expected a fit")
        };
        assert_eq!(context_at_floor, context_tokens == 2_048);
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
    fn a_requested_context_reports_what_it_cost_in_expert_slots() {
        // The tradeoff, in the planner's own sentence. `Context::Largest` asks for nothing
        // specific, so nothing is yielded and there is nothing to report.
        let (d, m) = (card(), model("qwen3-30b-a3b"));
        assert_eq!(plan(&d, &m, None).yield_note(), None);
        let long = plan(&d, &m, Some(32_768));
        let note = long.yield_note().expect("32k costs slots here");
        assert!(note.contains("gave up"), "{note}");
        assert!(note.contains("expert slots"), "{note}");
    }

    #[test]
    fn the_bias_policy_is_not_the_axis_the_dial_moves() {
        // 🔴 Measured, not assumed, and it is why the interface has a context dial and no bias
        // toggle. `Bias::Experts` and `Bias::Context` produce **byte-identical** allocations for
        // every model and every rung checked here: at `Context::Largest` the two take branches
        // that are the same arithmetic, and with a requested context the experts-first
        // yield-back loop lands on exactly the maximum residency the context-first path
        // computes directly. A toggle for it would be a control a viewer could not see working.
        //
        // The axis that *does* move the plan is `Largest` vs `Tokens(n)` — which is the dial.
        // Only the *explanation* differs between the two biases: the experts-first path emits
        // `Reason::ExpertsYieldedToContext` and the context-first path has nothing to yield.
        use moearc_engine::memory::{Bias, DeviceMemory, plan as engine};
        let d = card();
        let mem = DeviceMemory { total_bytes: d.total_bytes, free_bytes: d.free_bytes };
        for id in ["qwen3-30b-a3b", "gpt-oss-20b"] {
            let m = footprint(&model(id));
            for want in [Context::Largest, Context::Tokens(8_192), Context::Tokens(32_768)] {
                let by_experts =
                    engine(mem, &m, &Policy { bias: Bias::Experts, ..Policy::default() }, want)
                        .expect("plans");
                let by_context =
                    engine(mem, &m, &Policy { bias: Bias::Context, ..Policy::default() }, want)
                        .expect("plans");
                assert_eq!(
                    (
                        by_experts.resident_experts,
                        by_experts.context_tokens,
                        by_experts.kv_pages,
                        by_experts.expert_bytes,
                        by_experts.kv_bytes
                    ),
                    (
                        by_context.resident_experts,
                        by_context.context_tokens,
                        by_context.kv_pages,
                        by_context.expert_bytes,
                        by_context.kv_bytes
                    ),
                    "{id} at {want:?}: the bias changed the plan after all, so the interface \
                     needs a control for it"
                );
            }
        }
    }

    #[test]
    fn the_kv_width_is_the_one_the_header_was_read_at() {
        // Guards the seam against being wired up before the engine can honour it: today the
        // only width is the one `moearc-model` measured, so the scale is the identity.
        let m = model("qwen3-30b-a3b");
        assert_eq!(
            footprint(&m).kv_bytes_per_token,
            m.kv_bytes_per_token,
            "a scale applied before the engine supports it would misplan every context"
        );
        assert_eq!(KvPrecision::default().label(), "f16");
    }

    #[test]
    fn the_context_ladder_is_ascending_and_reaches_the_target() {
        // 64K is the length the product is aimed at, so it is a stop on the dial rather than
        // something a user has to land on with a linear step.
        let rungs: Vec<u32> = CONTEXT_LADDER.iter().flatten().copied().collect();
        assert!(rungs.windows(2).all(|w| w[0] < w[1]), "{rungs:?}");
        assert!(rungs.contains(&65_536));
        assert_eq!(CONTEXT_LADDER[0], None, "the first stop is `largest that fits`");
    }

    #[test]
    fn every_rung_maps_back_to_its_own_position() {
        for (i, rung) in CONTEXT_LADDER.iter().enumerate() {
            assert_eq!(ladder_index(*rung), i, "{rung:?}");
        }
        // A context typed with --ctx that is not on the ladder lands on the first rung that
        // covers it, so the next press is a step rather than a jump back to the bottom.
        assert_eq!(ladder_index(Some(3_000)), ladder_index(Some(4_096)));
        assert_eq!(ladder_index(Some(4_000_000)), CONTEXT_LADDER.len() - 1);
    }

    #[test]
    fn context_and_residency_trade_against_each_other_along_the_ladder() {
        // The coupling the second dial exists to show, asserted rather than asserted-at. The
        // arithmetic is the engine's; this only checks that walking the dial actually moves
        // both numbers, and in opposite directions.
        let (d, m) = (card(), model("qwen3-30b-a3b"));
        let mut last: Option<(u32, u32)> = None;
        for rung in CONTEXT_LADDER.iter().skip(1).flatten() {
            let FitOutcome::Fits { resident_experts, context_tokens, .. } =
                plan(&d, &m, Some(*rung)).outcome
            else {
                break; // the ladder runs past what this card can serve, which is the point
            };
            if let Some((slots, ctx)) = last {
                assert!(context_tokens > ctx, "a longer context should be longer");
                assert!(resident_experts <= slots, "and it should cost slots, never gain them");
            }
            last = Some((resident_experts, context_tokens));
        }
        assert!(last.is_some(), "at least one rung must plan");
    }

    #[test]
    fn the_tier_column_is_wide_enough_for_every_word_it_can_hold() {
        // A fixed width is only safe while it is checked against the strings it has to hold.
        use moearc_engine::host_budget::Tier;
        for tier in [Tier::RunsFromRam, Tier::RunsPagesFromDisk, Tier::WillNotFit] {
            assert!(
                tier.label().chars().count() <= Columns::TIER,
                "`{}` does not fit the {}-column host tier",
                tier.label(),
                Columns::TIER
            );
        }
    }

    #[test]
    fn the_residency_and_context_columns_do_not_move_when_a_plan_changes() {
        // The property the video depends on: dragging the budget must not reflow the table.
        // Both widths come from the model's own geometry, so a different plan cannot widen
        // them.
        let (d, m) = (card(), model("qwen3-30b-a3b"));
        let wide = Columns::of(std::slice::from_ref(&m));
        for ctx in [None, Some(8_192), Some(32_768)] {
            let f = plan(&d, &m, ctx);
            assert!(f.residency_cell().chars().count() <= wide.residency, "{}", f.residency_cell());
            assert!(f.context_cell().chars().count() <= wide.ctx, "{}", f.context_cell());
        }
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
