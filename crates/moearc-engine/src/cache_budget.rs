//! GPU cache budget planning: how the net GPU pool is split between the MoE expert cache
//! and the paged KV cache.
//!
//! Ported from FreeToken's `freetoken/engine/cache_budget.py` (Apache-2.0). Its logic is
//! pure integer byte arithmetic with no torch or device dependency, which makes it the
//! natural first component to own outright and the only one testable without hardware.
//!
//! Equivalence with the Python is *proven*, not assumed: `tools/gen_cache_budget_oracle.py`
//! runs the reference implementation over 4,133 structured and randomised inputs and records
//! every result — including the 1,900 it rejects — into `tests/data/`. The test in
//! `tests/cache_budget_oracle.rs` replays that fixture. The fixture is committed, so the
//! test is hermetic: contributors need neither Python nor a FreeToken checkout.
//!
//! One intentional deviation: Python raises `AssertionError` where this returns a typed
//! [`BudgetError`]. These are configuration errors a caller can act on — the upstream
//! comments say as much ("fails in arithmetic instead of OOMing in a later CUDA
//! allocation") — so a `Result` models them better than a panic.
//!
//! Not ported: `expert_bytes_per_slot`, which inspects live tensors and belongs with the
//! weight loader rather than with the arithmetic.

use std::fmt;

/// Why a cache geometry could not be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// `per_expert_bytes` must be positive.
    NonPositiveExpertBytes,
    /// `cache_per_page` must be positive; owned-KV models are not handled here.
    NonPositiveCachePerPage,
    /// The slot cap is below the minimum number of slots the model needs.
    SlotCapBelowMinimum { cap: i64, minimum: i64 },
    /// Even the minimum viable plan exceeds the budget.
    BudgetTooSmall { required: i64, budget: i64, moe_slots: i64, kv_pages: i64 },
    /// No room left for a usable KV cache after the MoE allocation.
    NoRoomForKv { pages: i64 },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveExpertBytes => write!(f, "per_expert_bytes must be positive"),
            Self::NonPositiveCachePerPage => {
                write!(f, "cache_per_page must be positive (owned-KV models unsupported here)")
            }
            Self::SlotCapBelowMinimum { cap, minimum } => {
                write!(f, "slot cap {cap} is below the {minimum} slots this model needs")
            }
            Self::BudgetTooSmall { required, budget, moe_slots, kv_pages } => write!(
                f,
                "cache budget too small: the minimum plan (moe={moe_slots} slots, \
                 kv={kv_pages} pages) needs {required} B but the budget is {budget} B \
                 (raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
            ),
            Self::NoRoomForKv { pages } => {
                write!(f, "not enough memory for a KV cache after the MoE allocation ({pages} pages)")
            }
        }
    }
}

impl std::error::Error for BudgetError {}

/// A planned split of the GPU pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePlan {
    /// Number of MoE expert slots held resident in VRAM.
    pub moe_cache_size: i64,
    /// Number of KV cache pages.
    pub num_pages: i64,
    /// Whether prefill/compute overlap is feasible with this geometry.
    pub prefill_overlap: bool,
}

/// Net GPU bytes available for the MoE and KV pools.
///
/// Takes `memory_ratio` of the persisted pre-model baseline, then subtracts weights and
/// fixed (non-paged) cache. The `(1 - memory_ratio)` remainder is graph/activation headroom
/// and is deliberately not subtracted here.
///
/// 🔴 `memory_ratio` is 0.9 upstream — a value tuned against the CUDA caching allocator.
/// It is on MoEArc's calibration list and must be measured per device, not inherited.
/// See `docs/calibration.md`.
pub fn net_cache_budget_bytes(
    memory_ratio: f64,
    baseline_free: i64,
    weights_bytes: i64,
    fixed_cache_size: i64,
) -> i64 {
    // Python `int()` truncates toward zero, which is what `as i64` does for finite values.
    (memory_ratio * baseline_free as f64) as i64 - weights_bytes - fixed_cache_size
}

/// GPU bytes a `(moe_cache_size, num_pages)` geometry occupies.
pub fn required_bytes(
    moe_cache_size: i64,
    num_pages: i64,
    per_expert_bytes: i64,
    cache_per_page: i64,
) -> i64 {
    moe_cache_size * per_expert_bytes + num_pages * cache_per_page
}

/// Split `budget_bytes` MoE-first into a cache geometry.
///
/// Experts greedily fill the budget after `kv_reserve_pages` is set aside for KV, clamped
/// to `[floor, min(total_experts, max_slots)]` — the floor being `2 * num_experts` when
/// prefill overlap is feasible, otherwise `num_experts`. KV pages take whatever is left.
#[allow(clippy::too_many_arguments)]
pub fn plan_cache_budget(
    budget_bytes: i64,
    per_expert_bytes: i64,
    cache_per_page: i64,
    num_experts: i64,
    total_experts: i64,
    prefill_overlap: bool,
    kv_reserve_pages: i64,
    max_slots: i64,
) -> Result<CachePlan, BudgetError> {
    // Upstream divides by these unguarded and raises ZeroDivisionError. Rust would panic on
    // integer division by zero, so they are rejected up front as the typed errors they are.
    if per_expert_bytes <= 0 {
        return Err(BudgetError::NonPositiveExpertBytes);
    }
    if cache_per_page <= 0 {
        return Err(BudgetError::NonPositiveCachePerPage);
    }

    let hi = total_experts.min(max_slots);
    // Prefill overlap borrows two full expert-layer buffers from the slot cache, so it needs
    // at least 2*num_experts slots. If the cap cannot fit that, disable it and drop the floor.
    let mut overlap = prefill_overlap && hi >= 2 * num_experts;
    let lo = if overlap { 2 * num_experts } else { num_experts };
    if hi < lo {
        return Err(BudgetError::SlotCapBelowMinimum { cap: hi, minimum: lo });
    }

    let kv_reserve_bytes = kv_reserve_pages * cache_per_page;
    // MoE-priority: set KV aside first, then experts greedily take the remainder.
    let raw = div_floor(budget_bytes - kv_reserve_bytes, per_expert_bytes);
    let moe_cache_size = lo.max(raw.min(hi));
    // A tight budget can force the size below 2*num_experts even with overlap requested.
    overlap = overlap && moe_cache_size >= 2 * num_experts;

    let remaining = budget_bytes - moe_cache_size * per_expert_bytes;
    let num_pages = div_floor(remaining, cache_per_page).max(kv_reserve_pages);

    // A tight budget can floor num_pages at kv_reserve_pages even when `remaining` is below
    // the reserve (or negative), producing a plan that overruns the budget. Reject it here so
    // auto-sizing fails in arithmetic rather than in a device allocation much later.
    let total = required_bytes(moe_cache_size, num_pages, per_expert_bytes, cache_per_page);
    if total > budget_bytes {
        return Err(BudgetError::BudgetTooSmall {
            required: total,
            budget: budget_bytes,
            moe_slots: moe_cache_size,
            kv_pages: num_pages,
        });
    }
    if num_pages <= 1 {
        return Err(BudgetError::NoRoomForKv { pages: num_pages });
    }

    Ok(CachePlan { moe_cache_size, num_pages, prefill_overlap: overlap })
}

/// The slot cap implied by a quantisation format.
///
/// 🔴 `nvfp4_marlin`'s 992 is a hardcoded property of NVIDIA's Marlin kernel, not of the
/// model. It is reproduced here only so this port is bit-identical to upstream. **MoEArc
/// must not inherit it** — on Arc the cap belongs to whichever kernel we ship, and is a
/// calibration output. Tracked in `docs/calibration.md`.
pub fn slot_cap_for_quant(quant_format: &str, total_experts: i64) -> i64 {
    if quant_format == "nvfp4_marlin" {
        992
    } else {
        total_experts
    }
}

/// Resolve `--moe-cache-auto` into a concrete cache geometry.
///
/// Applies `memory_ratio` to the persisted pre-model baseline exactly once, then defers the
/// MoE-vs-KV split to [`plan_cache_budget`].
#[allow(clippy::too_many_arguments)]
pub fn resolve_moe_cache_auto(
    baseline_free: i64,
    weights_bytes: i64,
    memory_ratio: f64,
    cache_per_page: i64,
    fixed_cache_size: i64,
    per_expert_bytes: i64,
    num_experts: i64,
    total_experts: i64,
    prefill_overlap: bool,
    kv_reserve_tokens: i64,
    page_size: i64,
    quant_format: &str,
) -> Result<CachePlan, BudgetError> {
    let budget_bytes =
        net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size);
    let max_slots = slot_cap_for_quant(quant_format, total_experts);
    let kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size);
    plan_cache_budget(
        budget_bytes,
        per_expert_bytes,
        cache_per_page,
        num_experts,
        total_experts,
        prefill_overlap,
        kv_reserve_pages,
        max_slots,
    )
}

/// Floor division matching Python's `//`.
///
/// Rust's `/` truncates toward zero where Python's `//` floors, and both numerators here
/// can go negative on a tight budget. Kept for fidelity to the reference -- but note it is
/// *not* currently load-bearing, and a mutation test proved it: replacing this with plain
/// `/` does not change a single one of the recorded outcomes.
///
/// The reason is that both results are swallowed by a clamp. `raw` is only read through
/// `lo.max(raw.min(hi))`, and a negative numerator gives `raw <= 0 < lo`, so the clamp
/// returns `lo` either way; likewise `-1` vs `0` at the second call site is lost to
/// `.max(kv_reserve_pages)`. Retained anyway because the clamps, not the division, are
/// what keep it safe -- and those are exactly the lines most likely to be rewritten.
fn div_floor(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Ceiling division, matching upstream's `freetoken.utils.div_ceil`.
fn div_ceil(a: i64, b: i64) -> i64 {
    div_floor(a + b - 1, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_floor_matches_python_semantics() {
        assert_eq!(div_floor(7, 2), 3);
        assert_eq!(div_floor(-7, 2), -4); // Rust `/` would give -3
        assert_eq!(div_floor(-1, 2), -1); // Rust `/` would give 0
        assert_eq!(div_floor(8, 2), 4);
    }

    #[test]
    fn div_ceil_rounds_up() {
        assert_eq!(div_ceil(0, 16), 0);
        assert_eq!(div_ceil(1, 16), 1);
        assert_eq!(div_ceil(16, 16), 1);
        assert_eq!(div_ceil(17, 16), 2);
    }

    #[test]
    fn net_budget_subtracts_weights_and_fixed() {
        assert_eq!(net_cache_budget_bytes(0.9, 1000, 300, 100), 500);
    }

    #[test]
    fn experts_take_priority_and_kv_gets_the_remainder() {
        let p = plan_cache_budget(10_000, 100, 10, 4, 64, false, 2, 64).unwrap();
        // 2 reserve pages = 20 B; (10000-20)/100 = 99 slots, clamped to hi = 64.
        assert_eq!(p.moe_cache_size, 64);
        // remaining = 10000 - 6400 = 3600 -> 360 pages.
        assert_eq!(p.num_pages, 360);
        assert!(!p.prefill_overlap);
    }

    #[test]
    fn overlap_is_disabled_when_the_cap_cannot_fit_two_layers() {
        // max_slots 6 < 2*num_experts (8) -> overlap off, floor drops to num_experts.
        let p = plan_cache_budget(10_000, 100, 10, 4, 64, true, 2, 6).unwrap();
        assert!(!p.prefill_overlap);
        assert_eq!(p.moe_cache_size, 6);
    }

    #[test]
    fn overlap_survives_when_the_cap_allows_it() {
        let p = plan_cache_budget(100_000, 100, 10, 4, 64, true, 2, 64).unwrap();
        assert!(p.prefill_overlap);
        assert!(p.moe_cache_size >= 8);
    }

    #[test]
    fn rejects_a_plan_that_would_exceed_the_budget() {
        let e = plan_cache_budget(100, 100, 10, 4, 64, false, 2, 64).unwrap_err();
        assert!(matches!(e, BudgetError::BudgetTooSmall { .. }));
    }

    #[test]
    fn rejects_non_positive_divisors_instead_of_panicking() {
        assert_eq!(
            plan_cache_budget(1000, 0, 10, 4, 64, false, 2, 64).unwrap_err(),
            BudgetError::NonPositiveExpertBytes
        );
        assert_eq!(
            plan_cache_budget(1000, 10, 0, 4, 64, false, 2, 64).unwrap_err(),
            BudgetError::NonPositiveCachePerPage
        );
    }

    #[test]
    fn rejects_a_cap_below_the_floor() {
        let e = plan_cache_budget(10_000, 100, 10, 8, 4, false, 2, 64).unwrap_err();
        assert!(matches!(e, BudgetError::SlotCapBelowMinimum { .. }));
    }

    #[test]
    fn marlin_slot_cap_is_quarantined_behind_its_own_function() {
        assert_eq!(slot_cap_for_quant("nvfp4_marlin", 128), 992);
        assert_eq!(slot_cap_for_quant("mxfp4", 128), 128);
    }
}
