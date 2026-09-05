//! GPU cache budget planning: how the net GPU pool is split between the MoE expert cache
//! and the paged KV cache.
//!
//! This is the machinery behind the product rule in `docs/ux.md` — *the split is computed,
//! never configured*. A user asks for a context length; deciding how many expert slots stay
//! resident and how many KV pages to allocate is this module's job.
//!
//! Ported from FreeToken's `freetoken/engine/cache_budget.py` (Apache-2.0). Its logic is
//! pure integer byte arithmetic with no torch or device dependency, which makes it the
//! natural first component to own outright and the only one testable without hardware.
//!
//! Equivalence with the Python is *proven*, not assumed: `tools/gen_cache_budget_oracle.py`
//! runs the reference over 4,145 structured and randomised inputs and records every result —
//! including the ~1,900 it rejects — into `tests/data/`. The test in
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
    /// `page_size` must be positive.
    NonPositivePageSize,
    /// The slot cap is below the minimum number of slots the model needs.
    SlotCapBelowMinimum { cap: i64, minimum: i64 },
    /// Even the minimum viable plan exceeds the budget.
    BudgetTooSmall {
        required: i64,
        budget: i64,
        moe_slots: i64,
        kv_pages: i64,
    },
    /// No room left for a usable KV cache after the MoE allocation.
    NoRoomForKv { pages: i64 },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveExpertBytes => write!(f, "per_expert_bytes must be positive"),
            Self::NonPositiveCachePerPage => {
                write!(
                    f,
                    "cache_per_page must be positive (owned-KV models unsupported here)"
                )
            }
            Self::NonPositivePageSize => write!(f, "page_size must be positive"),
            Self::SlotCapBelowMinimum { cap, minimum } => {
                write!(
                    f,
                    "slot cap {cap} is below the {minimum} slots this model needs"
                )
            }
            Self::BudgetTooSmall {
                required,
                budget,
                moe_slots,
                kv_pages,
            } => write!(
                f,
                "cache budget too small: the minimum plan (moe={moe_slots} slots, \
                 kv={kv_pages} pages) needs {required} B but the budget is {budget} B \
                 (raise memory_ratio, lower the reserved context, or free GPU memory)"
            ),
            Self::NoRoomForKv { pages } => {
                write!(
                    f,
                    "not enough memory for a KV cache after the MoE allocation ({pages} pages)"
                )
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

/// The quantisation format, insofar as it constrains the expert slot cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// NVIDIA's Marlin NVFP4 kernel, which caps resident expert slots at 992.
    Nvfp4Marlin,
    /// Anything else: the cap is the model's own expert count.
    Unconstrained,
}

impl QuantFormat {
    /// Parse the upstream format string. Unknown formats are [`Self::Unconstrained`],
    /// which is the safe direction — it defers to the model rather than to a kernel limit.
    pub fn parse(s: &str) -> Self {
        match s {
            "nvfp4_marlin" => Self::Nvfp4Marlin,
            _ => Self::Unconstrained,
        }
    }

    /// The slot cap this format implies.
    ///
    /// 🔴 The 992 is a hardcoded property of NVIDIA's Marlin kernel, not of any model. It is
    /// reproduced here only so this port is bit-identical to upstream, and it is deliberately
    /// walled off in its own function so it cannot leak into an Arc default. **MoEArc must not
    /// inherit it** — on Arc the cap belongs to whichever kernel we ship, and is a calibration
    /// output. See `docs/calibration.md`.
    ///
    /// Note it is nearly inert in practice: the cap only binds above 992 experts, and no
    /// shipping model has that many. A mutation test confirmed the branch was unobservable
    /// until cases with >992 experts were added to the fixture on purpose.
    pub fn slot_cap(self, total_experts: i64) -> i64 {
        match self {
            Self::Nvfp4Marlin => 992,
            Self::Unconstrained => total_experts,
        }
    }
}

/// A request to split an already-computed budget.
///
/// Named fields rather than positional arguments: the underlying quantities are almost all
/// `i64`, so a positional API silently accepts any two of them swapped. That is a real
/// hazard here, because a wrong-but-plausible plan does not fail at the call site — it fails
/// much later inside a device allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanRequest {
    /// Net bytes available for the MoE and KV pools.
    pub budget_bytes: i64,
    /// Bytes one resident expert slot occupies.
    pub per_expert_bytes: i64,
    /// Bytes one KV page occupies.
    pub cache_per_page: i64,
    /// Experts active per layer — the floor on resident slots.
    pub num_experts: i64,
    /// Experts the model has in total.
    pub total_experts: i64,
    /// Whether to try to reserve enough slots for prefill/compute overlap.
    pub prefill_overlap: bool,
    /// KV pages to set aside before experts take the remainder.
    pub kv_reserve_pages: i64,
    /// Hard cap on resident expert slots.
    pub max_slots: i64,
}

impl PlanRequest {
    /// Split the budget MoE-first into a cache geometry.
    ///
    /// Experts greedily fill the budget after `kv_reserve_pages` is set aside for KV, clamped
    /// to `[floor, min(total_experts, max_slots)]` — the floor being `2 * num_experts` when
    /// prefill overlap is feasible, otherwise `num_experts`. KV pages take whatever is left.
    pub fn plan(&self) -> Result<CachePlan, BudgetError> {
        // Upstream divides by these unguarded and raises ZeroDivisionError. Rust would panic
        // on integer division by zero, so reject them up front as the typed errors they are.
        if self.per_expert_bytes <= 0 {
            return Err(BudgetError::NonPositiveExpertBytes);
        }
        if self.cache_per_page <= 0 {
            return Err(BudgetError::NonPositiveCachePerPage);
        }

        let hi = self.total_experts.min(self.max_slots);
        // Prefill overlap borrows two full expert-layer buffers from the slot cache, so it
        // needs at least 2*num_experts slots. If the cap cannot fit that, disable it and drop
        // the floor accordingly.
        let mut overlap = self.prefill_overlap && hi >= 2 * self.num_experts;
        let lo = if overlap {
            2 * self.num_experts
        } else {
            self.num_experts
        };
        if hi < lo {
            return Err(BudgetError::SlotCapBelowMinimum {
                cap: hi,
                minimum: lo,
            });
        }

        let kv_reserve_bytes = self.kv_reserve_pages * self.cache_per_page;
        // MoE-priority: set KV aside first, then experts greedily take the remainder.
        let raw = div_floor(self.budget_bytes - kv_reserve_bytes, self.per_expert_bytes);
        let moe_cache_size = lo.max(raw.min(hi));
        // A tight budget can force the size below 2*num_experts even with overlap requested.
        overlap = overlap && moe_cache_size >= 2 * self.num_experts;

        let remaining = self.budget_bytes - moe_cache_size * self.per_expert_bytes;
        let num_pages = div_floor(remaining, self.cache_per_page).max(self.kv_reserve_pages);

        // A tight budget can floor num_pages at kv_reserve_pages even when `remaining` is
        // below the reserve (or negative), producing a plan that overruns the budget. Reject
        // it here so auto-sizing fails in arithmetic rather than in a device allocation much
        // later, where the cause is invisible.
        let total = required_bytes(
            moe_cache_size,
            num_pages,
            self.per_expert_bytes,
            self.cache_per_page,
        );
        if total > self.budget_bytes {
            return Err(BudgetError::BudgetTooSmall {
                required: total,
                budget: self.budget_bytes,
                moe_slots: moe_cache_size,
                kv_pages: num_pages,
            });
        }
        if num_pages <= 1 {
            return Err(BudgetError::NoRoomForKv { pages: num_pages });
        }

        Ok(CachePlan {
            moe_cache_size,
            num_pages,
            prefill_overlap: overlap,
        })
    }
}

/// A request to size the cache automatically from what the device reports.
///
/// This is the entry point behind `--moe-cache-auto`, and the one a user never sees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoCacheRequest {
    /// Free device bytes measured *before* the model was loaded.
    pub baseline_free: i64,
    /// Bytes the model weights occupy.
    pub weights_bytes: i64,
    /// Fraction of the baseline we are willing to commit.
    ///
    /// 🔴 Upstream is 0.9, tuned against the CUDA caching allocator. It must be measured per
    /// device, not inherited. See `docs/calibration.md`.
    pub memory_ratio: f64,
    /// Bytes one KV page occupies.
    pub cache_per_page: i64,
    /// Bytes of non-paged cache allocated regardless.
    pub fixed_cache_size: i64,
    /// Bytes one resident expert slot occupies.
    pub per_expert_bytes: i64,
    /// Experts active per layer.
    pub num_experts: i64,
    /// Experts the model has in total.
    pub total_experts: i64,
    /// Whether to try to reserve enough slots for prefill/compute overlap.
    pub prefill_overlap: bool,
    /// Context tokens to guarantee KV room for. **This is the unit a user thinks in.**
    pub kv_reserve_tokens: i64,
    /// Tokens per KV page.
    pub page_size: i64,
    /// Quantisation format, insofar as it caps resident slots.
    pub quant_format: QuantFormat,
}

impl AutoCacheRequest {
    /// Resolve into a concrete cache geometry.
    ///
    /// Applies `memory_ratio` to the persisted pre-model baseline exactly once, converts the
    /// caller's context reservation from tokens into pages, then defers the MoE-vs-KV split
    /// to [`PlanRequest::plan`].
    pub fn resolve(&self) -> Result<CachePlan, BudgetError> {
        if self.page_size <= 0 {
            return Err(BudgetError::NonPositivePageSize);
        }
        let budget_bytes = net_cache_budget_bytes(
            self.memory_ratio,
            self.baseline_free,
            self.weights_bytes,
            self.fixed_cache_size,
        );
        PlanRequest {
            budget_bytes,
            per_expert_bytes: self.per_expert_bytes,
            cache_per_page: self.cache_per_page,
            num_experts: self.num_experts,
            total_experts: self.total_experts,
            prefill_overlap: self.prefill_overlap,
            kv_reserve_pages: div_ceil(self.kv_reserve_tokens, self.page_size),
            max_slots: self.quant_format.slot_cap(self.total_experts),
        }
        .plan()
    }
}

/// Net GPU bytes available for the MoE and KV pools.
///
/// Takes `memory_ratio` of the pre-model baseline, then subtracts weights and fixed
/// (non-paged) cache. The `(1 - memory_ratio)` remainder is graph/activation headroom and is
/// deliberately not subtracted here.
///
/// Low-level; prefer [`AutoCacheRequest`], which applies this in the right order.
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

/// Floor division matching Python's `//`.
///
/// Rust's `/` truncates toward zero where Python's `//` floors, and both numerators here can
/// go negative on a tight budget. Kept for fidelity to the reference — but note it is *not*
/// currently load-bearing, and a mutation test proved it: replacing this with plain `/` does
/// not change a single one of the recorded outcomes.
///
/// The reason is that both results are swallowed by a clamp. `raw` is only read through
/// `lo.max(raw.min(hi))`, and a negative numerator gives `raw <= 0 < lo`, so the clamp returns
/// `lo` either way; likewise `-1` vs `0` at the second call site is lost to
/// `.max(kv_reserve_pages)`. Retained anyway because the clamps, not the division, are what
/// keep it safe — and those are exactly the lines most likely to be rewritten.
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

    /// A plan request that succeeds, for tests to vary one field of.
    fn req() -> PlanRequest {
        PlanRequest {
            budget_bytes: 10_000,
            per_expert_bytes: 100,
            cache_per_page: 10,
            num_experts: 4,
            total_experts: 64,
            prefill_overlap: false,
            kv_reserve_pages: 2,
            max_slots: 64,
        }
    }

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
        let p = req().plan().unwrap();
        // 2 reserve pages = 20 B; (10000-20)/100 = 99 slots, clamped to hi = 64.
        assert_eq!(p.moe_cache_size, 64);
        // remaining = 10000 - 6400 = 3600 -> 360 pages.
        assert_eq!(p.num_pages, 360);
        assert!(!p.prefill_overlap);
    }

    #[test]
    fn overlap_is_disabled_when_the_cap_cannot_fit_two_layers() {
        // max_slots 6 < 2*num_experts (8) -> overlap off, floor drops to num_experts.
        let p = PlanRequest {
            prefill_overlap: true,
            max_slots: 6,
            ..req()
        }
        .plan()
        .unwrap();
        assert!(!p.prefill_overlap);
        assert_eq!(p.moe_cache_size, 6);
    }

    #[test]
    fn overlap_survives_when_the_cap_allows_it() {
        let p = PlanRequest {
            budget_bytes: 100_000,
            prefill_overlap: true,
            ..req()
        }
        .plan()
        .unwrap();
        assert!(p.prefill_overlap);
        assert!(p.moe_cache_size >= 8);
    }

    #[test]
    fn rejects_a_plan_that_would_exceed_the_budget() {
        let e = PlanRequest {
            budget_bytes: 100,
            ..req()
        }
        .plan()
        .unwrap_err();
        assert!(matches!(e, BudgetError::BudgetTooSmall { .. }));
    }

    #[test]
    fn rejects_non_positive_divisors_instead_of_panicking() {
        assert_eq!(
            PlanRequest {
                per_expert_bytes: 0,
                ..req()
            }
            .plan()
            .unwrap_err(),
            BudgetError::NonPositiveExpertBytes
        );
        assert_eq!(
            PlanRequest {
                cache_per_page: 0,
                ..req()
            }
            .plan()
            .unwrap_err(),
            BudgetError::NonPositiveCachePerPage
        );
    }

    #[test]
    fn rejects_a_cap_below_the_floor() {
        let e = PlanRequest {
            num_experts: 8,
            total_experts: 4,
            ..req()
        }
        .plan()
        .unwrap_err();
        assert!(matches!(e, BudgetError::SlotCapBelowMinimum { .. }));
    }

    #[test]
    fn marlin_slot_cap_is_quarantined_behind_the_quant_format() {
        assert_eq!(QuantFormat::parse("nvfp4_marlin").slot_cap(128), 992);
        assert_eq!(QuantFormat::parse("mxfp4").slot_cap(128), 128);
        // An unknown format defers to the model, never to a foreign kernel's limit.
        assert_eq!(QuantFormat::parse("something_new").slot_cap(128), 128);
    }

    #[test]
    fn a_zero_page_size_is_rejected_rather_than_dividing_by_zero() {
        let r = AutoCacheRequest {
            baseline_free: 1 << 34,
            weights_bytes: 1 << 30,
            memory_ratio: 0.9,
            cache_per_page: 1 << 16,
            fixed_cache_size: 0,
            per_expert_bytes: 1 << 20,
            num_experts: 8,
            total_experts: 128,
            prefill_overlap: false,
            kv_reserve_tokens: 1024,
            page_size: 0,
            quant_format: QuantFormat::Unconstrained,
        };
        assert_eq!(r.resolve().unwrap_err(), BudgetError::NonPositivePageSize);
    }
}
