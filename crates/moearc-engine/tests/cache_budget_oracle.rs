//! Differential test: replay the reference implementation's recorded behaviour.
//!
//! `tools/gen_cache_budget_oracle.py` runs FreeToken's `cache_budget.py` over a structured
//! sweep plus randomised fuzz and records every outcome — accepted plans AND rejections —
//! into `tests/data/cache_budget_oracle.json`. This test replays all of it.
//!
//! Rejections are as load-bearing as acceptances: a port that accepts an input the
//! reference refuses will happily plan a geometry that overruns VRAM and fail much later,
//! inside a device allocation, where the cause is invisible. So every case is checked for
//! *agreement on outcome*, not merely on value.
//!
//! The fixture is committed, so this needs neither Python nor a FreeToken checkout.

use moearc_engine::cache_budget::{
    AutoCacheRequest, PlanRequest, QuantFormat, net_cache_budget_bytes, required_bytes,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct PlanArgs {
    budget_bytes: i64,
    per_expert_bytes: i64,
    cache_per_page: i64,
    num_experts: i64,
    total_experts: i64,
    prefill_overlap: bool,
    kv_reserve_pages: i64,
    max_slots: i64,
}

#[derive(Deserialize)]
struct ResolveArgs {
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
    quant_format: String,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Outcome {
    moe_cache_size: i64,
    num_pages: i64,
    prefill_overlap: bool,
}

#[derive(Deserialize)]
struct PlanCase {
    kind: String,
    args: PlanArgs,
    ok: Option<Outcome>,
}

#[derive(Deserialize)]
struct ResolveCase {
    args: ResolveArgs,
    ok: Option<Outcome>,
}

#[derive(Deserialize)]
struct NetArgs {
    memory_ratio: f64,
    baseline_free: i64,
    weights_bytes: i64,
    fixed_cache_size: i64,
}

#[derive(Deserialize)]
struct ReqArgs {
    moe_cache_size: i64,
    num_pages: i64,
    per_expert_bytes: i64,
    cache_per_page: i64,
}

#[derive(Deserialize)]
struct Checked<A> {
    args: A,
    want: i64,
}

#[derive(Deserialize)]
struct HelperCase {
    net: Checked<NetArgs>,
    req: Checked<ReqArgs>,
}

#[derive(Deserialize)]
struct Oracle {
    plan: Vec<PlanCase>,
    helpers: Vec<HelperCase>,
    resolve: Vec<ResolveCase>,
}

fn load() -> Oracle {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/data/cache_budget_oracle.json"
    );
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("oracle fixture missing at {path}: {e}"));
    serde_json::from_str(&raw).expect("oracle fixture is not valid JSON")
}

#[test]
fn plan_cache_budget_matches_the_reference() {
    let oracle = load();
    assert!(
        oracle.plan.len() > 3000,
        "fixture looks truncated: {} cases",
        oracle.plan.len()
    );

    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (i, case) in oracle.plan.iter().enumerate() {
        let a = &case.args;
        let got = PlanRequest {
            budget_bytes: a.budget_bytes,
            per_expert_bytes: a.per_expert_bytes,
            cache_per_page: a.cache_per_page,
            num_experts: a.num_experts,
            total_experts: a.total_experts,
            prefill_overlap: a.prefill_overlap,
            kv_reserve_pages: a.kv_reserve_pages,
            max_slots: a.max_slots,
        }
        .plan();
        match (&case.ok, got) {
            (Some(want), Ok(p)) => {
                assert_eq!(
                    (p.moe_cache_size, p.num_pages, p.prefill_overlap),
                    (want.moe_cache_size, want.num_pages, want.prefill_overlap),
                    "case {i} ({}) disagreed on the plan",
                    case.kind
                );
                accepted += 1;
            }
            (None, Err(_)) => rejected += 1,
            (Some(want), Err(e)) => panic!(
                "case {i} ({}): reference accepted {want:?}, port rejected with {e}",
                case.kind
            ),
            (None, Ok(p)) => panic!(
                "case {i} ({}): reference rejected, port accepted {p:?} — this is the \
                 dangerous direction, it plans a geometry that will overrun VRAM",
                case.kind
            ),
        }
    }
    // Guard against a fixture that is all one outcome, which would make agreement vacuous.
    assert!(
        accepted > 1000 && rejected > 1000,
        "lopsided fixture: {accepted} ok, {rejected} rejected"
    );
    eprintln!("plan_cache_budget: {accepted} accepted + {rejected} rejected all agree");
}

#[test]
fn auto_cache_request_matches_the_reference() {
    let oracle = load();
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (i, case) in oracle.resolve.iter().enumerate() {
        let a = &case.args;
        let got = AutoCacheRequest {
            baseline_free: a.baseline_free,
            weights_bytes: a.weights_bytes,
            memory_ratio: a.memory_ratio,
            cache_per_page: a.cache_per_page,
            fixed_cache_size: a.fixed_cache_size,
            per_expert_bytes: a.per_expert_bytes,
            num_experts: a.num_experts,
            total_experts: a.total_experts,
            prefill_overlap: a.prefill_overlap,
            kv_reserve_tokens: a.kv_reserve_tokens,
            page_size: a.page_size,
            quant_format: QuantFormat::parse(&a.quant_format),
        }
        .resolve();
        match (&case.ok, got) {
            (Some(want), Ok(p)) => {
                assert_eq!(
                    (p.moe_cache_size, p.num_pages, p.prefill_overlap),
                    (want.moe_cache_size, want.num_pages, want.prefill_overlap),
                    "resolve case {i} disagreed"
                );
                accepted += 1;
            }
            (None, Err(_)) => rejected += 1,
            (Some(want), Err(e)) => {
                panic!("resolve case {i}: reference gave {want:?}, port rejected: {e}")
            }
            (None, Ok(p)) => panic!("resolve case {i}: reference rejected, port accepted {p:?}"),
        }
    }
    assert!(
        accepted > 100 && rejected > 100,
        "lopsided: {accepted} ok, {rejected} rejected"
    );
    eprintln!("resolve_moe_cache_auto: {accepted} accepted + {rejected} rejected all agree");
}

#[test]
fn helper_arithmetic_matches_the_reference() {
    let oracle = load();
    assert!(!oracle.helpers.is_empty());
    for (i, h) in oracle.helpers.iter().enumerate() {
        let a = &h.net.args;
        assert_eq!(
            net_cache_budget_bytes(
                a.memory_ratio,
                a.baseline_free,
                a.weights_bytes,
                a.fixed_cache_size
            ),
            h.net.want,
            "net_cache_budget_bytes case {i}"
        );
        let b = &h.req.args;
        assert_eq!(
            required_bytes(
                b.moe_cache_size,
                b.num_pages,
                b.per_expert_bytes,
                b.cache_per_page
            ),
            h.req.want,
            "required_bytes case {i}"
        );
    }
    eprintln!("helpers: {} cases agree", oracle.helpers.len());
}
