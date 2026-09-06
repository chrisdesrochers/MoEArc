//! The forward pass against llama.cpp on Qwen3-30B-A3B — the first model that does not fit.
//!
//! Skipped unless both of these are set, so the suite still runs on a machine with no card and
//! no model:
//!
//! ```text
//! MOEARC_TEST_GPU=1
//! MOEARC_QWEN3MOE_MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf
//! ```
//!
//! # Why this file exists next to `olmoe_forward.rs` rather than instead of it
//!
//! The two models exercise different halves of the same code. OLMoE fits entirely in VRAM, has
//! no GQA, normalises QK over the whole projection and does **not** renormalise its router
//! weights. Qwen3-30B-A3B is the mirror image of all four. Keeping both gates is what makes
//! `moe.rs`'s architecture switches testable in both positions — a switch that is only ever
//! exercised one way is a constant with extra steps.
//!
//! # What is asserted, and why it is not "the logits are equal"
//!
//! 🔴 MoEArc and llama.cpp do not compute the same function, and the difference is llama.cpp's,
//! not this engine's: every K-quant matmul in `ggml-cpu` quantises the f32 activation to 8 bits
//! before the dot product, and `moearc-kernels` keeps it in f32. `olmoe_forward.rs` carries the
//! measurement of how large that unavoidable difference is. So this file asserts on the decision
//! the logits encode — the greedy id — which fails loudly if any of the ~600 operations per
//! token on 48 blocks is wrong.
//!
//! # 🔴 The prompt is part of the gate, and picking the obvious one would have been wrong
//!
//! A greedy continuation is only a *decision* where the top two logits are further apart than
//! the arithmetic difference above. On this model that difference reaches **max|d| = 1.06** on
//! the 151,936-wide logit vector (measured with `examples/probe` against a `MOEARC_DUMP_DIR`
//! dump), so a step whose top two are within ~0.1 is a coin flip, and once the two
//! implementations take different branches they never rejoin.
//!
//! That is not hypothetical. On `The capital of France is` — the prompt `olmoe_forward.rs`
//! uses — the model starts enumerating capitals and step 2 is a three-way near-tie. **llama.cpp's
//! own two backends disagree there**, measured on this exact file:
//!
//! ```text
//!   step 2, llama.cpp CPU     576 23.5524   15920 23.4591   3555 23.4533   <- picks 576
//!   step 2, llama.cpp SYCL    (follows 15920)
//!   step 2, MoEArc          15920 23.8202     576 23.7991   3555 23.3545   <- picks 15920
//! ```
//!
//! A 0.093 margin, inside the backend noise. MoEArc then tracks llama.cpp's **SYCL** backend for
//! 35 tokens before the next tie. Gating on that prompt would have been measuring the tie-break.
//!
//! So [`PROMPT`] is chosen for the opposite property: llama.cpp's CPU and SYCL backends produce
//! **identical** ids on it for all 64 tokens. A path both of llama.cpp's backends agree on is
//! robust to exactly the class of perturbation that separates MoEArc from either, which is what
//! makes MoEArc matching it a statement about the graph rather than about rounding.
//!
//! # Residency, in a file where residency is not optional
//!
//! Every session below names an explicit slot budget and context length. That is not tidiness:
//! this model's full pool is 6144 slots x 2.92 MiB = **17.51 GiB** against a B580's ~9.7 GiB of
//! usable device memory, and its trained context of 40,960 tokens is another 3.75 GiB of KV, so
//! the defaults (`Residency::All`, the model's own context) cannot be allocated at all. The
//! budgets here are deliberately small so that two sessions can be alive at once. That is a
//! real constraint and not caution: the shared session below holds `512 * 2.92 MiB = 1.46 GiB`
//! of pool plus 951 MiB of dense weights and stays alive for the whole run, so every
//! `constrained` session has to fit in what is left. The largest budget used here,
//! `static:12` at 1544 slots, brings the total to about 7.8 GiB — measured to work; the same
//! sweep with `static:20` does not.

#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::OnceLock;

use moearc_engine::moe::{Arch, Residency};
use moearc_engine::session::{Session, SessionOptions, StopConditions};

/// Context length for every session here. The prompt plus 64 tokens is 69, so this is ample,
/// and at 96 KiB of KV per token — 48 blocks x 4 KV heads x 256 channels x f16 — a larger one
/// would compete with the pool being tested for the same card.
const N_CTX: usize = 512;

/// The shared session's budget. Small on purpose: see the module header.
const SHARED_SLOTS: u32 = 512;

/// `def fibonacci(n):\n    `, tokenised by llama.cpp with `add_bos = false` — which is what this
/// file's `tokenizer.ggml.add_bos_token` says. Chosen for the reason in the module header: both
/// of llama.cpp's backends agree on its whole continuation.
const PROMPT: [u32; 5] = [750, 75698, 1445, 982, 257];

/// llama.cpp's greedy continuation of [`PROMPT`], ids in order.
///
/// Recorded from `llama-eval-callback` with `MOEARC_GREEDY_N=64`, llama.cpp
/// `e107984bcffcfd701e82738092a2b000b6fda7a2`, and **identical on `-ngl 0` (CPU) and
/// `-ngl 99 --n-cpu-moe 32` (SYCL)**. The argmax is written out by that patch rather than taken
/// from a sampler chain: at temperature 0 a chain still runs top-k, top-p and min-p, and a tie
/// broken differently there would look like a model difference. The same ids are in
/// `bench/references/qwen3-30b-a3b.fibonacci.ids`.
const LLAMA_CPP_GREEDY: [u32; 64] = [
    421, 308, 2651, 220, 16, 510, 260, 470, 308, 198, 257, 770, 510, 260, 470, 955, 579, 39345,
    1445, 12, 16, 8, 488, 75698, 1445, 12, 17, 4390, 77, 284, 526, 5384, 445, 6269, 279, 1372, 315,
    3793, 25, 330, 4390, 333, 308, 2651, 220, 15, 510, 257, 1173, 445, 5501, 3725, 264, 6785, 7546,
    1138, 1503, 510, 257, 1173, 445, 37, 579, 39345,
];

fn model_path() -> Option<PathBuf> {
    if std::env::var("MOEARC_TEST_GPU").ok().as_deref() != Some("1") {
        return None;
    }
    std::env::var("MOEARC_QWEN3MOE_MODEL").ok().map(PathBuf::from)
}

/// One session for the whole file: this model is nearly 3 GiB of VRAM even at a small budget,
/// and Rust runs tests in parallel.
fn session() -> Option<&'static Session> {
    static S: OnceLock<Option<Session>> = OnceLock::new();
    S.get_or_init(|| {
        let path = model_path()?;
        let opts = SessionOptions {
            n_ctx: Some(N_CTX),
            residency: Residency::Slots(SHARED_SLOTS),
            ..Default::default()
        };
        match Session::load_with(&path, opts) {
            Ok(s) => Some(s),
            Err(e) => panic!("MOEARC_QWEN3MOE_MODEL is set but the model would not load: {e}"),
        }
    })
    .as_ref()
}

macro_rules! session_or_skip {
    () => {
        match session() {
            Some(s) => s,
            None => {
                eprintln!(
                    "skipped: set MOEARC_TEST_GPU=1 and MOEARC_QWEN3MOE_MODEL=<file.gguf> to run"
                );
                return;
            }
        }
    };
}

fn generate(s: &Session, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let stop = StopConditions { max_tokens: n, stop_tokens: Vec::new() };
    s.generate(prompt, &stop, &mut |t| {
        out.push(t);
        true
    })
    .expect("generation failed");
    out
}

#[test]
fn the_geometry_is_read_from_the_file_and_not_assumed() {
    let s = session_or_skip!();
    let c = s.config();
    assert_eq!(c.arch, "qwen3moe");
    assert_eq!(c.kind, Arch::Qwen3Moe);
    assert_eq!(c.n_block, 48);
    assert_eq!(c.n_embd, 2048);
    assert_eq!(c.n_head, 32);

    // 🔴 Real GQA, 8 query heads to each KV head. OLMoE has none, so before this model the
    // grouped path in `attn_decode` had only ever run on synthetic shapes.
    assert_eq!(c.n_head_kv, 4);
    assert_eq!(c.n_head / c.n_head_kv, 8);

    // 🔴 The trap. 2048 / 32 = 64; the file says `attention.key_length = 128`, and a computed
    // head_dim would halve every head and still produce fluent English.
    assert_eq!(c.head_dim, 128);
    assert_ne!(c.head_dim, c.n_embd / c.n_head);
    assert_eq!(c.n_rot, 128, "RoPE must rotate the whole head");
    // And so the Q projection is twice the residual stream. Buffers sized `n_embd` would have
    // the QK-norm, RoPE and the output projection all reading half a projection.
    assert_eq!(c.n_head * c.head_dim, 4096);

    // 🔴 `expert_feed_forward_length`, not `feed_forward_length` (6144, a dense FFN this
    // architecture does not have).
    assert_eq!(c.n_ff, 768);

    assert_eq!(c.n_expert, 128);
    assert_eq!(c.n_expert_used, 8);
    assert_eq!(c.n_vocab, 151_936);
    assert_eq!(c.n_ctx_train, 40_960);

    // 🔴 1e-6 and 1e6. OLMoE is 1e-5 and 1e4, and both defaults are wrong here.
    assert!((c.rms_eps - 1e-6).abs() < 1e-12, "rms_eps was {}", c.rms_eps);
    assert_eq!(c.rope_freq_base, 1_000_000.0);

    // The two switches no GGUF key records.
    assert!(c.qk_norm_per_head, "qwen3moe normalises each head, not the whole projection");
    assert_eq!(
        c.gating,
        moearc_kernels::Gating::SoftmaxNormalised,
        "build_moe_ffn is called with norm_w = true"
    );

    // This file carries no `bos_token_id` at all, and `add_bos_token` is false.
    assert_eq!(c.bos, None);
    assert_eq!(c.eos, Some(151_645));
}

#[test]
fn greedy_generation_matches_llama_cpp_token_for_token() {
    let s = session_or_skip!();
    let got = generate(s, &PROMPT, LLAMA_CPP_GREEDY.len());
    assert_eq!(
        got,
        LLAMA_CPP_GREEDY.to_vec(),
        "greedy ids diverged from llama.cpp; first difference at index {:?}",
        got.iter().zip(LLAMA_CPP_GREEDY.iter()).position(|(a, b)| a != b)
    );
}

#[test]
fn a_prompt_longer_than_one_kv_page_still_matches() {
    // The KV cache pages at 32 tokens, so a 40-token generation crosses two page boundaries and
    // the block table stops being `[0]`. A kernel that assumed contiguous pages passes every
    // short test and fails here.
    let s = session_or_skip!();
    let got = generate(s, &PROMPT, 40);
    assert_eq!(got, LLAMA_CPP_GREEDY[..40].to_vec());
}

#[test]
fn the_router_renormalises_its_weights_on_every_block() {
    // 🔴 The mirror of `olmoe_forward.rs`'s assertion, and the only place `norm_w = true` is
    // tested. Under `norm_w = false` these would be raw softmax probabilities over 128 experts,
    // summing to well under one — plausible-looking numbers that scale every block's FFN output
    // down by a constant and change nothing else.
    let s = session_or_skip!();
    let (_, tap) = s.logits_tapped(&PROMPT).unwrap();
    let c = s.config();
    for b in 0..c.n_block {
        let w = tap
            .get(&format!("ffn_moe_weights-{b}"))
            .unwrap_or_else(|| panic!("block {b} produced no router weights"));
        assert_eq!(w.len(), c.n_expert_used);
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "block {b}'s router weights sum to {sum}, so they were not renormalised: {w:?}"
        );
        let experts = tap.get(&format!("ffn_moe_topk-{b}")).expect("no router choice");
        let mut ids: Vec<u32> = experts.iter().map(|v| *v as u32).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), c.n_expert_used, "block {b} routed to the same expert twice");
        assert!(ids.iter().all(|e| (*e as usize) < c.n_expert));
    }
}

#[test]
fn every_block_is_tapped_and_none_of_them_is_degenerate() {
    let s = session_or_skip!();
    let (logits, tap) = s.logits_tapped(&PROMPT).unwrap();
    let c = s.config();
    for b in 0..c.n_block {
        let h = tap
            .get(&format!("l_out-{b}"))
            .unwrap_or_else(|| panic!("block {b} produced no residual stream"));
        assert_eq!(h.len(), c.n_embd);
        assert!(h.iter().all(|v| v.is_finite()), "block {b} produced a non-finite activation");
        assert!(h.iter().any(|v| *v != 0.0), "block {b} produced all zeros");
    }
    assert_eq!(logits.len(), s.vocab_size());
    assert!(logits.iter().all(|v| v.is_finite()), "a NaN or infinity reached the logits");
    assert_eq!(moearc_engine::session::argmax(&logits), LLAMA_CPP_GREEDY[0]);
}

/// A session of its own, so a residency setting can be exercised without disturbing the shared
/// one. Returns `None` when the suite is being skipped.
fn constrained(residency: Residency) -> Option<Session> {
    let path = model_path()?;
    let opts = SessionOptions { n_ctx: Some(N_CTX), residency, ..Default::default() };
    Some(Session::load_with(&path, opts).expect("constrained session would not load"))
}

#[test]
fn constraining_residency_does_not_change_a_single_token_id() {
    // 🔴 The gate on the whole paging path, on the first model where paging is compulsory. A
    // step's working set here is 48 blocks x 8 experts = 384 slots, and none of these budgets
    // holds the model — so every row below is genuinely streaming, which was never true on
    // OLMoE. Residency decides what has to move, never what is computed, so an id that changes
    // with the budget is a paging bug: a slot read before it was filled, or two experts sharing
    // one. Sessions are built and dropped one at a time so this never holds two pools at once.
    if session().is_none() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_QWEN3MOE_MODEL=<file.gguf> to run");
        return;
    }
    let n = 24;
    for residency in [
        // Well above the per-token working set of 384 slots, at it, and below it.
        Residency::Slots(1024),
        Residency::Slots(384),
        Residency::Slots(64),
        // The incumbent, at two splits that fit. ⚠️ `olmoe_forward.rs` also sweeps
        // `static:n_block - 1`, the one setting where a ring entry survives from one visit to a
        // block to the next — the case that caught a slot collision no other could. **That
        // setting is unreachable on this model**: 47 pinned blocks is 6016 slots, 17.2 GiB, and
        // the card holds about 3000. It is covered instead by
        // `moe.rs`'s `the_ring_never_evicts_an_expert_the_same_step_still_needs`, which asserts
        // the same property on the policy directly, and by OLMoE's gate on real weights.
        Residency::StaticSplit { resident_blocks: 4 },
        Residency::StaticSplit { resident_blocks: 12 },
    ] {
        let s = constrained(residency).expect("checked above");
        let got = generate(&s, &PROMPT, n);
        assert_eq!(
            got,
            LLAMA_CPP_GREEDY[..n].to_vec(),
            "{residency:?} changed the output; first difference at index {:?}",
            got.iter().zip(LLAMA_CPP_GREEDY.iter()).position(|(a, b)| a != b)
        );
        let r = s.residency().expect("residency");
        assert!(r.resident_slots <= r.total_slots);
        assert_eq!(r.stats.demands, r.stats.hits + r.stats.misses);
    }
}

#[test]
fn the_pool_is_sized_in_slots_and_is_larger_than_the_bank_it_holds() {
    let s = session_or_skip!();
    let r = s.info().residency;
    // 🔴 48 blocks x 128 experts. The file says "128 experts"; the engine has to hold 6144
    // places to put one, and conflating the two is a factor-of-48 error in every budget.
    assert_eq!(r.total_slots, 6144);
    assert_eq!(r.resident_slots, SHARED_SLOTS);

    // A slot has to hold whichever block the router lands in, and `ffn_down_exps` is Q6_K in 24
    // blocks and Q4_K in the other 24, so a full pool is *larger* than the bank: 6144 slots of
    // 2.92 MiB is 17.51 GiB against 16.35 GiB of experts. Any budget derived from
    // `expert_bytes` rather than `slot_bytes` over-promises by that 7%.
    let full_pool = r.slot_bytes * u64::from(r.total_slots);
    assert!(
        full_pool > r.expert_bytes,
        "a full pool of {full_pool} B should exceed the {} B of experts it holds",
        r.expert_bytes
    );
    assert!(r.dense_bytes > 0 && r.dense_bytes < r.expert_bytes);
}

#[test]
fn a_pool_below_one_tokens_working_set_still_moves_exactly_the_bytes_it_claims() {
    // 48 blocks x 8 experts is 384 slots per token, so a 64-slot pool cannot hold a token's
    // working set and LRU evicts everything before it is wanted again. The interesting part is
    // not that the hit rate is low — it is that the accounting stays exact, because the whole
    // residency argument is made in these numbers.
    let Some(s) = constrained(Residency::Slots(64)) else {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_QWEN3MOE_MODEL=<file.gguf> to run");
        return;
    };
    let n = 8;
    s.clear_residency().unwrap();
    s.reset_cache_stats().unwrap();
    let _ = generate(&s, &PROMPT, n);
    let r = s.residency().unwrap();

    let cfg = s.config();
    // 🔴 `n - 1`, not `n`. The last token generated is emitted and never fed back, so a run of
    // `n` tokens is `prompt + n - 1` decode steps. Getting this wrong looks exactly like a
    // miscounting cache.
    let steps = (PROMPT.len() + n - 1) as u64;
    assert_eq!(r.stats.steps, steps * cfg.n_block as u64, "one admission per block per token");
    assert_eq!(r.stats.demands, steps * (cfg.n_block * cfg.n_expert_used) as u64);
    assert_eq!(
        r.stats.hits,
        0,
        "a pool below one token's working set cannot hit; it measured {}",
        r.stats.hit_rate()
    );
    assert_eq!(r.stats.misses, r.stats.demands);
    assert!(r.bytes_staged > 0);
    assert!(
        r.bytes_staged <= r.stats.misses * r.slot_bytes,
        "staged {} B for {} misses of at most {} B each",
        r.bytes_staged,
        r.stats.misses,
        r.slot_bytes
    );
}
