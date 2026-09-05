//! The forward pass against llama.cpp, on real hardware.
//!
//! Skipped unless both of these are set, so the suite still runs on a machine with no card and
//! no model:
//!
//! ```text
//! MOEARC_TEST_GPU=1
//! MOEARC_OLMOE_MODEL=/path/to/olmoe-1b-7b-0924-instruct-q4_k_m.gguf
//! ```
//!
//! # What is asserted, and why it is not "the logits are equal"
//!
//! 🔴 MoEArc and llama.cpp do not compute the same function, and the difference is llama.cpp's,
//! not this engine's. Every K-quant matmul in `ggml-cpu` has `vec_dot_type = GGML_TYPE_Q8_K`:
//! the f32 *activation* is quantised to 8 bits with one scale per 256 elements before the dot
//! product. `moearc-kernels` keeps the activation in f32. So a bit-exact comparison is not
//! available at any tolerance, and picking a tolerance that happens to pass would be picking a
//! number rather than measuring one.
//!
//! What is available is the decision the logits encode. The recorded ids below came from
//! llama.cpp's own greedy decode of the same prompt on the same file; matching them token for
//! token is the strongest statement that can be made, and it is the one that fails loudly if
//! any of the ~200 operations per token is wrong.
//!
//! # How large the unavoidable difference is — measured, not argued
//!
//! For the single-token prompt `12092` on this file, comparing final logits (`result_output`,
//! 50304 wide) three ways, where `1-cos` is the angle between the two logit vectors:
//!
//! ```text
//!   MoEArc (B580, SYCL)  vs llama.cpp CPU      max|d| 5.25e-1   1-cos 5.68e-3
//!   llama.cpp Vulkan     vs llama.cpp CPU      max|d| 5.29e-1   1-cos 6.81e-3
//!   MoEArc (B580, SYCL)  vs llama.cpp Vulkan   max|d| 1.58e-1   1-cos 6.22e-4
//! ```
//!
//! llama.cpp's own two backends disagree with each other slightly *more* than MoEArc disagrees
//! with either, on the same file and the same token, and MoEArc is an order of magnitude closer
//! to llama.cpp's GPU backend than that backend is to its own CPU one. That is the calibration
//! any tolerance here would have to be set against, and it is why this file asserts on decisions
//! rather than on floats.

#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::OnceLock;

use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions, StopReason};

/// One session for the whole file: the model is 3.9 GiB of VRAM and Rust runs tests in
/// parallel, so a session per test would try to hold several copies of it at once.
fn session() -> Option<&'static Session> {
    static S: OnceLock<Option<Session>> = OnceLock::new();
    S.get_or_init(|| {
        if std::env::var("MOEARC_TEST_GPU").ok().as_deref() != Some("1") {
            return None;
        }
        let path = PathBuf::from(std::env::var("MOEARC_OLMOE_MODEL").ok()?);
        match Session::load(&path) {
            Ok(s) => Some(s),
            Err(e) => panic!("MOEARC_OLMOE_MODEL is set but the model would not load: {e}"),
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
                    "skipped: set MOEARC_TEST_GPU=1 and MOEARC_OLMOE_MODEL=<file.gguf> to run"
                );
                return;
            }
        }
    };
}

/// `The capital of France is`, tokenised by llama.cpp with `add_bos = false` — which is what
/// this file's `tokenizer.ggml.add_bos_token` says.
const PROMPT: [u32; 5] = [510, 5347, 273, 6181, 310];

/// llama.cpp's greedy continuation of [`PROMPT`], ids in order.
///
/// Recorded from `llama-eval-callback` on the CPU backend with argmax written out by hand
/// rather than taken from a sampler chain — at temperature 0 a chain still runs top-k, top-p
/// and min-p, and a tie broken differently there would look like a model difference.
const LLAMA_CPP_GREEDY: [u32; 60] = [
    7785, 15, 187, 187, 510, 14731, 273, 6181, 310, 14029, 15, 187, 187, 510, 3072, 273, 6181, 310,
    9963, 15, 19, 3041, 952, 15, 187, 187, 510, 3565, 3448, 273, 6181, 310, 5112, 15, 187, 187,
    510, 3872, 802, 12404, 273, 6181, 310, 253, 11414, 280, 687, 7337, 15, 187, 187, 510, 3872,
    7908, 273, 6181, 310, 253, 492, 49122,
];

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
    assert_eq!(c.arch, "olmoe");
    assert_eq!(c.n_block, 16);
    assert_eq!(c.n_embd, 2048);
    assert_eq!(c.n_head, 16);
    // No GQA in this model: the grouped path in `attn_decode` is not exercised by it.
    assert_eq!(c.n_head_kv, 16);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.n_rot, 128, "RoPE must rotate the whole head");
    assert_eq!(c.n_ff, 1024);
    assert_eq!(c.n_expert, 64);
    assert_eq!(c.n_expert_used, 8);
    assert_eq!(c.n_vocab, 50304);
    assert_eq!(c.rms_eps, 1e-5);
    assert_eq!(c.rope_freq_base, 10_000.0);
    // 🔴 The same id is both. A generated 50279 ends the turn; a prompt one does not.
    assert_eq!(c.bos, Some(50279));
    assert_eq!(c.eos, Some(50279));
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
fn the_run_is_reproducible() {
    // A residency cache and a page allocator both carry state across calls. If either leaked
    // into the arithmetic, the second run of the same prompt would not be the first.
    let s = session_or_skip!();
    let a = generate(s, &PROMPT, 12);
    let b = generate(s, &PROMPT, 12);
    assert_eq!(a, b);
}

#[test]
fn a_prompt_longer_than_one_kv_page_still_matches() {
    // The KV cache pages at 32 tokens, so a 60-token generation crosses two page boundaries and
    // the block table stops being `[0]`. A kernel that assumed contiguous pages passes every
    // short test and fails here.
    let s = session_or_skip!();
    let got = generate(s, &PROMPT, 40);
    assert_eq!(got, LLAMA_CPP_GREEDY[..40].to_vec());
}

#[test]
fn a_false_callback_stops_generation_promptly() {
    let s = session_or_skip!();
    let mut seen = 0usize;
    let stop = StopConditions { max_tokens: 50, stop_tokens: Vec::new() };
    let stats = s
        .generate(&PROMPT, &stop, &mut |_| {
            seen += 1;
            seen < 3
        })
        .unwrap();
    assert_eq!(seen, 3, "the engine kept going after `on_token` said stop");
    assert_eq!(stats.stop_reason, StopReason::Cancelled);
    assert_eq!(stats.completion_tokens, 3);
}

#[test]
fn a_stop_token_ends_the_turn_and_is_not_emitted() {
    let s = session_or_skip!();
    // The second id llama.cpp produces for this prompt. Stopping on it must yield exactly the
    // first, and must not report it as generated.
    let stop = StopConditions { max_tokens: 20, stop_tokens: vec![LLAMA_CPP_GREEDY[1]] };
    let mut out = Vec::new();
    let stats = s
        .generate(&PROMPT, &stop, &mut |t| {
            out.push(t);
            true
        })
        .unwrap();
    assert_eq!(out, vec![LLAMA_CPP_GREEDY[0]]);
    assert_eq!(stats.stop_reason, StopReason::EndOfTurn);
    assert_eq!(stats.completion_tokens, 1);
    assert_eq!(stats.prompt_tokens, PROMPT.len());
}

#[test]
fn the_logits_are_a_vocabulary_and_the_argmax_is_the_first_generated_token() {
    let s = session_or_skip!();
    let logits = s.logits(&PROMPT).unwrap();
    assert_eq!(logits.len(), s.vocab_size());
    assert!(logits.iter().all(|v| v.is_finite()), "a NaN or infinity reached the logits");
    assert_eq!(moearc_engine::session::argmax(&logits), LLAMA_CPP_GREEDY[0]);
}

#[test]
fn every_block_is_tapped_and_none_of_them_is_degenerate() {
    let s = session_or_skip!();
    let (_, tap) = s.logits_tapped(&PROMPT).unwrap();
    let c = s.config();
    let mut sums = Vec::new();
    for b in 0..c.n_block {
        let h = tap
            .get(&format!("l_out-{b}"))
            .unwrap_or_else(|| panic!("block {b} produced no residual stream"));
        assert_eq!(h.len(), c.n_embd);
        assert!(h.iter().all(|v| v.is_finite()), "block {b} produced a non-finite activation");
        assert!(h.iter().any(|v| *v != 0.0), "block {b} produced all zeros");

        let experts = tap.get(&format!("ffn_moe_topk-{b}")).expect("no router choice");
        assert_eq!(experts.len(), c.n_expert_used);
        let mut ids: Vec<u32> = experts.iter().map(|v| *v as u32).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), c.n_expert_used, "block {b} routed to the same expert twice");
        assert!(ids.iter().all(|e| (*e as usize) < c.n_expert));

        let w = tap.get(&format!("ffn_moe_weights-{b}")).expect("no router weights");
        assert_eq!(w.len(), c.n_expert_used);
        assert!(w.iter().all(|v| (0.0..=1.0).contains(v)), "block {b} weight outside [0,1]");
        sums.push(w.iter().sum::<f32>());
    }

    // 🔴 `norm_w = false` for OLMoE, so the weights are raw softmax probabilities over all 64
    // experts and the eight selected ones do not have to sum to one. A renormalising router
    // would make *every* block sum to exactly one and would pass every other assertion in this
    // file, so this is the only place that distinction is tested. A single peaked block can
    // legitimately reach one, hence "not all of them".
    assert!(
        sums.iter().any(|s| (s - 1.0).abs() > 1e-3),
        "every block's router weights summed to one — they were renormalised: {sums:?}"
    );
}

/// A session of its own, so a residency setting can be exercised without disturbing the shared
/// one. Returns `None` when the suite is being skipped.
fn constrained(residency: Residency) -> Option<Session> {
    if std::env::var("MOEARC_TEST_GPU").ok().as_deref() != Some("1") {
        return None;
    }
    let path = PathBuf::from(std::env::var("MOEARC_OLMOE_MODEL").ok()?);
    // 512 tokens is more than this file's prompts need, and it keeps the KV cache small enough
    // that a constrained session can sit alongside the shared fully-resident one.
    let opts = SessionOptions { n_ctx: Some(512), residency };
    Some(Session::load_with(&path, opts).expect("constrained session would not load"))
}

#[test]
fn the_pool_is_sized_in_slots_not_experts() {
    let s = session_or_skip!();
    let r = s.info().residency;
    // 🔴 16 blocks x 64 experts. The file says "64 experts"; the engine has to hold 1024 places
    // to put one, and conflating the two is a factor-of-16 error in every budget.
    assert_eq!(r.total_slots, 1024);
    assert_eq!(r.resident_slots, 1024, "the default is everything resident");
    assert!((r.resident_fraction() - 1.0).abs() < f64::EPSILON);
    // A slot has to hold whichever block the router lands in, and `ffn_down_exps` is Q6_K in
    // half the blocks and Q4_K in the other half, so the pool is larger than the bank it holds.
    assert!(
        r.slot_bytes * u64::from(r.total_slots) >= r.expert_bytes,
        "a pool of {} slots x {} B cannot hold {} B of experts",
        r.total_slots,
        r.slot_bytes,
        r.expert_bytes
    );
    assert!(r.dense_bytes > 0 && r.dense_bytes < r.expert_bytes);
}

#[test]
fn constraining_residency_does_not_change_a_single_token_id() {
    // 🔴 The gate on the whole paging path. Residency decides what has to move, never what is
    // computed, so a token id that moves with the budget is a paging bug — a slot read before it
    // was filled, or two experts sharing one. Sessions are built and dropped one at a time so
    // this never holds two pools at once.
    if session().is_none() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_OLMOE_MODEL=<file.gguf> to run");
        return;
    }
    let n = 24;
    for residency in [
        // Above the per-token working set of 128 slots, at it, and far below it.
        Residency::Slots(256),
        Residency::Slots(128),
        Residency::Slots(16),
        // The incumbent, including the one setting where the ring survives between visits to
        // the same block — the case that caught a slot collision the others could not.
        Residency::StaticSplit { resident_blocks: 4 },
        Residency::StaticSplit { resident_blocks: 15 },
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
fn a_thrashing_cache_still_moves_exactly_the_bytes_it_claims() {
    // Sixteen blocks x eight experts is 128 slots per token, so a pool below that cannot hold a
    // token's working set and LRU evicts everything before it is wanted again. The interesting
    // part is not that the hit rate is low — it is that the accounting stays exact, because the
    // whole residency argument is made in these numbers.
    let Some(s) = constrained(Residency::Slots(16)) else {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_OLMOE_MODEL=<file.gguf> to run");
        return;
    };
    let n = 8;
    s.clear_residency().unwrap();
    s.reset_cache_stats().unwrap();
    let _ = generate(&s, &PROMPT, n);
    let r = s.residency().unwrap();

    let cfg = s.config();
    // 🔴 `n - 1`, not `n`. The last token generated is emitted and never fed back: a token the
    // caller has not accepted must not reach the KV cache, or a cancelled request would leave
    // the sequence a token ahead of what the client saw. So a run of `n` tokens is
    // `prompt + n - 1` decode steps, and getting this wrong looks exactly like a miscounting
    // cache.
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
    // Every miss moved three banks out of the mapping. The bytes are counted from the slices
    // uploaded, so this is a cross-check of the counter against the cache, not a restatement.
    assert!(r.bytes_staged > 0);
    assert!(
        r.bytes_staged <= r.stats.misses * r.slot_bytes,
        "staged {} B for {} misses of at most {} B each",
        r.bytes_staged,
        r.stats.misses,
        r.slot_bytes
    );
}
