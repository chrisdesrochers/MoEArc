//! The forward pass against llama.cpp on gpt-oss-120B — **5.6x the card, in one file**.
//!
//! Skipped unless both of these are set, so the suite still runs on a machine with no card and
//! no model:
//!
//! ```text
//! MOEARC_TEST_GPU=1
//! MOEARC_GPTOSS_MODEL=/path/to/gpt-oss-120b-MXFP4.gguf
//! ```
//!
//! # Why a third gate, when two already exist
//!
//! `olmoe_forward.rs` and `qwen3moe_forward.rs` cover a graph with no biases, no sinks, plain
//! SwiGLU, plain RoPE and a router that softmaxes before its top-k. gpt-oss is the opposite of
//! all five, and each difference is silent — the model runs and emits fluent English with any
//! one of them wrong. Transcribed from `llama_model_openai_moe` in
//! `/zfs/swift/projects/llama.cpp/src/models/openai-moe.cpp`:
//!
//! - **Biases on everything.** Q, K, V and the attention output; the router, added *before* the
//!   top-k so it changes which experts run; and every expert of every bank, added *inside* the
//!   router's weighting rather than after it.
//! - **A per-head attention sink** — one extra logit that joins the softmax denominator and has
//!   no value vector, so a head's weights do not sum to one. Omitting it makes every attention
//!   output uniformly too large, on every block, from the first token.
//! - **No QK-norm at all.**
//! - **`ggml_swiglu_oai`**: the gate clamped above only, the up branch clamped both ways, an
//!   alpha-scaled sigmoid, and a **`+ 1`** on the up branch.
//! - **The router softmaxes *after* the top-k**, over the four selected logits. A softmax over
//!   128 renormalised to 4 is a different vector.
//! - **YaRN RoPE** at `freq_base = 150000`, factor 32 — and llama.cpp's `rope_yarn` has **no
//!   position gate**, so it applies at position 0. There is no short-sequence regime in which
//!   plain RoPE is equivalent.
//!
//! And the experts are **MXFP4**, which is not a K-quant: 4-bit codes against a shared
//! power-of-two E8M0 exponent. Its dequantiser is checked bit-for-bit against llama.cpp's own
//! `to_float` in `moearc-kernels/tests/gguf_crosscheck.rs`.
//!
//! # Sliding-window attention, and why a short test proves nothing about it
//!
//! The file states `attention.sliding_window = 128`, applied by llama.cpp to **alternating**
//! blocks (`set_swa_pattern(2)`: even blocks windowed, odd blocks full causal). This pass
//! implements both regimes; [`N_CTX`] used to be 128 because it did not.
//!
//! 🔴 **Below 128 tokens sliding-window attention is a no-op, so a suite that only tested short
//! contexts would pass with SWA entirely unimplemented.** `is_masked_swa` masks a key when
//! `p1 - p0 >= n_swa`, and no pair of positions inside one window satisfies that — an SWA mask
//! and a plain causal mask are the same mask there. That is why
//! [`greedy_generation_past_the_sliding_window_matches_llama_cpp`] runs a **158-token prompt and
//! 256 generated tokens**, reaching position 413: every windowed block spends the whole run
//! dropping keys the full blocks keep.
//!
//! ⚠️ And the prompt is not incidental. llama.cpp's continuation **reproduces the passage** —
//! a copy of 158 tokens that only the full-attention blocks can see. An engine that windowed
//! every block could not produce it, and one that windowed none would produce it from a
//! different set of keys.
//!
//! # Residency, on the model residency exists for
//!
//! 4,608 slots x 12.607 MiB is **56.7 GiB** of experts against a B580's 11.33 GiB, of which
//! 2.29 GiB is dense weights that never leave. Nothing here can ask for a full pool, and the
//! budgets below are deliberately small so two sessions can be alive at once.

// 🔴 Without this the whole file is a compile error on default features: `moearc_engine::moe`,
// `moearc_engine::session` and the `moearc-kernels` dependency all exist only under `gpu`, so a
// stranger running a plain `cargo test` after cloning would meet `unresolved import` rather than
// a test suite. An inner attribute rather than a `[[test]] required-features` stanza, matching
// the three sibling gates — it leaves the target listed and building, as an empty binary,
// instead of silently absent from the run.
#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::Mutex;

use moearc_engine::moe::{Activation, Config, Residency};
use moearc_engine::session::{Session, SessionOptions, StopConditions};

// The default `SessionOptions` is still used by the refusal test, which needs a field the
// helper below does not expose.

/// Large enough for the long-context gate: a 380-token prompt plus 256 generated tokens reaches
/// position 635, **four windows** past where SWA starts to bite.
const N_CTX: usize = 768;

/// The default budget for a test that does not care about residency.
///
/// 🔴 Small, and the reason is arithmetic rather than taste. A slot here is **12.607 MiB** — the
/// largest of any model in this suite by an order of magnitude — and the dense half is
/// **2.29 GiB** that every session pays again. 160 slots is 1.97 GiB, which puts one session at
/// **4.26 GiB** and keeps it above one step's working set of 36 blocks x 4 experts = 144.
const SHARED_SLOTS: u32 = 160;

/// `The capital city of France is`, tokenised by llama.cpp with `add_bos = false`.
///
/// 🔴 Chosen the way `qwen3moe_forward.rs`'s was, and for the same reason: a greedy id is only
/// a *decision* where the top two logits are further apart than the arithmetic difference
/// between the two implementations. llama.cpp's CPU backend quantises the f32 activation to
/// Q8_0 before every matmul — `vec_dot_type = GGML_TYPE_Q8_0` for MXFP4 too — and
/// `moearc-kernels` keeps it in f32, which on this model reaches `max|d| = 0.29` on the logit
/// vector. This prompt's continuation is a repeating nine-token cycle whose winning logit
/// clears the runner-up by ~2.8, an order of magnitude above that.
///
/// Verified rather than assumed: **llama.cpp's own CPU and SYCL backends produce identical ids
/// on it for all 64 tokens**, measured on this file.
const PROMPT: [u32; 6] = [976, 9029, 5030, 328, 10128, 382];

/// llama.cpp's greedy continuation of [`PROMPT`], ids in order.
///
/// Recorded from `llama-eval-callback` with `MOEARC_GREEDY_N=64`, llama.cpp
/// `e107984bcffcfd701e82738092a2b000b6fda7a2`, and **identical on `-ngl 0` (CPU) and
/// `-ngl 99 --n-cpu-moe 31` (SYCL)**. The argmax is written out by that patch rather than taken
/// from a sampler chain: at temperature 0 a chain still runs top-k, top-p and min-p, and a tie
/// broken differently there would look like a model difference.
const LLAMA_CPP_GREEDY: [u32; 64] = [
    12650, 13, 279, 976, 9029, 5030, 328, 10128, 382, 12650, 13, 279, 976, 9029, 5030, 328, 10128,
    382, 12650, 13, 279, 976, 9029, 5030, 328, 10128, 382, 12650, 13, 279, 976, 9029, 5030, 328,
    10128, 382, 12650, 13, 279, 976, 9029, 5030, 328, 10128, 382, 12650, 13, 279, 976, 9029, 5030,
    328, 10128, 382, 12650, 13, 279, 976, 9029, 5030, 328, 10128, 382, 12650,
];

/// A 380-token prompt in gpt-oss's **harmony** format: a system line, a user turn holding a
/// 16-entry list and the instruction to repeat it verbatim, and an assistant turn opened
/// directly on the `final` channel with the list's first line already written.
///
/// 🔴 Three properties, each chosen and each measured, and the third is the one that took the
/// longest to get right:
///
/// - **It is a copy task**, so every generated token is determined by a source line 380 tokens
///   back — which only the full-attention blocks can see. The windowed blocks must simultaneously
///   *not* see it and not corrupt anything.
/// - **It is in harmony format.** Fed raw prose, this model breaks out of the passage into
///   assistant commentary within a dozen tokens, and commentary is exactly where greedy decoding
///   has no margin.
/// - 🔴 **Its smallest top-1/top-2 margin over the 256 steps is 5.81** (measured with
///   `llama-eval-callback`, which prints the runner-up). That matters because llama.cpp's CPU
///   backend quantises activations to Q8_0 before every matmul and `moearc-kernels` keeps them
///   in f32, a difference worth up to ~0.4 on this model's logits. Below about 1.0 of margin a
///   greedy id is a decision about the tie-break rather than about the model: three earlier
///   candidate prompts for this test had minimum margins of 0.16–0.19, and on one of them
///   **llama.cpp disagreed with itself** — its one-shot prefill and its incremental decode chose
///   different ids at the same position. A test built on such a prompt measures nothing.
///
/// 🔴 Long **on purpose**: the point of this file's long-context gate is that positions past
/// 128 exist at all, and a six-token prompt would have to generate its way there. Starting past
/// the window means every generated token is decoded with the two attention regimes already
/// disagreeing about which keys exist.
const SWA_PROMPT: [u32; 380] = [
    200006, 17360, 200008, 42743, 289, 25, 4465, 200007, 200006, 1428, 200008, 49704, 290, 3992,
    1562, 9707, 11, 483, 860, 1273, 2201, 364, 16, 13, 67062, 382, 290, 42173, 17921, 326, 290,
    31179, 316, 290, 11628, 13, 1225, 853, 220, 15, 182603, 558, 17, 13, 73794, 382, 290, 55357,
    17921, 11, 483, 261, 17154, 22137, 328, 15883, 70513, 13, 1225, 853, 220, 15, 182603, 558, 18,
    13, 16464, 382, 290, 1606, 17921, 5542, 316, 2498, 2615, 13, 1225, 853, 220, 16, 28479, 558,
    19, 13, 31259, 382, 4358, 290, 3592, 17921, 2236, 328, 290, 17069, 64480, 402, 1617, 9753, 13,
    1225, 853, 220, 17, 182603, 558, 20, 13, 79575, 382, 290, 10574, 17921, 11, 261, 7322, 27675,
    483, 261, 2212, 3592, 10021, 13, 1225, 853, 220, 4129, 182603, 558, 21, 13, 69868, 382, 290,
    7322, 27675, 483, 290, 125725, 12796, 2420, 13, 1225, 853, 220, 19302, 182603, 558, 22, 13,
    127440, 385, 382, 290, 12146, 376, 17921, 11, 326, 480, 155086, 402, 1617, 4307, 13, 1225, 853,
    220, 2029, 182603, 558, 23, 13, 141806, 382, 290, 286, 10847, 376, 17921, 11, 326, 290, 21779,
    129300, 591, 290, 11628, 13, 1225, 853, 220, 1125, 182603, 558, 24, 13, 363, 21042, 382, 261,
    124441, 17921, 6772, 290, 61353, 328, 141806, 13, 1225, 853, 220, 15, 182603, 558, 702, 13,
    136765, 382, 261, 124441, 17921, 484, 95853, 290, 61353, 328, 141806, 13, 1225, 853, 220, 20,
    182603, 558, 994, 13, 10694, 3095, 64, 382, 261, 124441, 17921, 40123, 1299, 448, 16102, 13,
    1225, 853, 220, 17, 182603, 558, 899, 13, 31283, 133846, 382, 261, 124441, 17921, 306, 290,
    23393, 19629, 28328, 13, 1225, 853, 220, 16, 28479, 558, 1311, 13, 4225, 276, 382, 261, 124441,
    17921, 945, 18965, 1572, 136765, 13, 1225, 853, 220, 16, 28479, 558, 1265, 13, 33794, 1503,
    382, 261, 51041, 2817, 483, 261, 1869, 1701, 61353, 13, 1225, 853, 220, 15, 182603, 558, 1055,
    13, 117716, 78, 277, 382, 261, 51041, 2817, 483, 261, 24125, 12796, 13, 1225, 853, 220, 16,
    28479, 558, 1125, 13, 145492, 385, 382, 261, 51041, 2817, 4783, 4358, 290, 10329, 95384, 3794,
    13, 1225, 853, 220, 16, 28479, 13, 200007, 200006, 173781, 200005, 17196, 200008, 16, 13,
    67062, 382, 290, 42173, 17921, 326, 290, 31179, 316, 290, 11628, 13, 1225, 853, 220, 15,
    182603, 13,
];

/// llama.cpp's greedy continuation of [`SWA_PROMPT`], ids in order — the list, copied out.
///
/// Recorded from `llama-eval-callback` with `MOEARC_GREEDY_N=256`, llama.cpp
/// `e107984bcffcfd701e82738092a2b000b6fda7a2`, and — **measured, not assumed** — identical
/// between `-ngl 0` (CPU) and `-ngl 99 --n-cpu-moe 31` (SYCL) across all 256 ids.
const SWA_GREEDY: [u32; 256] = [
    4066, 17, 13, 73794, 382, 290, 55357, 17921, 11, 483, 261, 17154, 22137, 328, 15883, 70513, 13,
    1225, 853, 220, 15, 182603, 13, 4066, 18, 13, 16464, 382, 290, 1606, 17921, 5542, 316, 2498,
    2615, 13, 1225, 853, 220, 16, 28479, 13, 4066, 19, 13, 31259, 382, 4358, 290, 3592, 17921,
    2236, 328, 290, 17069, 64480, 402, 1617, 9753, 13, 1225, 853, 220, 17, 182603, 13, 4066, 20,
    13, 79575, 382, 290, 10574, 17921, 11, 261, 7322, 27675, 483, 261, 2212, 3592, 10021, 13, 1225,
    853, 220, 4129, 182603, 13, 4066, 21, 13, 69868, 382, 290, 7322, 27675, 483, 290, 125725,
    12796, 2420, 13, 1225, 853, 220, 19302, 182603, 13, 4066, 22, 13, 127440, 385, 382, 290, 12146,
    376, 17921, 11, 326, 480, 155086, 402, 1617, 4307, 13, 1225, 853, 220, 2029, 182603, 13, 4066,
    23, 13, 141806, 382, 290, 286, 10847, 376, 17921, 11, 326, 290, 21779, 129300, 591, 290, 11628,
    13, 1225, 853, 220, 1125, 182603, 13, 4066, 24, 13, 363, 21042, 382, 261, 124441, 17921, 6772,
    290, 61353, 328, 141806, 13, 1225, 853, 220, 15, 182603, 13, 4066, 702, 13, 136765, 382, 261,
    124441, 17921, 484, 95853, 290, 61353, 328, 141806, 13, 1225, 853, 220, 20, 182603, 13, 4066,
    994, 13, 10694, 3095, 64, 382, 261, 124441, 17921, 40123, 1299, 448, 16102, 13, 1225, 853, 220,
    17, 182603, 13, 4066, 899, 13, 31283, 133846, 382, 261, 124441, 17921, 306, 290, 23393, 19629,
    28328, 13, 1225, 853, 220, 16, 28479, 13, 4066, 1311, 13, 4225, 276, 382, 261, 124441, 17921,
    945, 18965, 1572, 136765,
];

fn model_path() -> Option<PathBuf> {
    if std::env::var("MOEARC_TEST_GPU").ok().as_deref() != Some("1") {
        return None;
    }
    std::env::var("MOEARC_GPTOSS_MODEL").ok().map(PathBuf::from)
}

/// 🔴 **Exactly one session on the card at a time, across this whole file.**
///
/// The sibling gates keep one long-lived session in a `OnceLock` and build extra ones beside it.
/// That does not work here and the arithmetic says why: this model's dense half is **2.29 GiB**,
/// paid again by every session, so two of them are 4.6 GiB before a single expert is placed.
/// Rust runs a file's tests on eight threads and `cargo test --workspace` runs the three
/// forward-pass binaries as **concurrent processes**, so the peak is not two sessions but
/// several, on an 11.33 GiB card also holding Qwen3-30B-A3B and OLMoE.
///
/// ⚠️ And the failure is not an allocation error. Measured: three binaries at once put one
/// Level Zero thread into a **100%-CPU spin that does not return** — the same behaviour
/// `Residency::All` documents near the memory boundary, where allocations keep succeeding past
/// the point the memory exists. A hung test is worse than a failing one, so this file does not
/// go near that boundary: the lock below means one 4.26 GiB session, built and dropped, and
/// never a second.
///
/// The cost is that each test reloads the dense half — about 20 seconds off a warm page cache
/// against a 59 GiB file. That is the price of a gate that cannot hang.
static DEVICE: Mutex<()> = Mutex::new(());

/// Build one session, run `f` against it, and drop it before the lock is released.
///
/// Returns `None` when the suite is being skipped, so a test can report that rather than pass
/// silently.
fn with_session<T>(residency: Residency, host: &str, f: impl FnOnce(&Session) -> T) -> Option<T> {
    let path = model_path()?;
    // Poisoning is deliberately ignored: a panic in one test must not turn the other six into
    // spurious failures with a different message.
    let _guard = DEVICE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let opts =
        SessionOptions { n_ctx: Some(N_CTX), residency, host: host.parse().expect("host policy") };
    let session = match Session::load_with(&path, opts) {
        Ok(s) => s,
        Err(e) => panic!("MOEARC_GPTOSS_MODEL is set but the model would not load: {e}"),
    };
    Some(f(&session))
}

/// [`with_session`] at the default budget with no host executor.
fn with_default<T>(f: impl FnOnce(&Session) -> T) -> Option<T> {
    with_session(Residency::Slots(SHARED_SLOTS), "off", f)
}

fn skipped() {
    eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_GPTOSS_MODEL=<file.gguf> to run");
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
    let Some(()) = with_default(|s| {
        let c: &Config = s.config();

        assert_eq!(c.arch, "gpt-oss");
        assert_eq!(c.n_block, 36);
        assert_eq!(c.n_embd, 2880);
        assert_eq!(c.n_head, 64);
        assert_eq!(c.n_head_kv, 8);

        // 🔴 `attention.key_length`, not `n_embd / n_head`. The quotient is 45, which is not even a
        // whole head, and a file this shape would still produce fluent text with the wrong one.
        assert_eq!(c.head_dim, 64);
        // No `rope.dimension_count` key, so `n_rot` falls back to `head_dim` — full rotary.
        assert_eq!(c.n_rot, 64);
        // `expert_feed_forward_length`. This file states `feed_forward_length` as well and the two
        // happen to be equal, which is why the key is read rather than inferred from the equality.
        assert_eq!(c.n_ff, 2880);

        assert_eq!(c.n_expert, 128);
        assert_eq!(c.n_expert_used, 4);
        assert_eq!(c.n_vocab, 201_088);
        assert_eq!(c.n_ctx_train, 131_072);

        assert!((c.rms_eps - 1e-5).abs() < 1e-11, "rms_eps was {}", c.rms_eps);
        // 🔴 150000, not 10000 and not Qwen3's 1e6.
        assert_eq!(c.rope_freq_base, 150_000.0);

        // The switches no GGUF key records.
        assert!(!c.has_qk_norm, "gpt-oss has no attn_q_norm/attn_k_norm");
        assert_eq!(c.gating, moearc_kernels::Gating::SoftmaxAfterTopK);
        assert_eq!(c.act, Activation::SwigluOai { alpha: 1.702, limit: 7.0 });
        assert!(c.has_attn_bias && c.has_router_bias && c.has_expert_bias && c.has_sinks);

        // YaRN, and the fact that matters about it: it is present, so it applies from token 0.
        let y = c.rope_scaling.expect("rope.scaling.type = yarn");
        assert_eq!(y.freq_scale, 1.0 / 32.0);
        assert_eq!(y.ext_factor, 1.0);
        // 🔴 1.0, **not** the YaRN paper's 1.3466. llama.cpp divides the mscale back out of
        // `cparams.yarn_attn_factor` so that the kernel can multiply it in; passing 1.3466 here
        // squares it to 1.8133.
        assert_eq!(y.attn_factor, 1.0);
        // `ggml_rope_yarn_corr_dims(64, 4096, 150000, 32, 1)`, computed by hand:
        //   corr(32) = 64*ln(4096/(32*2pi))/(2*ln 150000) = 8.09 -> floor -> 8
        //   corr(1)  = 64*ln(4096/(1*2pi))/(2*ln 150000)  = 17.40 -> ceil -> 18
        assert_eq!((y.corr_lo, y.corr_hi), (8.0, 18.0));

        // Declared, carried, and not implemented — see the module header.
        assert_eq!(c.n_swa, Some(128));

        assert_eq!(c.bos, Some(199_998));
        assert_eq!(c.eos, Some(200_002));
    }) else {
        return skipped();
    };
}

#[test]
fn greedy_generation_matches_llama_cpp_token_for_token() {
    let Some(()) = with_default(|s| {
        let got = generate(s, &PROMPT, LLAMA_CPP_GREEDY.len());
        assert_eq!(
            got,
            LLAMA_CPP_GREEDY.to_vec(),
            "greedy ids diverged from llama.cpp; first difference at index {:?}",
            got.iter().zip(LLAMA_CPP_GREEDY.iter()).position(|(a, b)| a != b)
        );
    }) else {
        return skipped();
    };
}

#[test]
fn every_block_is_tapped_and_none_of_them_is_degenerate() {
    let Some(()) = with_default(|s| {
        let (logits, tap) = s.logits_tapped(&PROMPT).unwrap();
        let c = s.config();
        for b in 0..c.n_block {
            for stage in ["ffn_inp", "ffn_moe_logits", "ffn_moe_out", "l_out"] {
                let name = format!("{stage}-{b}");
                let h = tap.get(&name).unwrap_or_else(|| panic!("no tap for {name}"));
                assert!(h.iter().all(|v| v.is_finite()), "{name} holds a NaN or infinity");
                assert!(h.iter().any(|v| *v != 0.0), "{name} is all zeros");
            }
            // The tap emitted only where a router bias exists — which is this architecture, and is
            // why the bias is asserted to *change* the logits rather than merely to be present.
            let raw = tap.get(&format!("ffn_moe_logits-{b}")).expect("raw router logits");
            let biased = tap.get(&format!("ffn_moe_probs-{b}")).expect("biased router logits");
            assert_ne!(raw, biased, "block {b}: the router bias changed nothing");
        }
        assert_eq!(logits.len(), s.vocab_size());
        assert!(logits.iter().all(|v| v.is_finite()), "a NaN or infinity reached the logits");
        assert_eq!(moearc_engine::session::argmax(&logits), LLAMA_CPP_GREEDY[0]);
    }) else {
        return skipped();
    };
}

#[test]
fn the_router_softmaxes_after_the_top_k_and_its_weights_sum_to_one() {
    let Some(()) = with_default(|s| {
        let c = s.config();
        let (_, tap) = s.logits_tapped(&PROMPT).unwrap();
        for b in 0..c.n_block {
            let w = tap.get(&format!("ffn_moe_weights-{b}")).expect("router weights");
            assert_eq!(w.len(), c.n_expert_used);
            let sum: f64 = w.iter().map(|v| f64::from(*v)).sum();
            assert!((sum - 1.0).abs() < 1e-5, "block {b}: weights summed to {sum}");

            // 🔴 The distinguishing check. Under `SoftmaxAfterTopK` the weights are a softmax of the
            // four selected *logits*, so their ratios are set by those four alone. Under
            // `SoftmaxNormalised` they would be a softmax over all 128 renormalised to 4 — which
            // also sums to one, and is a different vector. Recomputing the first ratio from the
            // logits separates the two.
            let logits = tap.get(&format!("ffn_moe_probs-{b}")).expect("biased router logits");
            let idx = tap.get(&format!("ffn_moe_topk-{b}")).expect("router choice");
            let (l0, l1) = (logits[idx[0] as usize], logits[idx[1] as usize]);
            let want = f64::from(l1 - l0).exp();
            let got = f64::from(w[1]) / f64::from(w[0]);
            assert!(
                (got - want).abs() < 1e-4,
                "block {b}: w1/w0 = {got}, but exp(l1 - l0) over the selected logits = {want}"
            );
        }
    }) else {
        return skipped();
    };
}

#[test]
fn greedy_generation_past_the_sliding_window_matches_llama_cpp() {
    // 🔴 The gate this file exists for now. 380 prompt tokens + 256 generated is position 635,
    // and every windowed block is dropping keys that every full block still holds from the very
    // first step. A single wrong id anywhere in the 256 means the two regimes are not where
    // llama.cpp puts them — and below 128 the same test would pass with no SWA at all.
    //
    // ⚠️ **Measured, so that a future failure here is informative:** with `Config::is_swa_block`
    // forced to `false` — every block full causal, which is what this engine did before SWA —
    // this test fails at generated index **0**. Not eventually, and not by drift: the prompt is
    // already 252 tokens past the window when generation starts, so the very first token an
    // unwindowed engine produces is wrong.
    let Some(()) = with_default(|s| {
        let got = generate(s, &SWA_PROMPT, SWA_GREEDY.len());
        let first = got.iter().zip(SWA_GREEDY.iter()).position(|(a, b)| a != b);
        assert_eq!(
            got,
            SWA_GREEDY.to_vec(),
            "greedy ids diverged from llama.cpp; first difference at index {first:?} — the \
             prompt is already {} tokens past the window before a single token is generated",
            SWA_PROMPT.len().saturating_sub(128)
        );
    }) else {
        return skipped();
    };
}

#[test]
fn the_windowed_blocks_hold_a_short_kv_cache_and_the_full_ones_do_not() {
    // 🔴 The other half of sliding-window attention, and the half that is worth memory rather
    // than only correctness: a block that can only ever see 128 positions does not need a cache
    // of `n_ctx` positions. On this card the KV cache and the expert pool are drawn from the
    // same 11.33 GiB, so every byte given back here is a byte an expert slot can have.
    let Some(()) = with_default(|s| {
        let info = s.info();
        let kv = info.kv;
        let c = s.config();
        assert_eq!(c.n_swa, Some(128));
        assert_eq!(c.swa_pattern, 2);
        assert_eq!(kv.swa_blocks, 18, "18 of gpt-oss-120B's 36 blocks are windowed");
        assert_eq!(kv.n_block, 36);

        // 768 tokens is 24 pages; a 128-token window is 4.
        assert_eq!((kv.pages, kv.swa_pages), (24, 4));
        assert!(kv.bytes < kv.bytes_full);
        // 18 blocks at 4/24 of the cache: (18*24 + 18*4) / (36*24) = 7/12.
        assert_eq!(kv.bytes * 12, kv.bytes_full * 7);
        eprintln!(
            "KV at n_ctx {}: {} KiB against {} KiB unwindowed, {} KiB given back",
            info.n_ctx,
            kv.bytes >> 10,
            kv.bytes_full >> 10,
            kv.saved_bytes() >> 10
        );
    }) else {
        return skipped();
    };
}

#[test]
fn constraining_residency_does_not_change_a_single_token_id() {
    // 🔴 The gate on the paging path at 5.6x the card. A step's working set is
    // 36 blocks x 4 experts = 144 slots, and no budget here holds the model — 4,608 slots is
    // 56.7 GiB. Residency decides what has to move, never what is computed, so an id that
    // changes with the budget is a paging bug: a slot read before it was filled, or two experts
    // sharing one. Sessions are built and dropped one at a time so this never holds two pools.
    if model_path().is_none() {
        return skipped();
    }
    let n = 24;
    for residency in [
        // Just above the per-token working set of 144 slots, at it, and far below it.
        Residency::Slots(160),
        Residency::Slots(144),
        Residency::Slots(16),
        // The incumbent. ⚠️ Only one split fits at all: a pinned block is 128 slots,
        // 1.58 GiB, so two would exceed what is left beside the shared session.
        Residency::StaticSplit { resident_blocks: 1 },
    ] {
        with_session(residency, "off", |s| {
            let got = generate(s, &PROMPT, n);
            assert_eq!(
                got,
                LLAMA_CPP_GREEDY[..n].to_vec(),
                "{residency:?} changed the output; first difference at index {:?}",
                got.iter().zip(LLAMA_CPP_GREEDY.iter()).position(|(a, b)| a != b)
            );
            let r = s.residency().expect("residency");
            assert!(r.resident_slots <= r.total_slots);
            assert_eq!(r.stats.demands, r.stats.hits + r.stats.misses);
        })
        .expect("checked above");
    }
}

#[test]
fn a_host_policy_changes_where_an_expert_runs_and_not_what_it_computes() {
    // 🔴 The host executor has its own MXFP4 kernel, its own bias handling and its own copy of
    // `swiglu_oai`. Three chances to compute a different function from the device — and the
    // policy that selects it is a *performance* knob, so a divergence here would make
    // throughput and correctness the same setting.
    if model_path().is_none() {
        return skipped();
    }
    let n = 24;
    for spec in ["frac:0.5", "frac:1.0"] {
        with_session(Residency::Slots(144), spec, |s| {
            let got = generate(s, &PROMPT, n);
            assert_eq!(
                got,
                LLAMA_CPP_GREEDY[..n].to_vec(),
                "host policy {spec} changed the output; first difference at index {:?}",
                got.iter().zip(LLAMA_CPP_GREEDY.iter()).position(|(a, b)| a != b)
            );
        })
        .expect("checked above");
    }
}
