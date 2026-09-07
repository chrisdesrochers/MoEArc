//! Tests that need a linked llama.cpp.
//!
//! The whole file compiles to nothing without the `runtime` feature, so a default
//! `cargo test --workspace` neither builds nor runs any of it.
//!
//! ```text
//! source /opt/intel/oneapi/setvars.sh
//! ONEAPI_DEVICE_SELECTOR=level_zero:0 \
//!   cargo test -p moearc-llama --features runtime -- --include-ignored
//! ```

#![cfg(feature = "runtime")]

use std::path::PathBuf;

use moearc_llama::{ContextParams, Model, ModelParams};

/// The reference model: smallest MoE in the catalog, 3.9 GiB.
fn model_path() -> Option<PathBuf> {
    let p = std::env::var("MOEARC_TEST_MODEL").map_or_else(
        |_| PathBuf::from("/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf"),
        PathBuf::from,
    );
    p.is_file().then_some(p)
}

/// llama.cpp's defaults, read from the linked library rather than assumed.
///
/// 🔴 This test was written asserting `n_gpu_layers == 0` — the value llama.cpp
/// documentation and most third-party write-ups still give — and it **failed**.
/// At the pinned commit `e107984bc`, `llama_model_default_params()` in
/// `src/llama-model.cpp` returns `-1`: upstream now offloads everything by
/// default. The old value was not a harmless staleness. It is the difference
/// between "MoEArc's -1 is a load-bearing override" and "MoEArc's -1 agrees with
/// upstream today and will stop being checked by anyone".
///
/// That is the whole reason this test exists: a default that silently moves under
/// you produces a slow run, not a failed one, and this project has already had to
/// retract results for exactly that class of error.
#[test]
fn llama_defaults_are_what_moearc_assumes() {
    let d = moearc_llama::llama_defaults();
    assert_eq!(
        d.n_gpu_layers, -1,
        "llama.cpp's default n_gpu_layers moved; re-check whether MoEArc's -1 is still right"
    );
    assert_eq!(d.main_gpu, 0);
    // F16 == GGML_TYPE_F16 == 1, for both halves of the KV cache.
    assert_eq!(d.type_k, 1, "llama.cpp's default K cache type moved");
    assert_eq!(d.type_v, 1, "llama.cpp's default V cache type moved");
    assert!(d.offload_kqv, "KV offload is no longer on by default");
    assert!(d.n_batch > 0 && d.n_ubatch > 0);
}

/// Metadata and tokenizer, with no weights and no GPU.
#[test]
fn vocab_only_load_reads_metadata_and_tokenizes() {
    let Some(path) = model_path() else {
        eprintln!("skipping: reference model not present");
        return;
    };

    let model = Model::load(&path, &ModelParams::vocab_only()).expect("vocab-only load");

    assert_eq!(model.n_expert(), Some(64), "OLMoE has 64 experts per block");
    assert_eq!(model.n_expert_used(), Some(8), "OLMoE routes 8 experts per token");
    assert!(model.vocab_len() > 50_000);

    let toks = model.tokenize("The capital of France is", true).expect("tokenize");
    assert!(!toks.is_empty());

    // Round-trip: rendering every token back must reproduce the input text.
    // Pieces are bytes, so accumulate and decode once — a token boundary can
    // split a multi-byte character.
    let mut bytes = Vec::new();
    for t in &toks {
        bytes.extend_from_slice(&model.token_bytes(*t, false));
    }
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("capital of France"), "detokenized text did not round-trip: {text:?}");
}

/// A model asked about experts when it has none must answer `None`, not 0.
#[test]
fn expert_metadata_is_absent_not_zero_for_missing_keys() {
    let Some(path) = model_path() else {
        eprintln!("skipping: reference model not present");
        return;
    };
    let model = Model::load(&path, &ModelParams::vocab_only()).expect("vocab-only load");
    assert_eq!(model.meta("this.key.does.not.exist"), None);
}

/// The full path: weights on the Arc via SYCL, greedy decode, real tokens.
///
/// `#[ignore]` because it needs the card, and another agent may be measuring on
/// it. Run deliberately with `--include-ignored`.
#[test]
#[ignore = "requires the Arc B580; run deliberately"]
fn generates_tokens_on_the_gpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: reference model not present");
        return;
    };

    let mp = ModelParams { n_gpu_layers: -1, ..ModelParams::default() };
    let model = Model::load(&path, &mp).expect("load onto GPU");

    let cp = ContextParams { n_ctx: 512, ..ContextParams::default() };
    let mut ctx = moearc_llama::Context::new(&model, &cp).expect("context");
    let mut sampler = moearc_llama::Sampler::greedy();

    let run = moearc_llama::generate(&mut ctx, &mut sampler, "The capital of France is", 8)
        .expect("generate");

    assert!(!run.tokens.is_empty(), "generated no tokens");
    assert!(
        run.text.contains("Paris"),
        "greedy decode should be deterministic and name Paris; got {:?}",
        run.text
    );

    // Perf must refuse to report throughput it cannot justify.
    assert!(run.perf.decode_tok_per_s().is_some());
}

/// Greedy decode must be reproducible: the same prompt twice, the same token ids.
///
/// This is the property that makes cross-engine comparison meaningful at all.
#[test]
#[ignore = "requires the Arc B580; run deliberately"]
fn greedy_decode_is_deterministic() {
    let Some(path) = model_path() else {
        eprintln!("skipping: reference model not present");
        return;
    };

    let mp = ModelParams { n_gpu_layers: -1, ..ModelParams::default() };
    let model = Model::load(&path, &mp).expect("load");
    let cp = ContextParams { n_ctx: 512, ..ContextParams::default() };

    let mut first = None;
    for _ in 0..2 {
        let mut ctx = moearc_llama::Context::new(&model, &cp).expect("context");
        let mut s = moearc_llama::Sampler::greedy();
        let run = moearc_llama::generate(&mut ctx, &mut s, "The capital of France is", 8)
            .expect("generate");
        match &first {
            None => first = Some(run.tokens),
            Some(prev) => assert_eq!(prev, &run.tokens, "greedy decode is not reproducible"),
        }
    }
}

/// 🔴 The `-ncmoe` mapping, asserted against a real load.
///
/// `n_cpu_moe` is the one tuning knob with no scalar field behind it — it expands
/// to N tensor-buffer-override regexes — so "it compiles" proves nothing. This
/// asserts it actually moves weight off the GPU.
///
/// It also pins the finding that `-ncmoe` is **not** `-ngl`: the layer count
/// offloaded is unchanged, because the patterns match only `*_exps` tensors.
#[test]
#[ignore = "requires the Arc B580; run deliberately"]
fn n_cpu_moe_moves_expert_weight_off_the_gpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: reference model not present");
        return;
    };

    // load_mode None: with mmap, the reported CPU buffer is the *span* of the
    // mapping and includes GPU-resident tensors lying between the CPU-resident
    // ones, so it overstates host residency. Measured on OLMoE at -ncmoe 8:
    // 2254.16 MiB mapped against 1915.27 MiB actually host-side.
    let base = ModelParams {
        n_gpu_layers: -1,
        load_mode: moearc_llama::LoadMode::None,
        ..ModelParams::default()
    };

    let none = Model::load(&path, &base).expect("load with no offload");
    let total_none = none.size_bytes();
    drop(none);

    let offloaded = ModelParams { n_cpu_moe: 8, ..base };
    let some = Model::load(&path, &offloaded).expect("load with -ncmoe 8");

    // The model is the same model either way; offloading moves weights between
    // buffers, it does not change how much weight there is.
    assert_eq!(some.size_bytes(), total_none, "-ncmoe changed the model's size");
    assert_eq!(some.n_layer(), 16, "-ncmoe changed the block count");
}
