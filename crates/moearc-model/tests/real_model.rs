//! Integration test against a real model file.
//!
//! Skipped unless `MOEARC_TEST_GGUF` names a GGUF on disk, because the models this is meant for
//! are tens of gigabytes and cannot live in the repo or in CI. The hermetic coverage is the unit
//! tests in `src/lib.rs`, which build a GGUF byte-for-byte in memory; this test exists to catch
//! the things a synthetic file cannot — real quantisation mixes, real hybrid layer layouts, real
//! tokenizer arrays big enough to matter.
//!
//! ```text
//! MOEARC_TEST_GGUF=/path/to/model.gguf cargo test -p moearc-model -- --nocapture
//! ```

use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("MOEARC_TEST_GGUF")?);
    p.is_file().then_some(p)
}

#[test]
fn a_real_moe_model_yields_a_self_consistent_plan_input() {
    let Some(path) = model_path() else {
        eprintln!("skipped: set MOEARC_TEST_GGUF to a GGUF file to run this");
        return;
    };

    let info = moearc_model::inspect(&path).expect("inspect should succeed on a real model");
    println!("{info:#?}");

    // Everything below is a relation that must hold for any MoE model, checked against the file
    // itself rather than against a remembered number for one particular model.
    assert!(info.total_experts > 0);
    assert!(info.active_experts > 0);
    assert!(
        info.active_experts <= info.total_experts,
        "{} experts active out of {}",
        info.active_experts,
        info.total_experts
    );
    assert!(info.block_count > 0);
    assert!(info.embedding_length > 0);
    assert!(info.context_length > 0);
    assert!(info.per_expert_bytes > 0);
    assert!(info.kv_bytes_per_token > 0);
    assert!(info.kv_layers > 0 && info.kv_layers <= info.block_count);
    assert!(info.moe_block_count > 0 && info.moe_block_count <= info.block_count);

    // The tensor bytes must fit inside the file, and — since a GGUF is essentially all tensor
    // data — must account for nearly all of it. The 95% floor is the real check: it is what
    // would catch a byte-size computation that silently used the wrong block geometry, which
    // would otherwise still satisfy "less than the file size".
    assert!(info.weights_bytes < info.file_size);
    assert!(
        info.weights_bytes * 100 > info.file_size * 95,
        "weights {} B is implausibly small for a {} B file",
        info.weights_bytes,
        info.file_size
    );

    // The dense/expert split must be a genuine partition of the tensor index, and each side
    // must be non-empty — a model with no dense weights at all, or no expert weights, would mean
    // the name filter had stopped matching.
    assert_eq!(info.dense_weights_bytes + info.expert_weights_bytes, info.weights_bytes);
    assert!(info.dense_weights_bytes > 0);
    assert!(info.expert_weights_bytes > 0);

    // Experts dominate a MoE model; if they did not, the filter is matching too little.
    assert!(
        info.expert_weights_bytes > info.dense_weights_bytes,
        "expert weights {} B should exceed dense {} B in a MoE model",
        info.expert_weights_bytes,
        info.dense_weights_bytes
    );

    // The reason `expert_weights_bytes` is summed and not reconstructed: the product of the
    // per-block maximum is an upper bound, never below the truth, and strictly above it whenever
    // the file mixes quantisation types across blocks.
    let naive =
        info.per_expert_bytes * u64::from(info.total_experts) * u64::from(info.moe_block_count);
    assert!(naive >= info.expert_weights_bytes);
    if !info.per_expert_bytes_uniform {
        assert!(naive > info.expert_weights_bytes);
        println!(
            "naive estimate {naive} B overstates measured {} B by {:.2}%",
            info.expert_weights_bytes,
            100.0 * (naive - info.expert_weights_bytes) as f64 / info.expert_weights_bytes as f64
        );
    }

    // The whole point of the crate: these are the fields `AutoCacheRequest` needs, and every
    // resident expert must fit in the weights it came from.
    assert!(info.per_expert_bytes * u64::from(info.total_experts) < info.weights_bytes);
}
