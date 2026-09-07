//! End-to-end proof: Rust loads a GGUF, puts it on the Arc B580 through
//! llama.cpp's SYCL backend, and generates tokens.
//!
//! ```text
//! source /opt/intel/oneapi/setvars.sh          # never under `set -u`
//! export ONEAPI_DEVICE_SELECTOR=level_zero:0   # the B580, not the iGPU
//! cargo run -p moearc-llama --features runtime --example generate -- \
//!     /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf
//! ```
//!
//! 🔴 This prints timings, and they are **not benchmarks**. A single unpinned run
//! on a shared box is a smoke test; it says the path works, not how fast it is.
//! `moearc bench` is the thing that produces numbers anyone may quote.

use std::process::ExitCode;

use moearc_llama::{ContextParams, Model, ModelParams, Sampler};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: generate <model.gguf> [prompt] [max_tokens]");
        return ExitCode::FAILURE;
    };
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".to_string());
    let max_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(24);

    let verbose = std::env::var_os("MOEARC_VERBOSE").is_some();
    moearc_llama::init(verbose);

    // Everything on the GPU by default. OLMoE at 3.9 GiB fits the B580's 11.33 GiB
    // with room to spare, so there is nothing to trade.
    //
    // MOEARC_NCMOE exists so the `-ncmoe` expansion can be exercised against a real
    // load: it is the one parameter whose C-API mapping is not a scalar field, so
    // "it compiles" is not evidence that it does anything. Compare the reported
    // per-buffer sizes with and without it.
    let n_cpu_moe: i32 =
        std::env::var("MOEARC_NCMOE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    // MOEARC_NO_MMAP switches load_mode to None so that the reported CPU buffer
    // size is a sum of tensor bytes rather than the span of an mmap.
    let load_mode = if std::env::var_os("MOEARC_NO_MMAP").is_some() {
        moearc_llama::LoadMode::None
    } else {
        moearc_llama::LoadMode::Auto
    };
    let mp = ModelParams { n_gpu_layers: -1, n_cpu_moe, load_mode, ..ModelParams::default() };
    println!("  n_cpu_moe     {n_cpu_moe} -> {} override pattern(s)", mp.cpu_moe_patterns().len());

    println!("loading {path}");
    let model = match Model::load(&path, &mp) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("  desc          {}", model.describe());
    println!("  params        {:.2} B", model.n_params() as f64 / 1e9);
    println!("  size          {:.2} GiB", model.size_bytes() as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  blocks        {}", model.n_layer());
    println!("  n_embd        {}", model.n_embd());
    println!("  n_ctx_train   {}", model.n_ctx_train());
    println!("  vocab         {}", model.vocab_len());
    match (model.n_expert(), model.n_expert_used()) {
        (Some(n), Some(k)) => println!("  experts       {n} per block, {k} routed per token"),
        _ => println!("  experts       none (dense)"),
    }

    // n_threads is set explicitly and printed. It is never left implicit: an
    // unstated thread count is how every llama.cpp comparison in this project got
    // withdrawn.
    let cp = ContextParams { n_ctx: 2048, ..ContextParams::default() };
    println!("  n_threads     {} (explicit)", cp.n_threads);

    let mut ctx = match moearc_llama::Context::new(&model, &cp) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("  n_ctx         {}", ctx.n_ctx());

    let mut sampler = Sampler::greedy();

    println!("\nprompt: {prompt:?}");
    let run = match moearc_llama::generate(&mut ctx, &mut sampler, &prompt, max_tokens) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("output: {:?}", run.text);
    println!("tokens: {:?}", run.tokens);
    println!("stopped on: {}", if run.hit_eog { "end-of-generation" } else { "token budget" });

    // Report throughput only if there were tokens behind it.
    match run.perf.decode_tok_per_s() {
        Some(t) => println!(
            "\n[smoke timing, NOT a benchmark] {} tokens in {:.1} ms = {t:.2} tok/s",
            run.perf.n_decode, run.perf.t_decode_ms
        ),
        None => println!("\n[no tokens decoded — no throughput to report]"),
    }

    if run.tokens.is_empty() {
        eprintln!("error: generated nothing");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
