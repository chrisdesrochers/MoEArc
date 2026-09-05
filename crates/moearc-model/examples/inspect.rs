//! Print what `moearc-model` reads out of a GGUF file.
//!
//! ```text
//! cargo run -p moearc-model --example inspect -- /path/to/model.gguf
//! ```
//!
//! Exists so the crate's output can be eyeballed against another tool's — `llama-gguf`, or
//! anything else that reads the same file — without writing a throwaway binary each time.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: inspect <model.gguf>");
        return ExitCode::FAILURE;
    };

    let info = match moearc_model::inspect(&path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("architecture            {}", info.architecture);
    println!("name                    {}", info.name.as_deref().unwrap_or("<unset>"));
    println!("block_count             {}", info.block_count);
    println!(
        "moe_block_count         {} ({} residency slots)",
        info.moe_block_count,
        u64::from(info.moe_block_count) * u64::from(info.total_experts)
    );
    println!("embedding_length        {}", info.embedding_length);
    println!("context_length          {}", info.context_length);
    println!("total_experts           {}", info.total_experts);
    println!("active_experts          {}", info.active_experts);
    println!(
        "per_expert_bytes        {} ({:.3} MiB){}",
        info.per_expert_bytes,
        info.per_expert_bytes as f64 / (1 << 20) as f64,
        if info.per_expert_bytes_uniform { "" } else { "  [max; blocks disagree]" }
    );
    println!(
        "weights_bytes           {} ({:.3} GiB)",
        info.weights_bytes,
        info.weights_bytes as f64 / (1u64 << 30) as f64
    );
    println!(
        "  expert_weights_bytes  {} ({:.3} GiB, {:.1}%)",
        info.expert_weights_bytes,
        info.expert_weights_bytes as f64 / (1u64 << 30) as f64,
        100.0 * info.expert_weights_bytes as f64 / info.weights_bytes as f64
    );
    println!(
        "  dense_weights_bytes   {} ({:.3} GiB, {:.1}%)",
        info.dense_weights_bytes,
        info.dense_weights_bytes as f64 / (1u64 << 30) as f64,
        100.0 * info.dense_weights_bytes as f64 / info.weights_bytes as f64
    );
    println!(
        "kv_bytes_per_token      {} ({} of {} blocks cache)",
        info.kv_bytes_per_token, info.kv_layers, info.block_count
    );
    println!(
        "file_size               {} ({:.3} GiB)",
        info.file_size,
        info.file_size as f64 / (1u64 << 30) as f64
    );
    println!("overhead (file-weights) {} B", info.file_size - info.weights_bytes);
    ExitCode::SUCCESS
}
