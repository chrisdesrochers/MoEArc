//! Map a GGUF and report what it costs and what it contains.
//!
//! ```text
//! cargo run -p moearc-model --example map -- /path/to/model.gguf
//! ```
//!
//! Two jobs, both of which are evidence rather than description:
//!
//! 1. **It measures the memory-map claim.** `VmRSS` is printed before the map, after it, and
//!    after one expert has actually been touched. "Zero-copy" is a claim about resident memory
//!    and is only worth anything as a number: a 20.6 GiB model that adds 20 GiB of `VmSize` and
//!    a few MiB of `VmRSS` has proven it; one that does not, has not.
//! 2. **It reads the architecture off the tensor names.** Whether a model is a plain transformer
//!    MoE, and whether it has shared experts, decides which kernels an engine needs and how much
//!    of the FFN is pageable. Both are answered here from the file, not from a model card.

use std::path::PathBuf;
use std::process::ExitCode;

use moearc_model::tensors::{ExpertBank, MappedModel, names};

/// One line of `/proc/self/status`, in bytes, or `None` where there is no procfs.
fn vm(field: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with(field))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn mib(b: u64) -> String {
    format!("{:.2} MiB", b as f64 / (1u64 << 20) as f64)
}

/// Print the process's memory figures under a label.
fn report_memory(stage: &str) {
    match (vm("VmRSS:"), vm("VmSize:"), vm("VmHWM:")) {
        (Some(rss), Some(size), Some(hwm)) => println!(
            "  {stage:<28} VmRSS {:>12}   VmHWM {:>12}   VmSize {:>12}",
            mib(rss),
            mib(hwm),
            mib(size)
        ),
        _ => println!("  {stage:<28} (no /proc/self/status on this platform)"),
    }
}

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: map <model.gguf>");
        return ExitCode::FAILURE;
    };
    let file_size = match std::fs::metadata(&path) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("error: {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("file  {}", path.display());
    println!("size  {} B ({:.3} GiB)", file_size, file_size as f64 / (1u64 << 30) as f64);
    println!("\nmemory");
    report_memory("before open");

    let m = match MappedModel::open(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    report_memory("after header + mmap");

    // Touch one expert's bytes for real. `data` is a borrow into the mapping, so this is the
    // first thing in the program that forces the kernel to fault those pages in — and it is
    // therefore the only honest way to show what one expert actually costs in resident memory.
    let touched = match m.expert(0, ExpertBank::Gate, 0) {
        Ok(e) => {
            let sum = e.data.iter().fold(0u64, |a, &b| a + u64::from(b));
            println!("  (touched expert 0 of block 0: {} B, byte sum {sum})", e.len());
            report_memory("after touching 1 expert");
            e.len()
        }
        Err(e) => {
            println!("  (no expert bank to touch: {e})");
            0
        }
    };
    let _ = touched;

    let layout = m.layout();
    println!("\nlayout (from tensor names alone)");
    println!("  architecture              {}", m.architecture().unwrap_or("<unset>"));
    println!("  tensors                   {}", m.tensor_count());
    println!("  blocks                    {}", layout.block_count);
    println!("  attention blocks          {}", layout.attention_blocks);
    println!("  expert blocks             {}", layout.expert_blocks);
    println!("  shared-expert blocks      {}", layout.shared_expert_blocks);
    println!("  recurrent blocks          {}", layout.recurrent_blocks);
    if !layout.recurrent_suffixes.is_empty() {
        println!("  recurrent tensors         {}", layout.recurrent_suffixes.join(", "));
    }
    println!("  pure transformer MoE?     {}", layout.is_pure_transformer_moe());
    println!("  shared experts?           {}", layout.has_shared_experts());

    println!("\nper-block tensors (suffix, blocks carrying it of {})", layout.block_count);
    for (suffix, n) in &layout.per_block_suffixes {
        let flag = if *n == layout.block_count { " " } else { "*" };
        println!("  {flag} {n:>4}  blk.N.{suffix}");
    }
    println!("\nglobal tensors");
    for name in &layout.global_tensors {
        match m.tensor(name) {
            Ok(t) => {
                println!("    {:<28} {:?} {} {}", t.name, t.dims, t.quant.name, mib(t.len() as u64))
            }
            Err(e) => println!("    {name:<28} <{e}>"),
        }
    }

    println!("\nblock 0, by convention");
    for suffix in [
        names::ATTN_NORM,
        names::ATTN_Q,
        names::ATTN_Q_NORM,
        names::ATTN_K,
        names::ATTN_K_NORM,
        names::ATTN_V,
        names::ATTN_OUTPUT,
        names::FFN_NORM,
        names::FFN_GATE_INP,
        names::FFN_GATE_EXPS,
        names::FFN_UP_EXPS,
        names::FFN_DOWN_EXPS,
        names::FFN_GATE_SHEXP,
        names::FFN_UP_SHEXP,
        names::FFN_DOWN_SHEXP,
    ] {
        match m.optional_block_tensor(0, suffix) {
            Ok(Some(t)) => {
                println!(
                    "    {:<24} {:?} {:<6} {}",
                    suffix,
                    t.dims,
                    t.quant.name,
                    mib(t.len() as u64)
                )
            }
            Ok(None) => println!("    {suffix:<24} absent"),
            Err(e) => println!("    {suffix:<24} <{e}>"),
        }
    }

    if let Some(n) = m.expert_count() {
        println!("\none expert slot, block 0 ({n} experts stacked per bank)");
        let mut slot = 0u64;
        for bank in ExpertBank::ALL {
            match m.expert(0, bank, 0) {
                Ok(e) => {
                    slot += e.len() as u64;
                    println!(
                        "    {:<24} {:?} {:<6} {} at file offset {}",
                        bank.suffix(),
                        e.dims,
                        e.quant.name,
                        mib(e.len() as u64),
                        e.file_offset
                    );
                }
                Err(e) => println!("    {:<24} <{e}>", bank.suffix()),
            }
        }
        println!("    {:<24} {}", "slot total", mib(slot));
    }

    ExitCode::SUCCESS
}
