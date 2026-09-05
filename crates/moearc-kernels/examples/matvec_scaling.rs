//! How a quantised matvec's cost scales with `n_cols`, per quantisation type.
//!
//! ```text
//! MOEARC_TEST_GPU=1 cargo run --release -p moearc-kernels --example matvec_scaling
//! ```
//!
//! # Why this exists, and why a microbenchmark is defensible here
//!
//! 🔴 Two well-argued changes were rejected this project-day because a microbenchmark of "the
//! same kernel" turned out not to describe the kernel in the engine. So this one is **anchored**:
//! it runs the engine's own `Context::matvec_q` — not a copy of the inner loop — and its output
//! is only trusted at all because two of its points reproduce numbers measured in a live decode.
//!
//! The question it answers: in a real decode, Q6_K and Q4_K cost the **same** per element at
//! `n_cols = 768` (5.21 vs 5.12 ps/element, expert `down`), and Q6_K costs **2.5-3.5x more** at
//! `n_cols = 2048` (13.0-18.0 vs 4.5-6.9, the lm_head and `attn_v`). Both shapes run the same
//! kernel on the same quantiser. Something about the *shape* — not the quantisation — is
//! responsible, and a sweep over `n_cols` at fixed total work is the way to see it.
//!
//! Total elements are held constant across every row, so a column that rises is a real cost per
//! element and not more work. The footprint is kept well above the B580's 18 MiB last-level
//! cache so every run streams from DRAM, as the engine's do.
//!
//! ⚠️ **What this sweep found has since been acted on, so one column no longer means what it
//! did.** `moearc_matvec_q` now routes Q6_K through the batched kernel, so `Q6 unbat` and
//! `Q6 bat` measure the same code. The numbers that justified that change, measured here before
//! it, at `n_cols = 2048`:
//!
//! ```text
//!            unbatched   batched
//!   Q4_K        4.05       4.17    ps/element
//!   Q5_K        3.81       3.61
//!   Q6_K       13.05       4.13    <- 3.2x, and the reason for the routing
//! ```
//!
//! Flat across every `n_cols` from 256 to 4096. Keep the sweep: it is how the next person
//! re-checks the claim after a compiler or driver upgrade, which is exactly the sort of change
//! that could make the workaround unnecessary — or make it necessary somewhere else.

use std::time::Instant;

use moearc_kernels::{Context, QuantType};

/// Elements per configuration. 64M at Q6_K is ~52 MiB — comfortably past the LLC, so this
/// measures DRAM streaming rather than a cache-resident best case.
const ELEMS: usize = 64 << 20;
const REPS: usize = 20;

fn block_bytes(ty: QuantType) -> usize {
    match ty {
        QuantType::Q4K => 144,
        QuantType::Q5K => 176,
        QuantType::Q6K => 210,
        _ => 34,
    }
}

fn main() {
    if std::env::var("MOEARC_TEST_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 (and select a device) to run");
        return;
    }
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no device: {e}");
            return;
        }
    };
    println!("device {}", ctx.device_name().unwrap_or_default());
    println!("{ELEMS} elements per row, {REPS} reps, footprint kept above the 18 MiB LLC\n");
    // 🔴 `m1` is the load-bearing column. It runs the **batched** entry point with a single
    // matrix, so it differs from the unbatched one only in kernel structure and not in how much
    // work is batched. If Q6_K is fast there, the quantiser was never the problem.
    println!(
        "| n_cols | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
        "Q4 unbat", "Q5 unbat", "Q6 unbat", "Q4 bat", "Q5 bat", "Q6 bat"
    );
    println!("|---|---|---|---|---|---|---|");

    for n_cols in [256usize, 512, 768, 1024, 1536, 2048, 3072, 4096] {
        let mut cells: Vec<String> = Vec::new();
        for n_mat in [0usize, 1] {
            for ty in [QuantType::Q4K, QuantType::Q5K, QuantType::Q6K] {
                let mats = n_mat.max(1);
                let n_rows = ELEMS / n_cols / mats;
                let bb = block_bytes(ty);
                let w_bytes = n_rows * (n_cols / 256) * bb;
                let Ok(x) = ctx.alloc_n::<f32>(n_cols * mats) else {
                    cells.push("alloc".into());
                    continue;
                };
                let Ok(out) = ctx.alloc_n::<f32>(n_rows * mats) else {
                    cells.push("alloc".into());
                    continue;
                };
                let ws: Vec<_> = (0..mats).filter_map(|_| ctx.alloc(w_bytes).ok()).collect();
                if ws.len() != mats {
                    cells.push("alloc".into());
                    continue;
                }
                let _ = ctx.upload_slice(&x, &vec![1.0f32; n_cols * mats]);
                let refs: Vec<&_> = ws.iter().collect();
                let run = || {
                    if n_mat == 0 {
                        ctx.matvec_q(ty, &out, &ws[0], &x, n_rows, n_cols)
                    } else {
                        ctx.matvec_q_batched(ty, &out, &refs, &x, n_cols, n_rows, n_cols)
                    }
                };
                for _ in 0..3 {
                    let _ = run();
                }
                let _ = ctx.sync();
                let t = Instant::now();
                let mut ok = true;
                for _ in 0..REPS {
                    if run().is_err() {
                        ok = false;
                        break;
                    }
                }
                let _ = ctx.sync();
                if !ok {
                    cells.push("err".into());
                    continue;
                }
                let secs = t.elapsed().as_secs_f64() / REPS as f64;
                cells.push(format!("{:.2}", secs * 1e12 / ELEMS as f64));
            }
        }
        println!(
            "| {n_cols} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
            cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]
        );
    }
    println!("\nps/element, {ELEMS} elements per cell. `bat m1` is the batched kernel with one");
    println!("matrix: same amount of work as `unbat`, different kernel.");
}
