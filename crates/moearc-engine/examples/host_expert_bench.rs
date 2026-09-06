//! How fast the host expert kernels are, and against what.
//!
//! ```text
//! cargo run --release -p moearc-engine --features gpu --example host_expert_bench -- \
//!     <model.gguf> [threads] [experts-per-job] [jobs]
//! ```
//!
//! Reports three numbers, in this order, because each answers a different question:
//!
//! 1. **One core, one expert.** Weight bytes divided by wall time — directly comparable to the
//!    ~22.8 GB/s a single core reads host memory at (`docs/roadmap.md`). This is the number that
//!    says whether the kernel is memory-bound or arithmetic-bound, and there is no way to tell
//!    from a threaded figure.
//! 2. **The pool, a block's worth of experts at a time.** What the engine actually submits.
//! 3. **The implied ceiling**: if every one of a token's expert reads were served this way, how
//!    long would a token take.
//!
//! 🔴 Every expert is touched once before timing starts. A cold mmap read is a disk read, and a
//! first pass measures ZFS rather than the CPU — the first version of this bench reported
//! 410 us an expert for exactly that reason.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use moearc_engine::host_experts::{
    BankSpec, BlockSpec, Geometry, HostExecutor, default_threads, expert_ffn,
};
use moearc_model::ModelInfo;
use moearc_model::tensors::{ExpertBank, MappedModel};

fn bank_spec(m: &MappedModel, block: u32, bank: ExpertBank) -> Result<BankSpec, String> {
    let v = m.expert(block, bank, 0).map_err(|e| e.to_string())?;
    let ty = moearc_kernels::QuantType::from_type_id(v.quant.id)
        .ok_or_else(|| format!("{} is a quantisation this build cannot expand", v.quant.name))?;
    let (&n_cols, rest) = v.dims.split_first().ok_or("a matrix has dimensions")?;
    Ok(BankSpec { ty, n_rows: rest.iter().product::<u64>() as usize, n_cols: n_cols as usize })
}

fn block_spec(m: &MappedModel, block: u32) -> Result<BlockSpec, String> {
    Ok(BlockSpec {
        gate: bank_spec(m, block, ExpertBank::Gate)?,
        up: bank_spec(m, block, ExpertBank::Up)?,
        down: bank_spec(m, block, ExpertBank::Down)?,
    })
}

fn expert_bytes(m: &MappedModel, block: u32, e: u32) -> usize {
    [ExpertBank::Gate, ExpertBank::Up, ExpertBank::Down]
        .iter()
        .map(|b| m.expert(block, *b, e).map(|v| v.data.len()).unwrap_or(0))
        .sum()
}

fn activation(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 32) as u32 as f64 / u32::MAX as f64) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: host_expert_bench <model.gguf> [threads] [experts-per-job] [jobs]");
        return ExitCode::FAILURE;
    }
    let path = PathBuf::from(&args[1]);
    let threads: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or_else(default_threads);
    let per_job: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    let jobs: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(200);

    let m = match MappedModel::open(&path) {
        Ok(v) => Arc::new(v),
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let info = match ModelInfo::from_header(m.header()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut n_block = 0u32;
    while m.expert(n_block, ExpertBank::Gate, 0).is_ok() {
        n_block += 1;
    }
    let specs: Vec<BlockSpec> = match (0..n_block).map(|b| block_spec(&m, b)).collect() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let geom = Geometry {
        n_block: n_block as usize,
        n_expert: info.total_experts as usize,
        n_expert_used: info.active_experts as usize,
        n_embd: specs[0].gate.n_cols,
        n_ff: specs[0].gate.n_rows,
        expert_bias: false,
        act: moearc_engine::moe::Activation::Swiglu,
    };
    println!(
        "{} blocks, {} experts, {} active, n_embd {}, n_ff {}",
        n_block, geom.n_expert, geom.n_expert_used, geom.n_embd, geom.n_ff
    );
    println!("threads {threads}, {per_job} experts a job, {jobs} jobs\n");

    let x = activation(geom.n_embd, 12345);

    // ---- warm the pages we are about to time ------------------------------------------------
    //
    // Touching the bytes is not enough to be sure; the sum is used so the read cannot be
    // optimised away.
    let picks: Vec<u32> = (0..per_job as u32).map(|i| i * 7 % geom.n_expert as u32).collect();
    let mut warm = 0u64;
    let mut bytes_per_expert = 0usize;
    for b in 0..n_block {
        for e in &picks {
            for bank in [ExpertBank::Gate, ExpertBank::Up, ExpertBank::Down] {
                if let Ok(v) = m.expert(b, bank, *e) {
                    warm += v.data.iter().step_by(64).map(|b| u64::from(*b)).sum::<u64>();
                }
            }
        }
        if b == 0 {
            bytes_per_expert = expert_bytes(&m, 0, picks[0]);
        }
    }
    println!("(warm-up checksum {warm}, {bytes_per_expert} B an expert in block 0)\n");

    // ---- one core, one expert ----------------------------------------------------------------
    let spec = specs[0];
    let g = m.expert(0, ExpertBank::Gate, picks[0]).expect("gate");
    let u = m.expert(0, ExpertBank::Up, picks[0]).expect("up");
    let d = m.expert(0, ExpertBank::Down, picks[0]).expect("down");
    let reps = 200;
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..reps {
        sink += expert_ffn(spec, g.data, u.data, d.data, &x)[0];
    }
    let single = t.elapsed().as_secs_f64() / f64::from(reps);
    let one_bytes = g.data.len() + u.data.len() + d.data.len();
    println!(
        "one core, one expert  {:>8.1} us   {:>6.2} GB/s   (sink {sink:e})",
        single * 1e6,
        one_bytes as f64 / single / 1e9
    );

    // ---- the pool ----------------------------------------------------------------------------
    let exec = match HostExecutor::new(Arc::clone(&m), geom, &specs, threads) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let job_picks: Vec<(u16, f32)> = picks.iter().map(|e| (*e as u16, 0.1)).collect();
    let mut out = vec![0.0f32; geom.n_embd];
    // A few jobs to get the workers off the park path before anything is timed.
    for b in 0..8 {
        let job = exec
            .submit(b % n_block as usize, specs[b % n_block as usize], &job_picks, &x)
            .expect("submit");
        exec.sync(job, &mut out).expect("sync");
    }
    exec.reset_stats();

    let t = Instant::now();
    for i in 0..jobs {
        let b = i % n_block as usize;
        let job = exec.submit(b, specs[b], &job_picks, &x).expect("submit");
        exec.sync(job, &mut out).expect("sync");
    }
    let total = t.elapsed().as_secs_f64();
    let per = total / jobs as f64;

    // Bytes are counted from the slices themselves, not from a per-expert constant: the down
    // bank is Q6_K in some blocks and Q4_K in others and the two differ by 46%.
    let mut moved = 0usize;
    for i in 0..jobs {
        let b = (i % n_block as usize) as u32;
        for e in &picks {
            moved += expert_bytes(&m, b, *e);
        }
    }
    println!(
        "pool, {per_job} experts a job {:>8.1} us   {:>6.2} GB/s aggregate   {:>6.2} GB/s a thread",
        per * 1e6,
        moved as f64 / total / 1e9,
        moved as f64 / total / 1e9 / threads as f64
    );
    let s = exec.stats();
    println!(
        "  executor busy {:.1} us a job, caller waited {:.1} us a job",
        s.busy_nanos as f64 / 1000.0 / s.jobs as f64,
        s.wait_nanos as f64 / 1000.0 / s.jobs as f64
    );

    let per_expert = per / per_job as f64;
    println!(
        "\nimplied: {:.1} us an expert, so a whole token's {} experts would take {:.1} ms \
         ({:.1} tok/s) if the CPU did all of it",
        per_expert * 1e6,
        n_block as usize * geom.n_expert_used,
        per_expert * (n_block as usize * geom.n_expert_used) as f64 * 1e3,
        1.0 / (per_expert * (n_block as usize * geom.n_expert_used) as f64)
    );
    ExitCode::SUCCESS
}
