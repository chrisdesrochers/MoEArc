//! Attributing the cost of depth: attention, or a churning expert cache?
//!
//! ```text
//! MOEARC_PROFILE=1 MOEARC_PROFILE_EVENTS=1 \
//! cargo run --release -p moearc-engine --features gpu --example ctx_attrib -- \
//!     <model.gguf> <depth,...> <n-predict> <residency> <host-policy> <ids-file>
//! ```
//!
//! # The question this exists to answer, and why `ctx_curve` cannot
//!
//! Throughput falling as context grows has two candidate causes that a `tok/s` column cannot
//! tell apart. Attention over more keys costs more. And a deeper prompt names more distinct
//! experts, which leaves the resident pool holding a more diluted working set, so the decode
//! steps that follow miss more and stage more bytes. **The first is a cost of context; the
//! second is a cost of the memory strategy** — and they call for opposite fixes, so guessing
//! between them is worthless.
//!
//! `ctx_curve` reports `profile` phases, which are **host** wall time around a submit. On an
//! asynchronous queue that is close to the submit's own cost, and the device work it stands for
//! lands wherever the next synchronisation happens. 🔴 Read naively that says attention is free
//! and `moe.stage` is everything, which is an artefact of where the queue was drained, not a
//! finding. This example reads the **SYCL events themselves**, so every kernel is charged the
//! device time it actually took.
//!
//! # Differencing, and why it is exact rather than approximate
//!
//! Neither the event counters nor the cache counters can be zeroed part-way through a
//! generation: both are reached through the session, and `generate_with` holds its lock for the
//! whole call. So each depth is run **twice from the same warm pool**, differing in one thing
//! only:
//!
//! - `prefill` — the prompt, then a single token. `depth` decode-path steps, none of them a
//!   generated-token step.
//! - `full` — the same prompt, then `n` tokens. The same `depth` steps, plus `n - 1` more.
//!
//! The counters are cumulative, so `full - prefill` is exactly those `n - 1` steps. This is not
//! a model of the decode phase; it is the decode phase, measured by subtraction. ⚠️ The one
//! assumption is that the shared prefix costs the same in both runs, which is why both are
//! preceded by a full warm-up pass and neither is the first thing the pool sees.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use moearc_engine::host_experts::HostPolicy;
use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions};

fn load_avg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(f64::NAN)
}

fn read_ids(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut ids = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        for field in line.split([',', ' ', '\t']).filter(|f| !f.is_empty()) {
            ids.push(field.parse::<u32>().map_err(|_| format!("`{field}` is not a token id"))?);
        }
    }
    if ids.is_empty() {
        return Err(format!("{} holds no token ids", path.display()));
    }
    Ok(ids)
}

/// Cumulative counters after one generation.
struct Snapshot {
    /// Device nanoseconds per kernel key, from the SYCL events.
    device: BTreeMap<String, u64>,
    demands: u64,
    hits: u64,
    bytes_staged: u64,
    host_experts: u64,
}

fn snapshot(session: &Session, prompt: &[u32], n: usize) -> Result<Snapshot, String> {
    session.reset_event_profile().map_err(|e| e.to_string())?;
    session.reset_cache_stats().map_err(|e| e.to_string())?;
    let stop = StopConditions { max_tokens: n, stop_tokens: Vec::new() };
    session.generate(prompt, &stop, &mut |_| true).map_err(|e| e.to_string())?;
    let ev = session.event_profile().map_err(|e| e.to_string())?;
    let r = session.residency().map_err(|e| e.to_string())?;
    Ok(Snapshot {
        device: ev.into_iter().map(|(k, ns, _)| (k, ns)).collect(),
        demands: r.stats.demands,
        hits: r.stats.hits,
        bytes_staged: r.bytes_staged,
        host_experts: r.host.experts,
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "usage: ctx_attrib <model.gguf> <depth,...> <n-predict> <residency> <host-policy> \
             <ids-file>"
        );
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let mut depths: Vec<usize> = Vec::new();
    for d in args[2].split(',').filter(|s| !s.is_empty()) {
        match d.parse() {
            Ok(v) => depths.push(v),
            Err(_) => {
                eprintln!("`{d}` is not a depth");
                return ExitCode::FAILURE;
            }
        }
    }
    let Ok(n_predict) = args[3].parse::<usize>() else {
        eprintln!("n-predict must be a number");
        return ExitCode::FAILURE;
    };
    if n_predict < 3 {
        eprintln!("n-predict must leave at least two decode steps to difference");
        return ExitCode::FAILURE;
    }
    let residency: Residency = match args[4].parse() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let host: HostPolicy = match args[5].parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let pool = match read_ids(Path::new(&args[6])) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!("residency `{}`, host `{}`, {n_predict} generated tokens per depth", args[4], args[5]);
    println!("device time and cache counters, DECODE STEPS ONLY, by differencing\n");

    for &depth in &depths {
        if depth > pool.len() {
            println!("depth {depth}: only {} ids available\n", pool.len());
            continue;
        }
        let prompt = &pool[..depth];
        let n_ctx = depth + n_predict + 1;
        let opts = SessionOptions { n_ctx: Some(n_ctx), residency, host };
        let session = match Session::load_with(&model, opts) {
            Ok(s) => s,
            Err(e) => {
                println!("depth {depth}: load failed: {e}\n");
                continue;
            }
        };
        let stop = StopConditions { max_tokens: n_predict, stop_tokens: Vec::new() };
        // Warm the pool first, so neither differenced run is the one that pays for filling it.
        if let Err(e) = session.generate(prompt, &stop, &mut |_| true) {
            println!("depth {depth}: warm-up failed: {e}\n");
            continue;
        }
        let load = load_avg();
        let prefill = match snapshot(&session, prompt, 1) {
            Ok(s) => s,
            Err(e) => {
                println!("depth {depth}: prefill pass failed: {e}\n");
                continue;
            }
        };
        let full = match snapshot(&session, prompt, n_predict) {
            Ok(s) => s,
            Err(e) => {
                println!("depth {depth}: full pass failed: {e}\n");
                continue;
            }
        };
        let steps = (n_predict - 1) as f64;

        let d_demands = full.demands.saturating_sub(prefill.demands);
        let d_hits = full.hits.saturating_sub(prefill.hits);
        let d_staged = full.bytes_staged.saturating_sub(prefill.bytes_staged);
        let d_host = full.host_experts.saturating_sub(prefill.host_experts);
        let hit_rate = if d_demands == 0 { 0.0 } else { d_hits as f64 / d_demands as f64 };

        println!("## depth {depth} (load {load:.2} before the pair, {n_ctx} n_ctx)");
        println!(
            "\ndecode-only cache: **{:.1}% hit** over {d_demands} demands, {:.1} MiB staged per \
             step, {:.1} experts/step to the CPU",
            100.0 * hit_rate,
            d_staged as f64 / (1024.0 * 1024.0) / steps,
            d_host as f64 / steps,
        );
        println!("\n| kernel | ms/step (decode) | ms/step (prefill) |");
        println!("|---|---|---|");
        let mut keys: Vec<&String> = full.device.keys().collect();
        keys.sort_by_key(|k| {
            let f = full.device.get(*k).copied().unwrap_or(0);
            let p = prefill.device.get(*k).copied().unwrap_or(0);
            std::cmp::Reverse(f.saturating_sub(p))
        });
        let mut decode_total = 0.0;
        for k in keys {
            let f = full.device.get(k).copied().unwrap_or(0);
            let p = prefill.device.get(k).copied().unwrap_or(0);
            let dec = f.saturating_sub(p) as f64 / 1e6 / steps;
            let pre = p as f64 / 1e6 / depth.max(1) as f64;
            decode_total += dec;
            println!("| `{k}` | {dec:.3} | {pre:.3} |");
        }
        println!("| **tracked device busy** | **{decode_total:.3}** | |");
        println!(
            "\n⚠️ The prefill column divides by `depth` steps and is there to show a kernel's \
             cost *scaling*, not as a comparison of like with like: a prefill step at depth \
             {depth} attends over a cache that is filling, so it averages shallower positions \
             than the decode steps do.\n"
        );
    }
    ExitCode::SUCCESS
}
