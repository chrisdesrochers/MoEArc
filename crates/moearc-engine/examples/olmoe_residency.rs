//! Sweep expert residency and measure what it costs.
//!
//! ```text
//! cargo run -p moearc-engine --features gpu --example olmoe_residency -- \
//!     <model.gguf> <n-predict> <spec,spec,...> <token-id> [token-id ...]
//! ```
//!
//! A spec is one of:
//!
//! - `all`             — every slot resident; the baseline
//! - `<n>`             — `n` slots, LRU
//! - `plan:<bytes>`    — whatever `memory::plan` decides for a device with that much free
//! - `static:<blocks>` — the incumbent: `blocks` blocks pinned, the rest streamed
//!
//! Each row is run twice: once from a cold pool, once with whatever the first run left
//! resident. Both are reported, because they are different questions — a served request meets a
//! warm cache, but the first request after a load does not, and a table showing only one of
//! them would be choosing the flattering half.
//!
//! 🔴 Every timing here is a **floor, not a claim**. A miss is a synchronous host-to-device copy
//! with nothing overlapped behind it, and the prompt is decoded one token at a time. The cold
//! row in particular is dominated by staging rather than by arithmetic — which is the point of
//! reporting it next to the warm one.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::memory::DeviceMemory;
use moearc_engine::olmoe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions};

/// llama.cpp's greedy continuation of `[510, 5347, 273, 6181, 310]`
/// (`The capital of France is`), recorded from `llama-eval-callback` on the CPU backend.
/// Any row that does not reproduce this exactly has a paging bug, not a model difference.
const REFERENCE: [u32; 60] = [
    7785, 15, 187, 187, 510, 14731, 273, 6181, 310, 14029, 15, 187, 187, 510, 3072, 273, 6181, 310,
    9963, 15, 19, 3041, 952, 15, 187, 187, 510, 3565, 3448, 273, 6181, 310, 5112, 15, 187, 187,
    510, 3872, 802, 12404, 273, 6181, 310, 253, 11414, 280, 687, 7337, 15, 187, 187, 510, 3872,
    7908, 273, 6181, 310, 253, 492, 49122,
];

fn parse_spec(s: &str) -> Option<Residency> {
    if s == "all" {
        return Some(Residency::All);
    }
    if let Some(rest) = s.strip_prefix("plan:") {
        let free: u64 = rest.parse().ok()?;
        return Some(Residency::Planned(DeviceMemory { total_bytes: free, free_bytes: free }));
    }
    if let Some(rest) = s.strip_prefix("static:") {
        return Some(Residency::StaticSplit { resident_blocks: rest.parse().ok()? });
    }
    Some(Residency::Slots(s.parse().ok()?))
}

struct Run {
    ids: Vec<u32>,
    seconds: f64,
    steps: usize,
    hit_rate: f64,
    staged_mib: f64,
}

fn run(session: &Session, prompt: &[u32], n: usize) -> Result<Run, String> {
    session.reset_cache_stats().map_err(|e| e.to_string())?;
    let stop = StopConditions { max_tokens: n, stop_tokens: Vec::new() };
    let mut ids = Vec::new();
    let started = Instant::now();
    session
        .generate(prompt, &stop, &mut |t| {
            ids.push(t);
            true
        })
        .map_err(|e| e.to_string())?;
    let seconds = started.elapsed().as_secs_f64();
    let r = session.residency().map_err(|e| e.to_string())?;
    Ok(Run {
        ids,
        seconds,
        steps: prompt.len() + n,
        hit_rate: r.stats.hit_rate(),
        staged_mib: r.bytes_staged as f64 / (1024.0 * 1024.0),
    })
}

/// `n/total` if the run reproduces the reference, or where it first stops.
fn verdict(ids: &[u32]) -> String {
    let n = ids.len().min(REFERENCE.len());
    match (0..n).find(|i| ids[*i] != REFERENCE[*i]) {
        None => format!("{n}/{n}"),
        Some(i) => format!("DIVERGED at {i} (got {} want {})", ids[i], REFERENCE[i]),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: olmoe_residency <model.gguf> <n-predict> <spec,...> <token-id> [token-id ...]"
        );
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let Ok(n_predict) = args[2].parse::<usize>() else {
        eprintln!("n-predict must be a number");
        return ExitCode::FAILURE;
    };
    let specs: Vec<&str> = args[3].split(',').filter(|s| !s.is_empty()).collect();
    let prompt: Vec<u32> = args[4..].iter().filter_map(|s| s.parse().ok()).collect();

    println!("prompt {prompt:?}, {n_predict} tokens; cold rows are staging-dominated\n");
    println!(
        "| policy | slots | % of model | pool | cold hit | cold tok/s | warm hit | warm tok/s \
         | cold staged | warm staged | ids |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    let mut failures = 0;
    for spec in &specs {
        let Some(residency) = parse_spec(spec) else {
            eprintln!("unparseable spec `{spec}`");
            return ExitCode::FAILURE;
        };
        // 512 tokens of context: this sweep never exceeds it, and a smaller KV cache leaves the
        // card's memory to the thing being measured.
        let opts = SessionOptions { n_ctx: Some(512), residency };
        let session = match Session::load_with(&model, opts) {
            Ok(s) => s,
            Err(e) => {
                println!("| `{spec}` | — | — | — | — | — | — | — | — | — | LOAD FAILED: {e} |");
                failures += 1;
                continue;
            }
        };
        let r0 = session.residency().expect("residency");

        if let Err(e) = session.clear_residency() {
            eprintln!("clear failed: {e}");
            return ExitCode::FAILURE;
        }
        let cold = match run(&session, &prompt, n_predict) {
            Ok(r) => r,
            Err(e) => {
                println!("| `{spec}` | {} | | | | | | | | | FAILED: {e} |", r0.resident_slots);
                failures += 1;
                continue;
            }
        };
        let warm = match run(&session, &prompt, n_predict) {
            Ok(r) => r,
            Err(e) => {
                println!("| `{spec}` | {} | | | | | | | | | FAILED: {e} |", r0.resident_slots);
                failures += 1;
                continue;
            }
        };

        let v = verdict(&cold.ids);
        if v.starts_with("DIVERGED") || cold.ids != warm.ids {
            failures += 1;
        }
        println!(
            "| {} `{spec}` | {} | {:.1}% | {} MiB | {:.1}% | {:.2} | {:.1}% | {:.2} | {:.0} MiB \
             | {:.0} MiB | {v} |",
            r0.policy,
            r0.resident_slots,
            100.0 * r0.resident_fraction(),
            r0.pool_bytes >> 20,
            100.0 * cold.hit_rate,
            cold.steps as f64 / cold.seconds,
            100.0 * warm.hit_rate,
            warm.steps as f64 / warm.seconds,
            cold.staged_mib,
            warm.staged_mib,
        );
    }

    println!(
        "\n`% of model` is residency slots, not bytes. `staged` is expert bytes actually copied \
         host-to-device, counted from the slices uploaded."
    );
    if failures > 0 {
        eprintln!("\n{failures} row(s) failed or diverged");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
