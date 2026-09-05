//! Does host-side expert execution, overlapped with the GPU, pay?
//!
//! ```text
//! cargo run --release -p moearc-engine --features gpu --example hybrid_sweep -- \
//!     <model.gguf> <n-predict> <n-ctx> <residency,...> <host-policy,...> \
//!     <ref-ids-file|-> <token-id> [token-id ...]
//! ```
//!
//! A residency spec is whatever `Residency` parses; a host policy is whatever `HostPolicy`
//! parses (`off`, `frac:<f>`, `over:<n>`, `all`). Every residency is run against every policy,
//! and `off` is the stream-only control the rest are measured against.
//!
//! # What the columns mean, and which one is the experiment
//!
//! `tok/s` is the answer. The two that say *why* are **`busy`** — wall time the CPU pool spent
//! working, per token — and **`wait`** — wall time the device thread spent blocked in `sync`,
//! per token. Overlap is the difference: if `wait` is near `busy` the CPU is being waited for
//! and this is substitution, which `bench/baselines/qwen3-30b-a3b.md` already measures on
//! llama.cpp and which loses. If `wait` is far below `busy`, host work is being hidden behind
//! device work and the mechanism is doing what it was built for. **`tok/s` can still fall while
//! `wait` is near zero** — that means the CPU is not the cost, the experts it took are simply
//! not the expensive ones.
//!
//! 🔴 **`ids` is a gate, not a column.** Residency decides what has to move and the host policy
//! decides where a miss is computed; neither may change *what* is computed. The host path is not
//! bit-identical to the device's — different rounding, measured in `tests/host_experts_gpu.rs` —
//! so a late divergence on a near-tied logit is expected and is reported as the step it happened
//! at. A divergence at step 0, or a row whose cold and warm runs disagree with each other, is a
//! bug.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::host_experts::HostPolicy;
use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions};

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

struct Run {
    ids: Vec<u32>,
    seconds: f64,
    steps: usize,
    hit_rate: f64,
    staged_mib: f64,
    /// Experts computed host-side, per step.
    cpu_per_step: f64,
    /// Of every expert the router named, the fraction the CPU computed.
    cpu_share: f64,
    busy_ms: f64,
    wait_ms: f64,
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
    let steps = prompt.len() + n;
    let demands = r.stats.demands + r.host.experts;
    Ok(Run {
        ids,
        seconds,
        steps,
        hit_rate: r.stats.hit_rate(),
        staged_mib: r.bytes_staged as f64 / (1024.0 * 1024.0),
        cpu_per_step: r.host.experts as f64 / steps as f64,
        cpu_share: if demands == 0 { 0.0 } else { r.host.experts as f64 / demands as f64 },
        busy_ms: r.host.busy_nanos as f64 / 1e6 / steps as f64,
        wait_ms: r.host.wait_nanos as f64 / 1e6 / steps as f64,
    })
}

/// How this row's ids stand against the stream-only control for the same residency.
fn verdict(ids: &[u32], baseline: Option<&[u32]>, reference: Option<&[u32]>) -> String {
    let mut parts = Vec::new();
    match baseline {
        None => parts.push("control".to_string()),
        Some(b) => match (0..ids.len().min(b.len())).find(|i| ids[*i] != b[*i]) {
            None if ids.len() == b.len() => parts.push("identical".to_string()),
            None => parts.push(format!("LENGTH {} vs {}", ids.len(), b.len())),
            Some(i) => parts.push(format!("differs at {i}/{}", ids.len())),
        },
    }
    if let Some(reference) = reference {
        let n = ids.len().min(reference.len());
        match (0..n).find(|i| ids[*i] != reference[*i]) {
            None => parts.push(format!("ref {n}/{n}")),
            Some(i) => parts.push(format!("ref differs at {i}")),
        }
    }
    parts.join(", ")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 8 {
        eprintln!(
            "usage: hybrid_sweep <model.gguf> <n-predict> <n-ctx> <residency,...> \
             <host-policy,...> <ref-ids-file|-> <token-id> [token-id ...]"
        );
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let Ok(n_predict) = args[2].parse::<usize>() else {
        eprintln!("n-predict must be a number");
        return ExitCode::FAILURE;
    };
    let Ok(n_ctx) = args[3].parse::<usize>() else {
        eprintln!("n-ctx must be a number");
        return ExitCode::FAILURE;
    };
    let residencies: Vec<&str> = args[4].split(',').filter(|s| !s.is_empty()).collect();
    let policies: Vec<&str> = args[5].split(',').filter(|s| !s.is_empty()).collect();
    let reference = if args[6] == "-" {
        None
    } else {
        match read_ids(Path::new(&args[6])) {
            Ok(ids) => Some(ids),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    };
    let prompt: Vec<u32> = args[7..].iter().filter_map(|s| s.parse().ok()).collect();

    println!("prompt {prompt:?}, {n_predict} tokens, {n_ctx} ctx\n");
    println!(
        "| slots | host | threads | warm tok/s | vs stream | hit | staged MiB | cpu/step | \
         cpu share | busy ms/tok | wait ms/tok | ids |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");

    let mut failures = 0;
    for spec in &residencies {
        let residency: Residency = match spec.parse() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        // The stream-only control for this residency, so every comparison is like for like.
        let mut control_ids: Option<Vec<u32>> = None;
        let mut control_toks: Option<f64> = None;

        for pol in &policies {
            let host: HostPolicy = match pol.parse() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            };
            let opts = SessionOptions { n_ctx: Some(n_ctx), residency, host };
            let session = match Session::load_with(&model, opts) {
                Ok(s) => s,
                Err(e) => {
                    println!("| `{spec}` | `{pol}` | | | | | | | | | | LOAD FAILED: {e} |");
                    failures += 1;
                    continue;
                }
            };
            let r0 = session.residency().expect("residency");
            if let Err(e) = session.clear_residency() {
                eprintln!("clear failed: {e}");
                return ExitCode::FAILURE;
            }
            // The first pass fills the pool; the second is the steady state a served request
            // meets. Only the second is reported — the cold pass is the same measurement the
            // residency sweep already makes and it is dominated by staging either way.
            let cold = match run(&session, &prompt, n_predict) {
                Ok(r) => r,
                Err(e) => {
                    println!("| `{spec}` | `{pol}` | | | | | | | | | | FAILED: {e} |");
                    failures += 1;
                    continue;
                }
            };
            let warm = match run(&session, &prompt, n_predict) {
                Ok(r) => r,
                Err(e) => {
                    println!("| `{spec}` | `{pol}` | | | | | | | | | | FAILED: {e} |");
                    failures += 1;
                    continue;
                }
            };
            let toks = warm.steps as f64 / warm.seconds;
            let v = verdict(&warm.ids, control_ids.as_deref(), reference.as_deref());
            if cold.ids != warm.ids {
                failures += 1;
            }
            let rel = match control_toks {
                None => "—".to_string(),
                Some(c) => format!("{:+.1}%", 100.0 * (toks / c - 1.0)),
            };
            println!(
                "| {} | `{pol}` | {} | {:.2} | {rel} | {:.1}% | {:.0} | {:.2} | {:.1}% | {:.2} \
                 | {:.2} | {v} |",
                r0.resident_slots,
                r0.host_threads,
                toks,
                100.0 * warm.hit_rate,
                warm.staged_mib,
                warm.cpu_per_step,
                100.0 * warm.cpu_share,
                warm.busy_ms,
                warm.wait_ms,
            );
            if control_ids.is_none() {
                control_ids = Some(warm.ids.clone());
                control_toks = Some(toks);
            }
        }
    }

    println!(
        "\n`cpu/step` counts experts, not blocks. `busy` is the pool's own wall time per token; \
         `wait` is what the device thread lost to it. Their difference is the overlap."
    );
    if failures > 0 {
        eprintln!("\n{failures} row(s) failed or were unstable between the cold and warm passes");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
