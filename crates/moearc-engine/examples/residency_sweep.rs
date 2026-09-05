//! Sweep expert residency and measure what it costs.
//!
//! ```text
//! cargo run -p moearc-engine --features gpu --example residency_sweep -- \
//!     <model.gguf> <n-predict> <n-ctx> <spec,spec,...> <ref-ids-file|-> \
//!     <token-id> [token-id ...]
//! ```
//!
//! A spec is a [`Residency`] as `FromStr` parses it: `all`, `<slots>`, `plan:<bytes>` or
//! `static:<blocks>`.
//!
//! 🔴 **Every row must produce the same ids**, whatever its budget: residency decides what has
//! to *move*, never what is *computed*, so a row that disagrees with the others has a paging
//! bug — a slot read before it was filled, or two experts sharing one — and a sweep that
//! reported only throughput would present that bug as a result. That check needs no reference
//! and is what the `ids` column reports when there is none.
//!
//! `<ref-ids-file>` additionally checks the ids against llama.cpp's greedy continuation of the
//! same prompt — whitespace- or comma-separated token ids, `#` comments allowed, checked for as
//! far as it reaches — and `-` skips it. ⚠️ On a long run that check is **weaker than it looks**,
//! and deliberately optional: MoEArc keeps activations in f32 where `ggml-cpu` quantises them to
//! Q8_K before every K-quant matmul, so at a step where the top two logits are within that
//! difference the two implementations legitimately take different branches and never rejoin. See
//! `tests/qwen3moe_forward.rs` for the measurement of how often that happens and on which
//! prompts it does not. The ids are a file rather than a constant because this sweep now runs on
//! more than one model.
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

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions};

/// Read a reference continuation: token ids separated by whitespace or commas, `#` to end a
/// line. Anything unparseable is an error rather than a skipped token — a reference silently
/// short by one id would turn a real divergence into a pass.
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

/// How this row's ids stand against the other rows and, if there is one, the reference.
///
/// The first is the load-bearing check and needs nothing external: every budget must produce the
/// same ids. The second is the optional llama.cpp comparison, checked as far as the reference
/// reaches.
fn verdict(ids: &[u32], baseline: Option<&[u32]>, reference: Option<&[u32]>) -> String {
    let mut parts = Vec::new();
    match baseline {
        None => parts.push("baseline".to_string()),
        Some(b) => match (0..ids.len().min(b.len())).find(|i| ids[*i] != b[*i]) {
            None if ids.len() == b.len() => parts.push("same as baseline".to_string()),
            None => parts.push(format!("LENGTH DIFFERS {} vs {}", ids.len(), b.len())),
            Some(i) => parts.push(format!("DIVERGED from baseline at {i}")),
        },
    }
    if let Some(reference) = reference {
        let n = ids.len().min(reference.len());
        match (0..n).find(|i| ids[*i] != reference[*i]) {
            None => parts.push(format!("ref {n}/{n}")),
            Some(i) => {
                parts.push(format!("ref DIVERGED at {i} (got {} want {})", ids[i], reference[i]));
            }
        }
    }
    parts.join(", ")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: residency_sweep <model.gguf> <n-predict> <n-ctx> <spec,...> \
             <ref-ids-file|-> <token-id> [token-id ...]"
        );
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let Ok(n_predict) = args[2].parse::<usize>() else {
        eprintln!("n-predict must be a number");
        return ExitCode::FAILURE;
    };
    // 🔴 Explicit, not defaulted. The KV cache is allocated for exactly this many tokens, and on
    // Qwen3-30B-A3B one token costs 96 KiB across 48 blocks — the model's own 40,960-token
    // maximum is 3.75 GiB of card, a third of the budget the sweep is measuring.
    let Ok(n_ctx) = args[3].parse::<usize>() else {
        eprintln!("n-ctx must be a number");
        return ExitCode::FAILURE;
    };
    let specs: Vec<&str> = args[4].split(',').filter(|s| !s.is_empty()).collect();
    let reference = if args[5] == "-" {
        None
    } else {
        match read_ids(Path::new(&args[5])) {
            Ok(ids) => Some(ids),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    };
    let prompt: Vec<u32> = args[6..].iter().filter_map(|s| s.parse().ok()).collect();
    if let Some(r) = &reference {
        if r.len() < n_predict {
            // Said out loud rather than silently accepted: the `ref n/n` column would otherwise
            // read as a full pass while the tail of every row went unexamined.
            println!("note: the reference holds {} of the {n_predict} ids asked for", r.len());
        }
    }

    println!(
        "prompt {prompt:?}, {n_predict} tokens, {n_ctx} ctx; cold rows are staging-dominated\n"
    );
    println!(
        "| policy | slots | % of model | pool | cold hit | cold tok/s | warm hit | warm tok/s \
         | cold staged | warm staged | ids |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    let mut failures = 0;
    // The ids of the first row that ran. Every later row must reproduce them exactly — that is
    // the paging gate, and unlike the reference it holds on any model and any length.
    let mut baseline: Option<Vec<u32>> = None;
    for spec in &specs {
        let residency: Residency = match spec.parse() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        let opts = SessionOptions { n_ctx: Some(n_ctx), residency };
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

        let v = verdict(&cold.ids, baseline.as_deref(), reference.as_deref());
        if v.contains("DIVERGED") || v.contains("LENGTH DIFFERS") || cold.ids != warm.ids {
            failures += 1;
        }
        if baseline.is_none() {
            baseline = Some(cold.ids.clone());
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
