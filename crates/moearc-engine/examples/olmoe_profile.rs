//! Where a decode step's time goes, per phase, on real hardware.
//!
//! ```text
//! MOEARC_PROFILE=1 cargo run -p moearc-engine --features gpu --example olmoe_profile -- \
//!     <model.gguf> <n-predict> <token-id> [token-id ...]
//! ```
//!
//! The first few tokens are thrown away before the counters are read: token one pays a cold
//! expert cache and a first-touch of every device allocation, and averaging that into a
//! steady-state breakdown would attribute staging cost to whatever phase happened to run first.
//!
//! `decode.total` wraps every other phase, so it is reported separately and is not part of the
//! sum. The gap between it and the sum of the parts is host work no phase claimed.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::profile;
use moearc_engine::session::{Session, StopConditions};

/// Tokens decoded before the counters are cleared.
const WARMUP: usize = 5;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: olmoe_profile <model.gguf> <n-predict> <token-id> [token-id ...]");
        return ExitCode::FAILURE;
    }
    if !profile::enabled() {
        eprintln!("MOEARC_PROFILE=1 is not set; there would be nothing to report");
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let Ok(n_predict) = args[2].parse::<usize>() else {
        eprintln!("n-predict must be a number");
        return ExitCode::FAILURE;
    };
    let tokens: Vec<u32> = args[3..].iter().filter_map(|s| s.parse().ok()).collect();

    let session = match Session::load(&model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("device        {}", session.info().device);

    let stop = StopConditions { max_tokens: n_predict, stop_tokens: Vec::new() };
    let mut step = 0usize;
    // Wall clock for the measured window only, so tok/s and the breakdown describe the same
    // tokens.
    let mut measured_from = Instant::now();
    let mut ids = Vec::new();
    let r = session.generate(&tokens, &stop, &mut |t| {
        ids.push(t);
        step += 1;
        if step == WARMUP {
            profile::reset();
            measured_from = Instant::now();
        }
        true
    });
    let wall = measured_from.elapsed();
    if let Err(e) = r {
        eprintln!("generate failed: {e}");
        return ExitCode::FAILURE;
    }
    let measured = step.saturating_sub(WARMUP);
    if measured == 0 {
        eprintln!("nothing measured: ask for more than {WARMUP} tokens");
        return ExitCode::FAILURE;
    }

    let phases = profile::report();
    let total = phases.iter().find(|p| p.name == "decode.total");
    let sum: f64 = phases.iter().filter(|p| p.name != "decode.total").map(|p| p.seconds).sum();

    println!(
        "\nsteady state over {measured} tokens ({:.2} tok/s)",
        measured as f64 / wall.as_secs_f64()
    );
    println!(
        "{:<20} {:>10} {:>9} {:>9} {:>7}",
        "phase", "ms/token", "calls/tok", "us/call", "share"
    );
    let denom = total.map_or(sum, |t| t.seconds);
    for p in phases.iter().filter(|p| p.name != "decode.total") {
        let per_tok_ms = p.seconds * 1000.0 / measured as f64;
        let calls = p.calls as f64 / measured as f64;
        println!(
            "{:<20} {:>10.2} {:>9.1} {:>9.1} {:>6.1}%",
            p.name,
            per_tok_ms,
            calls,
            p.seconds * 1e6 / p.calls as f64,
            100.0 * p.seconds / denom
        );
    }
    println!("{:<20} {:>10.2}", "SUM OF PHASES", sum * 1000.0 / measured as f64);
    if let Some(t) = total {
        println!("{:<20} {:>10.2}", "decode.total", t.seconds * 1000.0 / measured as f64);
        println!(
            "{:<20} {:>10.2}   (host work no phase claimed)",
            "unattributed",
            (t.seconds - sum) * 1000.0 / measured as f64
        );
    }
    println!(
        "{:<20} {:>10.2}   (session plumbing outside decode)",
        "outside decode",
        (wall.as_secs_f64() - total.map_or(0.0, |t| t.seconds)) * 1000.0 / measured as f64
    );
    ExitCode::SUCCESS
}
