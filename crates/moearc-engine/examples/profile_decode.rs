//! Where a decode step's time goes, per phase, on real hardware.
//!
//! ```text
//! MOEARC_PROFILE=1 cargo run -p moearc-engine --features gpu --example profile_decode -- \
//!     <model.gguf> <n-predict> <residency> <n-ctx|-> <token-id> [token-id ...]
//! ```
//!
//! `<residency>` is a [`Residency`] spec (`all`, `<slots>`, `plan:<bytes>`, `static:<blocks>`)
//! and `<n-ctx>` a token count or `-` for the model's trained maximum. 🔴 Required rather than
//! defaulted: on a model that does not fit the card the defaults cannot be allocated, and a
//! profile of a configuration the caller did not choose is worse than no profile.
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

use moearc_engine::host_experts::HostPolicy;
use moearc_engine::moe::Residency;
use moearc_engine::profile;
use moearc_engine::session::{Session, SessionOptions, StopConditions};

/// Tokens decoded before the counters are cleared.
const WARMUP: usize = 5;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: profile_decode <model.gguf> <n-predict> <residency> <n-ctx|-> <token-id> \
             [token-id ...]"
        );
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
    let residency: Residency = match args[3].parse() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let n_ctx = if args[4] == "-" {
        None
    } else {
        match args[4].parse::<usize>() {
            Ok(n) => Some(n),
            Err(_) => {
                eprintln!("`{}` is not a context length", args[4]);
                return ExitCode::FAILURE;
            }
        }
    };
    let tokens: Vec<u32> = args[5..].iter().filter_map(|s| s.parse().ok()).collect();

    // An environment variable rather than a positional, because every positional after this
    // point is a token id and an optional one in the middle of them would be ambiguous.
    let host: HostPolicy = match std::env::var("MOEARC_HOST_EXPERTS") {
        Ok(v) => match v.parse() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("MOEARC_HOST_EXPERTS: {e}");
                return ExitCode::FAILURE;
            }
        },
        Err(_) => HostPolicy::Off,
    };

    let session = match Session::load_with(&model, SessionOptions { n_ctx, residency, host }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("device        {}", session.info().device);
    println!("host experts  {host} on {} threads", session.info().residency.host_threads);

    // 🔴 The warm-up is its own generation, and the device-event counters are cleared between
    // the two.
    //
    // Those counters are cumulative from process start, and — unlike `profile::reset()`, which
    // is a free function — clearing them means calling back into the `Session`. `on_token` runs
    // while `generate` holds the session's link mutex, so doing it from the callback deadlocks.
    // (It did. That is why this is two calls and not a branch inside one.)
    //
    // Left uncleared, the prompt pass and the cold-cache warm-up fold into a steady-state
    // per-token figure: measured, that inflated every kernel by 17% and reported `calls/tok` of
    // 55.3 on a model with 48 blocks. **`calls/tok` landing on a whole number is the check that
    // this is right**, and it is worth glancing at every time.
    let warm = StopConditions { max_tokens: WARMUP, stop_tokens: Vec::new() };
    if let Err(e) = session.generate(&tokens, &warm, &mut |_| true) {
        eprintln!("warm-up failed: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = session.reset_event_profile() {
        eprintln!("could not clear the device-event counters: {e}");
        return ExitCode::FAILURE;
    }

    // 🔴 Everything below is per **decode step**, not per emitted token, and the two are not the
    // same number: `generate` decodes the whole prompt and then feeds back each accepted token
    // except the last, so `n` tokens cost `prompt + n - 1` steps. The host phases and the device
    // events are now reset together, immediately before this generation, so both cover exactly
    // these steps. Dividing by tokens instead put `calls/step` at 55.3 on a 48-block model —
    // which is the check: **this number must come out at 48.**
    profile::reset();
    let stop = StopConditions { max_tokens: n_predict, stop_tokens: Vec::new() };
    let mut ids = Vec::new();
    let measured_from = Instant::now();
    let r = session.generate(&tokens, &stop, &mut |t| {
        ids.push(t);
        true
    });
    let wall = measured_from.elapsed();
    if let Err(e) = r {
        eprintln!("generate failed: {e}");
        return ExitCode::FAILURE;
    }
    let measured = tokens.len() + ids.len().saturating_sub(1);
    if measured == 0 {
        eprintln!("nothing measured");
        return ExitCode::FAILURE;
    }

    // Device timestamps from the events themselves, if the queue was built to carry them.
    // Printed alongside the host-side phases on purpose: the two disagreeing is the finding.
    match session.event_profile() {
        Ok(ev) if !ev.is_empty() => {
            let total_ns: u64 = ev.iter().map(|(_, ns, _)| *ns).sum();
            println!("\ndevice time from SYCL events (queue still asynchronous)");
            println!("{:<26} {:>10} {:>10} {:>9}", "kernel", "ms/step", "calls/step", "us/call");
            let mut rows = ev.clone();
            rows.sort_by_key(|(_, ns, _)| std::cmp::Reverse(*ns));
            for (key, ns, calls) in &rows {
                println!(
                    "{:<26} {:>10.2} {:>10.1} {:>9.1}",
                    key,
                    *ns as f64 / 1e6 / measured as f64,
                    *calls as f64 / measured as f64,
                    *ns as f64 / 1e3 / *calls as f64
                );
            }
            let busy_ms = total_ns as f64 / 1e6 / measured as f64;
            let step_ms = wall.as_secs_f64() * 1000.0 / measured as f64;
            println!(
                "{:<26} {:>10.2}   tracked device busy, {:.1}% of the {:.2} ms step",
                "TRACKED BUSY",
                busy_ms,
                100.0 * busy_ms / step_ms,
                step_ms
            );
        }
        Ok(_) => println!("\n(set MOEARC_PROFILE_EVENTS=1 for per-kernel device time)"),
        Err(e) => println!("\nevent profile unavailable: {e}"),
    }

    let phases = profile::report();
    let total = phases.iter().find(|p| p.name == "decode.total");
    let sum: f64 = phases.iter().filter(|p| p.name != "decode.total").map(|p| p.seconds).sum();

    println!(
        "\nsteady state over {measured} decode steps ({:.2} tok/s)",
        ids.len() as f64 / wall.as_secs_f64()
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
    if let Ok(r) = session.residency() {
        if r.host.jobs > 0 {
            // 🔴 The overlap, stated as one line. `busy` is the pool's own wall time; `wait` is
            // what the device thread lost to it. They are measured by different clocks on
            // different threads on purpose — if they were the same measurement the difference
            // could not exist.
            let busy = r.host.busy_nanos as f64 / 1e6 / measured as f64;
            let waited = r.host.wait_nanos as f64 / 1e6 / measured as f64;
            println!(
                "\nhost pool: {:.2} ms/step busy, {:.2} ms/step waited for -> {:.0}% of the \
                 host work was hidden behind device work ({} experts/step over {} jobs/step)",
                busy,
                waited,
                100.0 * (1.0 - (waited / busy).min(1.0)),
                r.host.experts / measured.max(1) as u64,
                r.host.jobs / measured.max(1) as u64,
            );
        }
    }
    println!(
        "{:<20} {:>10.2}   (session plumbing outside decode)",
        "outside decode",
        (wall.as_secs_f64() - total.map_or(0.0, |t| t.seconds)) * 1000.0 / measured as f64
    );
    ExitCode::SUCCESS
}
