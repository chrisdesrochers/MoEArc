//! Greedy generation from a fixed prompt, printing token ids.
//!
//! ```text
//! cargo run -p moearc-engine --features gpu --example olmoe_generate -- \
//!     <model.gguf> <n-predict> <token-id> [token-id ...]
//! ```
//!
//! Ids rather than text on purpose: two implementations can print the same string from
//! different tokenisations, and the acceptance question is whether the *ids* agree.
//!
//! Timing is printed and covers the whole run, prompt included, from a **cold expert pool** —
//! so it is a floor, not a steady state: the first tokens pay for staging every expert they
//! name. `examples/profile_decode.rs` reports the steady state and where the time goes.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::session::{Session, StopConditions};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: olmoe_generate <model.gguf> <n-predict> <token-id> [token-id ...]");
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
    println!(
        "moearc-greedy: prompt {} tokens:{}",
        tokens.len(),
        tokens.iter().map(|t| format!(" {t}")).collect::<String>()
    );

    // The engine stops on nothing but the count: the reference run prints every id it takes,
    // including an end-of-generation one, and a stop rule here would hide a divergence.
    let stop = StopConditions { max_tokens: n_predict, stop_tokens: Vec::new() };
    let mut step = 0usize;
    let started = Instant::now();
    let mut out = Vec::new();
    let stats = session.generate(&tokens, &stop, &mut |t| {
        println!("moearc-greedy: {step:3} {t:6}");
        out.push(t);
        step += 1;
        true
    });
    let elapsed = started.elapsed();

    match stats {
        Ok(s) => {
            println!("ids           {out:?}");
            println!(
                "{} prompt + {} generated in {:.2}s ({:.2} tok/s, cold pool) — {:?}",
                s.prompt_tokens,
                s.completion_tokens,
                elapsed.as_secs_f64(),
                (s.prompt_tokens + s.completion_tokens) as f64 / elapsed.as_secs_f64(),
                s.stop_reason
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("generate failed: {e}");
            ExitCode::FAILURE
        }
    }
}
