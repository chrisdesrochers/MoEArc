//! Fetch a GGUF from the Hugging Face Hub.
//!
//! ```text
//! cargo run -p moearc-model --example pull -- \
//!     --repo Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q2_k --dest /models
//! ```
//!
//! A stand-in for `moearc pull` while the real CLI is built elsewhere, and the demonstration
//! that the library keeps its side of the bargain: every character below is printed *here*, by
//! the caller. `moearc_model::pull` writes nothing — it reports through the callback on line
//! `progress`, which is exactly what a ratatui gauge will subscribe to instead.
//!
//! 🔴 **There is deliberately no `--token` flag.** A token on a command line is visible in
//! `ps`, in shell history, and in any CI log that echoes the command. `--token-file` and
//! `HF_TOKEN` are the supported routes.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Mutex;

use moearc_model::pull::{
    FileSelect, HfToken, ProgressSink, PullRequest, TokenSource, Verification, WHY_A_TOKEN_HELPS,
};

/// Renders a one-line gauge on stderr, so stdout stays machine-readable.
struct Bar {
    started: std::time::Instant,
    last_len: Mutex<usize>,
}

impl ProgressSink for Bar {
    fn on_progress(&self, downloaded: u64, total: u64) {
        let pct = if total == 0 { 0.0 } else { 100.0 * downloaded as f64 / total as f64 };
        let secs = self.started.elapsed().as_secs_f64();
        let rate = if secs > 0.0 { downloaded as f64 / secs / (1 << 20) as f64 } else { 0.0 };
        let line = format!(
            "  {pct:5.1}%  {:>9.1} / {:.1} MiB  {rate:6.1} MiB/s",
            downloaded as f64 / (1 << 20) as f64,
            total as f64 / (1 << 20) as f64,
        );
        let mut last = self.last_len.lock().unwrap();
        // Pad over the previous line rather than clearing: no terminal control codes, so the
        // output stays readable when redirected to a file.
        eprint!("\r{line}{:width$}", "", width = last.saturating_sub(line.len()));
        *last = line.len();
        let _ = std::io::stderr().flush();
    }
}

fn main() -> ExitCode {
    let mut repo = None;
    let mut file = None;
    let mut quant = None;
    let mut dest = None;
    let mut revision = None;
    let mut token_file: Option<PathBuf> = None;
    let mut verify = true;
    let mut force = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repo" => repo = args.next(),
            "--file" => file = args.next(),
            "--quant" => quant = args.next(),
            "--dest" => dest = args.next().map(PathBuf::from),
            "--revision" => revision = args.next(),
            "--token-file" => token_file = args.next().map(PathBuf::from),
            "--no-verify" => verify = false,
            "--force" => force = true,
            other => {
                eprintln!("unknown argument: {other}");
                return usage();
            }
        }
    }

    let (Some(repo), Some(dest)) = (repo, dest) else { return usage() };
    let select = match (file, quant) {
        (Some(f), None) => FileSelect::Exact(f),
        (None, Some(q)) => FileSelect::Quant(q),
        _ => {
            eprintln!("give exactly one of --file or --quant");
            return usage();
        }
    };

    // Read the token from a file, never from a flag. A failure to read is not fatal: an
    // anonymous download still works for a public repo, and saying so beats refusing.
    let token = match token_file {
        Some(p) => match std::fs::read_to_string(&p) {
            Ok(s) => HfToken::new(s),
            Err(e) => {
                eprintln!("could not read {}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let req = PullRequest { repo, select, revision, dest_dir: dest, token, verify, force };
    let bar = Bar { started: std::time::Instant::now(), last_len: Mutex::new(0) };

    let result = moearc_model::pull::pull(&req, Some(&bar));
    eprintln!();

    match result {
        Ok(m) => {
            println!("path                {}", m.path.display());
            println!(
                "file_size           {} ({:.1} MiB)",
                m.file_size,
                m.file_size as f64 / (1 << 20) as f64
            );
            println!("bytes_transferred   {}", m.bytes_transferred);
            println!(
                "resumed_from        {}{}",
                m.resumed_from,
                if m.was_resumed() { "   <-- RESUMED an interrupted transfer" } else { "" }
            );
            println!("token               {}", m.token_source);
            if m.token_source == TokenSource::Absent {
                println!("                    {WHY_A_TOKEN_HELPS}");
            }
            match m.verification {
                Verification::Skipped => println!("verification        skipped (--no-verify)"),
                Verification::Dense { architecture } => {
                    println!("verification        valid GGUF, dense model ({architecture})");
                }
                Verification::MixtureOfExperts(info) => {
                    println!("verification        valid GGUF, MoE");
                    println!("  architecture      {}", info.architecture);
                    println!(
                        "  experts           {} total, {} active",
                        info.total_experts, info.active_experts
                    );
                    println!("  per_expert_bytes  {}", info.per_expert_bytes);
                    println!("  weights_bytes     {}", info.weights_bytes);
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: pull --repo <owner/name> (--file <name> | --quant <substring>) --dest <dir>\n\
         \x20            [--revision <ref>] [--token-file <path>] [--no-verify] [--force]"
    );
    ExitCode::FAILURE
}
