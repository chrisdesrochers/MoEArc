//! Compare MoEArc's forward pass against a llama.cpp tensor dump, block by block.
//!
//! ```text
//! cargo run -p moearc-engine --features gpu --example probe -- \
//!     <model.gguf> <ref-dir|-> <residency> <n-ctx|-> <token-id> [token-id ...]
//! ```
//!
//! The reference directory is what `llama-eval-callback` writes when `MOEARC_DUMP_DIR` is set:
//! one `<seq>__<tensor-name>.f32` per graph node, raw little-endian f32, row-major. `-` skips
//! the comparison and just reports the logits.
//!
//! `<residency>` is a [`Residency`] spec (`all`, `<slots>`, `plan:<bytes>`, `static:<blocks>`)
//! and `<n-ctx>` is a token count or `-` for the model's trained maximum. 🔴 Both are required
//! rather than defaulted, because on a model that does not fit the card the defaults —
//! everything resident, the full trained context — are an allocation failure, and a probe that
//! silently picked something smaller would be comparing a configuration the caller did not
//! choose.
//!
//! 🔴 The point of the per-block report is that a forward pass which emits garbage cannot be
//! debugged from its output. A difference that first exceeds the noise floor at block 7 names
//! block 7; a single number at the end names nothing.
//!
//! # Two things the dump does not line up with on its own
//!
//! - **llama.cpp prefills the whole prompt in one graph; this engine decodes one token at a
//!   time and taps only the last.** So a `[n_embd, n_tokens]` dump is compared against its
//!   **final column**, which is the same token. `l_out-<last>` is already one column wide, because
//!   `inp_out_ids` gathers at the last block only — so both shapes occur in a single dump and
//!   both have to work.
//! - **`ffn_moe_weights` is the pre-normalisation tensor.** Under `norm_w = true` (`qwen3moe`)
//!   llama.cpp divides by the sum afterwards and calls the result `ffn_moe_weights_norm`; this
//!   engine's tap holds the value the combine actually uses, so it is compared against
//!   `ffn_moe_weights_norm` when the dump has one. Comparing against the raw name would report a
//!   spurious difference on every block of a normalising model.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, argmax};

/// The `<n>__<name>.f32` file for a tensor name, if the dump has one.
fn ref_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let want = format!("__{name}.f32");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(&want)))
        .collect();
    hits.sort();
    // The last one wins: a graph can name two nodes the same, and the later is the later stage.
    hits.pop()
}

fn read_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

struct Diff {
    max_abs: f64,
    rms_ref: f64,
    cosine: f64,
    worst_at: usize,
}

fn compare(got: &[f32], want: &[f32]) -> Diff {
    let n = got.len().min(want.len());
    let mut max_abs = 0.0f64;
    let mut worst_at = 0usize;
    let (mut dot, mut ga, mut wa) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (g, w) = (f64::from(got[i]), f64::from(want[i]));
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
            worst_at = i;
        }
        dot += g * w;
        ga += g * g;
        wa += w * w;
    }
    Diff {
        max_abs,
        rms_ref: (wa / n as f64).sqrt(),
        cosine: if ga > 0.0 && wa > 0.0 { dot / (ga.sqrt() * wa.sqrt()) } else { 0.0 },
        worst_at,
    }
}

/// The slice of a reference tensor that corresponds to the tapped token.
///
/// A prefill dump is `[n, n_tokens]` with the token index slowest-varying, so the last `n`
/// floats are the last token — the only one this engine taps. A length that is not a whole
/// multiple is a genuine shape disagreement and is left to `report` to call out.
fn last_token<'a>(got: &[f32], want: &'a [f32]) -> (&'a [f32], usize) {
    if !got.is_empty() && want.len() > got.len() && want.len() % got.len() == 0 {
        let tokens = want.len() / got.len();
        (&want[want.len() - got.len()..], tokens)
    } else {
        (want, 1)
    }
}

fn report(label: &str, got: &[f32], want: &[f32]) {
    let (want, tokens) = last_token(got, want);
    let of = if tokens > 1 { format!(" [col {}/{tokens}]", tokens - 1) } else { String::new() };
    if got.len() != want.len() {
        println!("{label:<22} SHAPE MISMATCH got {} want {}", got.len(), want.len());
        return;
    }
    // The expert lists are integers, and a summary statistic over them is meaningless: what
    // matters is whether the same experts were chosen, so both lists are printed.
    if label.starts_with("ffn_moe_topk") {
        let g: Vec<i32> = got.iter().map(|v| *v as i32).collect();
        let w: Vec<i32> = want.iter().map(|v| *v as i32).collect();
        let mut gs = g.clone();
        let mut ws = w.clone();
        gs.sort_unstable();
        ws.sort_unstable();
        let verdict = if g == w {
            "same"
        } else if gs == ws {
            "SAME SET, different order"
        } else {
            "DIFFERENT SET"
        };
        println!("{label:<22} {verdict:<26}{of} got {g:?} want {w:?}");
        return;
    }
    let d = compare(got, want);
    println!(
        "{label:<22} n={:<6} max|d|={:<11.3e} rms(ref)={:<11.3e} rel={:<11.3e} 1-cos={:<11.3e} \
         worst@{} ({:.5} vs {:.5}){of}",
        got.len(),
        d.max_abs,
        d.rms_ref,
        if d.rms_ref > 0.0 { d.max_abs / d.rms_ref } else { f64::NAN },
        1.0 - d.cosine,
        d.worst_at,
        got[d.worst_at],
        want[d.worst_at]
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: probe <model.gguf> <ref-dir|-> <residency> <n-ctx|-> <token-id> \
             [token-id ...]"
        );
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let refdir = if args[2] == "-" { None } else { Some(PathBuf::from(&args[2])) };
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

    let session = match Session::load_with(&model, SessionOptions { n_ctx, residency }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info = session.info();
    println!("device        {}", info.device);
    let r = info.residency;
    println!(
        "resident      {} MiB dense + {}/{} expert slots ({:.1}%, {} MiB pool, {} policy)",
        r.dense_bytes >> 20,
        r.resident_slots,
        r.total_slots,
        100.0 * r.resident_fraction(),
        r.pool_bytes >> 20,
        r.policy
    );
    println!("arch          {} n_ctx {}", info.config.arch, info.n_ctx);
    println!("tokens        {tokens:?}");

    let (logits, tap) = match session.logits_tapped(&tokens) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("decode failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(dir) = &refdir {
        println!("\n-- against {} --", dir.display());
        for (name, got) in &tap.items {
            // `ffn_moe_weights-<il>` is llama.cpp's pre-normalisation tensor; under
            // `norm_w = true` the value this engine's combine uses is `ffn_moe_weights_norm`.
            // Prefer that name when the dump carries it, so a normalising model is compared
            // against the same quantity rather than against the stage before it.
            let alias = name
                .strip_prefix("ffn_moe_weights-")
                .map(|il| format!("ffn_moe_weights_norm-{il}"));
            let file = alias.as_deref().and_then(|a| ref_file(dir, a));
            let (label, file) = match file {
                Some(p) => (alias.as_deref().unwrap_or(name), Some(p)),
                None => (name.as_str(), ref_file(dir, name)),
            };
            if let Some(p) = file {
                match read_f32(&p) {
                    Ok(want) => report(label, got, &want),
                    Err(e) => println!("{label:<22} unreadable: {e}"),
                }
            }
        }
        if let Some(p) = ref_file(dir, "result_output") {
            match read_f32(&p) {
                Ok(want) => {
                    report("result_output", &logits, &want);
                    let a = argmax(&want) as usize;
                    println!("llama.cpp argmax  {a} ({:.5})", want[a]);
                }
                Err(e) => println!("result_output unreadable: {e}"),
            }
        }
    }

    let best = argmax(&logits) as usize;
    println!("\nMoEArc argmax     {best} ({:.5})", logits[best]);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let top: Vec<String> = order.iter().take(8).map(|i| format!("{i}:{:.4}", logits[*i])).collect();
    println!("top-8             {}", top.join("  "));
    ExitCode::SUCCESS
}
