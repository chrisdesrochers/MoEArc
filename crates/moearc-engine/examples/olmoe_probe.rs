//! Compare MoEArc's forward pass against a llama.cpp tensor dump, block by block.
//!
//! ```text
//! cargo run -p moearc-engine --features gpu --example olmoe_probe -- \
//!     <model.gguf> <ref-dir|-> <token-id> [token-id ...]
//! ```
//!
//! The reference directory is what `llama-eval-callback` writes when `MOEARC_DUMP_DIR` is set:
//! one `<seq>__<tensor-name>.f32` per graph node, raw little-endian f32, row-major. `-` skips
//! the comparison and just reports the logits.
//!
//! 🔴 The point of the per-block report is that a forward pass which emits garbage cannot be
//! debugged from its output. A difference that first exceeds the noise floor at block 7 names
//! block 7; a single number at the end names nothing.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use moearc_engine::session::{Session, argmax};

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

fn report(label: &str, got: &[f32], want: &[f32]) {
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
        println!("{label:<22} {verdict:<26} got {g:?} want {w:?}");
        return;
    }
    let d = compare(got, want);
    println!(
        "{label:<22} n={:<6} max|d|={:<11.3e} rms(ref)={:<11.3e} rel={:<11.3e} 1-cos={:<11.3e} \
         worst@{} ({:.5} vs {:.5})",
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
    if args.len() < 4 {
        eprintln!("usage: olmoe_probe <model.gguf> <ref-dir|-> <token-id> [token-id ...]");
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let refdir = if args[2] == "-" { None } else { Some(PathBuf::from(&args[2])) };
    let tokens: Vec<u32> = args[3..].iter().filter_map(|s| s.parse().ok()).collect();

    let session = match Session::load(&model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info = session.info();
    println!("device        {}", info.device);
    println!("uploaded      {:.2} MiB", info.bytes_uploaded as f64 / (1024.0 * 1024.0));
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
            if let Some(p) = ref_file(dir, name) {
                match read_f32(&p) {
                    Ok(want) => report(name, got, &want),
                    Err(e) => println!("{name:<22} unreadable: {e}"),
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
