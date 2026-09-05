//! Print every expert slice's address and a hash of its bytes, for external verification.
//!
//! ```text
//! cargo run -p moearc-model --example expert_probe -- <model.gguf> <block> <gate|up|down>
//! ```
//!
//! 🔴 **This exists because an off-by-one expert stride is silent.** The wrong slice is the right
//! length and is still well-formed quantised data; nothing crashes, and the model simply answers
//! with a blend of two experts. A test written against this crate's own arithmetic cannot catch
//! that, because it would compute the same wrong offset twice and agree with itself.
//!
//! So this prints, per expert, the *absolute file offset*, the byte length and an FNV-1a-64 hash
//! of the bytes — everything a second, independent reader needs to compute the same three numbers
//! its own way and compare. The output is one tab-separated row per expert plus a `#` header, so
//! a script can diff it directly.

use std::path::PathBuf;
use std::process::ExitCode;

use moearc_model::tensors::{ExpertBank, MappedModel};

/// FNV-1a, 64-bit. Chosen because it is four lines in any language, so the checking side owes no
/// dependency and cannot accidentally share an implementation with this one.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, block, bank] = args.as_slice() else {
        eprintln!("usage: expert_probe <model.gguf> <block> <gate|up|down>");
        return ExitCode::FAILURE;
    };
    let Ok(block) = block.parse::<u32>() else {
        eprintln!("block must be a number");
        return ExitCode::FAILURE;
    };
    let bank = match bank.as_str() {
        "gate" => ExpertBank::Gate,
        "up" => ExpertBank::Up,
        "down" => ExpertBank::Down,
        other => {
            eprintln!("unknown bank `{other}`: expected gate, up or down");
            return ExitCode::FAILURE;
        }
    };

    let m = match MappedModel::open(&PathBuf::from(path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let parent = match m.block_tensor(block, bank.suffix()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(n_experts) = m.expert_count() else {
        eprintln!("error: this model declares no expert count");
        return ExitCode::FAILURE;
    };

    println!(
        "# tensor={} dims={:?} quant={} file_offset={} bytes={} experts={}",
        parent.name,
        parent.dims,
        parent.quant.name,
        parent.file_offset,
        parent.len(),
        n_experts
    );
    println!("# expert\tfile_offset\tbytes\tfnv1a64");
    for k in 0..n_experts {
        match m.expert(block, bank, k) {
            Ok(e) => {
                println!("{k}\t{}\t{}\t{:016x}", e.file_offset, e.len(), fnv1a64(e.data));
            }
            Err(e) => {
                eprintln!("error on expert {k}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
