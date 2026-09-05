//! Cross-check the dequantisation kernels against **llama.cpp's own output** for real tensors
//! from a real model.
//!
//! `kernels_gpu.rs` checks the GPU against a CPU reference in this repository. Both were
//! written by the same hand from the same source, so agreement between them rules out a
//! transcription slip in one of them but not a shared misreading of the format. This test
//! rules that out: the expected values come from `ggml_get_type_traits(type)->to_float`, the
//! function pointer llama.cpp's CPU backend itself calls, run over bytes lifted straight out
//! of a production GGUF file.
//!
//! It needs a golden directory, which `tools/ggml_dequant_dump.c` produces — see that file's
//! header for the build and run lines. Point `MOEARC_GGUF_GOLDEN` at the result. Without it
//! the test skips loudly rather than silently passing, because a cross-check that quietly
//! does nothing is worse than no cross-check.

mod common;

use common::{gpu_available, max_abs_diff};
use moearc_kernels::{Context, QuantType};
use std::path::{Path, PathBuf};

/// One line of the golden directory's `index.txt`.
struct Golden {
    file_stem: String,
    tensor: String,
    type_id: u32,
    n_elements: usize,
    q_bytes: usize,
}

fn read_index(dir: &Path) -> Vec<Golden> {
    let text = std::fs::read_to_string(dir.join("index.txt"))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join("index.txt").display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(f.len(), 5, "malformed index line: {l}");
            Golden {
                file_stem: f[0].to_string(),
                tensor: f[1].to_string(),
                type_id: f[2].parse().expect("type id"),
                n_elements: f[3].parse().expect("element count"),
                q_bytes: f[4].parse().expect("quantised byte count"),
            }
        })
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(bytes.len() % 4 == 0, "{}: not a whole number of f32", path.display());
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn the_dequantisers_agree_with_llama_cpp_on_real_model_weights() {
    if !gpu_available() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 to run against a real GPU");
        return;
    }
    let Some(dir) = std::env::var_os("MOEARC_GGUF_GOLDEN").map(PathBuf::from) else {
        eprintln!(
            "skipped: set MOEARC_GGUF_GOLDEN to a directory produced by \
             tools/ggml_dequant_dump.c (see that file for how to build and run it)"
        );
        return;
    };

    let index = read_index(&dir);
    assert!(!index.is_empty(), "{}: the golden index is empty", dir.display());

    let ctx = Context::new().unwrap();
    let mut checked = 0usize;

    for g in &index {
        let Some(ty) = QuantType::from_type_id(g.type_id) else {
            eprintln!(
                "{}: ggml type {} is not one this crate expands — skipped",
                g.tensor, g.type_id
            );
            continue;
        };
        assert!(
            g.n_elements % ty.block_elems() == 0,
            "{}: {} elements is not a whole number of {}-element blocks",
            g.tensor,
            g.n_elements,
            ty.block_elems()
        );
        let nblocks = g.n_elements / ty.block_elems();
        assert_eq!(
            nblocks * ty.block_bytes(),
            g.q_bytes,
            "{}: this crate's block size for {ty:?} disagrees with the size llama.cpp wrote",
            g.tensor
        );

        let q = std::fs::read(dir.join(format!("{}.q", g.file_stem))).expect("quantised bytes");
        assert_eq!(q.len(), g.q_bytes, "{}: truncated .q file", g.tensor);
        let want = read_f32(&dir.join(format!("{}.f32", g.file_stem)));
        assert_eq!(want.len(), g.n_elements, "{}: truncated .f32 file", g.tensor);

        let dsrc = ctx.alloc(q.len()).unwrap();
        let ddst = ctx.alloc_n::<f32>(g.n_elements).unwrap();
        ctx.upload_slice(&dsrc, &q).unwrap();
        ctx.dequant(ty, &ddst, &dsrc, nblocks).unwrap();
        let mut got = vec![0.0f32; g.n_elements];
        ctx.download_slice(&mut got, &ddst).unwrap();

        let (err, at) = max_abs_diff(&got, &want);
        let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
        // Both sides compute the same expression in the same association; the only licence
        // either compiler has is to contract a multiply-subtract into an FMA, worth about one
        // ulp. Four ulp of the tensor's largest magnitude is a 4x margin on that, and the
        // measured error is printed so the margin stays visible.
        let tol = 4.0 * 5.960_464_477_539_063e-8 * scale;
        eprintln!(
            "{} [{ty:?}, {} elements]: max |gpu - llama.cpp| = {err:.3e}, peak |value| = \
             {scale:.3e}, tolerance {tol:.3e}",
            g.tensor, g.n_elements
        );
        assert!(
            err <= tol,
            "{}: MoEArc and llama.cpp disagree by {err:.6e} at element {at} (gpu {}, llama.cpp \
             {}); tolerance was {tol:.6e}",
            g.tensor,
            got[at],
            want[at]
        );
        assert!(scale > 0.0, "{}: llama.cpp produced an all-zero tensor", g.tensor);
        checked += 1;
    }

    assert!(checked > 0, "the golden directory held no tensor this crate can expand");
    eprintln!("cross-checked {checked} real tensors against llama.cpp");
}
