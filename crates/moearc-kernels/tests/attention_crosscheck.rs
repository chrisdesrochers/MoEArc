//! Cross-check attention and the masked softmax against **ggml's own operators**.
//!
//! `forward_pass_gpu.rs` checks these kernels against a CPU reference in this repository. Both
//! were written by the same hand from the same reading of `build_attn_mha`, so agreement
//! between them cannot rule out a shared misunderstanding — whether the scale multiplies the
//! logits or the query, whether the mask is additive or multiplicative, which axis the softmax
//! runs along. This rules that out: the expected values come from `ggml_mul_mat` and
//! `ggml_soft_max_ext` executed on ggml's CPU backend.
//!
//! `tools/ggml_attn_ref.c` produces the golden directory; see its header for the build line and
//! for what the check does **not** cover. Point `MOEARC_ATTN_GOLDEN` at the output. Without it
//! the test skips loudly rather than passing quietly.
//!
//! One thing this gets for free and is worth naming: ggml computed its answer over a flat,
//! contiguous run of keys, while MoEArc reads the same keys through a deliberately scattered
//! page table. Agreement is therefore also evidence that the paging is transparent.

mod common;

use common::{assert_close, gpu_available};
use moearc_kernels::{Context, KvType, reference};
use std::path::{Path, PathBuf};

const U: f64 = 5.960_464_477_539_063e-8;

fn read_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(b.len() % 4 == 0, "{}: not a whole number of f32", path.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Meta {
    n_heads: usize,
    head_dim: usize,
    n_kv: usize,
    scale: f32,
    sm_rows: usize,
    sm_cols: usize,
    rn_rows: usize,
    rn_cols: usize,
    rn_eps: f32,
}

fn read_meta(dir: &Path) -> Meta {
    let t = std::fs::read_to_string(dir.join("attn.meta"))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join("attn.meta").display()));
    let mut lines = t.lines();
    let a: Vec<&str> = lines.next().expect("attn.meta line 1").split_whitespace().collect();
    let b: Vec<&str> = lines.next().expect("attn.meta line 2").split_whitespace().collect();
    let c: Vec<&str> = lines.next().expect("attn.meta line 3").split_whitespace().collect();
    assert_eq!(a.len(), 4);
    assert_eq!(b.len(), 3);
    assert_eq!(c.len(), 3);
    Meta {
        n_heads: a[0].parse().unwrap(),
        head_dim: a[1].parse().unwrap(),
        n_kv: a[2].parse().unwrap(),
        scale: a[3].parse().unwrap(),
        sm_rows: b[0].parse().unwrap(),
        sm_cols: b[1].parse().unwrap(),
        rn_rows: c[0].parse().unwrap(),
        rn_cols: c[1].parse().unwrap(),
        rn_eps: c[2].parse().unwrap(),
    }
}

fn golden_dir() -> Option<PathBuf> {
    std::env::var_os("MOEARC_ATTN_GOLDEN").map(PathBuf::from)
}

#[test]
fn paged_attention_agrees_with_ggmls_own_operators() {
    if !gpu_available() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 to run against a real GPU");
        return;
    }
    let Some(dir) = golden_dir() else {
        eprintln!(
            "skipped: set MOEARC_ATTN_GOLDEN to a directory produced by tools/ggml_attn_ref.c \
             (see that file for how to build and run it)"
        );
        return;
    };
    let m = read_meta(&dir);
    let q = read_f32(&dir.join("attn.q.f32"));
    let k = read_f32(&dir.join("attn.k.f32"));
    let v = read_f32(&dir.join("attn.v.f32"));
    let want = read_f32(&dir.join("attn.out.f32"));
    assert_eq!(q.len(), m.n_heads * m.head_dim);
    assert_eq!(k.len(), m.n_kv * m.n_heads * m.head_dim);
    assert_eq!(want.len(), m.n_heads * m.head_dim);

    let ctx = Context::new().unwrap();
    const PAGE_TOKENS: usize = 8;
    const PAGES: usize = 16;
    // Scattered on purpose. ggml saw a contiguous run; if the page walk is right, the answer is
    // the same one.
    let block_table: Vec<u32> = vec![11, 2, 15, 7, 0];
    assert!(block_table.len() * PAGE_TOKENS >= m.n_kv, "block table too short for {} keys", m.n_kv);

    let row = m.n_heads * m.head_dim;
    let pool = PAGES * PAGE_TOKENS * row;
    let dk = ctx.alloc_n::<f32>(pool).unwrap();
    let dv = ctx.alloc_n::<f32>(pool).unwrap();
    let dk_tok = ctx.alloc_n::<f32>(row).unwrap();
    let dv_tok = ctx.alloc_n::<f32>(row).unwrap();

    for j in 0..m.n_kv {
        let s = j * row;
        ctx.upload_slice(&dk_tok, &k[s..s + row]).unwrap();
        ctx.upload_slice(&dv_tok, &v[s..s + row]).unwrap();
        ctx.kv_append(
            &dk,
            &dv,
            &dk_tok,
            &dv_tok,
            block_table[j / PAGE_TOKENS],
            (j % PAGE_TOKENS) as u32,
            m.n_heads,
            m.head_dim,
            PAGE_TOKENS,
            KvType::F32,
        )
        .unwrap();
    }

    let dq = ctx.alloc_n::<f32>(q.len()).unwrap();
    let dbt = ctx.alloc_n::<u32>(block_table.len()).unwrap();
    let dout = ctx.alloc_n::<f32>(q.len()).unwrap();
    ctx.upload_slice(&dq, &q).unwrap();
    ctx.upload_slice(&dbt, &block_table).unwrap();
    ctx.attn_decode(
        &dout,
        &dq,
        &dk,
        &dv,
        &dbt,
        m.n_heads,
        m.n_heads, // the golden is generated without GQA, as OLMoE has none
        m.head_dim,
        m.n_kv,
        PAGE_TOKENS,
        m.scale,
        KvType::F32,
    )
    .unwrap();
    let mut got = vec![0.0f32; q.len()];
    ctx.download_slice(&mut got, &dout).unwrap();

    // ggml reduces in f32 with its own (SIMD, partially unrolled) ordering; MoEArc reduces in
    // f32 over a 128-lane tree and accumulates the softmax online. Neither is the exact answer,
    // so the bound is the sum of both sides' error models: a head_dim-term dot product, a
    // softmax over n_kv, and an n_kv-term weighted average of values bounded by 1.
    let dot = (m.head_dim as f64 / 32.0 + 5.0) * U;
    let tol = 8.0 * (dot + (m.n_kv as f64) * U + 8.0 * U);
    assert_close("attn_decode vs ggml", &got, &want, tol);

    // A kernel returning zeros would satisfy any tolerance if the golden were also near zero.
    let peak = want.iter().fold(0.0f64, |a, x| a.max(f64::from(*x).abs()));
    assert!(peak > 1e-3, "ggml's attention output is suspiciously flat (peak {peak:e})");
    eprintln!(
        "attention: {} heads x {} dims over {} keys through a scattered {}-page table, \
         peak |ggml| = {peak:.3e}",
        m.n_heads,
        m.head_dim,
        m.n_kv,
        block_table.len()
    );
}

#[test]
fn the_masked_softmax_agrees_with_ggml_soft_max_ext() {
    if !gpu_available() {
        return;
    }
    let Some(dir) = golden_dir() else {
        eprintln!("skipped: MOEARC_ATTN_GOLDEN is not set");
        return;
    };
    let m = read_meta(&dir);
    let x = read_f32(&dir.join("sm.x.f32"));
    let mask = read_f32(&dir.join("sm.mask.f32"));
    let want = read_f32(&dir.join("sm.out.f32"));
    assert_eq!(x.len(), m.sm_rows * m.sm_cols);

    let ctx = Context::new().unwrap();
    let dx = ctx.alloc_n::<f32>(x.len()).unwrap();
    let dm = ctx.alloc_n::<f32>(mask.len()).unwrap();
    let dout = ctx.alloc_n::<f32>(x.len()).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.upload_slice(&dm, &mask).unwrap();
    ctx.softmax_ext(&dout, &dx, Some(&dm), m.sm_rows, m.sm_cols, m.scale).unwrap();
    let mut got = vec![0.0f32; x.len()];
    ctx.download_slice(&mut got, &dout).unwrap();

    // Outputs are bounded by 1: two `exp`s at 4 ulp each on either side, plus a sum over
    // sm_cols terms in f32.
    let tol = 2.0 * (4.0 + 4.0 + m.sm_cols as f64 / 32.0 + 5.0) * U;
    assert_close("softmax_ext vs ggml_soft_max_ext", &got, &want, tol);

    // ggml and MoEArc must agree that a masked position is exactly zero, not merely small.
    let masked = mask.iter().filter(|v| v.is_infinite()).count();
    assert!(masked > 0, "the golden mask masks nothing — the check would prove nothing");
    for (i, mv) in mask.iter().enumerate() {
        if mv.is_infinite() {
            assert_eq!(got[i], 0.0, "position {i} is masked but got weight {}", got[i]);
            assert_eq!(want[i], 0.0, "ggml gave a masked position weight {}", want[i]);
        }
    }
    eprintln!("masked softmax: {} of {} positions masked, all exactly zero", masked, x.len());

    // And the CPU reference must agree with ggml too, independently of the GPU — otherwise a
    // matching pair could still both be wrong in the same way.
    let cpu = reference::softmax_ext(&x, Some(&mask), m.sm_rows, m.sm_cols, m.scale);
    assert_close("reference::softmax_ext vs ggml", &cpu, &want, tol);
}

#[test]
fn rmsnorm_agrees_with_ggmls_own_operator() {
    if !gpu_available() {
        return;
    }
    let Some(dir) = golden_dir() else {
        eprintln!("skipped: MOEARC_ATTN_GOLDEN is not set");
        return;
    };
    // 🔴 This is the operation that carries OLMoE's QK-norm. `blk.N.attn_q_norm` and
    // `blk.N.attn_k_norm` are f32 [n_embd] and `build_olmoe` applies them with
    // `build_norm(..., LLM_NORM_RMS)` across the whole 2048-wide vector, before the reshape into
    // heads and before RoPE — not per head. Most MoE implementations leave QK-norm out
    // altogether and the output degrades in a way that reads as a bug somewhere else, so the
    // norm itself is worth pinning against ggml rather than only against a reference of my own.
    let m = read_meta(&dir);
    let x = read_f32(&dir.join("rn.x.f32"));
    let w = read_f32(&dir.join("rn.w.f32"));
    let want = read_f32(&dir.join("rn.out.f32"));
    assert_eq!(x.len(), m.rn_rows * m.rn_cols);
    assert_eq!(w.len(), m.rn_cols);

    let ctx = Context::new().unwrap();
    let dx = ctx.alloc_n::<f32>(x.len()).unwrap();
    let dw = ctx.alloc_n::<f32>(w.len()).unwrap();
    let dout = ctx.alloc_n::<f32>(x.len()).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.upload_slice(&dw, &w).unwrap();
    ctx.rmsnorm(&dout, &dx, Some(&dw), m.rn_rows, m.rn_cols, m.rn_eps).unwrap();
    let mut got = vec![0.0f32; x.len()];
    ctx.download_slice(&mut got, &dout).unwrap();

    // ggml sums the squares in double; MoEArc sums them in f32 over a 32-lane tree, because
    // fp64 on Arc is emulated where it exists at all. The relative error of the sum is about
    // (cols/32 + 5) * U, halved by the square root, plus a few ulp for rsqrt and the multiply.
    let peak = want.iter().fold(0.0f64, |a, v| a.max(f64::from(*v).abs()));
    let tol = (0.5 * (m.rn_cols as f64 / 32.0 + 5.0) + 8.0) * U * peak;
    assert_close("rmsnorm vs ggml", &got, &want, tol);

    // And the CPU reference against ggml independently, so a matching pair of mine cannot both
    // be wrong the same way.
    let cpu = reference::rmsnorm(&x, Some(&w), m.rn_rows, m.rn_cols, m.rn_eps);
    assert_close("reference::rmsnorm vs ggml", &cpu, &want, tol);
    eprintln!("rmsnorm: {} rows x {} columns, ggml eps {:e}", m.rn_rows, m.rn_cols, m.rn_eps);
}
