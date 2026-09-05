//! Every kernel in this crate, run on the real device and checked against the CPU twin in
//! `moearc_kernels::reference`.
//!
//! Gated behind `MOEARC_TEST_GPU=1`, like `gpu.rs`: CI without an Arc card stays green, and a
//! machine with one proves the whole path in one command.
//!
//! # On tolerances
//!
//! Every tolerance below is *derived* from a stated error model and then compared against the
//! error actually measured, which each test prints. None of them is a number that was widened
//! until the test passed. The building block is the f32 unit roundoff, [`U`] = 2^-24, and three
//! facts about the two sides of each comparison:
//!
//! 1. The CPU reference reduces sequentially in `f64`; the GPU reduces in a `f32` tree over 32
//!    lanes. For an `n`-term sum that is a relative error of roughly `(n/32 + log2 32) * U`
//!    against the exact result, applied to the sum of the *absolute* values of the terms — the
//!    only scale that bounds a sum with cancellation in it.
//! 2. SYCL's `exp`, `sin` and `cos` are specified to 4 ulp or better (OpenCL C numerical
//!    compliance, which SYCL inherits); `pow` to 16.
//! 3. `icpx` compiles this crate's kernels at its default `-ffp-model=fast`, so it may contract
//!    `a*b - c` into an FMA and drop one rounding. That is worth about 1 ulp per expression.
//!
//! Where a bound needs slack beyond that model it is given a small integer factor and the
//! factor is named. The printed measurements are what makes this honest: if a tolerance is
//! 100x the observed error, that is visible in the output rather than hidden in a constant.

mod common;

use common::{Rng, assert_close, gpu_available, max_abs_diff, synth_blocks};
use moearc_kernels::{Context, KernelError, QK_K, QuantType, RopeKind, reference};

/// f32 unit roundoff, 2^-24. Half an ulp of 1.0.
const U: f64 = 5.960_464_477_539_063e-8;

const ALL_TYPES: [QuantType; 4] = [QuantType::Q80, QuantType::Q4K, QuantType::Q5K, QuantType::Q6K];

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(f64::from(*x).abs()))
}

// =======================================================================================
// 1. Dequantisation
// =======================================================================================

#[test]
fn dequantisation_matches_the_cpu_reference_for_every_supported_type() {
    if !gpu_available() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 to run against a real GPU");
        return;
    }
    let ctx = Context::new().unwrap();
    const NBLOCKS: usize = 128;

    for ty in ALL_TYPES {
        let mut rng = Rng::new(0x00C0_FFEE ^ u64::from(ty.type_id()));
        let blocks = synth_blocks(ty, NBLOCKS, &mut rng);
        let want = reference::dequant(ty, &blocks, NBLOCKS);

        let nelem = NBLOCKS * ty.block_elems();
        let dsrc = ctx.alloc(blocks.len()).unwrap();
        let ddst = ctx.alloc_n::<f32>(nelem).unwrap();
        ctx.upload_slice(&dsrc, &blocks).unwrap();
        ctx.dequant(ty, &ddst, &dsrc, NBLOCKS).unwrap();
        let mut got = vec![0.0f32; nelem];
        ctx.download_slice(&mut got, &ddst).unwrap();

        // Both sides evaluate the same expression in the same association. The only licence
        // the compiler has is to contract `d1*q - m1` into an FMA, worth about 1 ulp; 4 ulp of
        // the largest magnitude in the tensor is a 4x margin on that.
        let tol = 4.0 * U * max_abs(&want);
        assert_close(&format!("dequant {ty:?}"), &got, &want, tol);

        // A tensor of zeros would satisfy any tolerance. Prove there is signal.
        assert!(max_abs(&want) > 0.0, "{ty:?}: the reference produced all zeros");
        assert!(want.iter().any(|v| *v < 0.0), "{ty:?}: no negative values at all");
    }
}

#[test]
fn dequantisation_refuses_a_destination_that_is_too_small() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let src = ctx.alloc(QuantType::Q4K.block_bytes() * 4).unwrap();
    let dst = ctx.alloc_n::<f32>(QK_K).unwrap(); // room for one super-block, not four
    let err = ctx.dequant(QuantType::Q4K, &dst, &src, 4).unwrap_err();
    assert!(
        matches!(err, KernelError::BufferTooSmall { .. }),
        "expected a size refusal, got {err:?}"
    );
}

// =======================================================================================
// 2. Matrix-vector
// =======================================================================================

#[test]
fn quantised_matvec_matches_the_cpu_reference_for_every_supported_type() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    // 37 rows on purpose: not a multiple of the 32-wide work-group, so the row-to-group
    // mapping is exercised rather than assumed. 1024 columns is four super-blocks per row,
    // enough that each lane's strided loop runs more than once.
    const ROWS: usize = 37;
    const COLS: usize = 1024;

    for ty in ALL_TYPES {
        let nb = COLS / ty.block_elems();
        let mut rng = Rng::new(0x5EED ^ u64::from(ty.type_id()));
        let w = synth_blocks(ty, ROWS * nb, &mut rng);
        let x = rng.vec_unit(COLS);
        let want = reference::matvec_q(ty, &w, &x, ROWS, COLS);

        let dw = ctx.alloc(w.len()).unwrap();
        let dx = ctx.alloc_n::<f32>(COLS).unwrap();
        let dout = ctx.alloc_n::<f32>(ROWS).unwrap();
        ctx.upload_slice(&dw, &w).unwrap();
        ctx.upload_slice(&dx, &x).unwrap();
        ctx.matvec_q(ty, &dout, &dw, &dx, ROWS, COLS).unwrap();
        let mut got = vec![0.0f32; ROWS];
        ctx.download_slice(&mut got, &dout).unwrap();

        // The bound is on the sum of absolute products, not on the answer: a dot product with
        // cancellation can be near zero while its terms are large, and the rounding error
        // tracks the terms.
        let absum = (0..ROWS)
            .map(|r| {
                let row = reference::dequant(ty, &w[r * nb * ty.block_bytes()..], nb);
                row.iter().zip(&x).map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs()).sum::<f64>()
            })
            .fold(0.0f64, f64::max);
        // Each lane sums COLS/32 terms sequentially, then a 5-deep tree combines 32 lanes.
        // Factor 2 of slack on the model.
        let tol = 2.0 * (COLS as f64 / 32.0 + 5.0) * U * absum;
        assert_close(&format!("matvec {ty:?}"), &got, &want, tol);
        assert!(want.iter().any(|v| v.abs() > 0.0), "{ty:?}: the reference matvec is all zeros");
    }
}

#[test]
fn f32_matvec_matches_the_cpu_reference() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    const ROWS: usize = 33;
    const COLS: usize = 700; // not a multiple of 32 either: the strided loop must handle a tail
    let mut rng = Rng::new(0x000A_11CE);
    let w = rng.vec_unit(ROWS * COLS);
    let x = rng.vec_unit(COLS);
    let want = reference::matvec_f32(&w, &x, ROWS, COLS);

    let dw = ctx.alloc_n::<f32>(w.len()).unwrap();
    let dx = ctx.alloc_n::<f32>(COLS).unwrap();
    let dout = ctx.alloc_n::<f32>(ROWS).unwrap();
    ctx.upload_slice(&dw, &w).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.matvec_f32(&dout, &dw, &dx, ROWS, COLS).unwrap();
    let mut got = vec![0.0f32; ROWS];
    ctx.download_slice(&mut got, &dout).unwrap();

    let absum = (0..ROWS)
        .map(|r| {
            (0..COLS).map(|c| (f64::from(w[r * COLS + c]) * f64::from(x[c])).abs()).sum::<f64>()
        })
        .fold(0.0f64, f64::max);
    let tol = 2.0 * (COLS as f64 / 32.0 + 5.0) * U * absum;
    assert_close("matvec f32", &got, &want, tol);
}

#[test]
fn a_row_that_is_not_a_whole_number_of_blocks_is_refused() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let w = ctx.alloc(1 << 16).unwrap();
    let x = ctx.alloc_n::<f32>(300).unwrap();
    let out = ctx.alloc_n::<f32>(4).unwrap();
    let err = ctx.matvec_q(QuantType::Q4K, &out, &w, &x, 4, 300).unwrap_err();
    assert!(matches!(err, KernelError::BadArgument(_)), "expected a shape refusal, got {err:?}");
}

// =======================================================================================
// 3. RMSNorm, SiLU/SwiGLU, softmax, RoPE
// =======================================================================================

#[test]
fn rmsnorm_matches_the_cpu_reference_with_and_without_a_weight() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    // 100 columns is deliberately not a multiple of 32: some lanes contribute nothing to the
    // reduction, which is where an off-by-one in the strided loop would show up.
    for (rows, cols) in [(7usize, 1024usize), (3, 100)] {
        let mut rng = Rng::new(0xBEEF ^ cols as u64);
        let x = rng.vec_unit(rows * cols);
        let weight = rng.vec_unit(cols);
        let eps = 1e-5f32;

        let dx = ctx.alloc_n::<f32>(x.len()).unwrap();
        let dw = ctx.alloc_n::<f32>(cols).unwrap();
        let dout = ctx.alloc_n::<f32>(x.len()).unwrap();
        ctx.upload_slice(&dx, &x).unwrap();
        ctx.upload_slice(&dw, &weight).unwrap();

        for with_weight in [false, true] {
            ctx.rmsnorm(&dout, &dx, with_weight.then_some(&dw), rows, cols, eps).unwrap();
            let mut got = vec![0.0f32; x.len()];
            ctx.download_slice(&mut got, &dout).unwrap();
            let want = reference::rmsnorm(&x, with_weight.then_some(&weight[..]), rows, cols, eps);

            // The GPU sums squares in f32 where the reference uses f64, so the relative error
            // of the sum is about (cols/32 + 5) * U. `rsqrt` halves that and adds its own few
            // ulp; 8 ulp covers `rsqrt` and the final multiply with room to spare.
            let tol = (0.5 * (cols as f64 / 32.0 + 5.0) + 8.0) * U * max_abs(&want);
            assert_close(&format!("rmsnorm {rows}x{cols} weight={with_weight}"), &got, &want, tol);
        }
    }
}

#[test]
fn silu_and_swiglu_match_the_cpu_reference() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    const N: usize = 4001;
    let mut rng = Rng::new(0x0051_1000);
    // Scaled to +-8: far enough into both tails that a wrong sign in the exponent would be
    // obvious, without saturating.
    let gate: Vec<f32> = (0..N).map(|_| rng.unit() * 8.0).collect();
    let up: Vec<f32> = (0..N).map(|_| rng.unit() * 8.0).collect();

    let dg = ctx.alloc_n::<f32>(N).unwrap();
    let du = ctx.alloc_n::<f32>(N).unwrap();
    let dout = ctx.alloc_n::<f32>(N).unwrap();
    ctx.upload_slice(&dg, &gate).unwrap();
    ctx.upload_slice(&du, &up).unwrap();

    ctx.silu(&dout, &dg, N).unwrap();
    let mut got = vec![0.0f32; N];
    ctx.download_slice(&mut got, &dout).unwrap();
    let want = reference::silu(&gate);
    // `exp` is 4 ulp or better; the divide adds one. 8 ulp of the peak is a 1.6x margin.
    assert_close("silu", &got, &want, 8.0 * U * max_abs(&want));

    ctx.swiglu(&dout, &dg, &du, N).unwrap();
    ctx.download_slice(&mut got, &dout).unwrap();
    let want = reference::swiglu(&gate, &up);
    assert_close("swiglu", &got, &want, 16.0 * U * max_abs(&want));
}

#[test]
fn softmax_matches_the_cpu_reference_and_every_row_sums_to_one() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    const ROWS: usize = 5;
    const COLS: usize = 300;
    let mut rng = Rng::new(0x0050_0F70);
    // +-40 puts `exp(x)` at 2.4e17, which overflows nothing in f32 but makes the difference
    // between subtracting the row max and not subtracting it enormous.
    let x: Vec<f32> = (0..ROWS * COLS).map(|_| rng.unit() * 40.0).collect();

    let dx = ctx.alloc_n::<f32>(x.len()).unwrap();
    let dout = ctx.alloc_n::<f32>(x.len()).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.softmax(&dout, &dx, ROWS, COLS).unwrap();
    let mut got = vec![0.0f32; x.len()];
    ctx.download_slice(&mut got, &dout).unwrap();
    let want = reference::softmax(&x, ROWS, COLS);

    // Outputs are bounded by 1, so an absolute tolerance is the natural one. Two `exp`s at
    // 4 ulp each plus a sum of 300 terms in an f32 tree: (4 + 4 + 300/32 + 5) * U.
    let tol = (4.0 + 4.0 + COLS as f64 / 32.0 + 5.0) * U;
    assert_close("softmax", &got, &want, tol);

    for r in 0..ROWS {
        let sum: f64 = got[r * COLS..(r + 1) * COLS].iter().map(|v| f64::from(*v)).sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row {r} summed to {sum}, not 1");
        assert!(got[r * COLS..(r + 1) * COLS].iter().all(|v| *v >= 0.0), "row {r} went negative");
    }
}

#[test]
fn rope_matches_the_cpu_reference_for_both_conventions() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    const TOKENS: usize = 5;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 80;
    const N_DIMS: usize = 64; // < HEAD_DIM, so 16 channels per head must ride through untouched
    const FREQ_BASE: f32 = 10000.0;

    let mut rng = Rng::new(0x0000_B0BE);
    let src = rng.vec_unit(TOKENS * HEADS * HEAD_DIM);
    // Positions are kept small on purpose. The kernel computes the angle in closed form (as
    // llama.cpp's own SYCL backend does) while the reference iterates a product (as ggml's CPU
    // path does); the two agree mathematically and diverge by roughly one ulp *of the angle*,
    // and the ulp of a large angle is large. The norm test below covers the long-position case
    // in the way that is actually invariant.
    let pos: Vec<i32> = (0..TOKENS as i32).map(|t| t * 3).collect();

    let dsrc = ctx.alloc_n::<f32>(src.len()).unwrap();
    let ddst = ctx.alloc_n::<f32>(src.len()).unwrap();
    let dpos = ctx.alloc_n::<i32>(TOKENS).unwrap();
    ctx.upload_slice(&dsrc, &src).unwrap();
    ctx.upload_slice(&dpos, &pos).unwrap();

    for kind in [RopeKind::Normal, RopeKind::Neox] {
        ctx.rope(&ddst, &dsrc, &dpos, TOKENS, HEADS, HEAD_DIM, N_DIMS, FREQ_BASE, kind).unwrap();
        let mut got = vec![0.0f32; src.len()];
        ctx.download_slice(&mut got, &ddst).unwrap();
        let want = reference::rope(&src, &pos, TOKENS, HEADS, HEAD_DIM, N_DIMS, FREQ_BASE, kind);

        // `pow` is 16 ulp, `sin`/`cos` 4; the largest angle here is 12 radians, so 16 ulp of
        // the angle is ~1.1e-5 and the resulting coordinate error is that times the input
        // magnitude (<= 1). 4x margin on the model.
        let tol = 4.0 * (16.0 * 12.0 + 4.0) * U;
        assert_close(&format!("rope {kind:?}"), &got, &want, tol);

        // The untouched tail must be copied through byte for byte, not merely close.
        for t in 0..TOKENS {
            for h in 0..HEADS {
                let base = (t * HEADS + h) * HEAD_DIM;
                for d in N_DIMS..HEAD_DIM {
                    assert_eq!(
                        got[base + d],
                        src[base + d],
                        "{kind:?}: channel {d} is beyond n_dims and must pass through unchanged"
                    );
                }
            }
        }
        // And the rotated part must actually have moved, or "matches the reference" would be
        // satisfied by a kernel that copies its input. Token 0 sits at position 0, where the
        // rotation *is* the identity, so this looks at token 1.
        let b = HEADS * HEAD_DIM;
        let (moved, _) = max_abs_diff(&got[b..b + N_DIMS], &src[b..b + N_DIMS]);
        eprintln!("rope {kind:?}: token 1 moved by up to {moved:.3e}");
        assert!(moved > 1e-3, "{kind:?}: rope left token 1 unchanged");
    }
}

#[test]
fn rope_preserves_the_length_of_every_pair_it_rotates() {
    if !gpu_available() {
        return;
    }
    // A rotation is an isometry, whatever the angle. This holds at position 4095 where the
    // element-wise comparison against the iterated reference no longer does, so it is the check
    // that covers a realistic context length.
    let ctx = Context::new().unwrap();
    const TOKENS: usize = 3;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    let mut rng = Rng::new(0x0000_1507);
    let src = rng.vec_unit(TOKENS * HEADS * HEAD_DIM);
    let pos: Vec<i32> = vec![0, 2048, 4095];

    let dsrc = ctx.alloc_n::<f32>(src.len()).unwrap();
    let ddst = ctx.alloc_n::<f32>(src.len()).unwrap();
    let dpos = ctx.alloc_n::<i32>(TOKENS).unwrap();
    ctx.upload_slice(&dsrc, &src).unwrap();
    ctx.upload_slice(&dpos, &pos).unwrap();

    for kind in [RopeKind::Normal, RopeKind::Neox] {
        ctx.rope(&ddst, &dsrc, &dpos, TOKENS, HEADS, HEAD_DIM, HEAD_DIM, 10000.0, kind).unwrap();
        let mut got = vec![0.0f32; src.len()];
        ctx.download_slice(&mut got, &ddst).unwrap();

        let mut worst = 0.0f64;
        for t in 0..TOKENS {
            for h in 0..HEADS {
                let b = (t * HEADS + h) * HEAD_DIM;
                for p in 0..HEAD_DIM / 2 {
                    let (lo, hi) = match kind {
                        RopeKind::Normal => (2 * p, 2 * p + 1),
                        RopeKind::Neox => (p, p + HEAD_DIM / 2),
                    };
                    let before = f64::from(src[b + lo]).hypot(f64::from(src[b + hi]));
                    let after = f64::from(got[b + lo]).hypot(f64::from(got[b + hi]));
                    worst = worst.max((after - before).abs());
                }
            }
        }
        eprintln!("rope {kind:?}: max change in pair length = {worst:.3e}");
        // sin^2 + cos^2 departs from 1 by a few ulp of the transcendentals; 32 ulp on a length
        // bounded by sqrt(2) is generous and still far below any real error.
        assert!(worst < 32.0 * U * 1.5, "{kind:?}: rotation changed a pair's length by {worst:e}");
    }
}

// =======================================================================================
// 4. Router
// =======================================================================================

#[test]
fn the_router_picks_the_same_experts_and_weights_as_the_cpu_reference() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    // Qwen3's actual router shape: 128 experts, 8 used.
    const TOKENS: usize = 6;
    const EXPERTS: usize = 128;
    const K: usize = 8;
    let mut rng = Rng::new(0x0000_0D7E);
    let mut logits: Vec<f32> = (0..TOKENS * EXPERTS).map(|_| rng.unit() * 6.0).collect();
    // Plant an exact tie in token 0, above every random logit, so both tied experts are
    // certain to be selected and the tie-break rule is tested rather than assumed. A kernel
    // using `>=` would pick the higher index and a residency cache would then see a different
    // expert for the same input.
    logits[3] = 10.0;
    logits[77] = 10.0;

    let dl = ctx.alloc_n::<f32>(logits.len()).unwrap();
    let didx = ctx.alloc_n::<u32>(TOKENS * K).unwrap();
    let dw = ctx.alloc_n::<f32>(TOKENS * K).unwrap();
    ctx.upload_slice(&dl, &logits).unwrap();

    for normalize in [false, true] {
        ctx.topk_router(&didx, &dw, &dl, TOKENS, EXPERTS, K, normalize).unwrap();
        let mut idx = vec![0u32; TOKENS * K];
        let mut w = vec![0.0f32; TOKENS * K];
        ctx.download_slice(&mut idx, &didx).unwrap();
        ctx.download_slice(&mut w, &dw).unwrap();

        let (want_idx, want_w) = reference::topk_router(&logits, TOKENS, EXPERTS, K, normalize);
        assert_eq!(idx, want_idx, "normalize={normalize}: the router chose different experts");
        assert_eq!(idx[0], 3, "the planted tie must resolve to the lower expert index");
        assert_eq!(idx[1], 77, "the planted tie must place the higher index second");

        // Weights are probabilities, bounded by 1, so the tolerance is absolute: two `exp`s at
        // 4 ulp, a 128-term sum, and one divide.
        let tol = (4.0 + 4.0 + EXPERTS as f64 / 32.0 + 5.0 + 1.0) * U;
        assert_close(&format!("router weights normalize={normalize}"), &w, &want_w, tol);

        if normalize {
            for t in 0..TOKENS {
                let s: f64 = w[t * K..(t + 1) * K].iter().map(|v| f64::from(*v)).sum();
                assert!((s - 1.0).abs() < 1e-5, "token {t}: normalised weights summed to {s}");
            }
        }
        // Weights must come out in descending order: selection walks the experts largest first.
        for t in 0..TOKENS {
            for r in 1..K {
                assert!(
                    w[t * K + r] <= w[t * K + r - 1] + 1e-7,
                    "token {t}: weight {r} is larger than weight {}",
                    r - 1
                );
            }
        }
    }
}

#[test]
fn a_k_larger_than_the_router_supports_is_refused() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let logits = ctx.alloc_n::<f32>(16).unwrap();
    let idx = ctx.alloc_n::<u32>(64).unwrap();
    let w = ctx.alloc_n::<f32>(64).unwrap();
    let err = ctx.topk_router(&idx, &w, &logits, 1, 16, 64, true).unwrap_err();
    assert!(matches!(err, KernelError::BadArgument(_)), "expected a refusal, got {err:?}");
}
