//! The kernels a decode step needs beyond the linear algebra: embedding lookup, the paged KV
//! cache, single-query attention, the residual arithmetic, and the f16 path.
//!
//! Split from `kernels_gpu.rs` because these are the *shape* of a forward pass rather than its
//! arithmetic, and because attention needs fixtures — pages, block tables, a cache allocator —
//! that nothing else does.
//!
//! Tolerances follow the same rule as `kernels_gpu.rs`: derived from a stated error model, then
//! printed alongside what was actually measured. See that file's header for the model.

mod common;

use common::{Rng, assert_close, gpu_available, synth_blocks};
use moearc_kernels::{Context, KernelError, KvType, QuantType, reference};

/// f32 unit roundoff, 2^-24.
const U: f64 = 5.960_464_477_539_063e-8;

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(f64::from(*x).abs()))
}

// =======================================================================================
// Residual arithmetic
// =======================================================================================

#[test]
fn add_mul_and_axpy_match_the_cpu_reference() {
    if !gpu_available() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 to run against a real GPU");
        return;
    }
    let ctx = Context::new().unwrap();
    const N: usize = 5003; // prime-ish, so no launch geometry divides it evenly
    let mut rng = Rng::new(0x0000_ADD1);
    let a = rng.vec_unit(N);
    let b = rng.vec_unit(N);

    let da = ctx.alloc_n::<f32>(N).unwrap();
    let db = ctx.alloc_n::<f32>(N).unwrap();
    let dout = ctx.alloc_n::<f32>(N).unwrap();
    ctx.upload_slice(&da, &a).unwrap();
    ctx.upload_slice(&db, &b).unwrap();
    let mut got = vec![0.0f32; N];

    // Both operands are exact f32 and each output is a single rounded operation, so these are
    // bit-exact, not merely close. Asserting equality is the stronger and correct claim.
    ctx.add(&dout, &da, &db, N).unwrap();
    ctx.download_slice(&mut got, &dout).unwrap();
    assert_eq!(got, reference::add(&a, &b), "add is a single rounding and must be exact");

    ctx.mul(&dout, &da, &db, N).unwrap();
    ctx.download_slice(&mut got, &dout).unwrap();
    assert_eq!(got, reference::mul(&a, &b), "mul is a single rounding and must be exact");

    // axpy accumulates in place, so it is seeded with `a` and must read what it wrote.
    let alpha = 0.375f32; // exactly representable; keeps the comparison about the kernel
    ctx.upload_slice(&dout, &a).unwrap();
    ctx.axpy(&dout, &db, alpha, N).unwrap();
    ctx.download_slice(&mut got, &dout).unwrap();
    let mut want = a.clone();
    reference::axpy(&mut want, &b, alpha);
    // `a + alpha*b` is two roundings on the CPU and may be one fused one on the GPU, so this is
    // the one of the three that gets a tolerance rather than equality. The gap should be one
    // ulp of the result and measures exactly that; 4 leaves room for a toolchain that fuses
    // differently without letting a real error through.
    assert_close("axpy", &got, &want, 4.0 * U * max_abs(&want));
}

// =======================================================================================
// f16
// =======================================================================================

#[test]
fn the_f16_round_trip_matches_the_cpu_reference_over_every_bit_pattern() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();

    // Every one of the 65_536 f16 bit patterns, widened to f32 by the reference, then narrowed
    // again on the GPU. Skipping NaNs, whose payload is not required to survive a round trip.
    let inputs: Vec<f32> =
        (0..=u16::MAX).map(reference::f16_to_f32).filter(|v| !v.is_nan()).collect();
    let n = inputs.len();

    let dsrc = ctx.alloc_n::<f32>(n).unwrap();
    let ddst = ctx.alloc(n * 2).unwrap();
    ctx.upload_slice(&dsrc, &inputs).unwrap();
    ctx.quantize_f16(&ddst, &dsrc, n).unwrap();
    let mut bits = vec![0u16; n];
    ctx.download_slice(&mut bits, &ddst).unwrap();

    // Every value here came *from* an f16, so narrowing it must be exact and idempotent.
    let want: Vec<u16> = inputs.iter().map(|v| reference::f32_to_f16(*v)).collect();
    assert_eq!(bits, want, "GPU and CPU f32->f16 disagree on an exactly representable value");
    let back: Vec<f32> = bits.iter().map(|b| reference::f16_to_f32(*b)).collect();
    assert_eq!(back, inputs, "f16 -> f32 -> f16 -> f32 was not the identity");
    eprintln!("f16 round trip exact over {n} of 65536 bit patterns (NaNs excluded)");
}

#[test]
fn f16_narrowing_matches_the_cpu_reference_on_values_that_do_not_fit() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut rng = Rng::new(0x0000_F167);

    // The cases where the rounding rule actually shows: exact ties, subnormals, values just
    // past the largest finite half, and ordinary magnitudes that need rounding.
    let mut inputs: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        65504.0,
        -65504.0,
        65520.0, // the tie that rounds to infinity
        65519.996,
        1e-8,
        -1e-8,
        6e-8,         // half of the smallest subnormal: ties to even -> 0
        5.9604645e-8, // exactly half the smallest subnormal
        1.7881393e-7, // 3x that: rounds up
        f32::MIN_POSITIVE,
        1e30,
        -1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    inputs.extend((0..8192).map(|_| rng.unit() * 4.0));
    inputs.extend((0..2048).map(|_| rng.unit() * 1e-5));
    let n = inputs.len();

    let dsrc = ctx.alloc_n::<f32>(n).unwrap();
    let ddst = ctx.alloc(n * 2).unwrap();
    ctx.upload_slice(&dsrc, &inputs).unwrap();
    ctx.quantize_f16(&ddst, &dsrc, n).unwrap();
    let mut bits = vec![0u16; n];
    ctx.download_slice(&mut bits, &ddst).unwrap();

    let want: Vec<u16> = inputs.iter().map(|v| reference::f32_to_f16(*v)).collect();
    for (i, (g, w)) in bits.iter().zip(&want).enumerate() {
        assert_eq!(
            g, w,
            "f32->f16 disagreed on input {} ({:e}): gpu {:#06x}, cpu {:#06x}",
            i, inputs[i], g, w
        );
    }
    eprintln!("f32 -> f16 bit-identical to the CPU reference over {n} values including the ties");
}

#[test]
fn an_f16_tensor_expands_through_the_same_dequant_entry_point() {
    if !gpu_available() {
        return;
    }
    // f16 is a format in the table, not a special case: `dequant` widens one and `matvec_q`
    // consumes one. This is what makes the f16 path free rather than a parallel code path.
    let ctx = Context::new().unwrap();
    const N: usize = 4096;
    let mut rng = Rng::new(0x0000_F16E);
    let values = rng.vec_unit(N);
    let halves: Vec<u16> = values.iter().map(|v| reference::f32_to_f16(*v)).collect();
    let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();

    let dsrc = ctx.alloc(bytes.len()).unwrap();
    let ddst = ctx.alloc_n::<f32>(N).unwrap();
    ctx.upload_slice(&dsrc, &bytes).unwrap();
    ctx.dequant(QuantType::F16, &ddst, &dsrc, N).unwrap();
    let mut got = vec![0.0f32; N];
    ctx.download_slice(&mut got, &ddst).unwrap();

    let want = reference::dequant(QuantType::F16, &bytes, N);
    assert_eq!(got, want, "f16 widening must be exact — every half is an exact float");

    // And the same bytes as weights in a matvec.
    let x = rng.vec_unit(N);
    let dx = ctx.alloc_n::<f32>(N).unwrap();
    let dout = ctx.alloc_n::<f32>(1).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.matvec_q(QuantType::F16, &dout, &dsrc, &dx, 1, N).unwrap();
    let mut out = vec![0.0f32; 1];
    ctx.download_slice(&mut out, &dout).unwrap();
    let wantv = reference::matvec_q(QuantType::F16, &bytes, &x, 1, N);
    let absum: f64 = want.iter().zip(&x).map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs()).sum();
    assert_close("matvec f16", &out, &wantv, 2.0 * (N as f64 / 32.0 + 5.0) * U * absum);
}

// =======================================================================================
// Embedding lookup
// =======================================================================================

#[test]
fn the_embedding_lookup_gathers_the_right_rows_from_a_quantised_table() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    // Q4_K because that is what OLMoE's `token_embd.weight` is; Q8_0 because that is what the
    // Qwen3.6 file's is. Assuming either one would silently misread the other.
    for ty in [QuantType::Q4K, QuantType::Q80, QuantType::F16] {
        const VOCAB: usize = 96;
        const N_EMBD: usize = 512;
        let mut rng = Rng::new(0x000E_B0D0 ^ u64::from(ty.type_id()));
        let blocks_per_row = N_EMBD / ty.block_elems();
        let table = synth_blocks(ty, VOCAB * blocks_per_row, &mut rng);

        // Out of order, repeated, and including both endpoints of the vocabulary.
        let ids: Vec<u32> = vec![0, 95, 40, 40, 1, 77];
        let want = reference::embed_rows(ty, &table, &ids, N_EMBD);

        let dtab = ctx.alloc(table.len()).unwrap();
        let dids = ctx.alloc_n::<u32>(ids.len()).unwrap();
        let dout = ctx.alloc_n::<f32>(ids.len() * N_EMBD).unwrap();
        ctx.upload_slice(&dtab, &table).unwrap();
        ctx.upload_slice(&dids, &ids).unwrap();
        ctx.embed_rows(ty, &dout, &dtab, &dids, ids.len(), N_EMBD).unwrap();
        let mut got = vec![0.0f32; ids.len() * N_EMBD];
        ctx.download_slice(&mut got, &dout).unwrap();

        // Same expression, same association as `dequant`; only FMA contraction can differ.
        assert_close(&format!("embed_rows {ty:?}"), &got, &want, 4.0 * U * max_abs(&want));

        // A gather that returned the same row for every id would pass a loose comparison.
        let row = |i: usize| &got[i * N_EMBD..(i + 1) * N_EMBD];
        assert_eq!(row(2), row(3), "the repeated id must gather the same row twice");
        assert_ne!(row(0), row(1), "different ids must gather different rows");
    }
}

#[test]
fn an_embedding_row_that_is_not_a_whole_number_of_blocks_is_refused() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let tab = ctx.alloc(1 << 16).unwrap();
    let ids = ctx.alloc_n::<u32>(1).unwrap();
    let out = ctx.alloc_n::<f32>(300).unwrap();
    let err = ctx.embed_rows(QuantType::Q4K, &out, &tab, &ids, 1, 300).unwrap_err();
    assert!(matches!(err, KernelError::BadArgument(_)), "expected a shape refusal, got {err:?}");
}

// =======================================================================================
// Masked softmax
// =======================================================================================

#[test]
fn the_masked_softmax_matches_the_cpu_reference_and_gives_masked_keys_zero_weight() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    const N_Q: usize = 7;
    const N_KV: usize = 61;
    let scale = 1.0f32 / (128.0f32).sqrt(); // the 1/sqrt(head_dim) attention actually uses
    let mut rng = Rng::new(0x0000_5A5C);
    let x: Vec<f32> = (0..N_Q * N_KV).map(|_| rng.unit() * 30.0).collect();
    let mask = reference::causal_mask(N_Q, N_KV);

    let dx = ctx.alloc_n::<f32>(x.len()).unwrap();
    let dm = ctx.alloc_n::<f32>(mask.len()).unwrap();
    let dout = ctx.alloc_n::<f32>(x.len()).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.upload_slice(&dm, &mask).unwrap();
    ctx.softmax_ext(&dout, &dx, Some(&dm), N_Q, N_KV, scale).unwrap();
    let mut got = vec![0.0f32; x.len()];
    ctx.download_slice(&mut got, &dout).unwrap();

    let want = reference::softmax_ext(&x, Some(&mask), N_Q, N_KV, scale);
    let tol = (4.0 + 4.0 + N_KV as f64 / 32.0 + 5.0) * U;
    assert_close("softmax_ext causal", &got, &want, tol);

    // Structural checks the tolerance cannot make: masked positions must be exactly zero, and
    // each row must still be a distribution over what remains.
    let first = N_KV - N_Q;
    for i in 0..N_Q {
        for j in 0..N_KV {
            if j > first + i {
                assert_eq!(got[i * N_KV + j], 0.0, "row {i} attended to future key {j}");
            }
        }
        let sum: f64 = got[i * N_KV..(i + 1) * N_KV].iter().map(|v| f64::from(*v)).sum();
        assert!((sum - 1.0).abs() < 1e-5, "masked row {i} summed to {sum}");
    }
}

#[test]
fn a_masked_position_cannot_influence_the_row_even_when_it_dominates_it() {
    if !gpu_available() {
        return;
    }
    // 🔴 This test exists because a mutation survived without it.
    //
    // The max subtraction has to run on the *masked* values. Taking the maximum before applying
    // the mask is mathematically invisible on ordinary data — softmax is invariant to whatever
    // constant it subtracts — so an unmasked max pass passes every gentle test. It stops being
    // invisible when a masked entry is far larger than every visible one: the subtracted
    // constant is then enormous, `exp(visible - max)` underflows to zero across the whole row,
    // and a valid distribution comes back as zeros.
    //
    // A causal mask makes exactly that shape ordinary rather than exotic — a future token's
    // logit is under no obligation to resemble the past's.
    let ctx = Context::new().unwrap();
    const COLS: usize = 32;
    const VISIBLE: usize = 16;
    let mut rng = Rng::new(0x0000_D0A1);

    let mut x = vec![0.0f32; COLS];
    let mut mask = vec![0.0f32; COLS];
    for (i, v) in x.iter_mut().enumerate() {
        // 200 is far past where `exp` underflows in f32 (about -88), so an unmasked max makes
        // every visible weight exactly zero rather than merely small.
        *v = if i < VISIBLE { rng.unit() } else { 200.0 };
    }
    for m in mask.iter_mut().skip(VISIBLE) {
        *m = f32::NEG_INFINITY;
    }

    let dx = ctx.alloc_n::<f32>(COLS).unwrap();
    let dm = ctx.alloc_n::<f32>(COLS).unwrap();
    let dout = ctx.alloc_n::<f32>(COLS).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.upload_slice(&dm, &mask).unwrap();
    ctx.softmax_ext(&dout, &dx, Some(&dm), 1, COLS, 1.0).unwrap();
    let mut got = vec![0.0f32; COLS];
    ctx.download_slice(&mut got, &dout).unwrap();

    let want = reference::softmax_ext(&x, Some(&mask), 1, COLS, 1.0);
    assert_close("softmax_ext dominated by a masked entry", &got, &want, 16.0 * U);

    let sum: f64 = got.iter().map(|v| f64::from(*v)).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "the row summed to {sum}, not 1 — a masked entry set the max and underflowed the rest"
    );
    for (i, v) in got.iter().enumerate().skip(VISIBLE) {
        assert_eq!(*v, 0.0, "masked position {i} got weight {v}");
    }
    for (i, v) in got.iter().enumerate().take(VISIBLE) {
        assert!(*v > 0.0, "visible position {i} was flattened to zero");
    }
}

#[test]
fn a_fully_masked_row_comes_back_as_zeros_rather_than_nan() {
    if !gpu_available() {
        return;
    }
    // A padded batch slot produces this. One NaN row would propagate through every later
    // matmul and destroy the rows that were fine.
    let ctx = Context::new().unwrap();
    const N: usize = 40;
    let x = vec![1.0f32; N];
    let mask = vec![f32::NEG_INFINITY; N];
    let dx = ctx.alloc_n::<f32>(N).unwrap();
    let dm = ctx.alloc_n::<f32>(N).unwrap();
    let dout = ctx.alloc_n::<f32>(N).unwrap();
    ctx.upload_slice(&dx, &x).unwrap();
    ctx.upload_slice(&dm, &mask).unwrap();
    ctx.softmax_ext(&dout, &dx, Some(&dm), 1, N, 1.0).unwrap();
    let mut got = vec![0.0f32; N];
    ctx.download_slice(&mut got, &dout).unwrap();
    assert!(got.iter().all(|v| *v == 0.0), "a fully masked row produced {got:?}");
}

// =======================================================================================
// Paged KV cache and attention
// =======================================================================================

/// A synthetic sequence laid out across scattered pages, mirroring what the allocator produces.
struct KvFixture {
    k_host: Vec<f32>,
    v_host: Vec<f32>,
    block_table: Vec<u32>,
}

/// Fill a page pool one token at a time through the GPU append kernel, keeping a host mirror
/// built by the CPU reference. Returns the mirror and the block table.
#[allow(clippy::too_many_arguments)]
fn fill_cache(
    ctx: &Context,
    dk: &moearc_kernels::DeviceBuffer<'_>,
    dv: &moearc_kernels::DeviceBuffer<'_>,
    block_table: &[u32],
    n_kv: usize,
    n_kv_heads: usize,
    head_dim: usize,
    page_tokens: usize,
    n_pages_total: usize,
    kv: KvType,
    rng: &mut Rng,
) -> KvFixture {
    let pool = n_pages_total * page_tokens * n_kv_heads * head_dim;
    let mut k_host = vec![0.0f32; pool];
    let mut v_host = vec![0.0f32; pool];
    let row = n_kv_heads * head_dim;

    let dk_tok = ctx.alloc_n::<f32>(row).unwrap();
    let dv_tok = ctx.alloc_n::<f32>(row).unwrap();

    for j in 0..n_kv {
        let k: Vec<f32> = (0..row).map(|_| rng.unit()).collect();
        let v: Vec<f32> = (0..row).map(|_| rng.unit()).collect();
        let page = block_table[j / page_tokens];
        let slot = (j % page_tokens) as u32;

        ctx.upload_slice(&dk_tok, &k).unwrap();
        ctx.upload_slice(&dv_tok, &v).unwrap();
        ctx.kv_append(dk, dv, &dk_tok, &dv_tok, page, slot, n_kv_heads, head_dim, page_tokens, kv)
            .unwrap();
        reference::kv_append(
            &mut k_host,
            &mut v_host,
            &k,
            &v,
            page,
            slot,
            n_kv_heads,
            head_dim,
            page_tokens,
            kv,
        );
    }
    KvFixture { k_host, v_host, block_table: block_table.to_vec() }
}

#[test]
fn kv_append_writes_exactly_the_slot_it_was_given_and_nothing_else() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    const HEADS: usize = 3;
    const HEAD_DIM: usize = 16;
    const PAGE_TOKENS: usize = 4;
    const PAGES: usize = 5;
    let row = HEADS * HEAD_DIM;
    let pool = PAGES * PAGE_TOKENS * row;

    let mut rng = Rng::new(0x0000_C0DE);
    let dk = ctx.alloc_n::<f32>(pool).unwrap();
    let dv = ctx.alloc_n::<f32>(pool).unwrap();
    // Poison the pool so an untouched slot is recognisable rather than a plausible zero.
    let poison = vec![-7.5f32; pool];
    ctx.upload_slice(&dk, &poison).unwrap();
    ctx.upload_slice(&dv, &poison).unwrap();

    let k: Vec<f32> = (0..row).map(|_| rng.unit()).collect();
    let v: Vec<f32> = (0..row).map(|_| rng.unit()).collect();
    let dk_tok = ctx.alloc_n::<f32>(row).unwrap();
    let dv_tok = ctx.alloc_n::<f32>(row).unwrap();
    ctx.upload_slice(&dk_tok, &k).unwrap();
    ctx.upload_slice(&dv_tok, &v).unwrap();

    let (page, slot) = (3u32, 2u32);
    ctx.kv_append(
        &dk,
        &dv,
        &dk_tok,
        &dv_tok,
        page,
        slot,
        HEADS,
        HEAD_DIM,
        PAGE_TOKENS,
        KvType::F32,
    )
    .unwrap();

    let mut k_got = vec![0.0f32; pool];
    ctx.download_slice(&mut k_got, &dk).unwrap();
    let mut want = poison.clone();
    let mut want_v = poison.clone();
    reference::kv_append(
        &mut want,
        &mut want_v,
        &k,
        &v,
        page,
        slot,
        HEADS,
        HEAD_DIM,
        PAGE_TOKENS,
        KvType::F32,
    );
    assert_eq!(k_got, want, "kv_append wrote outside its slot, or into the wrong one");

    // Spelled out rather than trusted to the reference: the slot's own bytes changed and its
    // immediate neighbours did not.
    let base = reference::kv_index(page, slot, 0, 0, HEADS, HEAD_DIM, PAGE_TOKENS);
    assert_eq!(&k_got[base..base + row], &k[..], "the target slot does not hold the key");
    assert_eq!(k_got[base - 1], -7.5, "the slot before was overwritten");
    assert_eq!(k_got[base + row], -7.5, "the slot after was overwritten");
}

#[test]
fn paged_attention_matches_the_cpu_reference_across_scattered_pages() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    // OLMoE's real attention geometry: 16 heads, no GQA, head_dim 128.
    const HEADS: usize = 16;
    const KV_HEADS: usize = 16;
    const HEAD_DIM: usize = 128;
    const PAGE_TOKENS: usize = 8;
    const PAGES: usize = 16;
    // 37 keys is not a multiple of the page size, so the last page is partly filled — the
    // ordinary case, and the one where an off-by-one in the page walk shows.
    const N_KV: usize = 37;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    for kv in [KvType::F32, KvType::F16] {
        let mut rng = Rng::new(0x0000_A77E ^ kv.type_id() as u64);
        // Deliberately not 0,1,2,...: the pages a sequence gets are whatever the allocator has
        // free, and a kernel that assumed contiguity would pass on a fresh pool and corrupt any
        // sequence that outlived a neighbour.
        let block_table: Vec<u32> = vec![11, 2, 15, 7, 0];
        let pool = PAGES * PAGE_TOKENS * KV_HEADS * HEAD_DIM;
        let dk = ctx.alloc(pool * kv.elem_bytes()).unwrap();
        let dv = ctx.alloc(pool * kv.elem_bytes()).unwrap();

        let fx = fill_cache(
            &ctx,
            &dk,
            &dv,
            &block_table,
            N_KV,
            KV_HEADS,
            HEAD_DIM,
            PAGE_TOKENS,
            PAGES,
            kv,
            &mut rng,
        );

        let q = rng.vec_unit(HEADS * HEAD_DIM);
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
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            N_KV,
            PAGE_TOKENS,
            scale,
            kv,
        )
        .unwrap();
        let mut got = vec![0.0f32; q.len()];
        ctx.download_slice(&mut got, &dout).unwrap();

        let want = reference::attn_decode(
            &q,
            &fx.k_host,
            &fx.v_host,
            &fx.block_table,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            N_KV,
            PAGE_TOKENS,
            scale,
        );

        // Three f32 stages the reference does in f64: a 128-term dot product per key, the
        // softmax denominator over 37 keys, and the weighted sum of 37 values. The output is a
        // convex combination of values bounded by 1, so an absolute bound on that scale is the
        // natural one. `exp` contributes 4 ulp twice.
        let dot = (HEAD_DIM as f64 / 32.0 + 5.0) * U;
        let tol = 8.0 * (dot + (N_KV as f64) * U + 8.0 * U);
        assert_close(&format!("attn_decode {kv:?}"), &got, &want, tol);

        // Attention output is a weighted average of cached values, so it cannot leave their
        // range. A kernel that mixed up its denominator would.
        let vmax = max_abs(&fx.v_host);
        assert!(max_abs(&got) <= vmax + 1e-4, "attention output escaped the range of V");
    }
}

#[test]
fn paged_attention_handles_grouped_query_heads() {
    if !gpu_available() {
        return;
    }
    // 🔴 Synthetic only. OLMoE sets n_kv_heads == n_heads, so no real model in this project
    // exercises the grouped path; this checks the index arithmetic against the CPU reference
    // and nothing more.
    let ctx = Context::new().unwrap();
    const HEADS: usize = 8;
    const KV_HEADS: usize = 2; // group of 4
    const HEAD_DIM: usize = 64;
    const PAGE_TOKENS: usize = 4;
    const PAGES: usize = 6;
    const N_KV: usize = 10;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let mut rng = Rng::new(0x0000_6A11);

    let block_table: Vec<u32> = vec![5, 1, 3];
    let pool = PAGES * PAGE_TOKENS * KV_HEADS * HEAD_DIM;
    let dk = ctx.alloc_n::<f32>(pool).unwrap();
    let dv = ctx.alloc_n::<f32>(pool).unwrap();
    let fx = fill_cache(
        &ctx,
        &dk,
        &dv,
        &block_table,
        N_KV,
        KV_HEADS,
        HEAD_DIM,
        PAGE_TOKENS,
        PAGES,
        KvType::F32,
        &mut rng,
    );

    let q = rng.vec_unit(HEADS * HEAD_DIM);
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
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        N_KV,
        PAGE_TOKENS,
        scale,
        KvType::F32,
    )
    .unwrap();
    let mut got = vec![0.0f32; q.len()];
    ctx.download_slice(&mut got, &dout).unwrap();

    let want = reference::attn_decode(
        &q,
        &fx.k_host,
        &fx.v_host,
        &fx.block_table,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        N_KV,
        PAGE_TOKENS,
        scale,
    );
    let tol = 8.0 * ((HEAD_DIM as f64 / 32.0 + 5.0) * U + (N_KV as f64) * U + 8.0 * U);
    assert_close("attn_decode GQA 8/2", &got, &want, tol);

    // Heads 0..3 share KV head 0 and heads 4..7 share KV head 1. With different queries they
    // must still differ, or the grouping has collapsed the heads together.
    let head = |i: usize| &got[i * HEAD_DIM..(i + 1) * HEAD_DIM];
    assert_ne!(head(0), head(1), "two query heads in the same group produced identical output");
}

#[test]
fn attention_with_a_single_cached_key_returns_that_value_exactly() {
    if !gpu_available() {
        return;
    }
    // softmax over one score is 1 whatever the score, so the output must be V exactly. This is
    // the one attention case with a closed-form answer, and it pins the accumulator: a kernel
    // that forgot to divide by the denominator, or that seeded the running max wrongly, fails
    // here without any tolerance argument.
    let ctx = Context::new().unwrap();
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 32;
    const PAGE_TOKENS: usize = 4;
    let mut rng = Rng::new(0x0000_01E1);

    let row = HEADS * HEAD_DIM;
    let pool = 2 * PAGE_TOKENS * row;
    let dk = ctx.alloc_n::<f32>(pool).unwrap();
    let dv = ctx.alloc_n::<f32>(pool).unwrap();
    let block_table: Vec<u32> = vec![1];
    let fx = fill_cache(
        &ctx,
        &dk,
        &dv,
        &block_table,
        1,
        HEADS,
        HEAD_DIM,
        PAGE_TOKENS,
        2,
        KvType::F32,
        &mut rng,
    );

    let q = rng.vec_unit(row);
    let dq = ctx.alloc_n::<f32>(row).unwrap();
    let dbt = ctx.alloc_n::<u32>(1).unwrap();
    let dout = ctx.alloc_n::<f32>(row).unwrap();
    ctx.upload_slice(&dq, &q).unwrap();
    ctx.upload_slice(&dbt, &block_table).unwrap();
    ctx.attn_decode(
        &dout,
        &dq,
        &dk,
        &dv,
        &dbt,
        HEADS,
        HEADS,
        HEAD_DIM,
        1,
        PAGE_TOKENS,
        0.125,
        KvType::F32,
    )
    .unwrap();
    let mut got = vec![0.0f32; row];
    ctx.download_slice(&mut got, &dout).unwrap();

    for h in 0..HEADS {
        for d in 0..HEAD_DIM {
            let want = fx.v_host[reference::kv_index(1, 0, h, d, HEADS, HEAD_DIM, PAGE_TOKENS)];
            let got_v = got[h * HEAD_DIM + d];
            assert!(
                (got_v - want).abs() <= 1e-6 * want.abs().max(1.0),
                "head {h} channel {d}: single-key attention gave {got_v}, not V = {want}"
            );
        }
    }
    eprintln!("single-key attention reproduced V exactly across {HEADS} heads");
}

#[test]
fn attention_refuses_a_head_count_that_is_not_a_multiple_of_the_kv_head_count() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let b = ctx.alloc(1 << 16).unwrap();
    let bt = ctx.alloc_n::<u32>(4).unwrap();
    let err = ctx.attn_decode(&b, &b, &b, &b, &bt, 6, 4, 8, 4, 4, 1.0, KvType::F32).unwrap_err();
    assert!(matches!(err, KernelError::BadArgument(_)), "expected a refusal, got {err:?}");
}
