//! CPU reference implementations of every kernel in this crate.
//!
//! These exist so that no GPU kernel is ever asserted against itself. Each function here is a
//! second, independent expression of the same operation — written in the shape llama.cpp
//! writes it (a sequential walk down a row) rather than the shape the kernel needs (a closure
//! over one output index) — and the GPU tests assert the two agree. Where the two forms are
//! *deliberately* different, the difference is named in a comment on the function.
//!
//! They are also the executable specification. If a kernel's semantics are ever in doubt, the
//! answer is here, in Rust, readable without a SYCL toolchain.
//!
//! Accuracy convention: reductions accumulate in `f64` where ggml accumulates in `ggml_float`
//! (which is `double`), and in `f32` where ggml uses `float`. The GPU has no usable `f64`, so
//! this is the higher-precision side of every comparison, not a mirror of it.

/// The block-quantised weight formats this crate can expand.
///
/// The discriminants are the GGUF type ids from `gguf-py/gguf/constants.py`,
/// `class GGMLQuantizationType(IntEnum)` — the same numbers the `moearc-model` crate's `quant`
/// table carries, and the same numbers crossing the C ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantType {
    /// The legacy 8-bit format: 32 elements, one f16 delta, no minimum. A Q4_K_M quantisation
    /// of Qwen3 holds 251 of these against 80 Q4_K, 37 Q5_K and 4 Q6_K, so it is not optional.
    Q80 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
}

/// Elements per K-quant super-block. `QK_K` in `ggml/src/ggml-common.h`.
pub const QK_K: usize = 256;

/// Elements per Q8_0 block. `QK8_0` in `ggml/src/ggml-common.h`.
pub const QK8_0: usize = 32;

impl QuantType {
    /// Bytes one super-block occupies on disk.
    ///
    /// From the `static_assert`s beside each struct in `ggml/src/ggml-common.h`:
    /// Q8_0 is `2 + 32`, Q4_K is `2 + 2 + 12 + 128`, Q5_K is `2 + 2 + 12 + 32 + 128`, Q6_K is
    /// `128 + 64 + 16 + 2`.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q80 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Elements one block expands to. Not a constant across formats — the K-quants pack 256,
    /// Q8_0 packs 32 — so anything sizing a buffer must ask rather than assume `QK_K`.
    pub const fn block_elems(self) -> usize {
        match self {
            Self::Q80 => QK8_0,
            Self::Q4K | Self::Q5K | Self::Q6K => QK_K,
        }
    }

    /// The GGUF type id.
    pub const fn type_id(self) -> u32 {
        self as u32
    }

    /// Resolve a GGUF type id, or `None` if this crate cannot expand it.
    pub const fn from_type_id(id: u32) -> Option<Self> {
        match id {
            8 => Some(Self::Q80),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }
}

/// IEEE-754 binary16 to binary32. Lossless: every `f16` is exactly representable as an `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    // Rust has no `f16` in stable std, so the conversion is written out in integer arithmetic —
    // the same way `kernels.cpp` does it, so the two sides cannot disagree about a subnormal.
    let sign = u32::from(h & 0x8000) << 16;
    let mut exp = u32::from((h >> 10) & 0x1F);
    let mut mant = u32::from(h & 0x03FF);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 113; // 127 - 15 + 1: the exponent a half subnormal renormalises from
            while mant & 0x400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x3FF;
            sign | (exp << 23) | (mant << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        // 127 - 15 = 112, written folded: `exp - 15 + 127` underflows a u32 for
        // exponents below 15, which C's unsigned wraparound hides and Rust does not.
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn ld_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// `get_scale_min_k4` from `ggml/src/ggml-quants.c`, returning `(scale, min)`.
fn scale_min_k4(q: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Expand `nblocks` blocks into `nblocks * ty.block_elems()` floats.
///
/// Panics if `src` is too short — a truncated block would otherwise produce plausible garbage,
/// which is the failure mode this whole crate exists to avoid.
pub fn dequant(ty: QuantType, src: &[u8], nblocks: usize) -> Vec<f32> {
    let bb = ty.block_bytes();
    assert!(
        src.len() >= nblocks * bb,
        "need {} bytes for {nblocks} blocks, got {}",
        nblocks * bb,
        src.len()
    );
    let be = ty.block_elems();
    let mut out = vec![0.0f32; nblocks * be];
    for b in 0..nblocks {
        let blk = &src[b * bb..(b + 1) * bb];
        let y = &mut out[b * be..(b + 1) * be];
        match ty {
            QuantType::Q80 => dequant_block_q8_0(blk, y),
            QuantType::Q4K => dequant_block_q4_k(blk, y),
            QuantType::Q5K => dequant_block_q5_k(blk, y),
            QuantType::Q6K => dequant_block_q6_k(blk, y),
        }
    }
    out
}

/// `dequantize_row_q8_0` from `ggml/src/ggml-quants.c`, one block: `y[j] = qs[j] * d`.
fn dequant_block_q8_0(blk: &[u8], y: &mut [f32]) {
    let d = ld_f16(blk, 0);
    for (j, out) in y.iter_mut().enumerate() {
        *out = f32::from(blk[2 + j] as i8) * d;
    }
}

/// `dequantize_row_q4_K` from `ggml/src/ggml-quants.c`, one block, transcribed as written.
fn dequant_block_q4_k(blk: &[u8], y: &mut [f32]) {
    let d = ld_f16(blk, 0);
    let min = ld_f16(blk, 2);
    let scales = &blk[4..16];
    let qs = &blk[16..144];

    let mut w = 0usize; // write cursor, standing in for the C `*y++`
    let mut is = 0usize;
    for j in (0..QK_K).step_by(64) {
        let q = &qs[j / 2..]; // the C code advances `q += 32` each pass; j/2 == 32 * (j/64)
        let (sc, m) = scale_min_k4(scales, is);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(scales, is + 1);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        for b in &q[..32] {
            y[w] = d1 * f32::from(b & 0xF) - m1;
            w += 1;
        }
        for b in &q[..32] {
            y[w] = d2 * f32::from(b >> 4) - m2;
            w += 1;
        }
        is += 2;
    }
}

/// `dequantize_row_q5_K` from `ggml/src/ggml-quants.c`, one block, transcribed as written.
fn dequant_block_q5_k(blk: &[u8], y: &mut [f32]) {
    let d = ld_f16(blk, 0);
    let min = ld_f16(blk, 2);
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let qs = &blk[48..176];

    let mut w = 0usize;
    let mut is = 0usize;
    let (mut u1, mut u2) = (1u8, 2u8);
    for j in (0..QK_K).step_by(64) {
        let ql = &qs[j / 2..];
        let (sc, m) = scale_min_k4(scales, is);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(scales, is + 1);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16u16 } else { 0 };
            y[w] = d1 * f32::from(u16::from(ql[l] & 0xF) + hi) - m1;
            w += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16u16 } else { 0 };
            y[w] = d2 * f32::from(u16::from(ql[l] >> 4) + hi) - m2;
            w += 1;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// `dequantize_row_q6_K` from `ggml/src/ggml-quants.c`, one block, transcribed as written.
fn dequant_block_q6_k(blk: &[u8], y: &mut [f32]) {
    let d = ld_f16(blk, 208);
    for n in 0..2 {
        let ql = &blk[n * 64..];
        let qh = &blk[128 + n * 32..];
        let sc = &blk[192 + n * 8..];
        let y = &mut y[n * 128..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * f32::from(sc[is] as i8) * q1 as f32;
            y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
            y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
            y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
        }
    }
}

/// `out[row] = sum_col W[row][col] * x[col]` against quantised weights.
///
/// Dequantises the whole matrix first and accumulates in `f64`. That is the *opposite* of what
/// the kernel does — it dequantises lazily and reduces in a `f32` tree — which is why this is
/// a useful reference and why the two are compared with a tolerance rather than for equality.
pub fn matvec_q(ty: QuantType, w: &[u8], x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let be = ty.block_elems();
    assert_eq!(n_cols % be, 0, "a {ty:?} row must be a whole number of {be}-element blocks");
    assert_eq!(x.len(), n_cols);
    let nb = n_cols / be;
    let mut out = vec![0.0f32; n_rows];
    for (r, o) in out.iter_mut().enumerate() {
        let row = dequant(ty, &w[r * nb * ty.block_bytes()..], nb);
        let mut acc = 0.0f64;
        for (wv, xv) in row.iter().zip(x) {
            acc += f64::from(*wv) * f64::from(*xv);
        }
        *o = acc as f32;
    }
    out
}

/// `out[row] = sum_col W[row][col] * x[col]` against f32 weights, accumulated in `f64`.
pub fn matvec_f32(w: &[f32], x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), n_cols);
    assert_eq!(w.len(), n_rows * n_cols);
    (0..n_rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for c in 0..n_cols {
                acc += f64::from(w[r * n_cols + c]) * f64::from(x[c]);
            }
            acc as f32
        })
        .collect()
}

/// RMSNorm over the last axis, `f64` sum of squares as in `ggml_compute_forward_rms_norm_f32`.
pub fn rmsnorm(
    x: &[f32],
    weight: Option<&[f32]>,
    n_rows: usize,
    n_cols: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), n_rows * n_cols);
    let mut out = vec![0.0f32; x.len()];
    for r in 0..n_rows {
        let row = &x[r * n_cols..(r + 1) * n_cols];
        let sum: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let mean = (sum / n_cols as f64) as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for (c, v) in row.iter().enumerate() {
            let s = v * scale;
            out[r * n_cols + c] = match weight {
                Some(w) => s * w[c],
                None => s,
            };
        }
    }
    out
}

/// SiLU: `x / (1 + exp(-x))`.
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

/// SwiGLU: `silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter().zip(up).map(|(g, u)| (g / (1.0 + (-g).exp())) * u).collect()
}

/// Row-wise softmax, max-subtracted, summed in `f64`.
pub fn softmax(x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), n_rows * n_cols);
    let mut out = vec![0.0f32; x.len()];
    for r in 0..n_rows {
        let row = &x[r * n_cols..(r + 1) * n_cols];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|v| f64::from((v - mx).exp())).sum();
        for (c, v) in row.iter().enumerate() {
            out[r * n_cols + c] = (f64::from((v - mx).exp()) / sum) as f32;
        }
    }
    out
}

/// Which pairs of channels RoPE rotates together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// `GGML_ROPE_TYPE_NORMAL`: adjacent channels `(2i, 2i+1)`.
    Normal,
    /// `GGML_ROPE_TYPE_NEOX`: the two halves of the head, `(i, i + n_dims/2)`.
    Neox,
}

/// RoPE over `[n_tokens][n_heads][head_dim]`, head_dim fastest.
///
/// 🔴 Deliberately *not* written the way the kernel is. This builds the angle table by repeated
/// multiplication — `theta *= theta_scale` — which is what `ggml_rope_cache_init` does; the
/// kernel computes each angle in closed form so its work-items stay independent. The two are
/// mathematically equal and differ in the last bits, which is exactly what the test's tolerance
/// is there to bound.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    src: &[f32],
    pos: &[i32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_dims: usize,
    freq_base: f32,
    kind: RopeKind,
) -> Vec<f32> {
    assert_eq!(src.len(), n_tokens * n_heads * head_dim);
    assert_eq!(pos.len(), n_tokens);
    assert_eq!(n_dims % 2, 0);
    assert!(n_dims <= head_dim);

    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut out = src.to_vec();
    for (t, p) in pos.iter().enumerate() {
        // ggml builds the cos/sin cache once per token and reuses it across heads.
        let mut cache = Vec::with_capacity(n_dims);
        let mut theta = *p as f32;
        for _ in (0..n_dims).step_by(2) {
            cache.push((theta.cos(), theta.sin()));
            theta *= theta_scale;
        }
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for (p, (ct, st)) in cache.iter().enumerate() {
                let (lo, hi) = match kind {
                    RopeKind::Normal => (2 * p, 2 * p + 1),
                    RopeKind::Neox => (p, p + n_dims / 2),
                };
                let x0 = src[base + lo];
                let x1 = src[base + hi];
                out[base + lo] = x0 * ct - x1 * st;
                out[base + hi] = x0 * st + x1 * ct;
            }
        }
    }
    out
}

/// Top-k expert selection: `(indices, weights)`, both `n_tokens * k` long.
///
/// Mirrors `llm_graph_context::build_moe_ffn` for the softmax-gated case: softmax over all
/// experts, take the k largest, and if `normalize` divide by their sum clamped up to the
/// smallest normal f16. Ties break toward the lower expert index.
pub fn topk_router(
    logits: &[f32],
    n_tokens: usize,
    n_expert: usize,
    k: usize,
    normalize: bool,
) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(logits.len(), n_tokens * n_expert);
    assert!(k > 0 && k <= n_expert);
    let probs = softmax(logits, n_tokens, n_expert);
    let mut idx = Vec::with_capacity(n_tokens * k);
    let mut wts = Vec::with_capacity(n_tokens * k);
    for t in 0..n_tokens {
        let row = &logits[t * n_expert..(t + 1) * n_expert];
        // Sort by logit descending, index ascending — a stable sort on the negated key gives
        // the lower index precedence on a tie, which is what the kernel does.
        let mut order: Vec<usize> = (0..n_expert).collect();
        order.sort_by(|a, b| row[*b].partial_cmp(&row[*a]).expect("router logits must not be NaN"));
        let chosen = &order[..k];
        let mut w: Vec<f32> = chosen.iter().map(|&e| probs[t * n_expert + e]).collect();
        if normalize {
            // llama.cpp clamps to 6.103515625e-5 — the smallest normal f16, i.e. 2^-14 —
            // before dividing. Written as a power of two so the value is exact and the
            // provenance is still legible.
            let sum: f32 = w.iter().sum::<f32>().max(2.0f32.powi(-14));
            for v in &mut w {
                *v /= sum;
            }
        }
        idx.extend(chosen.iter().map(|e| *e as u32));
        wts.extend(w);
    }
    (idx, wts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_covers_zero_subnormals_and_the_extremes() {
        // Hand-computed from the binary16 encoding; no library consulted.
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // largest finite half
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14)); // smallest normal
        assert_eq!(f16_to_f32(0x03FF), 2f32.powi(-14) - 2f32.powi(-24)); // largest subnormal
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn the_block_sizes_are_the_ones_ggml_asserts() {
        assert_eq!(QuantType::Q80.block_bytes(), 2 + QK8_0);
        assert_eq!(QuantType::Q4K.block_bytes(), 2 + 2 + 12 + QK_K / 2);
        assert_eq!(QuantType::Q5K.block_bytes(), 2 + 2 + 12 + QK_K / 8 + QK_K / 2);
        assert_eq!(QuantType::Q6K.block_bytes(), 2 + QK_K / 16 + 3 * QK_K / 4);
        assert_eq!(QuantType::Q80.block_bytes(), 34);
        assert_eq!(QuantType::Q4K.block_bytes(), 144);
        assert_eq!(QuantType::Q5K.block_bytes(), 176);
        assert_eq!(QuantType::Q6K.block_bytes(), 210);
    }

    #[test]
    fn a_zero_scale_block_dequantises_to_zero() {
        // d = dmin = 0 means every value collapses to 0 whatever the quants say. Cheap, but it
        // catches a block laid out at the wrong offset: garbage scales would not give zeros.
        for ty in [QuantType::Q80, QuantType::Q4K, QuantType::Q5K, QuantType::Q6K] {
            let mut blk = vec![0xABu8; ty.block_bytes()];
            match ty {
                QuantType::Q6K => blk[208..210].copy_from_slice(&0u16.to_le_bytes()),
                QuantType::Q80 => blk[0..2].copy_from_slice(&[0, 0]),
                _ => blk[0..4].copy_from_slice(&[0, 0, 0, 0]),
            }
            assert!(dequant(ty, &blk, 1).iter().all(|v| *v == 0.0), "{ty:?} did not collapse");
        }
    }

    #[test]
    fn the_router_picks_the_largest_and_the_weights_sum_to_one() {
        let logits = vec![0.1, 5.0, -2.0, 4.0, 4.0];
        let (idx, w) = topk_router(&logits, 1, 5, 3, true);
        assert_eq!(idx, vec![1, 3, 4], "expected the two 4.0s to tie-break by index");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "normalised weights summed to {sum}");
        assert!(w[0] > w[1], "the largest logit must carry the largest weight");
    }
}
