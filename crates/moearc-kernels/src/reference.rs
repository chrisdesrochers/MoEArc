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
    /// Not quantised at all — a "block" of one element. Present so that f32 tensors (norm
    /// weights, the router's `ffn_gate_inp`) go through the same code path as quantised ones.
    F32 = 0,
    /// Half precision, also a block of one. This is the read side of the f16 path: the KV
    /// cache and any f16 weight expand through the same `dequant` entry point.
    F16 = 1,
    /// The legacy 8-bit format: 32 elements, one f16 delta, no minimum. A Q4_K_M quantisation
    /// of Qwen3 holds 251 of these against 80 Q4_K, 37 Q5_K and 4 Q6_K, so it is not optional.
    Q80 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    /// The OCP microscaling 4-bit float: 32 elements sharing one **power-of-two** E8M0
    /// exponent. Not a K-quant — there is no scale/min pair and no super-block — and it is what
    /// every expert of gpt-oss is stored in.
    Mxfp4 = 39,
}

/// Elements per K-quant super-block. `QK_K` in `ggml/src/ggml-common.h`.
pub const QK_K: usize = 256;

/// Elements per Q8_0 block. `QK8_0` in `ggml/src/ggml-common.h`.
pub const QK8_0: usize = 32;

/// Elements per MXFP4 block. `QK_MXFP4` in `ggml/src/ggml-common.h`.
pub const QK_MXFP4: usize = 32;

/// MXFP4's E2M1 code table — `kvalues_fp4` in `ggml/src/ggml-common.h`, aliased there as
/// `kvalues_mxfp4`.
///
/// 🔴 These are the true E2M1 values **doubled**, which is what lets the table be integral. The
/// scale must therefore be halved to match; see [`e8m0_half`]. Index 8 is negative zero.
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `ggml_e8m0_to_fp32_half` from `ggml/src/ggml-impl.h`: `2^(x - 128)`.
///
/// 🔴 Half of the true E8M0 value `2^(x - 127)`, deliberately, to cancel the doubling in
/// [`KVALUES_MXFP4`]. Using the unhalved scale with that table doubles every weight in the
/// model — an error that produces finite, fluent, wrong output.
///
/// `x < 2` is a separate arm because the result is then a **subnormal** f32, which the
/// normalised `(x - 1) << 23` form cannot express.
pub fn e8m0_half(x: u8) -> f32 {
    // 0x00200000 = 2^-128, 0x00400000 = 2^-127.
    let bits = if x < 2 { 0x0020_0000u32 << x } else { (u32::from(x) - 1) << 23 };
    f32::from_bits(bits)
}

impl QuantType {
    /// Bytes one super-block occupies on disk.
    ///
    /// From the `static_assert`s beside each struct in `ggml/src/ggml-common.h`:
    /// Q8_0 is `2 + 32`, Q4_K is `2 + 2 + 12 + 128`, Q5_K is `2 + 2 + 12 + 32 + 128`, Q6_K is
    /// `128 + 64 + 16 + 2`.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q80 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            // `sizeof(block_mxfp4)` = one E8M0 byte + `QK_MXFP4/2` nibble pairs.
            Self::Mxfp4 => 17,
        }
    }

    /// Elements one block expands to. Not a constant across formats — the K-quants pack 256,
    /// Q8_0 packs 32 — so anything sizing a buffer must ask rather than assume `QK_K`.
    pub const fn block_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q80 => QK8_0,
            Self::Mxfp4 => QK_MXFP4,
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
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q80),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            39 => Some(Self::Mxfp4),
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
            QuantType::F32 => y[0] = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]),
            QuantType::F16 => y[0] = ld_f16(blk, 0),
            QuantType::Q80 => dequant_block_q8_0(blk, y),
            QuantType::Q4K => dequant_block_q4_k(blk, y),
            QuantType::Q5K => dequant_block_q5_k(blk, y),
            QuantType::Q6K => dequant_block_q6_k(blk, y),
            QuantType::Mxfp4 => dequant_block_mxfp4(blk, y),
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

/// `dequantize_row_mxfp4` from `ggml/src/ggml-quants.c`, one block, transcribed as written.
///
/// 🔴 The two nibbles of `qs[j]` are elements `j` and `j + 16` — the **halves** of the block,
/// not an adjacent pair. The C writes `y[i*qk + j + 0]` from the low nibble and
/// `y[i*qk + j + qk/2]` from the high one. Reading them as `2j`/`2j+1` shuffles every expert's
/// weights into a permutation of themselves, which is exactly the kind of error that leaves the
/// model fluent.
fn dequant_block_mxfp4(blk: &[u8], y: &mut [f32]) {
    let d = e8m0_half(blk[0]);
    for j in 0..QK_MXFP4 / 2 {
        let b = blk[1 + j];
        y[j] = f32::from(KVALUES_MXFP4[(b & 0x0F) as usize]) * d;
        y[j + QK_MXFP4 / 2] = f32::from(KVALUES_MXFP4[(b >> 4) as usize]) * d;
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

/// `ggml_swiglu_oai`, elementwise — the gpt-oss expert activation.
///
/// From `ggml/src/ggml-cpu/ops.cpp`:
///
/// ```text
///   x = min(gate, limit)
///   y = clamp(up, -limit, limit)
///   out = (x / (1 + exp(-alpha * x))) * (y + 1)
/// ```
///
/// 🔴 Three departures from [`swiglu`], each survivable and each wrong: the gate is clamped
/// **above only**, the sigmoid is alpha-scaled, and the up branch carries a **`+ 1`**.
pub fn swiglu_oai(gate: &[f32], up: &[f32], alpha: f32, limit: f32) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            let x = g.min(limit);
            let y = u.clamp(-limit, limit);
            (x / (1.0 + (alpha * -x).exp())) * (y + 1.0)
        })
        .collect()
}

/// Row-wise softmax, max-subtracted, summed in `f64`.
pub fn softmax(x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    softmax_ext(x, None, n_rows, n_cols, 1.0)
}

/// `softmax(x * scale + mask)` — `ggml_soft_max_ext` with no ALiBi.
///
/// The mask is additive and holds `-inf` where a key must not be seen, which is how causality
/// is expressed. Summed in `f64`; the GPU sums in `f32`.
pub fn softmax_ext(
    x: &[f32],
    mask: Option<&[f32]>,
    n_rows: usize,
    n_cols: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), n_rows * n_cols);
    if let Some(m) = mask {
        assert_eq!(m.len(), n_rows * n_cols);
    }
    let at = |i: usize| x[i] * scale + mask.map_or(0.0, |m| m[i]);
    let mut out = vec![0.0f32; x.len()];
    for r in 0..n_rows {
        let lo = r * n_cols;
        let mx = (lo..lo + n_cols).map(at).fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = (lo..lo + n_cols).map(|i| f64::from((at(i) - mx).exp())).sum();
        // A fully-masked row would divide by zero and poison everything downstream with NaN;
        // clamping to the smallest normal f32 makes it come back as zeros instead. The kernel
        // does the same.
        let denom = sum.max(f64::from(f32::MIN_POSITIVE));
        for c in 0..n_cols {
            out[lo + c] = (f64::from((at(lo + c) - mx).exp()) / denom) as f32;
        }
    }
    out
}

/// An additive causal mask for `n_q` queries against `n_kv` keys, the queries being the last
/// `n_q` of them. Zero where a key is visible, `-inf` where it is not.
pub fn causal_mask(n_q: usize, n_kv: usize) -> Vec<f32> {
    assert!(n_kv >= n_q);
    let first = n_kv - n_q;
    let mut m = vec![0.0f32; n_q * n_kv];
    for i in 0..n_q {
        for j in 0..n_kv {
            if j > first + i {
                m[i * n_kv + j] = f32::NEG_INFINITY;
            }
        }
    }
    m
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

/// How a router turns logits into the weights the combine applies.
///
/// 🔴 All three select the same experts — softmax is monotonic — and produce **different
/// weights**. Nothing in a GGUF records which one an architecture wants; each comes from
/// llama.cpp's `build_moe_ffn` call site, which is why `moe.rs` allowlists architectures by
/// name rather than inferring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Gating {
    /// Softmax over **all** experts, the k largest taken as-is. Their weights do not sum to
    /// one. `build_moe_ffn(..., norm_w = false, ..., SOFTMAX)` — OLMoE.
    Softmax = 0,
    /// Softmax over all experts, then the k selected divided by their sum (clamped up to the
    /// smallest normal f16). `norm_w = true` — Qwen3-MoE.
    SoftmaxNormalised = 1,
    /// Top-k on the **raw** logits, then a softmax over just those k.
    /// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`, which sets `probs = logits` and defers
    /// `ggml_soft_max` until after `ggml_get_rows` — gpt-oss.
    ///
    /// 🔴 Not the same as [`Self::SoftmaxNormalised`]. A softmax over 128 experts renormalised
    /// to 4 is not a softmax over those 4: the former's ratios are set by the full logit
    /// spread, the latter's only by the four selected. Both sum to one, and they differ.
    SoftmaxAfterTopK = 2,
}

/// YaRN context-extension parameters, as the RoPE kernel wants them.
///
/// Built by the engine from `rope.scaling.*`; see [`RopeScaling::yarn`] for where each number
/// comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScaling {
    /// `1 / rope.scaling.factor`.
    pub freq_scale: f32,
    /// llama.cpp forces this to 1.0 whenever the scaling type is YaRN, and 0.0 otherwise. At
    /// 0.0 the kernel degenerates to plain RoPE.
    pub ext_factor: f32,
    /// 🔴 **Not the YaRN paper's mscale.** llama.cpp computes `0.1*ln(s)+1` into
    /// `cparams.yarn_attn_factor` and then divides it straight back out, so that the *kernel*
    /// can multiply it in; the value that crosses this boundary is therefore **1.0** for a
    /// model with no `rope.scaling.attn_factor` key. Passing the paper's 1.3466 here squares
    /// it.
    pub attn_factor: f32,
    /// `ggml_rope_yarn_corr_dims()[0]` — below this channel, pure extrapolation.
    pub corr_lo: f32,
    /// `ggml_rope_yarn_corr_dims()[1]` — above this channel, pure interpolation.
    pub corr_hi: f32,
}

impl RopeScaling {
    /// No scaling: the kernel computes `cos(theta_extrap) * 1.0`, both exact, so a model
    /// carrying this is bit-for-bit what plain RoPE gives.
    pub const NONE: Self =
        Self { freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0, corr_lo: 0.0, corr_hi: 0.0 };

    /// YaRN, from the four GGUF keys and the model's geometry.
    ///
    /// `ggml_rope_yarn_corr_dims`, `ggml/src/ggml.c`:
    ///
    /// ```text
    ///   corr(beta) = n_dims * ln(n_ctx_orig / (beta * 2*pi)) / (2 * ln(freq_base))
    ///   lo = max(0, floor(corr(beta_fast)))   hi = min(n_dims - 1, ceil(corr(beta_slow)))
    /// ```
    pub fn yarn(
        n_dims: usize,
        n_ctx_orig: usize,
        freq_base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Self {
        let corr = |beta: f32| -> f32 {
            n_dims as f32 * (n_ctx_orig as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * freq_base.ln())
        };
        Self {
            freq_scale: 1.0 / factor,
            ext_factor: 1.0,
            attn_factor: 1.0,
            corr_lo: corr(beta_fast).floor().max(0.0),
            corr_hi: corr(beta_slow).ceil().min(n_dims as f32 - 1.0),
        }
    }
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
    gating: Gating,
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
        if gating == Gating::SoftmaxAfterTopK {
            // A softmax over the k selected **logits**, not a restriction of the full softmax.
            let sel: Vec<f32> = chosen.iter().map(|&e| row[e]).collect();
            idx.extend(chosen.iter().map(|e| *e as u32));
            wts.extend(softmax(&sel, 1, k));
            continue;
        }
        let mut w: Vec<f32> = chosen.iter().map(|&e| probs[t * n_expert + e]).collect();
        if gating == Gating::SoftmaxNormalised {
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

/// IEEE-754 binary32 to binary16, round-to-nearest-even.
///
/// Written out for the same reason [`f16_to_f32`] is: Rust has no stable `f16`, and the
/// rounding rule is part of the contract rather than an implementation detail. ggml rounds with
/// the hardware `_cvtss_sh`, which is RNE, so RNE is what this does — ties to even, subnormals
/// rounded rather than flushed to zero, overflow to infinity, and a NaN that stays a NaN.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e32 = (x >> 23) & 0xFF;
    let mut mant = x & 0x007F_FFFF;

    if e32 == 0xFF {
        // Preserve NaN-ness: collapsing a NaN to infinity would turn a loud failure quiet.
        let payload = if mant != 0 { 0x0200 | (mant >> 13) as u16 } else { 0 };
        return sign | 0x7C00 | payload;
    }

    let mut exp = e32 as i32 - 127 + 15;
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x0080_0000;
        let shift = 14 - exp; // 14..=24
        let t = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let round = u32::from(rem > half || (rem == half && t & 1 == 1));
        // A carry out of the mantissa lands in the exponent field by itself, which is exactly
        // right: the value rounds up to the smallest normal.
        return sign | (t + round) as u16;
    }

    let mut t = mant >> 13;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && t & 1 == 1) {
        t += 1;
        if t == 0x400 {
            t = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | t as u16
}

/// `out[i] = a[i] + b[i]` — the residual add.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `out[i] = a[i] * b[i]`.
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// `out[i] += alpha * x[i]` — the MoE combine, folding one expert's output into the total.
pub fn axpy(out: &mut [f32], x: &[f32], alpha: f32) {
    assert_eq!(out.len(), x.len());
    for (o, v) in out.iter_mut().zip(x) {
        *o += alpha * v;
    }
}

/// Gather and expand `token_ids.len()` rows of an embedding table.
pub fn embed_rows(ty: QuantType, table: &[u8], token_ids: &[u32], n_embd: usize) -> Vec<f32> {
    let be = ty.block_elems();
    assert_eq!(n_embd % be, 0, "an embedding row must be a whole number of {be}-element blocks");
    let row_bytes = (n_embd / be) * ty.block_bytes();
    let mut out = Vec::with_capacity(token_ids.len() * n_embd);
    for &t in token_ids {
        let off = t as usize * row_bytes;
        out.extend(dequant(ty, &table[off..off + row_bytes], n_embd / be));
    }
    out
}

/// How a KV cache page stores its numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    F32,
    /// Half the footprint and half the bandwidth, at the cost of ~3 decimal digits on each
    /// cached key and value. The reference models that loss rather than ignoring it.
    F16,
}

impl KvType {
    /// The GGUF type id that crosses the C ABI.
    pub const fn type_id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
        }
    }

    /// Bytes one cached number occupies.
    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    /// The value that actually lands in the cache — for f16, after a round trip through the
    /// narrower format.
    pub fn store(self, v: f32) -> f32 {
        match self {
            Self::F32 => v,
            Self::F16 => f16_to_f32(f32_to_f16(v)),
        }
    }
}

/// Index of channel `d` of head `h` at (`page`, `slot`) in a KV page pool.
///
/// The layout is `[page][slot][kv_head][head_dim]`, head_dim contiguous. Written once, here, so
/// the kernel and the reference cannot disagree about it — a layout mismatch between writer and
/// reader is silent and produces plausible-looking garbage.
pub fn kv_index(
    page: u32,
    slot: u32,
    head: usize,
    d: usize,
    n_kv_heads: usize,
    head_dim: usize,
    page_tokens: usize,
) -> usize {
    ((page as usize * page_tokens + slot as usize) * n_kv_heads + head) * head_dim + d
}

/// Write one token's K and V into a page slot.
#[allow(clippy::too_many_arguments)]
pub fn kv_append(
    k_pages: &mut [f32],
    v_pages: &mut [f32],
    k: &[f32],
    v: &[f32],
    page_id: u32,
    slot: u32,
    n_kv_heads: usize,
    head_dim: usize,
    page_tokens: usize,
    kv: KvType,
) {
    assert_eq!(k.len(), n_kv_heads * head_dim);
    assert_eq!(v.len(), n_kv_heads * head_dim);
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let i = kv_index(page_id, slot, h, d, n_kv_heads, head_dim, page_tokens);
            k_pages[i] = kv.store(k[h * head_dim + d]);
            v_pages[i] = kv.store(v[h * head_dim + d]);
        }
    }
}

/// Single-query attention over a paged KV cache: `softmax(scale * q.K^T) . V`.
///
/// 🔴 Deliberately not written the way the kernel is. This materialises every score, takes the
/// maximum, exponentiates, and divides — the textbook two-pass softmax — in `f64`. The kernel
/// accumulates online in `f32`, rescaling as the running maximum moves. The two agree
/// mathematically and are different programs, which is the only reason comparing them means
/// anything.
///
/// Causality is the loop bound: `n_kv` counts the keys at or before the query, which for decode
/// is every key in the cache including the query's own, already appended.
///
/// 🔴 `kv_begin` is the other end of that bound, and it is what **sliding-window attention**
/// means here: the span is `[kv_begin, n_kv)`, so a window of `n_swa` is
/// `kv_begin = n_kv.saturating_sub(n_swa)`. `0` is full causal attention. Deliberately written
/// as a span rather than as an additive `-inf` mask, so that this and the kernel are two
/// different programs for the same function rather than one program written twice.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode(
    q: &[f32],
    k_pages: &[f32],
    v_pages: &[f32],
    block_table: &[u32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_begin: usize,
    n_kv: usize,
    page_tokens: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert!(n_kv > 0 && n_heads % n_kv_heads == 0);
    assert!(kv_begin < n_kv, "an empty key span is a window computed wrong");
    assert!(block_table.len() >= n_kv.div_ceil(page_tokens));
    let group = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        let scores: Vec<f64> = (kv_begin..n_kv)
            .map(|j| {
                let page = block_table[j / page_tokens];
                let slot = (j % page_tokens) as u32;
                let base = kv_index(page, slot, kvh, 0, n_kv_heads, head_dim, page_tokens);
                let dot: f64 =
                    (0..head_dim).map(|d| f64::from(qh[d]) * f64::from(k_pages[base + d])).sum();
                dot * f64::from(scale)
            })
            .collect();

        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let denom: f64 = exps.iter().sum();

        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for (i, e) in exps.iter().enumerate() {
                let j = kv_begin + i;
                let page = block_table[j / page_tokens];
                let slot = (j % page_tokens) as u32;
                let base = kv_index(page, slot, kvh, 0, n_kv_heads, head_dim, page_tokens);
                acc += e * f64::from(v_pages[base + d]);
            }
            out[h * head_dim + d] = (acc / denom) as f32;
        }
    }
    out
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
        let (idx, w) = topk_router(&logits, 1, 5, 3, Gating::SoftmaxNormalised);
        assert_eq!(idx, vec![1, 3, 4], "expected the two 4.0s to tie-break by index");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "normalised weights summed to {sum}");
        assert!(w[0] > w[1], "the largest logit must carry the largest weight");
    }
}
