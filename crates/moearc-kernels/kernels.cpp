// MoEArc SYCL kernels, behind a plain C ABI.
//
// The C ABI is the whole design. SYCL is C++ and its headers transitively pull
// <sycl/sycl.hpp>, so anything that saw those types would force every consumer to have oneAPI
// installed. Nothing above this file knows SYCL exists: Rust sees opaque pointers and integers.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace sycl;

// =======================================================================================
// GGUF block-quantisation layouts
// =======================================================================================
//
// 🔴 Every constant and every line of arithmetic below was read out of a checkout of
// `ggml-org/llama.cpp`, not reconstructed from memory. Provenance, file by file:
//
//   * `ggml/src/ggml-common.h` — `#define QK_K 256`, `#define K_SCALE_SIZE 12`, and the
//     `block_q4_K` / `block_q5_K` / `block_q6_K` struct definitions with their
//     `static_assert`ed sizes. Field order in a C struct is field order on disk, and GGUF is
//     little-endian, so the byte offsets named below are read straight off those structs.
//   * `ggml/src/ggml-quants.c` — `get_scale_min_k4()` (line 880 in the checkout read) and
//     `dequantize_row_q4_K` / `dequantize_row_q5_K` / `dequantize_row_q6_K`. The element
//     formulas here are those functions re-expressed as a closure over one output index
//     instead of a sequential walk, so a work-item can compute any element on its own.
//
// The re-expression is the only liberty taken, and it is checked rather than trusted: the
// `gguf_crosscheck` test dequantises real tensors out of a real model with llama.cpp's own
// `to_float` and compares element for element.
//
// 🔴 Association is deliberate. `d1 * q - m1` where `d1 = d * sc` is written in exactly that
// order because that is the order `ggml-quants.c` writes it in, and floating-point addition
// and multiplication are not associative. Rewriting it as `d * (sc * q)` would be
// mathematically identical and numerically different, which is the kind of difference that
// makes a cross-check fail for a reason nobody can find.

// Super-block size for the K-quants. Source: `ggml-common.h`, `#define QK_K 256`. The legacy
// Q8_0 format uses a smaller, independent block: `#define QK8_0 32`.
static constexpr int QK_K = 256;
static constexpr int QK8_0 = 32;
// MXFP4's block, from `ggml-common.h`: `#define QK_MXFP4 32`, and a `block_mxfp4` of one E8M0
// scale byte followed by `QK_MXFP4/2` nibble pairs.
static constexpr int QK_MXFP4 = 32;

// Bytes per block, from the `static_assert`s in `ggml-common.h`:
//   Q4_K: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2       = 2 + 2 + 12 + 128 = 144
//   Q5_K: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8 = 2 + 2 + 12 + 128 + 32 = 176
//   Q6_K: sizeof(ggml_half) + QK_K/16 + 3*QK_K/4            = 2 + 16 + 192 = 210
//   Q8_0: sizeof(ggml_half) + QK8_0                         = 2 + 32 = 34
static constexpr int Q4_K_BYTES = 144;
static constexpr int Q5_K_BYTES = 176;
static constexpr int Q6_K_BYTES = 210;
static constexpr int Q8_0_BYTES = 34;
//   MXFP4: sizeof(uint8_t) + QK_MXFP4/2                     = 1 + 16 = 17
static constexpr int MXFP4_BYTES = 17;

// GGUF type ids, from `gguf-py/gguf/constants.py`, `class GGMLQuantizationType(IntEnum)`.
// The same ids the `moearc-model` crate's `quant` table carries.
static constexpr unsigned int GGML_TYPE_F32 = 0;
static constexpr unsigned int GGML_TYPE_F16 = 1;
static constexpr unsigned int GGML_TYPE_Q8_0 = 8;
static constexpr unsigned int GGML_TYPE_Q4_K = 12;
static constexpr unsigned int GGML_TYPE_Q5_K = 13;
static constexpr unsigned int GGML_TYPE_Q6_K = 14;
static constexpr unsigned int GGML_TYPE_MXFP4 = 39;

/// Load a little-endian 16-bit word from an arbitrary byte address.
///
/// Byte-wise on purpose. A `ggml_half` inside a Q6_K block sits at offset 208 of a 210-byte
/// block, so blocks are only 2-byte aligned even when the array is; and GGUF is defined as
/// little-endian regardless of the host, which this makes explicit rather than assumed.
static inline unsigned int ld_u16le(const unsigned char *p) {
    return (unsigned int) p[0] | ((unsigned int) p[1] << 8);
}

/// IEEE-754 binary16 to binary32, written out rather than delegated.
///
/// Every half value — including subnormals — is exactly representable as a float, so this is a
/// lossless conversion and any correct implementation gives the same answer as ggml's
/// `GGML_FP16_TO_FP32`. Doing it in integer arithmetic avoids depending on how a particular
/// SYCL backend implements `half`, and works on a device with no native fp16 support.
/// ⚠️ **Do not replace this with the device's own `half` conversion in the matvec. Measured
/// slower.** The integer path looks like the expensive option — it has a `while` loop for
/// subnormals that runs on every block — and an isolated microbenchmark of a Q4_K matvec on this
/// card agreed, putting `(float) sycl::bit_cast<sycl::half>(h)` 7% ahead and claiming it was the
/// gate on a further 13% from coalescing.
///
/// It does not transfer. Swapping only the four hot-loop call sites in `unit_acc`, measured in
/// the engine on Qwen3-30B-A3B at 2952 resident slots with `MOEARC_SYNC_EACH=1`, three runs each
/// and a spread under 0.2%:
///
/// ```text
///                            software f16   device half
///   moe.expert_matvec  Q4_K      7.42 ms       7.47 ms    +0.7%
///   moe.expert_down    Q6_K      3.29 ms       3.60 ms    +9.4%
///   out.matvec         Q6_K      4.02 ms       4.44 ms   +10.4%
///   decode.total                40.4  ms      41.0  ms    +1.6%
/// ```
///
/// 🔴 Uniformly worse, and worst on the **Q6_K** kernels. The likely reason is that these kernels
/// are not ALU-bound at all: the integer work here hides under memory latency on a pipe the float
/// MACs are not using, and moving it onto the float/conversion pipe puts it on the contended one.
/// (Mechanism is inference; the numbers are measured.)
///
/// The wider lesson is the one worth keeping: **a microbenchmark of "the same kernel" is not this
/// kernel.** The 2x2 that produced the 7% also produced the 21% coalescing figure, so that figure
/// is unsupported here too until it is measured in place.
///
/// Correctness was never the obstacle — over all 65,536 half bit patterns the two differ on 1,022,
/// every one a signalling NaN, which a GGUF scale is never. It is simply slower.
static inline float f16_to_f32(unsigned int h) {
    const unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;  // signed zero
        } else {
            // Subnormal: renormalise until the implicit bit appears, decrementing the exponent.
            exp = 113u;  // 127 - 15 + 1: where a half subnormal renormalises from
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);  // 127 - 15 = 112
    }
    return sycl::bit_cast<float>(bits);
}

/// IEEE-754 binary32 to binary16, round-to-nearest-even.
///
/// The inverse of `f16_to_f32`, and the harder direction: it is lossy, so the rounding rule is
/// part of the contract. ggml uses the hardware `_cvtss_sh` where F16C exists, which is RNE, so
/// RNE is what this implements — ties to even, subnormals rounded rather than flushed, overflow
/// to infinity rather than wrapping. `tests/gguf_crosscheck.rs` checks it against
/// `ggml_fp32_to_fp16_row` over every f16 bit pattern and a sweep of hard f32 cases, so the
/// claim is verified rather than asserted.
static inline unsigned short f32_to_f16(float f) {
    const unsigned int x = sycl::bit_cast<unsigned int>(f);
    const unsigned int sign = (x >> 16) & 0x8000u;
    const unsigned int e32 = (x >> 23) & 0xFFu;
    unsigned int mant = x & 0x7FFFFFu;

    if (e32 == 0xFFu) {  // inf or NaN: keep NaN-ness, and never let a NaN collapse to infinity
        return (unsigned short) (sign | 0x7C00u | (mant ? (0x200u | (mant >> 13)) : 0u));
    }

    int exp = (int) e32 - 127 + 15;
    if (exp >= 0x1F) return (unsigned short) (sign | 0x7C00u);  // overflow
    if (exp <= 0) {
        if (exp < -10) return (unsigned short) sign;  // underflows past the last subnormal
        // Subnormal: restore the implicit bit and shift down to a multiple of 2^-24.
        mant |= 0x800000u;
        const int shift = 14 - exp;  // 14..24
        const unsigned int t = mant >> shift;
        const unsigned int rem = mant & ((1u << shift) - 1u);
        const unsigned int half = 1u << (shift - 1);
        // A carry out of the mantissa lands in the exponent field on its own, which is exactly
        // right: the value rounds up to the smallest normal.
        return (unsigned short) (sign | (t + ((rem > half || (rem == half && (t & 1u))) ? 1u : 0u)));
    }

    unsigned int t = mant >> 13;
    const unsigned int rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (t & 1u))) {
        if (++t == 0x400u) {  // mantissa carried; bump the exponent and renormalise
            t = 0;
            if (++exp >= 0x1F) return (unsigned short) (sign | 0x7C00u);
        }
    }
    return (unsigned short) (sign | ((unsigned int) exp << 10) | t);
}

/// One element of an unquantised f32 "block". A block of one, so the format table below can
/// treat f32 and f16 tensors exactly like quantised ones and every consumer — dequantise,
/// matvec, embedding lookup — works on all six formats with no special case.
static inline float f32_elem(const unsigned char *blk, int) {
    return sycl::bit_cast<float>((unsigned int) blk[0] | ((unsigned int) blk[1] << 8)
                                 | ((unsigned int) blk[2] << 16) | ((unsigned int) blk[3] << 24));
}

/// One element of an f16 "block" of one.
static inline float f16_elem(const unsigned char *blk, int) { return f16_to_f32(ld_u16le(blk)); }

/// The 6-bit scale/min unpacker shared by Q4_K and Q5_K.
///
/// Transcribed from `get_scale_min_k4` in `ggml/src/ggml-quants.c`. `j` indexes the eight
/// 32-element sub-blocks of a super-block; twelve bytes hold sixteen 6-bit values (eight
/// scales, eight mins), the first four of each packed plainly and the last four split across
/// the high bits of the earlier bytes.
static inline void q45k_scale_min(const unsigned char *q, int j, int *d, int *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// One dequantised element of a Q4_K block. `i` is 0..255 within the super-block.
///
/// Block layout (`block_q4_K`, `ggml-common.h`):
///   [0..2)   ggml_half d      — super-block scale for the quantised scales
///   [2..4)   ggml_half dmin   — super-block scale for the quantised mins
///   [4..16)  uint8_t scales[12]
///   [16..144) uint8_t qs[128] — two 4-bit quants per byte
///
/// `dequantize_row_q4_K` walks the block in four passes of 64 outputs. Pass `n` reads
/// `qs[32n .. 32n+31]`: the low nibbles supply outputs `64n..64n+31` under scale index `2n`,
/// the high nibbles supply outputs `64n+32..64n+63` under scale index `2n+1`. Writing `sub`
/// for the 32-element sub-block `i/32`, that is exactly `n = sub/2`, `half = sub%2`.
static inline float q4k_elem(const unsigned char *blk, int i) {
    const float d = f16_to_f32(ld_u16le(blk));
    const float dmin = f16_to_f32(ld_u16le(blk + 2));
    const unsigned char *scales = blk + 4;
    const unsigned char *qs = blk + 16;

    const int sub = i >> 5;   // 0..7
    const int n = sub >> 1;   // 0..3, which 32-byte run of qs
    const int half = sub & 1; // low nibble or high nibble
    const int l = i & 31;

    int sc, m;
    q45k_scale_min(scales, sub, &sc, &m);
    const float d1 = d * (float) sc;
    const float m1 = dmin * (float) m;

    const unsigned int byte = qs[n * 32 + l];
    const unsigned int q = half ? (byte >> 4) : (byte & 0xF);
    return d1 * (float) q - m1;
}

/// One dequantised element of a Q5_K block. `i` is 0..255.
///
/// Block layout (`block_q5_K`, `ggml-common.h`):
///   [0..2)    ggml_half d
///   [2..4)    ggml_half dmin
///   [4..16)   uint8_t scales[12]
///   [16..48)  uint8_t qh[32]   — the fifth bit of every quant
///   [48..176) uint8_t qs[128]  — the low four bits
///
/// `dequantize_row_q5_K` differs from Q4_K only in the extra bit: it carries masks `u1 = 1`,
/// `u2 = 2` and shifts both left by two after each 64-output pass, testing them against
/// `qh[l]`. Pass `n` therefore tests bit `2n` for the low-nibble half and bit `2n+1` for the
/// high-nibble half — i.e. bit `2n + half`, which is what this computes. `qh` is indexed by
/// `l` alone and is not advanced between passes.
static inline float q5k_elem(const unsigned char *blk, int i) {
    const float d = f16_to_f32(ld_u16le(blk));
    const float dmin = f16_to_f32(ld_u16le(blk + 2));
    const unsigned char *scales = blk + 4;
    const unsigned char *qh = blk + 16;
    const unsigned char *ql = blk + 48;

    const int sub = i >> 5;
    const int n = sub >> 1;
    const int half = sub & 1;
    const int l = i & 31;

    int sc, m;
    q45k_scale_min(scales, sub, &sc, &m);
    const float d1 = d * (float) sc;
    const float m1 = dmin * (float) m;

    const unsigned int byte = ql[n * 32 + l];
    const unsigned int lo = half ? (byte >> 4) : (byte & 0xF);
    const unsigned int hi = (qh[l] >> (2 * n + half)) & 1u;
    return d1 * (float) (lo + 16u * hi) - m1;
}

/// One dequantised element of a Q6_K block. `i` is 0..255.
///
/// Block layout (`block_q6_K`, `ggml-common.h`) — note the scale lives at the *end*:
///   [0..128)   uint8_t ql[128]  — low four bits
///   [128..192) uint8_t qh[64]   — high two bits, four quants per byte
///   [192..208) int8_t  scales[16] — one signed 8-bit scale per 16 elements
///   [208..210) ggml_half d
///
/// `dequantize_row_q6_K` walks the block in two passes of 128 outputs. Within pass `n`, for
/// `l` in 0..31 it emits four outputs at `l`, `l+32`, `l+64`, `l+96` drawn from `ql[l]`,
/// `ql[l+32]`, `ql[l]>>4`, `ql[l+32]>>4` with `qh[l]` shifted by 0, 2, 4, 6 and scale indices
/// `is`, `is+2`, `is+4`, `is+6` where `is = l/16`. Writing `k = (i%128)/32` for which of those
/// four an output is, the pattern collapses to: byte `ql[64n + 32*(k%2) + l]`, nibble low for
/// `k < 2`, `qh` shift `2k`, scale `8n + 2k + l/16`. The quant is biased by -32 — Q6_K is
/// `x = a*q` with a signed `q`, not `a*q + b`.
static inline float q6k_elem(const unsigned char *blk, int i) {
    const unsigned char *ql = blk;
    const unsigned char *qh = blk + 128;
    const signed char *sc = (const signed char *) (blk + 192);
    const float d = f16_to_f32(ld_u16le(blk + 208));

    const int n = i >> 7;  // 0..1, which 128-output pass
    const int r = i & 127;
    const int k = r >> 5;  // 0..3, which quarter of the pass
    const int l = r & 31;

    const unsigned int qlb = ql[n * 64 + (k & 1) * 32 + l];
    const unsigned int lo = (k < 2) ? (qlb & 0xF) : (qlb >> 4);
    const unsigned int hb = (qh[n * 32 + l] >> (2 * k)) & 3u;
    const int q = (int) (lo | (hb << 4)) - 32;
    const int s = sc[n * 8 + 2 * k + (l >> 4)];

    return d * (float) s * (float) q;
}

/// The E2M1 code table MXFP4 indexes with each nibble.
///
/// 🔴 These are the true E2M1 values **doubled** — `ggml-common.h` calls the table
/// `kvalues_fp4` and aliases `kvalues_mxfp4` to it, precisely so it can be `int8_t`. Index 8 is
/// negative zero and decodes to 0. Pairing this table with the *undoubled* scale gives every
/// expert weight twice its value: finite, fluent, wrong.
static constexpr signed char KVALUES_MXFP4[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                                  0, -1, -2, -3, -4, -6, -8, -12};

/// The E8M0 scale byte, **halved** — `ggml_e8m0_to_fp32_half` from `ggml/src/ggml-impl.h`.
///
/// 🔴 The halving is not a rounding convenience, it is the other half of the doubled table
/// above: `half_scale * doubled_value == true_scale * true_value`. `GGML_E8M0_TO_FP32` (without
/// `_HALF`) is a different function and is *not* the one `dequantize_row_mxfp4` calls.
///
/// `x < 2` is a separate arm because 2^(x-128) for x in {0,1} is a **subnormal** f32, which the
/// normalised `(x-1) << 23` form cannot express — it would underflow the exponent field and
/// produce a large number rather than a tiny one. Transcribed from the C, not derived.
static inline float e8m0_half(unsigned int x) {
    // 0x00200000 = 2^-128, 0x00400000 = 2^-127.
    const unsigned int bits = (x < 2u) ? (0x00200000u << x) : ((x - 1u) << 23);
    return sycl::bit_cast<float>(bits);
}

/// One dequantised element of an MXFP4 block. `i` is 0..31.
///
/// Block layout (`block_mxfp4`, `ggml-common.h`): `[0]` the E8M0 scale, `[1..17)` sixteen bytes
/// of two 4-bit codes each.
///
/// 🔴 The two nibbles of a byte are elements `j` and `j + 16` — **split halves, not adjacent**.
/// `dequantize_row_mxfp4` writes `y[i*qk + j]` from the low nibble and `y[i*qk + j + qk/2]` from
/// the high one. Reading them as `2j` and `2j+1` is the same bytes in the wrong order, which
/// gives a model that runs and is wrong; llama.cpp's own HF converter carries a
/// `transform_nibble_layout()` for exactly this reason.
static inline float mxfp4_elem(const unsigned char *blk, int i) {
    const float d = e8m0_half(blk[0]);
    const unsigned int b = blk[1 + (i & 15)];
    const int q = KVALUES_MXFP4[(i < 16) ? (b & 0xFu) : (b >> 4)];
    return (float) q * d;
}

/// One dequantised element of a Q8_0 block. `i` is 0..31.
///
/// Block layout (`block_q8_0`, `ggml-common.h`): `[0..2)` a `ggml_half` delta, `[2..34)` thirty-
/// two signed 8-bit quants. There is no per-sub-block scale and no minimum: `dequantize_row_q8_0`
/// is the single line `y[i*qk + j] = x[i].qs[j]*d`, with the quant on the left, which is the
/// order kept here.
///
/// Q8_0 earns its place because the target model needs it. A Q4_K_M quantisation of Qwen3 holds
/// 251 Q8_0 tensors alongside its 80 Q4_K, 37 Q5_K and 4 Q6_K — the K-quant kernels alone cannot
/// run a forward pass over that file.
static inline float q80_elem(const unsigned char *blk, int i) {
    const float d = f16_to_f32(ld_u16le(blk));
    const signed char q = (signed char) blk[2 + i];
    return (float) q * d;
}

/// Bytes one block of `type_id` occupies, or 0 if this file cannot read that type.
static inline int block_bytes(unsigned int type_id) {
    switch (type_id) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_Q8_0: return Q8_0_BYTES;
        case GGML_TYPE_Q4_K: return Q4_K_BYTES;
        case GGML_TYPE_Q5_K: return Q5_K_BYTES;
        case GGML_TYPE_Q6_K: return Q6_K_BYTES;
        case GGML_TYPE_MXFP4: return MXFP4_BYTES;
        default: return 0;
    }
}

/// Elements one block of `type_id` expands to. Not a constant across formats: the K-quants pack
/// 256 elements per super-block, Q8_0 packs 32, and f32/f16 are blocks of one.
static inline int block_elems(unsigned int type_id) {
    switch (type_id) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_MXFP4: return QK_MXFP4;
        default: return QK_K;
    }
}

/// One element of a block of any supported format.
///
/// The single dispatch point. Every consumer of a weight — the expansion kernel, both matvecs,
/// the embedding gather — goes through here, so none of them can hold a different opinion about
/// what a byte means. `type_id` is uniform across a launch, so the switch costs nothing per
/// element beyond what the compiler hoists out of the loop.
static inline float elem_at(unsigned int type_id, const unsigned char *blk, int i) {
    switch (type_id) {
        case GGML_TYPE_F32: return f32_elem(blk, i);
        case GGML_TYPE_F16: return f16_elem(blk, i);
        case GGML_TYPE_Q8_0: return q80_elem(blk, i);
        case GGML_TYPE_Q4_K: return q4k_elem(blk, i);
        case GGML_TYPE_Q5_K: return q5k_elem(blk, i);
        case GGML_TYPE_MXFP4: return mxfp4_elem(blk, i);
        default: return q6k_elem(blk, i);  // Q6_K; block_bytes rejected everything else
    }
}

/// Read one element of a KV cache page, which is stored as f32 or f16.
///
/// f16 halves the cache's footprint and its bandwidth, and the KV cache is the largest thing on
/// the card after the weights. The precision cost lands on attention scores, which are about to
/// go through a softmax that is far less sensitive than a weight would be — which is why
/// llama.cpp's own flash-attention path casts K and V to f16 unconditionally.
static inline float kv_load(const void *base, unsigned long i, unsigned int kv_type) {
    if (kv_type == GGML_TYPE_F16) {
        return f16_to_f32(static_cast<const unsigned short *>(base)[i]);
    }
    return static_cast<const float *>(base)[i];
}

static inline void kv_store(void *base, unsigned long i, float v, unsigned int kv_type) {
    if (kv_type == GGML_TYPE_F16) {
        static_cast<unsigned short *>(base)[i] = f32_to_f16(v);
    } else {
        static_cast<float *>(base)[i] = v;
    }
}

// Work-group width for the row-per-group reductions (matvec, rmsnorm, softmax). 32 is a
// native sub-group width on Intel Xe and keeps `reduce_over_group` cheap. It is a tuning
// constant, not a correctness one: any power of two gives the same answer up to the order of
// the reduction.
static constexpr size_t WG = 32;

// Largest `k` the router will select. Bounds a per-work-item array, since kernels cannot
// allocate. Real MoE models use 8 or fewer; 32 leaves headroom without spilling.
static constexpr int MAX_TOPK = 32;

// Lanes per row for RMSNorm.
//
// `WG` (32) is one sub-group, which is the right width for a short row and nowhere near enough
// for a long one: a single work-group has one hardware thread's worth of outstanding loads, so
// a 2048-wide row spends its time waiting on memory rather than doing arithmetic. Measured on
// a B580, one RMSNorm over `n_embd = 2048` cost 28 us at WG=32 — for 8 KiB of traffic.
//
// 🔴 This changes the *grouping* of the sum of squares, and floating-point addition is not
// associative, so a wider group is not bit-identical to a narrower one. It is not less accurate
// either — a wider tree over the same values has a shorter dependence chain and no more error —
// but it is a change to the arithmetic, and the forward-pass tests are what say it is a safe
// one.
static inline size_t norm_wg(unsigned long n_cols) {
    if (n_cols >= 1024) return 256;
    if (n_cols >= 256) return 64;
    return WG;
}

// Return codes shared by the entry points below.
//   0  ok
//  -1  a device call threw, or an argument was null
//  -2  an argument was out of range (unsupported quant type, bad shape, k too large)
static constexpr int OK = 0;
static constexpr int ERR = -1;
static constexpr int ERR_ARG = -2;


// =======================================================================================
// The quantised matvec's inner loop
// =======================================================================================
//
// Every format above has a natural **32-element unit** over which the dequantisation constants
// are fixed: a K-quant super-block is eight of them, a Q8_0 block is exactly one. `elem_at`
// derives those constants afresh for every element — two f16 conversions and a 6-bit scale
// unpack for a single multiply-accumulate. Deriving them once per unit instead is worth about
// 25 instructions per MAC, and a decode step spends most of its time here.
//
// 🔴 The per-element expression is unchanged, term for term and in the same association:
// `d1 * q - m1` with `d1 = d * sc` and `m1 = dmin * m` for Q4_K and Q5_K, `(d * s) * q` for
// Q6_K, `(q * d)` for Q8_0 — exactly as `ggml-quants.c` writes them, and exactly as the
// `*_elem` functions above compute them. Only *where* those constants are computed moves. The
// accumulation is likewise still `acc += weight * x` one element at a time, so a lane that
// covers the same elements in the same order gets a bit-identical answer.
//
// `elem_at` and the `*_elem` functions stay, and stay the single definition of what a byte
// means: `moearc_dequant`, `moearc_embed_rows` and the reference tests all still go through
// them, and `tests/gguf_crosscheck.rs` checks them against llama.cpp's own `to_float`. What is
// below is the same formulas re-associated for a hot loop, and the forward-pass tests are what
// hold the two in agreement.

/// 32-element units per block of `TY`. A K-quant super-block holds eight; a Q8_0 block is one.
template <unsigned int TY>
static constexpr int units_per_block() {
    return (TY == GGML_TYPE_Q8_0 || TY == GGML_TYPE_MXFP4) ? 1 : QK_K / 32;
}

/// Bytes per block of `TY`, as a compile-time constant so the address arithmetic folds.
template <unsigned int TY>
static constexpr int const_block_bytes() {
    return TY == GGML_TYPE_Q4_K   ? Q4_K_BYTES
           : TY == GGML_TYPE_Q5_K ? Q5_K_BYTES
           : TY == GGML_TYPE_Q6_K ? Q6_K_BYTES
           : TY == GGML_TYPE_MXFP4 ? MXFP4_BYTES
                                  : Q8_0_BYTES;
}

/// Accumulate `x . w` over one 32-element unit: sub-block `sub` of the block at `blk`, against
/// the 32 activations at `xs`.
template <unsigned int TY>
static inline void unit_acc(float &acc, const unsigned char *blk, const float *xs, int sub) {
    if constexpr (TY == GGML_TYPE_Q4_K) {
        int sc, m;
        q45k_scale_min(blk + 4, sub, &sc, &m);
        const float d1 = f16_to_f32(ld_u16le(blk)) * (float) sc;
        const float m1 = f16_to_f32(ld_u16le(blk + 2)) * (float) m;
        const unsigned char *qs = blk + 16 + (sub >> 1) * 32;
        // The nibble half is uniform across the unit, so it leaves the loop as a branch on a
        // scalar rather than a select on every element.
        if (sub & 1) {
            for (int l = 0; l < 32; ++l) acc += (d1 * (float) (qs[l] >> 4) - m1) * xs[l];
        } else {
            for (int l = 0; l < 32; ++l) acc += (d1 * (float) (qs[l] & 0xFu) - m1) * xs[l];
        }
    } else if constexpr (TY == GGML_TYPE_Q5_K) {
        int sc, m;
        q45k_scale_min(blk + 4, sub, &sc, &m);
        const float d1 = f16_to_f32(ld_u16le(blk)) * (float) sc;
        const float m1 = f16_to_f32(ld_u16le(blk + 2)) * (float) m;
        const unsigned char *qh = blk + 16;
        const unsigned char *ql = blk + 48 + (sub >> 1) * 32;
        // `q5k_elem` shifts `qh[l]` by `2 * n + half` with `n = sub >> 1`, `half = sub & 1` —
        // which is `sub` itself.
        const int shift = sub;
        const bool hi = (sub & 1) != 0;
        for (int l = 0; l < 32; ++l) {
            const unsigned int byte = ql[l];
            const unsigned int lo = hi ? (byte >> 4) : (byte & 0xFu);
            const unsigned int h = (qh[l] >> shift) & 1u;
            acc += (d1 * (float) (lo + 16u * h) - m1) * xs[l];
        }
    } else if constexpr (TY == GGML_TYPE_Q6_K) {
        // A Q6_K unit is one `(n, k)` quarter of `q6k_elem`'s indexing: `sub = 4n + k`.
        const int n = sub >> 2;
        const int k = sub & 3;
        const unsigned char *ql = blk + n * 64 + (k & 1) * 32;
        const unsigned char *qh = blk + 128 + n * 32;
        const signed char *sc = (const signed char *) (blk + 192);
        const float d = f16_to_f32(ld_u16le(blk + 208));
        const bool hi = k >= 2;
        const int shift = 2 * k;
        // ⚠️ The `hi` select is deliberately left **inside** the loop, unlike the Q4_K branch
        // above which hoists its equivalent into a branch on a scalar. Hoisting it here was
        // tried and is measurably worse: batched Q6_K went 4.13 -> 6.06 ps/element and the
        // unbatched path barely moved (13.05 -> 12.53). Do not "fix" this to match Q4_K.
        //
        // Q6_K scales one 16-element half at a time, so the unit splits in two.
        for (int half = 0; half < 2; ++half) {
            const float ds = d * (float) sc[n * 8 + 2 * k + half];
            for (int l = half * 16; l < half * 16 + 16; ++l) {
                const unsigned int byte = ql[l];
                const unsigned int lo = hi ? (byte >> 4) : (byte & 0xFu);
                const unsigned int hb = (qh[l] >> shift) & 3u;
                acc += (ds * (float) ((int) (lo | (hb << 4)) - 32)) * xs[l];
            }
        }
    } else if constexpr (TY == GGML_TYPE_MXFP4) {
        // One E8M0 scale for the whole 32, and the unit *is* the block. The two halves are
        // walked in one pass over `qs` rather than two, so each of the sixteen bytes is loaded
        // once; the per-element expression is `mxfp4_elem`'s, term for term.
        const float d = e8m0_half(blk[0]);
        const unsigned char *qs = blk + 1;
        for (int l = 0; l < 16; ++l) {
            const unsigned int b = qs[l];
            acc += ((float) KVALUES_MXFP4[b & 0xFu] * d) * xs[l];
            acc += ((float) KVALUES_MXFP4[b >> 4] * d) * xs[l + 16];
        }
    } else {  // Q8_0: no sub-block structure, one delta for the whole 32
        const float d = f16_to_f32(ld_u16le(blk));
        const signed char *q = (const signed char *) (blk + 2);
        for (int l = 0; l < 32; ++l) acc += ((float) q[l] * d) * xs[l];
    }
}

/// Rows one work-group covers.
///
/// 🔴 This is not about the weights. A matvec reads each weight byte exactly once; what it reads
/// `n_rows` times is the **activation vector**, and on these shapes that is the larger number by
/// an order of magnitude — an expert's gate matrix is 1.2 MiB against 8 MiB of re-read `x`. That
/// showed up as a matvec running at what looked like full memory bandwidth while moving almost
/// no weights: `attn_q`, `attn_k` and `attn_v` together measured 55 MiB of traffic in 121 us,
/// which is 456 GB/s on a card whose peak is 456 GB/s — and only 7 MiB of it was weights.
///
/// Eight rows to a work-group means one trip through `x` serves eight of them. The work-group is
/// eight sub-groups of `WG`, one row each, so the *total* number of work-items is unchanged and
/// so is occupancy; only the sharing changes.
static constexpr int MATVEC_ROWS = 8;

/// Submit a matvec against block-quantised weights: one sub-group per row, `MATVEC_ROWS` rows
/// to a work-group.
///
/// 🔴 The work split within a row is over **32-element units, not blocks**, and that is the
/// other half of the point. A block-per-lane split leaves a work-group mostly idle on the shapes
/// this model actually runs: an expert's `n_cols` is 2048, which is eight Q4_K super-blocks, so
/// eight of thirty-two lanes had work and twenty-four sat out. Splitting by unit gives sixty-four
/// pieces to thirty-two lanes.
template <unsigned int TY>
static event matvec_q_submit(queue &q, float *out, const void *w, const float *x,
                            unsigned long n_rows, unsigned long n_cols) {
    constexpr int BB = const_block_bytes<TY>();
    constexpr int UPB = units_per_block<TY>();
    const unsigned long nb = n_cols / (unsigned long) (UPB * 32);
    const unsigned long units = nb * (unsigned long) UPB;
    const unsigned long row_bytes = nb * (unsigned long) BB;
    const unsigned long groups = (n_rows + MATVEC_ROWS - 1) / (unsigned long) MATVEC_ROWS;
    const auto *base = static_cast<const unsigned char *>(w);
    return q.parallel_for(
        nd_range<1>{range<1>{groups * (MATVEC_ROWS * WG)}, range<1>{MATVEC_ROWS * WG}},
        [=](nd_item<1> it) [[sycl::reqd_sub_group_size(WG)]] {
            const auto sg = it.get_sub_group();
            const size_t row = it.get_group(0) * MATVEC_ROWS + sg.get_group_linear_id();
            const size_t lane = sg.get_local_linear_id();
            // A tail group has rows past the end. They read row 0 and throw the answer away,
            // rather than branching out — every lane has to reach the reduction below.
            const bool live = row < n_rows;
            const unsigned char *rowp = base + (live ? row : 0) * row_bytes;
            float acc = 0.0f;
            for (size_t u = lane; u < units; u += WG) {
                unit_acc<TY>(acc, rowp + (u / UPB) * (size_t) BB, x + u * 32, (int) (u % UPB));
            }
            const float total = reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0 && live) out[row] = total;
        });
}

/// Matrices one batched launch can cover.
///
/// The array below is captured into the kernel by value, so this bounds a kernel argument
/// rather than an allocation: 32 pointers is 256 bytes, well inside what Level Zero will take.
/// A caller with more matrices than this is refused rather than silently truncated.
static constexpr int MAX_BATCHED_MATS = 32;

/// The weight matrices of one batched launch, as a value the kernel can carry.
///
/// 🔴 By value on purpose. The alternative — a table in device memory — would have to be
/// uploaded before every launch, and `moearc_copy_h2d` waits, which on an in-order queue drains
/// everything already submitted. Passing the pointers as a kernel argument is ordered by
/// construction: there is no table for the kernel to race.
struct mat_table {
    const unsigned char *p[MAX_BATCHED_MATS];
};

/// The per-matrix bias row indices of one batched launch, carried by value for the same reason
/// `mat_table` is. See `moearc_add_bias_id`.
struct idx_table {
    unsigned int i[MAX_BATCHED_MATS];
};

/// The same product as `matvec_q_submit`, over `n_mat` matrices of one shape and type, in one
/// launch.
///
/// 🔴 This exists because of *parallelism*, not because of launch overhead. One expert's matvec
/// on this model is 1024 rows of 2048 columns — 32768 work-items, which is roughly one pass over
/// a B580's resident thread capacity. A kernel one wave deep has no second wave to run while the
/// first waits on memory, so it spends its life on load latency: measured, an expert matvec
/// moved 1.18 MB in 12 us, which is 98 GB/s on a card that reaches 456. Eight experts in one
/// launch is eight waves, and the tail of one hides under the head of the next.
///
/// `x_stride` is in elements. Zero means every matrix reads the same activation vector, which is
/// what the gate and up projections do; the down projection passes `n_ff` because each expert
/// consumes its own.
template <unsigned int TY>
static event matvec_q_batched_submit(queue &q, float *out, mat_table w, unsigned int n_mat,
                                    const float *x, unsigned long x_stride,
                                    unsigned long n_rows, unsigned long n_cols) {
    constexpr int BB = const_block_bytes<TY>();
    constexpr int UPB = units_per_block<TY>();
    const unsigned long nb = n_cols / (unsigned long) (UPB * 32);
    const unsigned long units = nb * (unsigned long) UPB;
    const unsigned long row_bytes = nb * (unsigned long) BB;
    const unsigned long per_mat = (n_rows + MATVEC_ROWS - 1) / (unsigned long) MATVEC_ROWS;
    const unsigned long groups = per_mat * (unsigned long) n_mat;
    return q.parallel_for(
        nd_range<1>{range<1>{groups * (MATVEC_ROWS * WG)}, range<1>{MATVEC_ROWS * WG}},
        [=](nd_item<1> it) [[sycl::reqd_sub_group_size(WG)]] {
            const auto sg = it.get_sub_group();
            const size_t g = it.get_group(0);
            // The matrix index is uniform across the work-group, so the indirection into the
            // captured table is one load per group rather than one per lane.
            const size_t mat = g / per_mat;
            const size_t row = (g - mat * per_mat) * MATVEC_ROWS + sg.get_group_linear_id();
            const size_t lane = sg.get_local_linear_id();
            const bool live = row < n_rows;
            const unsigned char *rowp = w.p[mat] + (live ? row : 0) * row_bytes;
            const float *xs = x + mat * x_stride;
            float acc = 0.0f;
            for (size_t u = lane; u < units; u += WG) {
                unit_acc<TY>(acc, rowp + (u / UPB) * (size_t) BB, xs + u * 32, (int) (u % UPB));
            }
            const float total = reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0 && live) out[mat * n_rows + row] = total;
        });
}

/// The unhoisted path: one element at a time through `elem_at`, for the formats that have no
/// 32-element unit to hoist anything out of (f16, f32). Shared by the single and batched entry
/// points so the two cannot drift.
static event matvec_generic_submit(queue &q, unsigned int type_id, int bb, float *out,
                                  const void *w, const float *x, unsigned long n_rows,
                                  unsigned long n_cols) {
    const auto *base = static_cast<const unsigned char *>(w);
    return q.parallel_for(nd_range<1>{range<1>{n_rows * WG}, range<1>{WG}}, [=](nd_item<1> it) {
        const size_t row = it.get_group(0);
        const size_t lid = it.get_local_id(0);
        float acc = 0.0f;
        for (size_t i = lid; i < n_cols; i += WG) {
            acc += elem_at(type_id, base + (row * n_cols + i) * (size_t) bb, 0) * x[i];
        }
        const float total = reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        if (lid == 0) out[row] = total;
    });
}

extern "C" {

/// One kernel's accumulated device time, keyed by kernel and shape.
struct kernel_time {
    std::string key;
    unsigned long long ns;
    unsigned long long calls;
};

struct moearc_ctx {
    queue q;
    /// Whether the queue was built with `enable_profiling`. See `moearc_ctx_create`.
    bool profiling = false;
    /// Events submitted but not yet folded into `totals`. Drained lazily, never waited on
    /// early: reading a profiling timestamp requires the event to have completed, so draining
    /// at submission time would reintroduce exactly the synchronisation this exists to avoid.
    std::vector<std::pair<std::string, event>> pending;
    std::vector<kernel_time> totals;
};

/// Fold every completed event into `totals`.
///
/// 🔴 Called only from `moearc_profile_events_report`, i.e. after the caller has already
/// finished the work it wants to measure. `get_profiling_info` blocks until the event
/// completes, so calling this mid-stream would serialise the queue and produce the same
/// distortion `MOEARC_SYNC_EACH` produces.
static void moearc_flush_events(moearc_ctx *c) {
    for (auto &pe : c->pending) {
        unsigned long long ns = 0;
        try {
            const auto t0 = pe.second.get_profiling_info<info::event_profiling::command_start>();
            const auto t1 = pe.second.get_profiling_info<info::event_profiling::command_end>();
            ns = (t1 > t0) ? (unsigned long long) (t1 - t0) : 0ull;
        } catch (...) {
            continue;  // a backend that will not profile this event is skipped, not guessed at
        }
        bool found = false;
        for (auto &kt : c->totals) {
            if (kt.key == pe.first) {
                kt.ns += ns;
                kt.calls += 1;
                found = true;
                break;
            }
        }
        if (!found) c->totals.push_back(kernel_time{pe.first, ns, 1});
    }
    c->pending.clear();
}

/// Record one submission's event under a key that carries the kernel's shape.
///
/// The key carries the quantisation **and** the shape on purpose. Every quantised mat-vec in
/// this engine goes through one of two entry points, so without the shape `out.matvec`,
/// `attn.qkv` and `attn.proj` collapse into one number; and without the type, a bank that is
/// Q6_K in half its blocks and Q4_K in the other half -- which is every expert `down` bank and
/// every `attn_v` in this file -- reports the **average** of two kernels that differ by 2x. That
/// average is exactly what made Q6_K look fine in `expert_down` and slow in `lm_head`.
static void moearc_track(moearc_ctx *c, const char *what, unsigned long n_rows, unsigned int n_mat,
                         const event &e) {
    if (!c->profiling) return;
    char key[96];
    std::snprintf(key, sizeof(key), "%s r%lu m%u", what, n_rows, n_mat);
    c->pending.emplace_back(std::string(key), e);
    // A token submits a few hundred events; a long run would otherwise hold every one of them
    // alive. Anything this far back is long complete, so folding it in costs nothing.
    if (c->pending.size() >= 16384) moearc_flush_events(c);
}

// ---- lifecycle ------------------------------------------------------------------------
moearc_ctx *moearc_ctx_create() {
    try {
        // 🔴 In-order, and that is load-bearing. The kernels below submit and return
        // without waiting, so what keeps a matmul from reading a buffer another kernel is
        // still writing is the queue's ordering, not a synchronisation after every launch.
        // An out-of-order queue with these submissions would produce output that is finite,
        // fluent and wrong.
        // `MOEARC_PROFILE_EVENTS=1` adds `enable_profiling`, which makes every submission
        // carry device timestamps. That is the only way to attribute device time per kernel
        // **without** serialising -- `MOEARC_SYNC_EACH` buys the same attribution by waiting
        // after every launch, which destroys exactly the overlap it is then used to reason
        // about. Off by default because enabling profiling on a queue is not free.
        const char *ev = std::getenv("MOEARC_PROFILE_EVENTS");
        const bool want_profiling = ev != nullptr && ev[0] == '1';
        auto *c = want_profiling
                      ? new moearc_ctx{queue{gpu_selector_v,
                                             property_list{property::queue::in_order(),
                                                           property::queue::enable_profiling()}}}
                      : new moearc_ctx{queue{gpu_selector_v, property::queue::in_order()}};
        c->profiling = want_profiling;
        return c;
    } catch (...) {
        return nullptr;
    }
}

void moearc_ctx_destroy(moearc_ctx *c) { delete c; }

// ---- event profiling -------------------------------------------------------------------
//
// Per-kernel device time, taken from the SYCL events themselves, on a queue that is still
// asynchronous. `MOEARC_SYNC_EACH` answers the same question by waiting after every launch,
// which is sound for attribution and useless for anything that depends on kernels overlapping.
// This is the instrument to prefer; see `moearc_ctx_create`.

int moearc_profile_events_enabled(moearc_ctx *c) { return (c && c->profiling) ? 1 : 0; }

int moearc_profile_events_reset(moearc_ctx *c) {
    if (!c) return ERR;
    c->pending.clear();
    c->totals.clear();
    return OK;
}

// Write one `key nanoseconds calls` line per kernel shape into `out`.
//
// Drains outstanding events first, which blocks until they complete -- call it after the work
// being measured, never inside it.
int moearc_profile_events_report(moearc_ctx *c, char *out, unsigned long cap) {
    if (!c || !out || cap == 0) return ERR;
    if (!c->profiling) {
        out[0] = '\0';
        return OK;
    }
    try {
        moearc_flush_events(c);
        unsigned long used = 0;
        for (const auto &kt : c->totals) {
            char line[160];
            const int n = std::snprintf(line, sizeof(line), "%s %llu %llu\n", kt.key.c_str(),
                                        kt.ns, kt.calls);
            if (n <= 0 || used + (unsigned long) n + 1 > cap) break;
            std::memcpy(out + used, line, (unsigned long) n);
            used += (unsigned long) n;
        }
        out[used] = '\0';
        return OK;
    } catch (...) { return ERR; }
}


// Block until everything submitted to the queue has finished.
//
// The queue is in-order, so this is also the point at which an exception thrown by any kernel
// submitted since the last synchronisation is delivered. Kernels below therefore return OK for
// "accepted", not for "completed" — the completion verdict arrives here, or at the next
// device-to-host copy, which waits for the same reason.
int moearc_sync(moearc_ctx *c) {
    if (!c) return ERR;
    try { c->q.wait_and_throw(); return OK; } catch (...) { return ERR; }
}

int moearc_device_name(moearc_ctx *c, char *out, unsigned long cap) {
    if (!c || !out || cap == 0) return -1;
    try {
        auto n = c->q.get_device().get_info<info::device::name>();
        std::strncpy(out, n.c_str(), cap - 1);
        out[cap - 1] = '\0';
        return 0;
    } catch (...) { return -1; }
}

// ---- device memory --------------------------------------------------------------------
void *moearc_alloc_device(moearc_ctx *c, unsigned long bytes) {
    if (!c) return nullptr;
    try { return malloc_device(bytes, c->q); } catch (...) { return nullptr; }
}

void moearc_free_device(moearc_ctx *c, void *p) {
    if (c && p) free(p, c->q);
}

int moearc_copy_h2d(moearc_ctx *c, void *dst, const void *src, unsigned long bytes) {
    if (!c) return -1;
    try { c->q.memcpy(dst, src, bytes).wait_and_throw(); return 0; } catch (...) { return -1; }
}

int moearc_copy_d2h(moearc_ctx *c, void *dst, const void *src, unsigned long bytes) {
    if (!c) return -1;
    try { c->q.memcpy(dst, src, bytes).wait_and_throw(); return 0; } catch (...) { return -1; }
}

// Host-to-device copy that submits and returns.
//
// 🔴 **The mechanism is not what it looks like, and the obvious explanation is wrong.**
//
// It is tempting to say the blocking `moearc_copy_h2d` "drains the queue", which it does in
// general. But trace the in-order queue through one block of `moearc-engine`'s decode: the
// router's top-k is submitted, then its result is **read back** — and that read drains
// everything. Admission is host-only. So by the time staging runs **the queue is already
// empty**, and each blocking upload waits for nothing but itself.
//
// What the wait actually cost was **overlap**. Serialised, copy n+1 could not be submitted
// until copy n had landed, so the copy engine never had a queue to stream and the host never
// ran ahead into the next block's kernels. Measured on Qwen3-30B-A3B at 2952 resident slots,
// 95 steady-state tokens, changing only this:
//
//     moe.stage       17.33 -> 10.11 ms/token
//     decode.total    43.19 -> 37.51 ms/token
//     throughput      22.87 -> 26.28 tok/s          (+15%)
//
// with every greedy token id, every cache hit rate and every staged-byte count unchanged.
//
// ⚠️ **Do not compare what remains against the 13.4 GB/s in `docs/roadmap.md`.** That figure
// comes from `tools/stream_bench.cpp`, which allocates its host side with `malloc_host` — it is
// a **pinned** number. This path copies out of a memory-mapped GGUF: ordinary pageable,
// file-backed pages, which Level Zero cannot DMA from directly and must bounce through a
// driver staging buffer. Measured here, per ~930 KiB bank copy: **136 us at a 93% hit rate,
// falling to 91 us at 0%** as the copies pipeline — an effective 6.7 GB/s and 10.5 GB/s
// respectively, against a phase that is now within ~20% of the pinned link rate at volume.
// Closing that last gap means a pinned staging ring, and the arithmetic does not obviously
// favour one: an extra mmap->pinned host copy at the measured 22.8 GB/s costs more than it
// saves unless the pageable path is worse than 10.5 GB/s. Measure before building it.
//
// **Ordering is preserved by the queue, not by the wait.** The memcpy is submitted before the
// matvec that reads the slot, so the matvec runs after it — the same argument every kernel in
// this file already relies on, and the reason `moearc_ctx_create` asks for `in_order()`.
//
// ⚠️ Two things the caller takes on, and both are real:
//
//   - **`src` must stay alive and unmodified until the copy completes**, and the caller cannot
//     see when that is. `moearc-engine`'s `stage()` copies straight out of the memory-mapped
//     GGUF, which outlives every buffer here; `session.rs` makes that ordering explicit rather
//     than incidental, and says why. A caller copying from a temporary, a stack array, or a
//     buffer it is about to reuse **must** use the blocking `moearc_copy_h2d`. Nothing in the
//     type system stops it; that comment and this one are the only guard.
//   - **Failures surface later, at the next synchronisation.** A pool that over-committed
//     device memory used to fail on the copy itself with a legible message; it will now fail on
//     whichever kernel or readback comes next.
int moearc_copy_h2d_async(moearc_ctx *c, void *dst, const void *src, unsigned long bytes) {
    if (!c) return -1;
    try { c->q.memcpy(dst, src, bytes); return 0; } catch (...) { return -1; }
}

// ---- the first real kernel ------------------------------------------------------------
// Gather `count` expert slots of `slot_bytes` each from a resident pool into a packed
// staging buffer, given their slot indices.
//
// This is not a toy. It is the operation the residency cache performs on every token: the
// router names 8 experts per block, some resident, and their weights must be presented
// contiguously to the matmul. Doing it as a device-side gather avoids a round trip per
// expert, which at 320 activations per token is the difference between one launch and 320.
int moearc_gather_experts(moearc_ctx *c, void *dst, const void *pool, const unsigned int *idx,
                          unsigned int count, unsigned long slot_bytes) {
    if (!c || !dst || !pool || !idx) return -1;
    try {
        // Copy indices to the device so the kernel can read them.
        unsigned int *d_idx = malloc_device<unsigned int>(count, c->q);
        if (!d_idx) return -1;
        c->q.memcpy(d_idx, idx, count * sizeof(unsigned int)).wait_and_throw();

        const unsigned long words = slot_bytes / sizeof(unsigned int);
        auto *d_dst = static_cast<unsigned int *>(dst);
        const auto *d_pool = static_cast<const unsigned int *>(pool);

        c->q.parallel_for(range<2>{count, words}, [=](id<2> it) {
            const unsigned long slot = it[0];
            const unsigned long w = it[1];
            d_dst[slot * words + w] = d_pool[(unsigned long)d_idx[slot] * words + w];
        }).wait_and_throw();

        free(d_idx, c->q);
        return 0;
    } catch (...) { return -1; }
}

// ---- dequantisation -------------------------------------------------------------------
// Expand `nblocks` block-quantised blocks into `nblocks * block_elems(type)` f32.
//
// One work-item per output element. Each recomputes its block's scales, which is redundant
// work — 256 work-items per block repeat the same two f16 conversions and the same 6-bit
// scale unpack. That is a deliberate first-version trade: the element formula lives in one
// function, `qXk_elem`, which the matvec kernels below also call, so the two paths cannot
// drift apart. Hoisting the per-block constants is the obvious optimisation and is not done
// yet.
int moearc_dequant(moearc_ctx *c, unsigned int type_id, float *dst, const void *src,
                   unsigned long nblocks) {
    if (!c || !dst || !src) return ERR;
    const int bb = block_bytes(type_id);
    if (bb == 0) return ERR_ARG;
    if (nblocks == 0) return OK;
    const int be = block_elems(type_id);
    try {
        const auto *base = static_cast<const unsigned char *>(src);
        c->q.parallel_for(range<1>{nblocks * (unsigned long) be}, [=](id<1> it) {
            const size_t g = it[0];
            const size_t b = g / be;
            const int i = (int) (g % be);
            dst[g] = elem_at(type_id, base + b * (size_t) bb, i);
        });
        return OK;
    } catch (...) { return ERR; }
}

// ---- matrix-vector against quantised weights ------------------------------------------
// out[row] = sum_col W[row][col] * x[col], with W stored as GGUF stores it: row-major, each
// row an independent run of `n_cols / 256` super-blocks.
//
// Decode is matvec-dominated — one token in, one row of activations — so this, not a GEMM,
// is the shape that matters first. The weights are never materialised as f32: each work-item
// dequantises the elements it needs as it consumes them, so the only traffic is the
// quantised bytes. A dequantise-then-GEMM path would move 8x the bytes and need a scratch
// buffer the size of the expert, which on a 12 GB card is the difference between an expert
// fitting and not.
//
// 🔴 Unoptimised, and knowingly so. One work-group of 32 per row, strided over the row's
// blocks, then a group reduction. Each element re-derives its block's scales (see
// `moearc_dequant`); there is no sub-group shuffle, no vectorised load, no dot-product
// instruction. Correctness first.
//
// The reduction is a tree, so the summation order differs from a sequential CPU dot product
// and the results will not be bit-identical. See the tolerance note in `tests/`.
int moearc_matvec_q(moearc_ctx *c, unsigned int type_id, float *out, const void *w,
                    const float *x, unsigned long n_rows, unsigned long n_cols) {
    if (!c || !out || !w || !x) return ERR;
    const int bb = block_bytes(type_id);
    if (bb == 0) return ERR_ARG;
    const int be = block_elems(type_id);
    if (n_cols % (unsigned long) be != 0) return ERR_ARG;
    if (n_rows == 0) return OK;
    try {
        switch (type_id) {
            case GGML_TYPE_Q4_K:
                moearc_track(c, "mvq Q4_K", n_rows, 1,
                             matvec_q_submit<GGML_TYPE_Q4_K>(c->q, out, w, x, n_rows, n_cols));
                break;
            case GGML_TYPE_Q5_K:
                moearc_track(c, "mvq Q5_K", n_rows, 1,
                             matvec_q_submit<GGML_TYPE_Q5_K>(c->q, out, w, x, n_rows, n_cols));
                break;
            case GGML_TYPE_Q6_K: {
                // 🔴 Q6_K goes through the **batched** kernel with a single matrix, and that is
                // not a tidying-up — it is worth 3.2x.
                //
                // The two kernels have token-identical inner loops. They differ only in how the
                // row pointer is formed: `base + row * row_bytes` from a kernel argument here,
                // against `w.p[mat] + row * row_bytes` through a by-value table there. On Q6_K,
                // and *only* on Q6_K, that is the difference between ~13.1 and ~4.1
                // ps/element -- flat across every `n_cols` from 256 to 4096. Q4_K and Q5_K are
                // unaffected by the same swap, and Q5_K reads two byte streams exactly as Q6_K
                // does, so it is neither "Q6_K is expensive" nor "two streams break coalescing".
                //
                // ⚠️ **The source-level cause is not established.** Three hypotheses were
                // measured and refuted: `n_cols` (flat), the second byte stream (Q5_K is fine),
                // and the in-loop nibble select (hoisting it made things worse -- see the note
                // in `unit_acc`). What remains is an IGC codegen difference that an opaque
                // pointer suppresses. It is characterised, reproducible via
                // `examples/matvec_scaling`, and this is a workaround, not a fix.
                //
                // Routing only Q6_K, because the swap is *not* free at every shape: at
                // `n_cols = 512` the batched kernel costs Q4_K 4.98 -> 7.57 ps/element. No
                // shape in this engine uses it, but a future one might.
                mat_table t{};
                t.p[0] = static_cast<const unsigned char *>(w);
                moearc_track(c, "mvq Q6_K", n_rows, 1,
                             matvec_q_batched_submit<GGML_TYPE_Q6_K>(c->q, out, t, 1, x, 0,
                                                                     n_rows, n_cols));
                break;
            }
            case GGML_TYPE_Q8_0:
                moearc_track(c, "mvq Q8_0", n_rows, 1,
                             matvec_q_submit<GGML_TYPE_Q8_0>(c->q, out, w, x, n_rows, n_cols));
                break;
            case GGML_TYPE_MXFP4:
                moearc_track(c, "mvq MXFP4", n_rows, 1,
                             matvec_q_submit<GGML_TYPE_MXFP4>(c->q, out, w, x, n_rows, n_cols));
                break;
            default:
                // f32 and f16 are "blocks" of one element, so there is no unit to hoist
                // anything out of and no constraint that `n_cols` be a multiple of 32. One
                // element per step, straight through `elem_at`.
                moearc_track(c, "mv_generic", n_rows, 1,
                             matvec_generic_submit(c->q, type_id, bb, out, w, x, n_rows, n_cols));
                break;
        }
        return OK;
    } catch (...) { return ERR; }
}

// The same operation against unquantised weights. Not a fallback for the above — it is what
// the small dense projections (norm weights, the router itself) need, and it is the control
// the quantised path is measured against.
int moearc_matvec_f32(moearc_ctx *c, float *out, const float *w, const float *x,
                      unsigned long n_rows, unsigned long n_cols) {
    if (!c || !out || !w || !x) return ERR;
    if (n_rows == 0) return OK;
    try {
        c->q.parallel_for(
               nd_range<1>{range<1>{n_rows * WG}, range<1>{WG}},
               [=](nd_item<1> it) {
                   const size_t row = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   float acc = 0.0f;
                   for (size_t i = lid; i < n_cols; i += WG) {
                       acc += w[row * n_cols + i] * x[i];
                   }
                   const float total = reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                   if (lid == 0) out[row] = total;
               });
        return OK;
    } catch (...) { return ERR; }
}

// The same product over several weight matrices at once.
//
// `w` is a host array of `n_mat` device pointers, one weight matrix each, all of `type_id` and
// all `n_rows` x `n_cols`. Matrix `m` writes `out[m * n_rows .. (m+1) * n_rows]` and reads
// `x + m * x_stride`; `x_stride == 0` shares one activation vector between all of them.
//
// This is the MoE FFN's shape. The router names k experts and each is a matrix of its own, so
// the unbatched version is k launches of a kernel that is one wave deep — see
// `matvec_q_batched_submit` for what that costs.
int moearc_matvec_q_batched(moearc_ctx *c, unsigned int type_id, float *out,
                            const void *const *w, unsigned int n_mat, const float *x,
                            unsigned long x_stride, unsigned long n_rows,
                            unsigned long n_cols) {
    if (!c || !out || !w || !x) return ERR;
    if (n_mat == 0 || n_rows == 0) return OK;
    if (n_mat > (unsigned int) MAX_BATCHED_MATS) return ERR_ARG;
    const int bb = block_bytes(type_id);
    if (bb == 0) return ERR_ARG;
    const int be = block_elems(type_id);
    if (n_cols % (unsigned long) be != 0) return ERR_ARG;
    mat_table t{};
    for (unsigned int i = 0; i < n_mat; ++i) {
        if (!w[i]) return ERR;
        t.p[i] = static_cast<const unsigned char *>(w[i]);
    }
    // Every unused entry aims at matrix 0. Nothing reads them — `mat < n_mat` by construction —
    // but a table of uninitialised pointers is a bad thing to hand a kernel.
    for (int i = (int) n_mat; i < MAX_BATCHED_MATS; ++i) t.p[i] = t.p[0];
    try {
        switch (type_id) {
            case GGML_TYPE_Q4_K:
                moearc_track(c, "mvq_batched Q4_K", n_rows, n_mat,
                             matvec_q_batched_submit<GGML_TYPE_Q4_K>(
                                 c->q, out, t, n_mat, x, x_stride, n_rows, n_cols));
                break;
            case GGML_TYPE_Q5_K:
                moearc_track(c, "mvq_batched Q5_K", n_rows, n_mat,
                             matvec_q_batched_submit<GGML_TYPE_Q5_K>(
                                 c->q, out, t, n_mat, x, x_stride, n_rows, n_cols));
                break;
            case GGML_TYPE_Q6_K:
                moearc_track(c, "mvq_batched Q6_K", n_rows, n_mat,
                             matvec_q_batched_submit<GGML_TYPE_Q6_K>(
                                 c->q, out, t, n_mat, x, x_stride, n_rows, n_cols));
                break;
            case GGML_TYPE_Q8_0:
                moearc_track(c, "mvq_batched Q8_0", n_rows, n_mat,
                             matvec_q_batched_submit<GGML_TYPE_Q8_0>(
                                 c->q, out, t, n_mat, x, x_stride, n_rows, n_cols));
                break;
            case GGML_TYPE_MXFP4:
                moearc_track(c, "mvq_batched MXFP4", n_rows, n_mat,
                             matvec_q_batched_submit<GGML_TYPE_MXFP4>(
                                 c->q, out, t, n_mat, x, x_stride, n_rows, n_cols));
                break;
            default:
                // No unit structure to batch over; issue the generic path per matrix. Correct,
                // and no slower than the unbatched call it replaces.
                for (unsigned int m = 0; m < n_mat; ++m) {
                    moearc_track(c, "mv_generic", n_rows, 1,
                                 matvec_generic_submit(c->q, type_id, bb,
                                                       out + (size_t) m * n_rows, t.p[m],
                                                       x + (size_t) m * x_stride, n_rows,
                                                       n_cols));
                }
                break;
        }
        return OK;
    } catch (...) { return ERR; }
}

// The MoE combine: out[i] = sum over m of weights[m] * parts[m * n + i].
//
// One launch in place of the k `moearc_axpy` calls it replaces, and `moearc_zero` with them —
// this writes rather than accumulates, so the destination does not have to be cleared first.
//
// 🔴 The accumulation runs m ascending from 0.0f, which is exactly the order a zero followed by
// k axpys visited. Floating-point addition is not associative and this model's greedy output is
// asserted token for token against llama.cpp's, so the order is a correctness property, not a
// detail.
//
// `weights` is read on the device. It is the router's own output, still where `moearc_topk_router`
// left it, so nothing has to travel to the host and back for the combine to know its scalars.
int moearc_moe_combine(moearc_ctx *c, float *out, const float *parts, const float *weights,
                       unsigned int n_mat, unsigned long n) {
    if (!c || !out || !parts || !weights) return ERR;
    if (n == 0) return OK;
    if (n_mat > (unsigned int) MAX_BATCHED_MATS) return ERR_ARG;
    try {
        const unsigned int k = n_mat;
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            const size_t i = it[0];
            float acc = 0.0f;
            for (unsigned int m = 0; m < k; ++m) acc += weights[m] * parts[(size_t) m * n + i];
            out[i] = acc;
        });
        return OK;
    } catch (...) { return ERR; }
}

// ---- normalisation, activation, attention pieces --------------------------------------
// RMSNorm over the last axis: y = x * rsqrt(mean(x^2) + eps), optionally scaled by a
// per-column weight. `weight` may be null, which is the bare normalisation ggml's
// `ggml_rms_norm` computes before its separate `ggml_mul`.
//
// 🔴 The sum of squares accumulates in f32. ggml accumulates it in `ggml_float`, which is
// double. That is a real difference and it is deliberate: fp64 on Arc is emulated where it
// exists at all, and a kernel that needs it would be unusable. The tests quantify the gap
// against an f64 CPU reference rather than assuming it away.
int moearc_rmsnorm(moearc_ctx *c, float *out, const float *x, const float *weight,
                   unsigned long n_rows, unsigned long n_cols, float eps) {
    if (!c || !out || !x) return ERR;
    if (n_cols == 0) return ERR_ARG;
    if (n_rows == 0) return OK;
    try {
        const bool have_w = weight != nullptr;
        const size_t wg = norm_wg(n_cols);
        c->q.parallel_for(
               nd_range<1>{range<1>{n_rows * wg}, range<1>{wg}},
               [=](nd_item<1> it) {
                   const size_t row = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   const size_t width = it.get_local_range(0);
                   const float *xr = x + row * n_cols;
                   float *outr = out + row * n_cols;

                   float ss = 0.0f;
                   for (size_t i = lid; i < n_cols; i += width) ss += xr[i] * xr[i];
                   ss = reduce_over_group(it.get_group(), ss, sycl::plus<float>());

                   const float scale = sycl::rsqrt(ss / (float) n_cols + eps);
                   for (size_t i = lid; i < n_cols; i += width) {
                       const float v = xr[i] * scale;
                       outr[i] = have_w ? v * weight[i] : v;
                   }
               });
        return OK;
    } catch (...) { return ERR; }
}

// SiLU (a.k.a. swish): x / (1 + exp(-x)). ggml's `ggml_silu_f32` verbatim.
int moearc_silu(moearc_ctx *c, float *out, const float *x, unsigned long n) {
    if (!c || !out || !x) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            const float v = x[it[0]];
            out[it[0]] = v / (1.0f + sycl::exp(-v));
        });
        return OK;
    } catch (...) { return ERR; }
}

// SwiGLU: silu(gate) * up, the gated FFN activation every expert in a modern MoE uses.
// Fused because the two halves are produced by two matvecs and consumed once; materialising
// silu(gate) separately doubles the traffic for nothing.
int moearc_swiglu(moearc_ctx *c, float *out, const float *gate, const float *up,
                  unsigned long n) {
    if (!c || !out || !gate || !up) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            const float g = gate[it[0]];
            out[it[0]] = (g / (1.0f + sycl::exp(-g))) * up[it[0]];
        });
        return OK;
    } catch (...) { return ERR; }
}

// `silu(gu[i]) * gu[n + i]` — the same activation as `moearc_swiglu`, for the case where both
// halves came out of one launch and are laid end to end in one buffer.
//
// It exists so the gate and up projections can be batched together: they read the same
// activation vector and differ only in their weights, so k experts x 2 banks is one launch of
// 2k matrices — but one launch writes one buffer, and the halves of that buffer are what this
// reads. The arithmetic is `moearc_swiglu`'s, term for term.
int moearc_swiglu_halves(moearc_ctx *c, float *out, const float *gu, unsigned long n) {
    if (!c || !out || !gu) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            const float g = gu[it[0]];
            out[it[0]] = (g / (1.0f + sycl::exp(-g))) * gu[n + it[0]];
        });
        return OK;
    } catch (...) { return ERR; }
}

// Row-wise softmax of `x * scale + mask`, max-subtracted.
//
// This is `ggml_soft_max_ext(a, mask, scale, 0.0f)` — the operation llama.cpp's attention
// actually uses, not a bare softmax. Both extras are load-bearing: `scale` folds in the
// 1/sqrt(head_dim) that would otherwise need its own pass over the scores, and `mask` is how
// causality is expressed. A causal mask holds 0 where a key is visible and -inf where it is
// not, so the softmax assigns it exactly zero weight.
//
// `mask` may be null, and with `scale = 1` this degenerates to the plain row softmax the
// router uses.
//
// The max subtraction is not an optimisation. Attention logits and router logits both reach
// magnitudes where a bare exp overflows to inf and the whole row comes back NaN.
//
// 🔴 A fully-masked row is a real possibility (a padded batch slot) and would divide by zero.
// The sum is clamped to the smallest normal f32 first, which turns that row into zeros rather
// than NaNs — a NaN would propagate through the rest of the network and destroy every other
// row's output too.
int moearc_softmax(moearc_ctx *c, float *out, const float *x, const float *mask,
                   unsigned long n_rows, unsigned long n_cols, float scale) {
    if (!c || !out || !x) return ERR;
    if (n_cols == 0) return ERR_ARG;
    if (n_rows == 0) return OK;
    try {
        const bool have_mask = mask != nullptr;
        c->q.parallel_for(
               nd_range<1>{range<1>{n_rows * WG}, range<1>{WG}},
               [=](nd_item<1> it) {
                   const size_t row = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   const float *xr = x + row * n_cols;
                   const float *mr = have_mask ? mask + row * n_cols : nullptr;
                   float *outr = out + row * n_cols;

                   float mx = -3.402823466e+38f;  // -FLT_MAX; a valid identity for max
                   for (size_t i = lid; i < n_cols; i += WG) {
                       const float v = xr[i] * scale + (have_mask ? mr[i] : 0.0f);
                       mx = sycl::fmax(mx, v);
                   }
                   mx = reduce_over_group(it.get_group(), mx, sycl::maximum<float>());

                   float sum = 0.0f;
                   for (size_t i = lid; i < n_cols; i += WG) {
                       sum += sycl::exp(xr[i] * scale + (have_mask ? mr[i] : 0.0f) - mx);
                   }
                   sum = reduce_over_group(it.get_group(), sum, sycl::plus<float>());

                   const float inv = 1.0f / sycl::fmax(sum, 1.175494351e-38f);
                   for (size_t i = lid; i < n_cols; i += WG) {
                       outr[i] = sycl::exp(xr[i] * scale + (have_mask ? mr[i] : 0.0f) - mx) * inv;
                   }
               });
        return OK;
    } catch (...) { return ERR; }
}

// Rotary position embedding over a [n_tokens][n_heads][head_dim] tensor, head_dim fastest.
//
// Two conventions, and they are not interchangeable — applying the wrong one produces a model
// that emits fluent nonsense rather than an error. Source for both:
// `ggml/src/ggml-cpu/ops.cpp`, `rotate_pairs()` and `ggml_rope_cache_init()`.
//   neox = 0 (GGML_ROPE_TYPE_NORMAL): pair (2i, 2i+1) — adjacent elements.
//   neox = 1 (GGML_ROPE_TYPE_NEOX):   pair (i, i + n_dims/2) — halves of the head.
// Channels at or above `n_dims` are copied through unrotated, as ggml does.
//
// One divergence from ggml's *CPU* path, and it is the same divergence llama.cpp's own GPU
// backends make. `ggml_rope_cache_init` builds its angle table by repeated multiplication —
// `theta *= theta_scale` down the row — which is inherently sequential. A work-item cannot do
// that, so this uses the closed form `pos * theta_scale^(i0/2)`, which is exactly what
// `ggml/src/ggml-sycl/rope.cpp` computes (`pos[i2] * dpct::pow(theta_scale, iw / 2.0f)`, with
// `theta_scale = powf(freq_base, -2.0f/n_dims)` evaluated on the host). The two are
// mathematically equal and differ in the last bits, and that difference grows with the
// position, because a large angle has a large ulp. The CPU reference in `reference.rs`
// deliberately keeps ggml's iterated form so the test measures the gap rather than hiding it.
int moearc_rope(moearc_ctx *c, float *dst, const float *src, const int *pos,
                unsigned long n_tokens, unsigned long n_heads, unsigned long head_dim,
                unsigned long n_dims, float freq_base, float freq_scale, float ext_factor,
                float attn_factor, float corr_lo, float corr_hi, int neox) {
    if (!c || !dst || !src || !pos) return ERR;
    if (n_dims == 0 || n_dims % 2 != 0 || n_dims > head_dim) return ERR_ARG;
    if (n_tokens == 0 || n_heads == 0) return OK;
    try {
        const unsigned long half = n_dims / 2;
        // Host-side, exactly as `ggml_sycl_rope` computes it before launching.
        const float theta_scale = std::pow(freq_base, -2.0f / (float) n_dims);
        c->q.parallel_for(range<1>{n_tokens * n_heads * head_dim}, [=](id<1> it) {
            const size_t g = it[0];
            const size_t d = g % head_dim;
            const size_t h = (g / head_dim) % n_heads;
            const size_t t = g / (head_dim * n_heads);
            const float *s = src + (t * n_heads + h) * head_dim;
            float *o = dst + (t * n_heads + h) * head_dim;

            if (d >= n_dims) {  // untouched channels ride along unchanged
                o[d] = s[d];
                return;
            }

            // `i0` is ggml's index into its angle cache: the even element of the pair.
            size_t i0, lo, hi;
            bool is_lo;
            if (neox) {
                is_lo = d < half;
                lo = is_lo ? d : d - half;
                hi = lo + half;
                i0 = 2 * lo;
            } else {
                is_lo = (d % 2) == 0;
                lo = is_lo ? d : d - 1;
                hi = lo + 1;
                i0 = lo;
            }

            // `rope_yarn` from `ggml/src/ggml-cpu/ops.cpp`, transcribed.
            //
            // 🔴 There is **no position gate**. `theta_interp = freq_scale * theta_extrap` runs
            // before the branch, and the only thing the ramp consults is `i0` — the *dimension*.
            // A model with `rope.scaling.type = yarn` is scaled from position 0; there is no
            // regime below `original_context_length` in which plain RoPE is equivalent.
            //
            // 🔴 The magnitude scaling is likewise unconditional whenever `ext_factor != 0`, and
            // it is applied **here**, not by the caller: llama.cpp computes
            // `cparams.yarn_attn_factor` and then divides it by `1 + 0.1*ln(1/freq_scale)`
            // precisely so this line can multiply it back. The value arriving as `attn_factor`
            // is 1.0 for gpt-oss and the effective mscale is 1.3466. Passing the paper's mscale
            // in from outside squares it.
            //
            // With `freq_scale = 1` and `ext_factor = 0` — every non-YaRN model — this is
            // `theta_extrap` and a multiply by 1.0f, both exact, so no existing model moves.
            const float theta_extrap = (float) pos[t] * sycl::pow(theta_scale, (float) i0 / 2.0f);
            float theta = freq_scale * theta_extrap;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float y = ((float) (i0 / 2) - corr_lo) / sycl::fmax(0.001f, corr_hi - corr_lo);
                const float ramp = 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
                const float ramp_mix = ramp * ext_factor;
                theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
            }
            const float ct = sycl::cos(theta) * mscale;
            const float st = sycl::sin(theta) * mscale;
            const float x0 = s[lo];
            const float x1 = s[hi];
            o[d] = is_lo ? (x0 * ct - x1 * st) : (x0 * st + x1 * ct);
        });
        return OK;
    } catch (...) { return ERR; }
}

// ---- router ---------------------------------------------------------------------------
// Top-k expert selection: given `n_expert` logits per token, produce the k chosen expert
// indices and their weights.
//
// Semantics follow `llm_graph_context::build_moe_ffn` in llama.cpp's `src/llama-graph.cpp`
// for the softmax-gated, normalised case (Qwen3's path): softmax over *all* experts, take the
// k largest, then if `normalize` divide by their sum. The sum is clamped to 6.103515625e-5 —
// the smallest normal f16 — before the division, which is llama.cpp's guard against a
// division by zero, transcribed rather than invented.
//
// Selection is by logit rather than by probability. Softmax is strictly monotonic, so the two
// orderings are identical, and comparing logits avoids the case where several experts'
// probabilities round to the same f32 and the ordering becomes an artefact of the exp.
//
// Ties break toward the lower expert index, so the same logits always name the same experts.
// That matters more than it looks: expert choice drives which weights get paged in, and a
// nondeterministic router makes a residency cache impossible to reason about.
//
// 🔴 Unoptimised: one work-item per token doing k passes over the experts. At batch 1 that is
// a single work-item, which is a terrible use of a GPU and completely irrelevant next to the
// matvecs it gates.
int moearc_topk_router(moearc_ctx *c, unsigned int *idx, float *weights, const float *logits,
                       unsigned long n_tokens, unsigned long n_expert, unsigned int k,
                       unsigned int gating) {
    if (!c || !idx || !weights || !logits) return ERR;
    if (k == 0 || k > (unsigned int) MAX_TOPK || (unsigned long) k > n_expert) return ERR_ARG;
    if (gating > 2u) return ERR_ARG;
    if (n_tokens == 0) return OK;
    try {
        c->q.parallel_for(
               nd_range<1>{range<1>{n_tokens * WG}, range<1>{WG}},
               [=](nd_item<1> it) {
                   const size_t t = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   const auto g = it.get_group();
                   const float *l = logits + t * n_expert;
                   constexpr float NEG_MAX = -3.402823466e+38f;

                   // The maximum is exact under any grouping: `fmax` is associative.
                   float mymax = NEG_MAX;
                   for (size_t j = lid; j < n_expert; j += WG) mymax = sycl::fmax(mymax, l[j]);
                   const float mx = reduce_over_group(g, mymax, sycl::maximum<float>());

                   // Selection, k rounds of a masked argmax. The serial version rescanned every
                   // expert against every already-chosen one in a single lane — k*n_expert*k
                   // comparisons on one work-item, which is where a decode step's router time
                   // went. Every lane keeps its own copy of `sel`; they agree because the two
                   // reductions below are group-wide.
                   int sel[MAX_TOPK];
                   for (unsigned int r = 0; r < k; ++r) {
                       int bidx = -1;
                       float bestv = NEG_MAX;
                       for (size_t j = lid; j < n_expert; j += WG) {
                           bool taken = false;
                           for (unsigned int p = 0; p < r; ++p) {
                               if (sel[p] == (int) j) { taken = true; break; }
                           }
                           if (taken) continue;
                           // Strict `>`, ascending scan: the lower index wins a tie in a lane.
                           if (bidx < 0 || l[j] > bestv) {
                               bidx = (int) j;
                               bestv = l[j];
                           }
                       }
                       const float gmax =
                           reduce_over_group(g, bidx < 0 ? NEG_MAX : bestv, sycl::maximum<float>());
                       // ...and the lower index wins a tie across lanes, which is what makes the
                       // same logits always name the same experts.
                       const int cand = (bidx >= 0 && bestv == gmax) ? bidx : 0x7FFFFFFF;
                       sel[r] = reduce_over_group(g, cand, sycl::minimum<int>());
                   }

                   // 🔴 The denominator and the weights stay in one lane, summed in index order.
                   // Floating-point addition is not associative, so reducing them over the group
                   // would change the router's weights in the last bits — a change to the
                   // model's arithmetic, bought for a few microseconds on a 64-element sum.
                   if (lid == 0) {
                       for (unsigned int r = 0; r < k; ++r) {
                           idx[t * k + r] = (unsigned int) sel[r];
                       }
                       if (gating == 2u) {
                           // 🔴 gpt-oss: the softmax is over the **k selected logits only**,
                           // taken *after* the top-k — `LAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`
                           // sets `probs = logits` and defers `ggml_soft_max` until after
                           // `ggml_get_rows`. The selection is identical either way (softmax is
                           // monotonic) but the weights are not: a softmax over 128 experts
                           // renormalised to 4 is a different vector from a softmax over those
                           // 4, and the difference lands on every expert's contribution.
                           float smax = NEG_MAX;
                           for (unsigned int r = 0; r < k; ++r) smax = sycl::fmax(smax, l[sel[r]]);
                           float d2 = 0.0f;
                           for (unsigned int r = 0; r < k; ++r) d2 += sycl::exp(l[sel[r]] - smax);
                           for (unsigned int r = 0; r < k; ++r) {
                               weights[t * k + r] = sycl::exp(l[sel[r]] - smax) / d2;
                           }
                           return;
                       }
                       float denom = 0.0f;
                       for (size_t j = 0; j < n_expert; ++j) denom += sycl::exp(l[j] - mx);

                       float wsum = 0.0f;
                       for (unsigned int r = 0; r < k; ++r) {
                           const float w = sycl::exp(l[sel[r]] - mx) / denom;
                           weights[t * k + r] = w;
                           wsum += w;
                       }
                       if (gating == 1u) {
                           const float clamped = sycl::fmax(wsum, 6.103515625e-5f);
                           for (unsigned int r = 0; r < k; ++r) weights[t * k + r] /= clamped;
                       }
                   }
               });
        return OK;
    } catch (...) { return ERR; }
}

// The OpenAI MoE activation — `ggml_swiglu_oai`, `ggml/src/ggml-cpu/ops.cpp`:
//
//     x = min(gate, limit)
//     y = clamp(up, -limit, limit)
//     out = (x / (1 + exp(-alpha * x))) * (y + 1)
//
// 🔴 Three departures from plain SwiGLU, each of which leaves a running model that is quietly
// wrong rather than an error:
//
//  - the gate is clamped **on one side only** (`min`, unbounded below) while the up branch is
//    clamped on both;
//  - there is a **`+ 1` on the up branch**, so an up projection of zero passes the gate
//    through rather than annihilating it;
//  - the sigmoid is **alpha-scaled** (`alpha = 1.702`), which is the tanh-free GELU
//    approximation, not SiLU.
//
// `alpha` and `limit` are hard-coded constants in llama.cpp (`llama-graph.cpp`,
// `LLM_FFN_SWIGLU_OAI_MOE`) rather than hparams, but they are passed in here so the kernel
// states its arithmetic rather than hiding two magic numbers.
static inline float swiglu_oai_one(float g, float u, float alpha, float limit) {
    const float x = sycl::fmin(g, limit);
    const float y = sycl::fmin(sycl::fmax(u, -limit), limit);
    return (x / (1.0f + sycl::exp(alpha * (-x)))) * (y + 1.0f);
}

int moearc_swiglu_oai(moearc_ctx *c, float *out, const float *gate, const float *up,
                      unsigned long n, float alpha, float limit) {
    if (!c || !out || !gate || !up) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            out[it[0]] = swiglu_oai_one(gate[it[0]], up[it[0]], alpha, limit);
        });
        return OK;
    } catch (...) { return ERR; }
}

// `moearc_swiglu_oai` for the case where both halves came out of one batched launch and lie end
// to end in one buffer — the counterpart of `moearc_swiglu_halves`.
int moearc_swiglu_oai_halves(moearc_ctx *c, float *out, const float *gu, unsigned long n,
                             float alpha, float limit) {
    if (!c || !out || !gu) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            out[it[0]] = swiglu_oai_one(gu[it[0]], gu[n + it[0]], alpha, limit);
        });
        return OK;
    } catch (...) { return ERR; }
}

// Add a per-expert bias row to each matrix of a batched matvec's output:
//
//     out[m * n_rows + r] += bias[idx[m] * n_rows + r]
//
// gpt-oss carries an f32 bias for every expert of every bank, and llama.cpp applies it with
// `ggml_add_id` — a gather by the router's own selection — immediately after each
// `mul_mat_id` and *before* the activation and the weighted combine. That ordering is why this
// is a separate launch rather than something folded into `moearc_moe_combine`: the down bias is
// inside the weighting, not outside it.
//
// `idx` is a host array of `n_mat` bank-relative row-group indices, copied into a by-value
// struct for the same reason `mat_table` is one — a table in device memory would need an
// upload, and `moearc_copy_h2d` drains the in-order queue.
int moearc_add_bias_id(moearc_ctx *c, float *out, const float *bias, const unsigned int *idx,
                       unsigned int n_mat, unsigned long n_rows) {
    if (!c || !out || !bias || !idx) return ERR;
    if (n_mat == 0 || n_rows == 0) return OK;
    if (n_mat > (unsigned int) MAX_BATCHED_MATS) return ERR_ARG;
    idx_table t{};
    for (unsigned int i = 0; i < n_mat; ++i) t.i[i] = idx[i];
    try {
        c->q.parallel_for(range<1>{(size_t) n_mat * n_rows}, [=](id<1> it) {
            const size_t g = it[0];
            const size_t m = g / n_rows;
            const size_t r = g - m * n_rows;
            out[g] += bias[(size_t) t.i[m] * n_rows + r];
        });
        return OK;
    } catch (...) { return ERR; }
}

// ---- elementwise --------------------------------------------------------------------
// The residual stream's two operators. Trivial individually; named here because a transformer
// block is mostly these and getting one of them backwards is invisible until the output is
// subtly wrong.
int moearc_add(moearc_ctx *c, float *out, const float *a, const float *b, unsigned long n) {
    if (!c || !out || !a || !b) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) { out[it[0]] = a[it[0]] + b[it[0]]; });
        return OK;
    } catch (...) { return ERR; }
}

int moearc_mul(moearc_ctx *c, float *out, const float *a, const float *b, unsigned long n) {
    if (!c || !out || !a || !b) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) { out[it[0]] = a[it[0]] * b[it[0]]; });
        return OK;
    } catch (...) { return ERR; }
}

// Fill a device buffer with zeros.
//
// The MoE combine needs its accumulator cleared once per block. Doing that by uploading a host
// vector of zeros is a host-to-device copy, and a copy is a synchronisation point: it drains
// the queue that everything else is now free to run ahead of. A kernel is not.
int moearc_zero(moearc_ctx *c, float *dst, unsigned long n) {
    if (!c || !dst) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) { dst[it[0]] = 0.0f; });
        return OK;
    } catch (...) { return ERR; }
}

// out += alpha * x, accumulating in place.
//
// This is the MoE combine, and it is why it exists as its own kernel rather than as an `add`
// after a `scale`. Each of the k active experts produces a vector that must be folded into one
// running total with its router weight; doing that as scale-then-add would allocate and write a
// full intermediate per expert, k times per token per layer.
int moearc_axpy(moearc_ctx *c, float *out, const float *x, float alpha, unsigned long n) {
    if (!c || !out || !x) return ERR;
    if (n == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n}, [=](id<1> it) { out[it[0]] += alpha * x[it[0]]; });
        return OK;
    } catch (...) { return ERR; }
}

// f32 -> f16, the write side of the half-precision path.
//
// The read side needs no kernel of its own: f16 is a format in the table above, so
// `moearc_dequant` with `GGML_TYPE_F16` already expands one.
int moearc_quantize_f16(moearc_ctx *c, void *dst, const float *src, unsigned long n) {
    if (!c || !dst || !src) return ERR;
    if (n == 0) return OK;
    try {
        auto *d = static_cast<unsigned short *>(dst);
        c->q.parallel_for(range<1>{n}, [=](id<1> it) { d[it[0]] = f32_to_f16(src[it[0]]); });
        return OK;
    } catch (...) { return ERR; }
}

// ---- embedding lookup ----------------------------------------------------------------
// Gather `n_tokens` rows out of a token-embedding table and expand them to f32.
//
// A gather and a dequantisation fused, because the table is the single largest tensor in a
// small MoE and expanding it would be absurd: OLMoE's `token_embd.weight` is 50k x 2048 in
// Q4_K, so materialising it as f32 costs 412 MB to read one row of 8 KB.
//
// 🔴 The table's format is not assumed. It is Q4_K in OLMoE and Q8_0 in the Qwen3.6 file, and
// `type_id` comes from the GGUF tensor header rather than from a constant here.
//
// Rows must begin on a block boundary, which is exactly ggml's own requirement that the
// fastest-varying dimension be a multiple of the block size.
int moearc_embed_rows(moearc_ctx *c, unsigned int type_id, float *out, const void *table,
                      const unsigned int *token_ids, unsigned long n_tokens,
                      unsigned long n_embd) {
    if (!c || !out || !table || !token_ids) return ERR;
    const int bb = block_bytes(type_id);
    if (bb == 0) return ERR_ARG;
    const int be = block_elems(type_id);
    if (n_embd == 0 || n_embd % (unsigned long) be != 0) return ERR_ARG;
    if (n_tokens == 0) return OK;
    try {
        const unsigned long row_bytes = (n_embd / be) * (unsigned long) bb;
        const auto *base = static_cast<const unsigned char *>(table);
        c->q.parallel_for(range<1>{n_tokens * n_embd}, [=](id<1> it) {
            const size_t g = it[0];
            const size_t t = g / n_embd;
            const size_t i = g % n_embd;
            const unsigned char *row = base + (size_t) token_ids[t] * row_bytes;
            out[g] = elem_at(type_id, row + (i / be) * (size_t) bb, (int) (i % be));
        });
        return OK;
    } catch (...) { return ERR; }
}

// ---- paged KV cache ------------------------------------------------------------------
// Write one token's K and V into the slot the cache allocator handed out.
//
// The page layout is `[page][slot][kv_head][head_dim]`, head_dim contiguous. That ordering is
// chosen for the read side: attention walks keys in time order within a head, so keeping a
// whole head's vector contiguous makes each step one coalesced run rather than a stride.
//
// `page_id` and `slot` come straight from `PagedKvCache::append` in `moearc-engine`; nothing
// here decides placement. Keeping the allocator on the host and off the device is what makes
// it testable without a GPU — see the note at the top of `moearc-engine/src/kv.rs`.
int moearc_kv_append(moearc_ctx *c, void *k_pages, void *v_pages, const float *k, const float *v,
                     unsigned int page_id, unsigned int slot, unsigned long n_kv_heads,
                     unsigned long head_dim, unsigned long page_tokens, unsigned int kv_type) {
    if (!c || !k_pages || !v_pages || !k || !v) return ERR;
    if (kv_type != GGML_TYPE_F32 && kv_type != GGML_TYPE_F16) return ERR_ARG;
    if (page_tokens == 0 || slot >= page_tokens) return ERR_ARG;
    if (n_kv_heads == 0 || head_dim == 0) return ERR_ARG;
    try {
        const unsigned long n = n_kv_heads * head_dim;
        const unsigned long base = ((unsigned long) page_id * page_tokens + slot) * n;
        c->q.parallel_for(range<1>{n}, [=](id<1> it) {
            const size_t i = it[0];
            kv_store(k_pages, base + i, k[i], kv_type);
            kv_store(v_pages, base + i, v[i], kv_type);
        });
        return OK;
    } catch (...) { return ERR; }
}

// ---- attention -----------------------------------------------------------------------
// Single-query attention over a paged KV cache: softmax(scale * q.K^T) . V, one token in.
//
// 🔴 What "causal" means here, precisely. This is the decode shape — one query, which is the
// newest token, whose K and V have already been appended. Every one of the `n_kv` cached keys
// is therefore at or before the query, so the causal mask is satisfied by the loop bound and
// no mask tensor is needed. That is a property of single-query decode, not a shortcut: for a
// multi-token prefill the mask is genuinely necessary, and `moearc_softmax`'s `mask` argument
// is what supplies it. This kernel is not a prefill kernel and does not pretend to be one.
//
// 🔴 **Sliding-window attention is the other half of that bound.** `kv_begin` is the first
// logical key this query may see, so the span is `[kv_begin, n_kv)`. llama.cpp masks a key when
// `p1 - p0 >= n_swa` (`llama_hparams::is_masked_swa`, `LLAMA_SWA_TYPE_STANDARD`), which for a
// single query at `p1 = n_kv - 1` is exactly `kv_begin = n_kv - n_swa`. Expressing it as a loop
// bound rather than an additive mask is not a shortcut either: a masked key contributes
// `exp(-inf) = 0` to both the numerator and the denominator, which is the same arithmetic as
// never visiting it — and visiting it would cost a full `head_dim` reduction per key, which is
// the whole run-time saving.
//
// 🔴 The keys are read through `block_table`, not from a contiguous run. Logical key `j` lives
// in page `block_table[j / page_tokens]` at slot `j % page_tokens`, and those pages are
// scattered across the pool in whatever order the allocator handed them out. A kernel that
// assumed contiguity would work perfectly on a freshly started sequence and corrupt every
// sequence that outlived a neighbour.
//
// GQA: query head `h` reads KV head `h / (n_heads / n_kv_heads)`. OLMoE sets `n_kv_heads ==
// n_heads`, so the model this is being built for exercises only the identity case; the
// grouped case is implemented and is tested against the CPU reference on synthetic shapes, but
// it has not been run against a real grouped-query model.
//
// The softmax is computed online — running max, running denominator, running weighted sum —
// which is the flash-attention accumulation. Here it is not for speed but because the
// alternative is a scratch buffer of `n_kv` scores per head, and `n_kv` is the context length.
//
// 🔴 Unoptimised: one work-group of `head_dim` lanes per head, one whole-group reduction per
// cached key. There is no tiling, no shared memory staging of K, and no vectorised load.
int moearc_attn_decode(moearc_ctx *c, float *out, const float *q, const void *k_pages,
                       const void *v_pages, const unsigned int *block_table, const float *sinks,
                       unsigned long n_heads, unsigned long n_kv_heads, unsigned long head_dim,
                       unsigned long kv_begin, unsigned long n_kv, unsigned long page_tokens,
                       float scale, unsigned int kv_type) {
    if (!c || !out || !q || !k_pages || !v_pages || !block_table) return ERR;
    if (kv_type != GGML_TYPE_F32 && kv_type != GGML_TYPE_F16) return ERR_ARG;
    if (n_heads == 0 || n_kv_heads == 0 || head_dim == 0 || n_kv == 0 || page_tokens == 0)
        return ERR_ARG;
    if (n_heads % n_kv_heads != 0) return ERR_ARG;
    // An empty span is not "attend to nothing" — it is a caller that computed the window
    // wrong, and softmax over no keys is 0/0. Refused rather than returned as NaNs.
    if (kv_begin >= n_kv) return ERR_ARG;
    // One lane per channel of the head, so the accumulator lives in registers. 1024 is the
    // smallest maximum work-group size any conformant device may report.
    if (head_dim > 1024) return ERR_ARG;
    try {
        const unsigned long group = n_heads / n_kv_heads;
        const unsigned long kv_row = n_kv_heads * head_dim;
        c->q.parallel_for(
               nd_range<1>{range<1>{n_heads * head_dim}, range<1>{head_dim}},
               [=](nd_item<1> it) {
                   const size_t h = it.get_group(0);
                   const size_t d = it.get_local_id(0);
                   const size_t kvh = h / group;
                   const float qv = q[h * head_dim + d];

                   // 🔴 An attention **sink** is one extra logit per query head that enters the
                   // softmax denominator and has no value vector — `ggml_soft_max_add_sinks`,
                   // which llama.cpp's `build_attn` folds in for gpt-oss. Seeding the online
                   // softmax with it (`m = s_h`, `l = exp(s_h - s_h) = 1`, `acc = 0`) is exactly
                   // the `max = MAX(max, sk)` / `sum += expf(sk - max)` pair in
                   // `ggml-cpu/ops.cpp`, expressed in the running form.
                   //
                   // The consequence is that the attention weights **do not sum to one**: the
                   // sink drains mass. Omitting it leaves every head's output uniformly too
                   // large on every block from the first token, with nothing to point at.
                   //
                   // 🔴 The sink is compared against scores that have already been scaled by
                   // `scale`, and is itself raw. That is llama.cpp's order, not a choice here.
                   const bool have_sink = sinks != nullptr;
                   float m = have_sink ? sinks[h] : -3.402823466e+38f;  // running max
                   float l = have_sink ? 1.0f : 0.0f;  // running softmax denominator
                   float acc = 0.0f;             // running sum of p_j * V_j, this lane's channel

                   for (size_t j = kv_begin; j < n_kv; ++j) {
                       const size_t page = block_table[j / page_tokens];
                       const size_t slot = j % page_tokens;
                       const unsigned long base =
                           (page * page_tokens + slot) * kv_row + kvh * head_dim;

                       const float kd = kv_load(k_pages, base + d, kv_type);
                       const float s =
                           reduce_over_group(it.get_group(), qv * kd, sycl::plus<float>()) * scale;

                       // Rescale what has accumulated so far to the new maximum, then fold in
                       // this key. On the first step `m` is -FLT_MAX and `corr` underflows to
                       // zero, which correctly discards the empty accumulator.
                       const float m_new = sycl::fmax(m, s);
                       const float corr = sycl::exp(m - m_new);
                       const float p = sycl::exp(s - m_new);
                       l = l * corr + p;
                       acc = acc * corr + p * kv_load(v_pages, base + d, kv_type);
                       m = m_new;
                   }

                   out[h * head_dim + d] = acc / l;
               });
        return OK;
    } catch (...) { return ERR; }
}


}  // extern "C"
