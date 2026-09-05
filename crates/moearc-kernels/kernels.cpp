// MoEArc SYCL kernels, behind a plain C ABI.
//
// The C ABI is the whole design. SYCL is C++ and its headers transitively pull
// <sycl/sycl.hpp>, so anything that saw those types would force every consumer to have oneAPI
// installed. Nothing above this file knows SYCL exists: Rust sees opaque pointers and integers.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>
#include <new>

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

// Bytes per block, from the `static_assert`s in `ggml-common.h`:
//   Q4_K: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2       = 2 + 2 + 12 + 128 = 144
//   Q5_K: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8 = 2 + 2 + 12 + 128 + 32 = 176
//   Q6_K: sizeof(ggml_half) + QK_K/16 + 3*QK_K/4            = 2 + 16 + 192 = 210
//   Q8_0: sizeof(ggml_half) + QK8_0                         = 2 + 32 = 34
static constexpr int Q4_K_BYTES = 144;
static constexpr int Q5_K_BYTES = 176;
static constexpr int Q6_K_BYTES = 210;
static constexpr int Q8_0_BYTES = 34;

// GGUF type ids, from `gguf-py/gguf/constants.py`, `class GGMLQuantizationType(IntEnum)`.
// The same ids the `moearc-model` crate's `quant` table carries.
static constexpr unsigned int GGML_TYPE_Q8_0 = 8;
static constexpr unsigned int GGML_TYPE_Q4_K = 12;
static constexpr unsigned int GGML_TYPE_Q5_K = 13;
static constexpr unsigned int GGML_TYPE_Q6_K = 14;

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

/// Bytes one block of `type_id` occupies, or 0 if this file cannot dequantise that type.
static inline int block_bytes(unsigned int type_id) {
    switch (type_id) {
        case GGML_TYPE_Q8_0: return Q8_0_BYTES;
        case GGML_TYPE_Q4_K: return Q4_K_BYTES;
        case GGML_TYPE_Q5_K: return Q5_K_BYTES;
        case GGML_TYPE_Q6_K: return Q6_K_BYTES;
        default: return 0;
    }
}

/// Elements one block of `type_id` expands to. Not a constant across formats: the K-quants pack
/// 256 elements per super-block, Q8_0 packs 32.
static inline int block_elems(unsigned int type_id) {
    return type_id == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
}

// Work-group width for the row-per-group reductions (matvec, rmsnorm, softmax). 32 is a
// native sub-group width on Intel Xe and keeps `reduce_over_group` cheap. It is a tuning
// constant, not a correctness one: any power of two gives the same answer up to the order of
// the reduction.
static constexpr size_t WG = 32;

// Largest `k` the router will select. Bounds a per-work-item array, since kernels cannot
// allocate. Real MoE models use 8 or fewer; 32 leaves headroom without spilling.
static constexpr int MAX_TOPK = 32;

// Return codes shared by the entry points below.
//   0  ok
//  -1  a device call threw, or an argument was null
//  -2  an argument was out of range (unsupported quant type, bad shape, k too large)
static constexpr int OK = 0;
static constexpr int ERR = -1;
static constexpr int ERR_ARG = -2;

extern "C" {

struct moearc_ctx {
    queue q;
};

// ---- lifecycle ------------------------------------------------------------------------
moearc_ctx *moearc_ctx_create() {
    try {
        return new moearc_ctx{queue{gpu_selector_v}};
    } catch (...) {
        return nullptr;
    }
}

void moearc_ctx_destroy(moearc_ctx *c) { delete c; }

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
            const unsigned char *blk = base + b * (size_t) bb;
            float v;
            switch (type_id) {
                case GGML_TYPE_Q4_K: v = q4k_elem(blk, i); break;
                case GGML_TYPE_Q5_K: v = q5k_elem(blk, i); break;
                case GGML_TYPE_Q6_K: v = q6k_elem(blk, i); break;
                default: v = q80_elem(blk, i); break;  // Q8_0; block_bytes rejected everything else
            }
            dst[g] = v;
        }).wait_and_throw();
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
        const unsigned long nb = n_cols / be;
        const auto *base = static_cast<const unsigned char *>(w);
        c->q.parallel_for(
               nd_range<1>{range<1>{n_rows * WG}, range<1>{WG}},
               [=](nd_item<1> it) {
                   const size_t row = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   float acc = 0.0f;
                   for (size_t b = lid; b < nb; b += WG) {
                       const unsigned char *blk = base + (row * nb + b) * (size_t) bb;
                       const float *xs = x + b * be;
                       for (int i = 0; i < be; ++i) {
                           float v;
                           switch (type_id) {
                               case GGML_TYPE_Q4_K: v = q4k_elem(blk, i); break;
                               case GGML_TYPE_Q5_K: v = q5k_elem(blk, i); break;
                               case GGML_TYPE_Q6_K: v = q6k_elem(blk, i); break;
                               default: v = q80_elem(blk, i); break;  // Q8_0
                           }
                           acc += v * xs[i];
                       }
                   }
                   const float total = reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                   if (lid == 0) out[row] = total;
               })
            .wait_and_throw();
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
               })
            .wait_and_throw();
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
        c->q.parallel_for(
               nd_range<1>{range<1>{n_rows * WG}, range<1>{WG}},
               [=](nd_item<1> it) {
                   const size_t row = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   const float *xr = x + row * n_cols;
                   float *outr = out + row * n_cols;

                   float ss = 0.0f;
                   for (size_t i = lid; i < n_cols; i += WG) ss += xr[i] * xr[i];
                   ss = reduce_over_group(it.get_group(), ss, sycl::plus<float>());

                   const float scale = sycl::rsqrt(ss / (float) n_cols + eps);
                   for (size_t i = lid; i < n_cols; i += WG) {
                       const float v = xr[i] * scale;
                       outr[i] = have_w ? v * weight[i] : v;
                   }
               })
            .wait_and_throw();
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
        }).wait_and_throw();
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
        }).wait_and_throw();
        return OK;
    } catch (...) { return ERR; }
}

// Row-wise softmax, max-subtracted. The subtraction is not an optimisation: attention logits
// and router logits both reach magnitudes where a bare exp overflows to inf and the row comes
// back as NaN.
int moearc_softmax(moearc_ctx *c, float *out, const float *x, unsigned long n_rows,
                   unsigned long n_cols) {
    if (!c || !out || !x) return ERR;
    if (n_cols == 0) return ERR_ARG;
    if (n_rows == 0) return OK;
    try {
        c->q.parallel_for(
               nd_range<1>{range<1>{n_rows * WG}, range<1>{WG}},
               [=](nd_item<1> it) {
                   const size_t row = it.get_group(0);
                   const size_t lid = it.get_local_id(0);
                   const float *xr = x + row * n_cols;
                   float *outr = out + row * n_cols;

                   float mx = -3.402823466e+38f;  // -FLT_MAX; a valid identity for max
                   for (size_t i = lid; i < n_cols; i += WG) mx = sycl::fmax(mx, xr[i]);
                   mx = reduce_over_group(it.get_group(), mx, sycl::maximum<float>());

                   float sum = 0.0f;
                   for (size_t i = lid; i < n_cols; i += WG) sum += sycl::exp(xr[i] - mx);
                   sum = reduce_over_group(it.get_group(), sum, sycl::plus<float>());

                   const float inv = 1.0f / sum;
                   for (size_t i = lid; i < n_cols; i += WG) {
                       outr[i] = sycl::exp(xr[i] - mx) * inv;
                   }
               })
            .wait_and_throw();
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
                unsigned long n_dims, float freq_base, int neox) {
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

            const float theta = (float) pos[t] * sycl::pow(theta_scale, (float) i0 / 2.0f);
            const float ct = sycl::cos(theta);
            const float st = sycl::sin(theta);
            const float x0 = s[lo];
            const float x1 = s[hi];
            o[d] = is_lo ? (x0 * ct - x1 * st) : (x0 * st + x1 * ct);
        }).wait_and_throw();
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
                       int normalize) {
    if (!c || !idx || !weights || !logits) return ERR;
    if (k == 0 || k > (unsigned int) MAX_TOPK || (unsigned long) k > n_expert) return ERR_ARG;
    if (n_tokens == 0) return OK;
    try {
        c->q.parallel_for(range<1>{n_tokens}, [=](id<1> it) {
            const size_t t = it[0];
            const float *l = logits + t * n_expert;

            float mx = -3.402823466e+38f;
            for (size_t j = 0; j < n_expert; ++j) mx = sycl::fmax(mx, l[j]);
            float denom = 0.0f;
            for (size_t j = 0; j < n_expert; ++j) denom += sycl::exp(l[j] - mx);

            int sel[MAX_TOPK];
            for (unsigned int r = 0; r < k; ++r) {
                int best = -1;
                float bestv = 0.0f;
                for (size_t j = 0; j < n_expert; ++j) {
                    bool taken = false;
                    for (unsigned int p = 0; p < r; ++p) {
                        if (sel[p] == (int) j) { taken = true; break; }
                    }
                    if (taken) continue;
                    // Strict `>` so the first — lowest-indexed — of equal logits wins.
                    if (best < 0 || l[j] > bestv) {
                        best = (int) j;
                        bestv = l[j];
                    }
                }
                sel[r] = best;
            }

            float wsum = 0.0f;
            for (unsigned int r = 0; r < k; ++r) {
                const float w = sycl::exp(l[sel[r]] - mx) / denom;
                idx[t * k + r] = (unsigned int) sel[r];
                weights[t * k + r] = w;
                wsum += w;
            }
            if (normalize) {
                const float clamped = sycl::fmax(wsum, 6.103515625e-5f);
                for (unsigned int r = 0; r < k; ++r) weights[t * k + r] /= clamped;
            }
        }).wait_and_throw();
        return OK;
    } catch (...) { return ERR; }
}

}  // extern "C"
