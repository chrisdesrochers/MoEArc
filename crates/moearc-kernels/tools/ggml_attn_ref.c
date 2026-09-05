// Compute single-query attention and a masked softmax with **ggml's own operators**, and dump
// the inputs and results.
//
// `tests/attention_crosscheck.rs` reads what this writes, runs MoEArc's kernels over the same
// inputs, and compares. The point is the same as `ggml_dequant_dump.c`: a CPU reference written
// in this repository and a GPU kernel written in this repository agreeing rules out a
// transcription slip but not a shared misunderstanding of the operation. ggml is the third
// opinion.
//
// What is actually being checked is llama.cpp's `build_attn_mha` non-flash path, reduced to the
// decode case, transcribed from `src/llama-graph.cpp`:
//
//     kq  = ggml_mul_mat(k, q);                              // [n_kv, n_q]
//     kq  = ggml_soft_max_ext(kq, kq_mask, kq_scale, 0.0f);
//     kqv = ggml_mul_mat(v_t, kq);                           // [head_dim, n_q]
//
// with `kq_scale = 1/sqrt(head_dim)`, which is what `build_olmoe` passes. The single-query case
// needs no mask — every cached key precedes the query — so the mask is exercised separately, on
// its own multi-row softmax, where it is the thing under test.
//
// It also emits an RMSNorm golden — `ggml_mul(ggml_rms_norm(x, eps), w)`, which is what
// `build_norm(..., LLM_NORM_RMS)` expands to. That is the operation carrying OLMoE's QK-norm.
//
// 🔴 What this does NOT cover: grouped-query attention (the golden is generated with
// n_kv_heads == n_heads, as OLMoE has), the f16 KV cache (ggml is given f32), and paging. The
// page walk is MoEArc's own concern and is checked against the CPU reference instead.
//
// Build (needs a built llama.cpp tree; use `icx` if llama.cpp was built with Intel's compiler,
// see the note in `ggml_dequant_dump.c`):
//
//     icx -O2 -o ggml_attn_ref tools/ggml_attn_ref.c \
//         -I"$LLAMA_CPP/ggml/include" -L"$LLAMA_CPP/build/bin" \
//         -lggml-base -lggml-cpu -Wl,-rpath,"$LLAMA_CPP/build/bin"
//
// Run: ./ggml_attn_ref <outdir>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

// Shapes. These are OLMoE's attention geometry: 16 heads, head_dim 128, no GQA. N_KV is
// deliberately not a multiple of anything, so a page walk on the reading side has a partly
// filled last page.
#define N_HEADS  16
#define HEAD_DIM 128
#define N_KV     37

// The masked-softmax case, kept separate because a single query needs no mask.
#define SM_ROWS 7
#define SM_COLS 61

// RMSNorm, at OLMoE's n_embd. This is the op that carries OLMoE's QK-norm: `attn_q_norm` and
// `attn_k_norm` are f32 [n_embd] and `build_olmoe` applies them with `build_norm(..., LLM_NORM_RMS)`
// over the whole 2048-wide vector, before the reshape into heads and before RoPE. Getting it
// checked against ggml matters because most MoE implementations omit QK-norm entirely and the
// resulting degradation looks like a bug somewhere else.
#define RN_ROWS 4
#define RN_COLS 2048
#define RN_EPS  1e-5f

static unsigned long long rng_state = 0x243F6A8885A308D3ULL;

static float rnd_unit(void) {  // uniform in [-1, 1)
    rng_state += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (float) ((double) (unsigned int) (z >> 32) / 4294967296.0) * 2.0f - 1.0f;
}

static int dump(const char *dir, const char *name, const void *p, size_t bytes) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    const size_t n = fwrite(p, 1, bytes, f);
    fclose(f);
    if (n != bytes) { fprintf(stderr, "%s: short write\n", path); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <outdir>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];

    const float scale = 1.0f / sqrtf((float) HEAD_DIM);

    // Inputs, in MoEArc's layouts: q is [head][dim]; K and V are [key][head][dim], which is the
    // order a KV page stores a slot in, so the reading side can drop them straight into pages.
    static float q[N_HEADS * HEAD_DIM];
    static float k[N_KV * N_HEADS * HEAD_DIM];
    static float v[N_KV * N_HEADS * HEAD_DIM];
    static float out[N_HEADS * HEAD_DIM];
    for (size_t i = 0; i < sizeof q / sizeof *q; ++i) q[i] = rnd_unit();
    for (size_t i = 0; i < sizeof k / sizeof *k; ++i) k[i] = rnd_unit();
    for (size_t i = 0; i < sizeof v / sizeof *v; ++i) v[i] = rnd_unit();

    static float sm_x[SM_ROWS * SM_COLS];
    static float sm_mask[SM_ROWS * SM_COLS];
    for (size_t i = 0; i < sizeof sm_x / sizeof *sm_x; ++i) sm_x[i] = rnd_unit() * 30.0f;
    // A causal mask over the last SM_ROWS of SM_COLS keys: 0 where visible, -inf where not.
    for (int r = 0; r < SM_ROWS; ++r) {
        for (int c = 0; c < SM_COLS; ++c) {
            sm_mask[r * SM_COLS + c] = (c > (SM_COLS - SM_ROWS) + r) ? -INFINITY : 0.0f;
        }
    }

    struct ggml_init_params ip = { 256u * 1024 * 1024, NULL, false };
    struct ggml_context *ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    // One head at a time. Building the batched 4-D permuted form llama.cpp uses would add
    // exactly the kind of index reasoning this cross-check exists to avoid trusting.
    struct ggml_tensor *kqv[N_HEADS];
    for (int h = 0; h < N_HEADS; ++h) {
        struct ggml_tensor *tq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD_DIM, 1);
        struct ggml_tensor *tk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD_DIM, N_KV);
        struct ggml_tensor *tv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_KV, HEAD_DIM);

        memcpy(tq->data, q + (size_t) h * HEAD_DIM, HEAD_DIM * sizeof(float));
        for (int j = 0; j < N_KV; ++j) {
            const float *src = k + ((size_t) j * N_HEADS + h) * HEAD_DIM;
            memcpy((float *) tk->data + (size_t) j * HEAD_DIM, src, HEAD_DIM * sizeof(float));
            // v arrives transposed: ggml wants [n_kv, head_dim] so that mul_mat(v_t, kq)
            // contracts over the key axis. This is llama.cpp's `v_trans` layout.
            const float *sv = v + ((size_t) j * N_HEADS + h) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) ((float *) tv->data)[(size_t) d * N_KV + j] = sv[d];
        }

        struct ggml_tensor *kq = ggml_mul_mat(ctx, tk, tq);          // [N_KV, 1]
        kq = ggml_soft_max_ext(ctx, kq, NULL, scale, 0.0f);
        kqv[h] = ggml_mul_mat(ctx, tv, kq);                          // [HEAD_DIM, 1]
        ggml_build_forward_expand(gf, kqv[h]);
    }

    struct ggml_tensor *tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SM_COLS, SM_ROWS);
    struct ggml_tensor *tm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SM_COLS, SM_ROWS);
    memcpy(tx->data, sm_x, sizeof sm_x);
    memcpy(tm->data, sm_mask, sizeof sm_mask);
    struct ggml_tensor *sm = ggml_soft_max_ext(ctx, tx, tm, scale, 0.0f);
    ggml_build_forward_expand(gf, sm);

    static float rn_x[RN_ROWS * RN_COLS];
    static float rn_w[RN_COLS];
    for (size_t i = 0; i < sizeof rn_x / sizeof *rn_x; ++i) rn_x[i] = rnd_unit();
    for (size_t i = 0; i < RN_COLS; ++i) rn_w[i] = rnd_unit();
    struct ggml_tensor *trx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, RN_COLS, RN_ROWS);
    struct ggml_tensor *trw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, RN_COLS);
    memcpy(trx->data, rn_x, sizeof rn_x);
    memcpy(trw->data, rn_w, sizeof rn_w);
    // `build_norm(cur, mw, NULL, LLM_NORM_RMS, il)` is exactly these two nodes.
    struct ggml_tensor *rn = ggml_mul(ctx, ggml_rms_norm(ctx, trx, RN_EPS), trw);
    ggml_build_forward_expand(gf, rn);

    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_graph_compute failed\n");
        return 1;
    }

    for (int h = 0; h < N_HEADS; ++h) {
        memcpy(out + (size_t) h * HEAD_DIM, kqv[h]->data, HEAD_DIM * sizeof(float));
    }

    int rc = 0;
    char meta[512];
    snprintf(meta, sizeof meta, "%d %d %d %.9g\n%d %d %.9g\n%d %d %.9g\n", N_HEADS, HEAD_DIM,
             N_KV, (double) scale, SM_ROWS, SM_COLS, (double) scale, RN_ROWS, RN_COLS,
             (double) RN_EPS);
    rc |= dump(dir, "attn.meta", meta, strlen(meta));
    rc |= dump(dir, "attn.q.f32", q, sizeof q);
    rc |= dump(dir, "attn.k.f32", k, sizeof k);
    rc |= dump(dir, "attn.v.f32", v, sizeof v);
    rc |= dump(dir, "attn.out.f32", out, sizeof out);
    rc |= dump(dir, "sm.x.f32", sm_x, sizeof sm_x);
    rc |= dump(dir, "sm.mask.f32", sm_mask, sizeof sm_mask);
    rc |= dump(dir, "sm.out.f32", sm->data, sizeof sm_x);
    rc |= dump(dir, "rn.x.f32", rn_x, sizeof rn_x);
    rc |= dump(dir, "rn.w.f32", rn_w, sizeof rn_w);
    rc |= dump(dir, "rn.out.f32", rn->data, sizeof rn_x);

    fprintf(stderr,
            "wrote ggml golden: attention %d heads x %d dims over %d keys, scale %.9g; "
            "masked softmax %d x %d; rmsnorm %d x %d eps %.9g\n",
            N_HEADS, HEAD_DIM, N_KV, (double) scale, SM_ROWS, SM_COLS, RN_ROWS, RN_COLS,
            (double) RN_EPS);
    ggml_free(ctx);
    return rc;
}
