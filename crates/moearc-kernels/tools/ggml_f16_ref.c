// Convert a sweep of f32 values to f16 with **ggml's own converter** and dump both sides.
//
// `tests/f16_crosscheck.rs` reads what this writes and compares against MoEArc's kernel. The
// rounding rule is the whole content of an f32->f16 conversion — ties to even, subnormals
// rounded rather than flushed, overflow to infinity — and it is the kind of thing two
// implementations by the same author can get consistently wrong together. `ggml_fp32_to_fp16_row`
// is what llama.cpp itself uses, so it is the arbiter.
//
// Build (see `ggml_dequant_dump.c` for the `icx` note):
//
//     icx -O2 -o ggml_f16_ref tools/ggml_f16_ref.c \
//         -I"$LLAMA_CPP/ggml/include" -L"$LLAMA_CPP/build/bin" \
//         -lggml-base -Wl,-rpath,"$LLAMA_CPP/build/bin"
//
// Run: ./ggml_f16_ref <outdir>   ->   f16.in.f32, f16.out.u16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"

static unsigned long long st = 0x853C49E6748FEA9BULL;

static float rnd(void) {
    st += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = st;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (float) ((double) (unsigned int) (z >> 32) / 4294967296.0) * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <outdir>\n", argv[0]); return 2; }

    // Sized to cover the cases where the rounding rule is visible: every f16 widened back to
    // f32 (which must narrow again exactly), every exact midpoint between adjacent f16 values
    // (which must tie to even), the subnormal range, the overflow boundary, and a random sweep.
    const size_t n_exact = 65536;
    const size_t n_mid = 65535;
    const size_t n_rand = 32768;
    const size_t n_edge = 16;
    const size_t n = n_exact + n_mid + n_rand + n_edge;

    float *in = malloc(n * sizeof(float));
    ggml_fp16_t *out = malloc(n * sizeof(ggml_fp16_t));
    if (!in || !out) { fprintf(stderr, "out of memory\n"); return 1; }

    size_t w = 0;
    for (size_t i = 0; i < n_exact; ++i) {
        in[w++] = ggml_fp16_to_fp32((ggml_fp16_t) i);
    }
    // The exact midpoint between consecutive f16 bit patterns. Rounding these is where "nearest"
    // stops being enough and "ties to even" starts to matter.
    for (size_t i = 0; i + 1 < n_exact; ++i) {
        const float a = ggml_fp16_to_fp32((ggml_fp16_t) i);
        const float b = ggml_fp16_to_fp32((ggml_fp16_t) (i + 1));
        in[w++] = (isfinite(a) && isfinite(b)) ? (float) (0.5 * ((double) a + (double) b)) : a;
    }
    for (size_t i = 0; i < n_rand; ++i) {
        const float r = rnd();
        // Spread across ten orders of magnitude so normals, subnormals and overflow all appear.
        in[w++] = r * powf(10.0f, (float) ((i % 21) - 10));
    }
    const float edges[16] = { 0.0f,     -0.0f,      65504.0f,  -65504.0f, 65520.0f,  -65520.0f,
                              65519.0f, 5.9604645e-8f, 1.7881393e-7f, 1e-45f, -1e-45f,
                              INFINITY, -INFINITY,  NAN,       1e30f,     -1e30f };
    for (size_t i = 0; i < n_edge; ++i) in[w++] = edges[i];
    if (w != n) { fprintf(stderr, "internal: wrote %zu of %zu\n", w, n); return 1; }

    ggml_fp32_to_fp16_row(in, out, (int64_t) n);

    char path[1024];
    snprintf(path, sizeof path, "%s/f16.in.f32", argv[1]);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    fwrite(in, sizeof(float), n, f);
    fclose(f);

    snprintf(path, sizeof path, "%s/f16.out.u16", argv[1]);
    f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    fwrite(out, sizeof(ggml_fp16_t), n, f);
    fclose(f);

    fprintf(stderr, "wrote %zu f32 -> f16 conversions from ggml_fp32_to_fp16_row\n", n);
    free(in);
    free(out);
    return 0;
}
