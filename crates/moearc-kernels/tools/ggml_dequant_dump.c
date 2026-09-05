// Dump a GGUF tensor twice: as the raw quantised bytes, and as llama.cpp itself dequantises
// them.
//
// This exists so that `tests/gguf_crosscheck.rs` can compare MoEArc's dequantisation kernels
// against a *third party's* answer for real weights out of a real model, rather than against
// a CPU reference this project also wrote. Two implementations by the same author agreeing is
// weak evidence; agreeing with the reference implementation on a 22 GB production model is
// not.
//
// The heavy lifting is llama.cpp's. `gguf_init_from_file` with `no_alloc = true` parses the
// header without reading the tensor data — necessary, because these files do not fit in RAM —
// and `ggml_get_type_traits(type)->to_float` is the same function pointer the CPU backend
// calls to expand a block. Nothing is reimplemented here.
//
// Build (needs a built llama.cpp tree; no paths are baked in):
//
//     cc -O2 -o ggml_dequant_dump tools/ggml_dequant_dump.c \
//        -I"$LLAMA_CPP/ggml/include" \
//        -L"$LLAMA_CPP/build/bin" -lggml-base -Wl,-rpath,"$LLAMA_CPP/build/bin"
//
// If that link fails on `_intel_fast_memcpy`, `__kmpc_*` or `__svml_*`, the llama.cpp being
// linked against was itself built with Intel's compiler and pulls in Intel runtime libraries
// that `cc` knows nothing about. Build with `icx` instead of `cc` (after sourcing
// `setvars.sh`) and it links clean — the same lesson `build.rs` records for this crate's own
// kernel object.
//
// Run:
//
//     ./ggml_dequant_dump model.gguf                                  # list every tensor
//     ./ggml_dequant_dump model.gguf outdir tensor.name [tensor.name ...]
//
// `GGML_DUMP_MAX_ELEMENTS` caps how much of each tensor is dumped, rounded down to whole
// blocks. Expert tensors in a 35B model are hundreds of megabytes and expand eightfold; a few
// million elements is already tens of thousands of independent blocks.
//
// Writes, per tensor, `<sanitised>.q` (the exact bytes on disk), `<sanitised>.f32` (little-
// endian f32), and one line in `index.txt`:
//
//     <sanitised> <name> <ggml_type_id> <n_elements> <q_bytes>

#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "gguf.h"

// GGUF tensor names contain dots and slashes; file names should not.
static void sanitise(const char *in, char *out, size_t cap) {
    size_t i = 0;
    for (; in[i] && i + 1 < cap; ++i) {
        const char c = in[i];
        out[i] = (c == '.' || c == '/' || c == ' ') ? '_' : c;
    }
    out[i] = '\0';
}

static int dump_one(FILE *f, const struct gguf_context *gguf, const char *name,
                    const char *outdir, FILE *index) {
    const int64_t id = gguf_find_tensor(gguf, name);
    if (id < 0) {
        fprintf(stderr, "tensor not found: %s\n", name);
        return 1;
    }

    const enum ggml_type type = gguf_get_tensor_type(gguf, id);
    size_t nbytes = gguf_get_tensor_size(gguf, id);
    const size_t blck = (size_t) ggml_blck_size(type);
    const size_t tsz = ggml_type_size(type);
    size_t nelem = nbytes / tsz * blck;

    // Expert tensors in a 35B model run to hundreds of megabytes, and the f32 expansion is
    // eight times that again. `GGML_DUMP_MAX_ELEMENTS` takes a prefix instead. That is a
    // legitimate subset rather than a shortcut: super-blocks are self-contained and stored
    // contiguously, so the first N blocks of a tensor dequantise to exactly the first N*256
    // elements of the whole.
    const char *cap_env = getenv("GGML_DUMP_MAX_ELEMENTS");
    if (cap_env) {
        const size_t cap = strtoull(cap_env, NULL, 10);
        if (cap > 0 && cap < nelem) {
            const size_t nblk = cap / blck;  // whole blocks only
            if (nblk > 0) {
                nelem = nblk * blck;
                nbytes = nblk * tsz;
            }
        }
    }

    const struct ggml_type_traits *tt = ggml_get_type_traits(type);
    if (!tt || !tt->to_float) {
        fprintf(stderr, "%s: ggml has no to_float for type %d (%s)\n", name, (int) type,
                ggml_type_name(type));
        return 1;
    }

    unsigned char *q = malloc(nbytes);
    float *y = malloc(nelem * sizeof(float));
    if (!q || !y) {
        fprintf(stderr, "%s: out of memory (%zu quantised bytes, %zu elements)\n", name, nbytes,
                nelem);
        free(q);
        free(y);
        return 1;
    }

    const off_t off = (off_t) gguf_get_data_offset(gguf) + (off_t) gguf_get_tensor_offset(gguf, id);
    if (fseeko(f, off, SEEK_SET) != 0 || fread(q, 1, nbytes, f) != nbytes) {
        fprintf(stderr, "%s: could not read %zu bytes at offset %lld\n", name, nbytes,
                (long long) off);
        free(q);
        free(y);
        return 1;
    }

    // llama.cpp's own expansion, via the same function pointer its CPU backend uses.
    tt->to_float(q, y, (int64_t) nelem);

    char safe[512];
    sanitise(name, safe, sizeof safe);

    char path[1024];
    snprintf(path, sizeof path, "%s/%s.q", outdir, safe);
    FILE *o = fopen(path, "wb");
    if (!o) { perror(path); free(q); free(y); return 1; }
    fwrite(q, 1, nbytes, o);
    fclose(o);

    snprintf(path, sizeof path, "%s/%s.f32", outdir, safe);
    o = fopen(path, "wb");
    if (!o) { perror(path); free(q); free(y); return 1; }
    fwrite(y, sizeof(float), nelem, o);
    fclose(o);

    fprintf(index, "%s %s %d %zu %zu\n", safe, name, (int) type, nelem, nbytes);
    fprintf(stderr, "%s: type %s, %zu elements, %zu quantised bytes\n", name,
            ggml_type_name(type), nelem, nbytes);

    free(q);
    free(y);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>                              # list tensors\n",
                argv[0]);
        fprintf(stderr, "       %s <model.gguf> <outdir> <tensor> [tensor ...]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];

    struct gguf_init_params p = { /*no_alloc =*/ true, /*ctx =*/ NULL };
    struct gguf_context *gguf = gguf_init_from_file(model, p);
    if (!gguf) {
        fprintf(stderr, "could not parse %s as GGUF\n", model);
        return 1;
    }

    // With no tensor names, list the index so a caller can pick representatives of each
    // quantisation type without guessing at naming conventions.
    if (argc == 2) {
        const int64_t n = gguf_get_n_tensors(gguf);
        for (int64_t i = 0; i < n; ++i) {
            const enum ggml_type t = gguf_get_tensor_type(gguf, i);
            printf("%s %s %zu\n", gguf_get_tensor_name(gguf, i), ggml_type_name(t),
                   gguf_get_tensor_size(gguf, i));
        }
        gguf_free(gguf);
        return 0;
    }

    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <outdir> <tensor> [tensor ...]\n", argv[0]);
        gguf_free(gguf);
        return 2;
    }
    const char *outdir = argv[2];

    // Opened read-only and never written: these models are shared, expensive to fetch, and
    // nothing here has any business modifying one.
    FILE *f = fopen(model, "rb");
    if (!f) { perror(model); gguf_free(gguf); return 1; }

    char path[1024];
    snprintf(path, sizeof path, "%s/index.txt", outdir);
    FILE *index = fopen(path, "w");
    if (!index) { perror(path); fclose(f); gguf_free(gguf); return 1; }

    int rc = 0;
    for (int i = 3; i < argc; ++i) {
        rc |= dump_one(f, gguf, argv[i], outdir, index);
    }

    fclose(index);
    fclose(f);
    gguf_free(gguf);
    return rc;
}
