// MoEArc SYCL kernels, behind a plain C ABI.
//
// The C ABI is the whole design. SYCL is C++ and its headers transitively pull
// <sycl/sycl.hpp>, so anything that saw those types would force every consumer to have oneAPI
// installed. Nothing above this file knows SYCL exists: Rust sees opaque pointers and integers.
#include <sycl/sycl.hpp>
#include <cstring>
#include <new>

using namespace sycl;

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

}  // extern "C"
