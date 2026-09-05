// What does a token's expert miss-path actually cost on this card?
//
// The bandwidth-bound ceiling was arithmetic: misses/token x bytes/expert / link speed. This
// measures it. Real geometry from the GGUF (Qwen3.6-35B-A3B-UD-Q4_K_M): 40 blocks, 8 experts
// active per block = 320 activations per token, 2,039,808 B per expert slot.
//
// Two regimes, because they are not the same cost:
//   BULK  - all of a token's misses fetched as one contiguous transfer. The optimistic bound.
//   BLOCK - 40 sequential fetches, one per block, which is what actually happens: block N+1's
//           router runs on block N's output, so its experts cannot be named until N completes.
//
// The gap between them is the price of MoE's serialisation, and it is the number that decides
// whether prefetch is worth building.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
using namespace sycl;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    queue q{gpu_selector_v};
    std::printf("device: %s\n\n", q.get_device().get_info<info::device::name>().c_str());

    const size_t slot = 2039808;      // measured per-expert bytes
    const int blocks = 40;
    const int active = 8;             // per block
    const int reps = 20;

    for (double hit : {0.400, 0.659, 0.801}) {
        const int miss_per_block = (int)((1.0 - hit) * active + 0.5);
        const size_t bulk_bytes = (size_t)miss_per_block * blocks * slot;
        if (bulk_bytes == 0) continue;

        void *host = malloc_host(bulk_bytes, q);
        void *dev = malloc_device(bulk_bytes, q);
        if (!host || !dev) { std::printf("alloc failed at hit %.3f\n", hit); continue; }
        std::memset(host, 0x5A, bulk_bytes);

        // BULK: one transfer for the whole token.
        q.memcpy(dev, host, bulk_bytes).wait();
        double t0 = now_s();
        for (int r = 0; r < reps; ++r) q.memcpy(dev, host, bulk_bytes).wait();
        double bulk_ms = (now_s() - t0) / reps * 1000.0;

        // BLOCK: 40 sequential transfers, each waited on, mimicking the dependency chain.
        t0 = now_s();
        for (int r = 0; r < reps; ++r) {
            const size_t per = (size_t)miss_per_block * slot;
            for (int b = 0; b < blocks; ++b) {
                q.memcpy((char *)dev + b * per, (char *)host + b * per, per).wait();
            }
        }
        double block_ms = (now_s() - t0) / reps * 1000.0;

        std::printf("hit %.1f%%  %d miss/block  %.1f MB/token\n",
                    hit * 100, miss_per_block, bulk_bytes / 1e6);
        std::printf("   BULK  (one transfer)      %7.2f ms -> %6.1f tok/s ceiling\n",
                    bulk_ms, 1000.0 / bulk_ms);
        std::printf("   BLOCK (40 sequential)     %7.2f ms -> %6.1f tok/s ceiling   [%.2fx slower]\n\n",
                    block_ms, 1000.0 / block_ms, block_ms / bulk_ms);

        free(host, q); free(dev, q);
    }
    return 0;
}
