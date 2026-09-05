// How much VRAM can actually be COMMITTED on this device?
//
// A first version of this probe allocated expert-sized blocks until failure and "succeeded"
// 20,001 times -- 38 GiB on an 11.33 GiB card. That is the finding, not a bug in the driver:
// malloc_device returns a pointer long past physical capacity because pages are not committed
// until touched. Allocation success is therefore NOT evidence of capacity, and a cache that
// sized itself by allocating until failure would sail past the end of the card and fail later,
// somewhere else, as a hang or a reset rather than an allocation error.
//
// So every block is written to before it is counted.
//
// Build: icpx -fsycl -O2 vram_probe2.cpp -o vram_probe2
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

using namespace sycl;

static size_t reported_free(const device &d) {
    if (d.has(aspect::ext_intel_free_memory))
        return d.get_info<ext::intel::info::device::free_memory>();
    return 0;
}

int main() {
    device dev;
    try { dev = device(gpu_selector_v); }
    catch (const exception &e) { std::printf("no GPU: %s\n", e.what()); return 1; }
    queue q(dev);

    const size_t total = dev.get_info<info::device::global_mem_size>();
    const size_t free0 = reported_free(dev);
    const size_t baseline = free0 ? free0 : total;
    std::printf("device   : %s\n", dev.get_info<info::device::name>().c_str());
    std::printf("free     : %zu (%.2f GiB)\n", baseline, baseline / 1073741824.0);

    // 2,039,808 B: the measured per-expert size of Qwen3.6-35B-A3B-UD-Q4_K_M.
    const size_t chunk = 2039808;
    // Hard stop at 120% of reported free. Without it, an over-committing allocator lets this
    // run until the driver resets, which takes the machine's GPU with it.
    const size_t cap = baseline + baseline / 5;

    std::vector<void *> blocks;
    size_t committed = 0;
    bool failed = false;
    const char *how = "hit the safety cap";

    while (committed + chunk <= cap) {
        void *p = nullptr;
        try { p = malloc_device(chunk, q); }
        catch (const exception &e) { how = "malloc_device threw"; failed = true; break; }
        if (!p) { how = "malloc_device returned null"; failed = true; break; }

        // Force commitment. This is the step the first probe was missing.
        try {
            q.memset(p, 0x5A, chunk).wait_and_throw();
        } catch (const exception &e) {
            free(p, q);
            how = "write failed (this is where over-commitment actually bites)";
            failed = true;
            break;
        }
        blocks.push_back(p);
        committed += chunk;
    }

    const size_t free_after = reported_free(dev);
    std::printf("\ncommitted: %zu blocks x %zu = %zu B (%.2f GiB)\n",
                blocks.size(), chunk, committed, committed / 1073741824.0);
    std::printf("stopped  : %s\n", how);
    std::printf("free now : %zu (%.2f GiB)\n", free_after, free_after / 1073741824.0);

    // Branch on whether a failure actually occurred, not on how the total compares to the
    // baseline. A first version tested `committed >= baseline` and printed "WITHOUT failing"
    // directly beneath a line reading "stopped: write failed" -- a self-contradicting report,
    // which is worse than a wrong one because both halves look authoritative.
    if (!failed) {
        std::printf("\nRESULT: reached the safety cap at %.2f%% of reported free without any\n",
                    100.0 * committed / baseline);
        std::printf("        failure. Capacity was never found; raise the cap to measure it.\n");
    } else if (committed >= baseline) {
        std::printf("\nRESULT: committed %.2f%% of reported free before the first failure.\n",
                    100.0 * committed / baseline);
        std::printf("        More than the driver reports as free is committable, so allocator\n");
        std::printf("        overhead is not a reason to hold memory back -- the reported free\n");
        std::printf("        figure is already conservative by %.2f%%.\n",
                    100.0 * (double)(committed - baseline) / (double)baseline);
        std::printf("        Headroom above 0%% must be justified by activation and scratch\n");
        std::printf("        memory, which this probe does not measure.\n");
    } else {
        const double overhead = 100.0 * (double)(baseline - committed) / (double)baseline;
        std::printf("\nRESULT: %zu B unusable (%.2f GiB) = %.2f%% of reported free.\n",
                    baseline - committed, (baseline - committed) / 1073741824.0, overhead);
        std::printf("        That is a measured FLOOR for headroom -- allocator overhead and\n");
        std::printf("        fragmentation only. Activation and scratch memory are NOT included.\n");
    }

    for (void *p : blocks) free(p, q);
    return 0;
}
