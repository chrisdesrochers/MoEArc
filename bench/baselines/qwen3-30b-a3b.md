# Baseline: Qwen3-30B-A3B Q4_K_M — the model that does not fit

**17.28 GiB of weights against 11.33 GiB of VRAM.** This is the regime MoEArc exists for, and
the first baseline where the incumbent's strategy is visible rather than assumed.

**Hardware:** Arc B580 12 GB, Core Ultra 7 265K, 96 GB DDR5. **Date:** 2026-09-05.
**Command:** `llama-bench -m <model> -n 128 -r 2 -ncmoe <N>`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
Device selection verified: the B580 is `level_zero:0`; the machine also has an Arrow Lake iGPU,
deliberately left enabled for display and transcoding, which benches at 25.5 tok/s on Vulkan and
therefore cannot be confused with these numbers.

## The sweep

`-ncmoe N` moves the first N MoE layers to the CPU **permanently** — they are computed host-side
every token. It is not streaming.

| `-ncmoe` | prefill pp512 | decode tg128 |
| ---: | ---: | ---: |
| 0 | **crash** | `UR_RESULT_ERROR_DEVICE_LOST` |
| 16 | **crash** | `UR_RESULT_ERROR_OUT_OF_RESOURCES` |
| **24** | **466.38 ± 7.00** | **50.13 ± 0.08** ← best |
| 32 | 380.44 ± 3.40 | 42.38 ± 0.03 |
| 40 | 337.33 ± 1.57 | 37.14 ± 0.06 |
| 48 | 319.83 ± 1.33 | 33.19 ± 0.00 |

## Two findings, and the second sets our target

**1. llama.cpp cannot run this model below `ncmoe=24`.** Half the MoE layers must go to the CPU
or it does not fit. Host execution here is not an optimisation it chooses; it is the only way
the model runs at all.

**2. Throughput falls monotonically as more work moves to the CPU.** So llama.cpp offloads the
*minimum it must* — its CPU path is a fallback, not a preference. 🔴 This corrects a claim made
earlier in this project's notes: the argument that "computing an expert host-side beats shipping
it over PCIe" is not supported by this curve. Every layer moved to the CPU costs throughput.

## What this means for MoEArc

**The target on this model is 50.13 tok/s, not the 283 measured on OLMoE.** Those are different
models in different regimes and must never be compared.

MoEArc's strategy is different in kind: keep every layer on the GPU and stream only the experts
that miss the cache. On a real routing trace of this model (`bench/traces/qwen3-30b-prose`),
LRU hits **94.7%** at the capacity the B580 holds. The transfer cost of that:

| hit rate | misses/token | MB/token | PCIe time |
| ---: | ---: | ---: | ---: |
| 94.7% (measured, simulator) | 20.4 | 62.3 | **4.65 ms** |
| 83.3% (static split) | 64.1 | 196.2 | 14.64 ms |

**4.65 ms of a 19.9 ms budget — 23%.** The streaming approach leaves three quarters of
llama.cpp's per-token time for compute. That is the case for the design.

🔴 **And the case against it, today.** MoEArc's kernels currently read weights at ~17% of the
card's peak bandwidth where llama.cpp manages 63%. Carrying that efficiency onto this model:

- at MoEArc's current efficiency → roughly **38 tok/s**, *below* llama.cpp's 50.13
- at llama.cpp's efficiency → roughly **95 tok/s**, nearly double it

**The memory strategy is better and the kernel efficiency is worse, and right now the kernel
deficit dominates.** Both numbers above are arithmetic from measured inputs, not engine runs —
the engine cannot load this architecture yet. They are a prediction to be checked, and they are
recorded here so that check is honest either way.
