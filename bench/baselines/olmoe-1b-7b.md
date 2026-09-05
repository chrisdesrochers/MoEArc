# Baseline: OLMoE-1B-7B-0924-Instruct Q4_K_M

The comparable baseline for every MoEArc number measured on this model. It did not exist
until now, and its absence was actively misleading: the repo's other baselines are
Qwen3.5-35B-A3B, and comparing a 6.92 B model's throughput against a 34.66 B model's would
have manufactured a win out of nothing.

**Hardware:** Arc B580 12 GB, Core Ultra 7 265K, 96 GB DDR5.
**Model:** `olmoe-1b-7b-0924-instruct-q4_k_m.gguf`, 3.92 GiB, 6.92 B params.
**Command:** `llama-bench -m <model> -p 512 -n 128 -r 3`, 3 repetitions, default `-ngl`.
**Date:** 2026-09-05.

| backend | prefill (pp512) tok/s | decode (tg128) tok/s |
| --- | ---: | ---: |
| llama.cpp **SYCL** | 3218.22 ± 48.99 | **283.31 ± 0.49** |
| llama.cpp **Vulkan** | 3761.69 ± 53.40 | 133.55 ± 0.16 |
| **MoEArc** | *(no batched prefill)* | **77.00** |

## 🔴 Read this before quoting MoEArc's number

MoEArc is **3.7× slower than llama.cpp SYCL** and **1.7× slower than its Vulkan backend** on
this model, on this card, today.

That is after two sessions of work: 6.25 → 64.62 → 77.00 tok/s, a 12× improvement, every step
independently verified with token ids unchanged. **It does not make MoEArc competitive.** The
gap moved from roughly 45× to 3.7×; it did not close.

⚠️ **Two numbers, and they measure different things.** 77.00 is *steady state* — warm cache,
warm-up tokens discarded, which is what `llama-bench`'s `tg128` also reports and therefore the
only fair comparison. `olmoe_generate` on a **cold pool**, which includes staging the model,
measures **55.25**. A served request meets a warm cache; the first one after a load does not.
Quoting the steady-state figure against a cold-start figure would be exactly the
apples-to-oranges error this file exists to prevent.

📌 Two honest caveats, in both directions:

- MoEArc has **no batched prefill** at all, so its prefill number is absent rather than poor.
  llama.cpp's 3218 tok/s prefill has no counterpart here.
- The whole model fits in VRAM at this size, so **none of MoEArc's residency machinery is
  doing anything for this benchmark.** The thesis this project exists to test is not being
  exercised by the number above. See `bench/traces/` for where it is.

## Where MoEArc's time goes at 64.6 tok/s

Device-side, 19.09 ms/token, cold pool:

| phase | ms/token | achieved bandwidth |
| --- | ---: | ---: |
| expert matvecs (gate+up, down) | 3.58 | 136 GB/s — **29.9% of peak** |
| dense matvecs (qkv, proj, head) | ~3.0 | |
| everything else | ~6.2 | |

🔴 **The remaining 3.7× is now a specific, identified cause rather than a mystery.** Each lane
covers a contiguous 32-element unit, so the activation load `x[u*32 + l]` is strided by 128
bytes across the sub-group — fully uncoalesced. And the activation is f32 while the weights are
4-bit, so the kernel issues roughly **7 bytes of activation load for every byte of weight**.
llama.cpp quantises the activation to Q8_K before the dot product and moves about 3× fewer
bytes for the same arithmetic. That is the gap.
