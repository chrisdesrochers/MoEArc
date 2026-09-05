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
| **MoEArc** | *(no batched prefill)* | **64.62** |

## 🔴 Read this before quoting MoEArc's number

MoEArc is **4.4× slower than llama.cpp SYCL** and **2.1× slower than its Vulkan backend** on
this model, on this card, today.

That is after a 10× improvement in one session (6.25 → 64.62 tok/s). The improvement is real
and independently verified with token ids unchanged. **It does not make MoEArc competitive.**
The gap moved from roughly 45× to 4.4×; it did not close.

📌 Two honest caveats, in both directions:

- MoEArc has **no batched prefill** at all, so its prefill number is absent rather than poor.
  llama.cpp's 3218 tok/s prefill has no counterpart here.
- The whole model fits in VRAM at this size, so **none of MoEArc's residency machinery is
  doing anything for this benchmark.** The thesis this project exists to test is not being
  exercised by the number above. See `bench/traces/` for where it is.

## Where MoEArc's time goes at 64.6 tok/s

Device-side, 19.09 ms/token, cold pool:

| phase | share |
| --- | ---: |
| expert matvecs | 37.8% |
| expert staging (H2D) | 24.8% |
| dense matvecs | 17.8% |
| everything else | 19.6% |

Staging falls to ~1 ms/token once warm, so it is warm-up cost rather than a fixed one.
