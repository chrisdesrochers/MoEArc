# Archaeology — mining ipex-llm / FlashMoE

**Milestone M0.5. Status: not started.**

Intel wrote years of Arc-specific inference code, shipped FlashMoE (May 2025)
running DeepSeek-V3 671B and Qwen3-235B-A22B on one or two Arc cards via CPU/GPU
expert offload — our exact problem, on our hardware — and then archived the repo
on 2026-01-28. It is frozen, readable and Apache-2.0.

We read it before we write anything.

> Upstream: https://github.com/intel/ipex-llm (archived)

## Questions this document must answer

### 1. FlashMoE offload policy
How did it decide which experts live on GPU vs CPU? Static per-layer split or
dynamic? Any caching or prefetch? What differed between prefill and decode?

*This directly shapes q\*. If Intel found something that works on Arc, we start
there rather than from CUDA-tuned heuristics designed for a different bottleneck.*

### 2. Kernels to lift
Catalog their SYCL / oneDNN paths for quantised GEMV/GEMM (Q4_0, Q4_K, FP8, their
`sym_int4` / `woq` formats), attention, and RMSNorm/RoPE/SiLU fusions. For each:

- does it build against current oneAPI 2026.x?
- does it use XMX?
- rough perf on the reference box vs llama.cpp SYCL master
- verdict: **lift / adapt / rewrite**

### 3. Xe-specific tricks
Sub-group sizes, work-group tiling for Xe2, USM vs buffer usage, how they handled
the Alchemist → Battlemage split, driver workarounds. These are the things that
cost weeks to rediscover.

### 4. Weight format
Their low-bit storage layout, and whether it is closer to what we want than GGUF.

### 5. What to avoid
Why it fell behind — pinned to old llama.cpp/Ollama, Python-heavy runtime, a
PyTorch dependency. **We take the kernels, not the architecture.**

## Also
Grep their forks of llama.cpp and Ollama. The "Portable Zip" builds carried
patches to llama.cpp's SYCL backend that never went upstream — diff them against
current master.

## Gate
This document merged, with a shortlist of **at least three** kernels or patterns
we will lift into M1/M2. Time-box porting to one week per kernel; lift the *ideas*
(tiling, sub-group layout, offload policy) even where the code is dead.
