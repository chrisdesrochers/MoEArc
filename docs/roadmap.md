# Where the performance is, and how to get it

Written 2026-09-05, after MoEArc went from 6.25 to 64.6 tok/s and the first honest
comparison against llama.cpp on the same model.

## The gap, as one number

| | tok/s | achieved bandwidth | % of B580 peak (456 GB/s) |
| --- | ---: | ---: | ---: |
| llama.cpp SYCL | 283.31 | 289 GB/s | **63.4%** |
| llama.cpp Vulkan | 133.55 | 136 GB/s | 29.9% |
| **MoEArc** | 64.62 | 66 GB/s | **14.5%** |

**Decode at batch 1 is memory-bound.** Each token reads roughly 973 MiB of active weights and
performs about one multiply–add per byte. This has a consequence worth stating early because it
rules out a whole category of work: **XMX and the matrix engines are irrelevant to decode.**
There is no matrix to multiply. They matter for prefill and nowhere else.

There are **two independent gaps**, and conflating them wastes effort.

---

## Gap 1 — kernel efficiency. We read weights at 14.5% of what the card can do.

Expert matvecs read **465 MiB/token in 7.21 ms = 67.6 GB/s = 14.8% of peak** — the same figure
as the whole engine, so they are the bottleneck.

📌 **The tell:** attention's q/k/v projection was measured at **456 GB/s, 100% of peak**, by the
same kernel on the same card. The kernel *can* saturate memory. It does not for experts, and
the reason is shape: each expert matvec reads ~1.2 MiB and is launched separately —
**8 experts × 3 tensors × 16 blocks = 384 launches per token**. A 1.2 MiB read has nowhere near
enough parallelism in flight to hide memory latency.

**In progress:** batch the 8 experts per block into one launch, so the read is ~9.7 MiB with 8×
the parallelism. Needs a device-side slot table so the kernel finds each expert without a host
round trip.

This gap is closable with the model resident. It has nothing to do with offload.

---

## Gap 2 — a mechanism we have not built: CPU co-execution

This is the one that matters for models that do not fit, and it is not a kernel optimisation.

### The bandwidth argument

Measured on this machine:

| path | bandwidth |
| --- | ---: |
| PCIe host→device | **13.4 GB/s** |
| host memory, single core (read half of memcpy) | ~22.8 GB/s |
| host memory, 20 cores | several times that |

🔴 **One CPU core reads host memory faster than the PCIe link can ship it to the GPU.**

So for an expert that is *not* resident, there are two ways to use it, and shipping it is not
obviously the better one:

- **Stream it:** 3.891 MiB over PCIe at 13.4 GB/s ≈ **0.29 ms**, then the GPU computes in
  microseconds.
- **Compute it host-side:** read the same 3.891 MiB from DRAM at several tens of GB/s ≈
  **0.10 ms**, and the CPU does the arithmetic — which for a memory-bound matvec is nearly free.

The CPU is far slower at floating-point than the GPU and it does not matter, because **neither
side is compute-bound.** This is the same reason llama.cpp's `--n-cpu-moe` exists, and it is
why a static split is a real competitor rather than a strawman.

🔴 **Scored 2026-09-05, and the last sentence of that paragraph is wrong.** The host expert
kernel *is* compute-bound: one core computes an expert at **6.5 GB/s** against the 22.8 GB/s it
reads memory at, so a single core needs **465 us** for a 3.06 MB expert, not 0.10 ms. The
conclusion survives anyway, by a different route — **19 threads bring it to ~50 us**, which beats
the streaming path not by the 3x this arithmetic claimed but by **6–9x**, because the 13.4 GB/s
above is a *pinned* figure and `stage()` copies out of a memory-mapped file at 6.7–10.5 GB/s.
The full measurement, and what it is worth in the engine, is in
`bench/baselines/qwen3-30b-a3b.md`.

### What this means for the thesis

Our claim has been *dynamic residency beats a static split*. The residency measurements support
it: on Qwen3-30B at the B580's capacity, LRU hits **94.7%** against a static split's **83.3%**
and moves **3.1× less data**.

But a static split's residency number was never its whole story. **llama.cpp does not stall on
a miss — it computes that layer on the CPU.** So the honest comparison is not "our hit rate
against theirs", it is **total time**, where their misses cost host compute and ours cost PCIe
transfers. Until MoEArc can also execute an expert host-side, we are comparing our best case
against their worst, and the 4.4× throughput deficit is the evidence that this framing has been
flattering us.

---

## What FreeToken built here (architecture, surveyed — not code we take)

Their `moe/` module is where the mechanism lives, and the file sizes say what mattered to them:

| module | size | what it does |
| --- | ---: | --- |
| `offload_cache.py` | 57 KB | the expert cache, including a **hybrid** path (`ensure_experts_hybrid`) where some experts come from VRAM and others are computed host-side |
| `benchbw.py` | 44 KB | **bandwidth benchmarking**, which is what makes the policy adaptive rather than guessed |
| `cpu_executor.py` | 33 KB | `CpuMoeExecutor` — physical-core affinity, thread pinning, `decode_submit`/`decode_sync` for **async overlap with the GPU**, and a watchdog |
| `expert_banks.py`, `host_banks.py` | 45 KB | pinned host staging so transfers are fast when they do happen |
| `fused_*.py` | 48 KB | fused MoE kernels, one per quantisation format — the same batched-expert launch we are building now |

Two design decisions worth taking:

1. **The CPU/GPU choice is made per layer, not per expert** (`is_cpu_layer`). Per-expert would
   need a synchronisation point inside each block; per-layer keeps the decision on the host
   where it is cheap.
2. **Prefill prefetches, decode cannot.** They double-buffer and prefetch layer N+1's experts
   during layer N (`begin_prefill`, `prefetch_prefill_layer`, `wait_prefill_layer`). That works
   in prefill because every token's routing is known at once. In decode it is impossible: block
   N+1's router runs on block N's output. We measured the cost of that serialisation at ~2%, so
   it is not worth chasing in decode — but prefill is a different regime and the prefetch is
   real there.

We are writing our own. The value of the survey is knowing which mechanisms exist and which of
their shapes are forced by the problem rather than chosen.

---

## Plan

**Stage 1 — close the kernel gap (in progress).** Batch the expert matmuls. Target: expert
matvecs from 14.8% toward llama.cpp's ~63% of peak. If it reaches even 50%, the engine goes
from 15.25 ms/token to roughly 10 ms — about **96 tok/s**. Re-measure after; the bottleneck
will move and the next target should be chosen from the profile, not from this document.

**Stage 2 — build a host-side expert executor. ✅ Built and measured 2026-09-05; the policy
half is still outstanding.** `crates/moearc-engine/src/host_experts.rs`: Q4_K and Q6_K matvecs on
the CPU, 19 pinned threads, `submit`/`sync` around the block's device work so the two run at
once. Measured on Qwen3-30B-A3B, every configuration reproducing identical token ids:

- **Overlap works.** 94% of the host arithmetic runs while the device is busy, by device-event
  timestamps, and it is worth **+2% at 48% residency rising to +36% at 2%** — because what it
  hides behind is staging, and staging is what a low-residency step is made of.
- **The larger effect at high residency is not overlap at all.** A miss routed host-side is never
  admitted, so it never evicts a resident expert: at 2952 slots the hit rate goes 91.7% → 99.8%
  and expert traffic falls **46x**.
- **Throughput stops depending on residency.** Stream-only falls 30 → 7.7 tok/s across the sweep;
  with a host policy every residency reaches ~29 tok/s or better, peaking at **43.3 against
  llama.cpp's 50.13** — the gap from 1.7x to 1.16x.

⬜ **Still to do, and it is the half this document asked for:** the three policies measured are
constants (`off`, `frac:<f>`, `over:<n>`), chosen so the overlap question had a readable answer
before anything was tuned. **The policy must be driven by measurement, not constants** — the PCIe
and DRAM figures above are properties of this machine, and a user's box will differ. Two things
the sweep says such a policy has to handle: it must **switch itself off when the hit rate is
high** (the mechanism costs 1–4% when it routes almost nothing), and it must **not starve the
cache** (`frac:1.0` admits nothing, so the pool never fills and its VRAM is wasted).

**Stage 3 — batched prefill.** We have none at all; llama.cpp's 3218 tok/s prefill has no
counterpart here. This is where GEMM and XMX finally pay, and where cross-layer prefetch is
possible.

## Models to measure against

The fraction of a model a token touches is `active_experts / total_experts`; layer count
cancels. Lower is better for residency.

| model | experts | active | touched/token | fits in 11.33 GiB? |
| --- | ---: | ---: | ---: | --- |
| OLMoE-1B-7B | 64 | 8 | 12.5% | yes — residency does nothing |
| **Qwen3-30B-A3B** | 128 | 8 | **6.2%** | **no** — running; 27.0 tok/s at 2952 slots |
| **Gemma-4-26B-A4B** | 128 | 8 | **6.2%** | **no** — candidate |
| Qwen3.6-35B-A3B | 256 | 8 | 3.1% | no — but hybrid, needs SSM kernels |

🔴 **Every model needs its own llama.cpp baseline before any comparison.** The repo already
carried a 46 tok/s figure that was Qwen3.5-35B being implicitly compared against OLMoE numbers.
Baselines live in `bench/baselines/`, one file per model, with the command that produced them.
