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

---

## The check — 2026-09-05, and the prediction was optimistic

The engine runs this architecture now (`crates/moearc-engine/src/moe.rs`, gated by
`crates/moearc-engine/tests/qwen3moe_forward.rs`), so the predictions above can be scored.

**Method.** Prompt `def fibonacci(n):` + newline + four spaces → `750 75698 1445 982 257`, 192
greedy tokens, `n_ctx = 512`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`. Every row below reproduces
the same 192 ids, and their first 64 match llama.cpp exactly on **both** its backends. Reproduce:

```sh
cargo run --release -p moearc-engine --features gpu --example residency_sweep -- \
  <model.gguf> 192 512 2952,static:23,2056,static:16,1032,static:8,520,static:4,264,static:2 \
  bench/references/qwen3-30b-a3b.fibonacci.ids 750 75698 1445 982 257
```

| slots | % of model | pool | LRU warm hit | LRU warm tok/s | static warm hit | static warm tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2952 | 48.0% | 8613 MiB | **93.0%** | **24.04** | 47.9% | 8.42 |
| 2056 | 33.5% | 5999 MiB | 85.3% | 18.02 | 33.3% | 7.04 |
| 1032 | 16.8% | 3011 MiB | 67.6% | 11.52 | 16.7% | 5.84 |
| 520 | 8.5% | 1517 MiB | 47.9% | 8.24 | 8.3% | 5.41 |
| 264 | 4.3% | 770 MiB | **0.0%** | 4.93 | 4.2% | 5.17 |

⚠️ Unoptimised, and every figure is a floor: a miss is a synchronous host-to-device copy with
nothing overlapped behind it, and prompt tokens go through the single-token decode path.

**Score.**

- **"roughly 38 tok/s at MoEArc's current efficiency" — no. 24.04.** The prediction was 1.6x
  optimistic. It was arithmetic on a *bandwidth* model, and bandwidth is not what this engine is
  short of.
- **"LRU beats the static split" — yes, and by more than the offline study said.** 93.0% against
  47.9% at matched capacity, and **2.9x the throughput**. The gap is wider than
  `bench/traces`' 83.3%-vs-94.7% because the offline `widest_static_split` counts only the
  experts a trace *touched* in a resident block, while the engine — like real `--n-cpu-moe` —
  must hold all 128 of them. 2952 slots buys 23 whole blocks, not 40.
- **llama.cpp's 50.13 tok/s stands, at 2.1x.** Closing it is kernel work, not residency work.

**Where the time goes, from the sweep's own staged-bytes column.** The engine counts the bytes it
actually copies, so the transfer share of a token is measurable rather than modelled. Over
197 decode steps, against the `13.4 GB/s` host→device figure in `docs/roadmap.md`:

| slots | warm staged | per step | transfer at 13.4 GB/s | measured step | not transfer |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2952 | 14 477 MiB | 73.5 MiB | 5.8 ms | 41.6 ms | **35.8 ms** |
| 264 | 205 065 MiB | 1 041 MiB | 81.5 ms | 202.8 ms | **121.3 ms** |

🔴 **A fixed cost of roughly 36 ms a token sits underneath every row**, and at the useful end of
the sweep it is 86% of the step. Residency buys the difference between 4.9 and 24.0 tok/s — real,
and larger than the static split by 2.9x — but the thing standing between 24.0 and llama.cpp's
50.13 is not the bus. It is kernel and synchronisation work, and no residency policy touches it.

🔴 **And the capacity in the top row is not the one the planner picks.** `memory::plan` chooses
**3157** slots for this card and model, and 3157 does not run — the measured ceiling is between
3050 and 3100, about 85% of the 11.33 GiB the device reports free, where `Headroom::PROVISIONAL`
leaves 88%. Nothing detects it at load: `malloc_device` returns valid pointers past the point
where the memory exists, so the pool reports its size and the first token fails. Measured rows
are in `Headroom::PROVISIONAL` and `Residency::All`.
