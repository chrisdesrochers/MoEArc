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

**Method.** Prompt `def fibonacci(n):` + newline + four spaces -> `750 75698 1445 982 257`, 192
greedy tokens, `n_ctx = 512`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`. Every row below reproduces
the same 192 ids, and their first 64 match llama.cpp exactly on **both** its backends. Reproduce:

```sh
cargo run --release -p moearc-engine --features gpu --example residency_sweep -- \
  <model.gguf> 192 512 2952,static:23,2056,static:16,1032,static:8,520,static:4,264,static:2 \
  bench/references/qwen3-30b-a3b.fibonacci.ids 750 75698 1445 982 257
```

| slots | % of model | pool | LRU warm hit | LRU warm tok/s | static warm hit | static warm tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2952 | 48.0% | 8613 MiB | **93.0%** | **26.97** | 47.9% | 12.10 |
| 2056 | 33.5% | 5999 MiB | 85.3% | 21.73 | 33.3% | 10.26 |
| 1032 | 16.8% | 3011 MiB | 67.6% | 15.28 | 16.7% | 8.61 |
| 520 | 8.5% | 1517 MiB | 47.9% | 11.62 | 8.3% | 8.07 |
| 264 | 4.3% | 770 MiB | **0.0%** | 7.38 | 4.2% | 7.84 |

⚠️ Unoptimised, and every figure is a floor: prompt tokens go through the single-token decode
path and there is no batched prefill.

**Score.**

- **"roughly 38 tok/s at MoEArc's current efficiency" — no. 26.97.** The prediction was 1.4x
  optimistic. It modelled the *bus*, and the bus is not the binding constraint.
- **"LRU beats the static split" — yes, and by more than the offline study said.** 93.0% against
  47.9% at matched capacity, and **2.2x the throughput**. The gap is wider than
  `bench/traces`' 83.3%-vs-94.7% because the offline `widest_static_split` counts only the
  experts a trace *touched* in a resident block, while the engine — like real `--n-cpu-moe` —
  must hold all 128 of them. 2952 slots buys 23 whole blocks, not 40.
- 🔴 **Below one token's working set of 384 slots, LRU loses to the static split.** At 264 slots
  LRU thrashes to exactly **0.0%** while two pinned blocks still hit 4.2%, and static is faster
  (7.84 vs 7.38 tok/s). The dynamic policy is not universally better and this is where it stops
  being better.
- **llama.cpp's 50.13 tok/s stands, at 1.9x.**

---

## 🔴 Where the time actually goes — and a retraction

An earlier revision of this file reported, from a host-side profile, that **"compute is 1-6% of
the step"** and that ~44% was synchronisation in `moe.readback`. **Both claims were wrong, and
they were wrong for a reason this repo had already written down.**

`crates/moearc-kernels/src/lib.rs` documents it in `Context::new`: with an asynchronous queue a
host timer around a launch measures the **submission**, and the device time piles up at whichever
call next blocks. `moe.readback` was not stalling — it was the first blocking call after 48
blocks of submitted work, so it was *billed* for that work. The engine ships the instrument that
settles this and it had never been pointed at this configuration.

Both runs below are the same binary, same prompt, 59 steady-state tokens at 2952 slots; the only
difference is `MOEARC_SYNC_EACH=1`, which makes every kernel wait so device time lands on the
call that caused it.

| phase | async queue | `MOEARC_SYNC_EACH=1` | what it really is |
| --- | ---: | ---: | --- |
| `moe.readback` | 19.47 ms | **0.31 ms** | 6.5 us/call — a fence, as originally designed |
| `out.readback` | 4.32 ms | **0.08 ms** | ditto |
| `moe.expert_matvec` | 0.11 ms | **7.43 ms** | gate+up, 679 MB/token, **91 GB/s** |
| `moe.expert_down` | 0.09 ms | **3.29 ms** | 438 MB/token, **133 GB/s** |
| `out.matvec` | 0.00 ms | **4.02 ms** | the lm_head, 255 MB, **63 GB/s** |
| `attn.qkv` | 0.27 ms | **3.41 ms** | 290 MB/token, **85 GB/s** |
| `attn.attend` | 0.10 ms | **2.57 ms** | |
| `attn.proj` | 0.19 ms | **2.40 ms** | |
| `moe.stage` | 11.07 ms | **11.31 ms** | unchanged — it was always really blocking |
| total | 37.17 ms | 40.41 ms | the 3.2 ms delta is the added fences |

**The corrected decomposition of a ~37 ms token at 2952 slots:**

| | ms | share |
| --- | ---: | ---: |
| expert streaming (`moe.stage`) | 11.1 | 30% |
| expert FFN compute | 11.2 | 30% |
| attention compute | 9.8 | 26% |
| lm_head | 4.0 | 11% |
| router | 2.5 | 7% |
| **actual fences** | **0.7** | **2%** |

🔴 **Compute is ~65% of the step, not 6%, and synchronisation is 2%, not 44%.** Every matvec runs
at **14-29% of the B580's 456 GB/s peak**, which independently reproduces the 14.8-29% already
recorded in `docs/roadmap.md` — and llama.cpp SYCL reaches 63.4% on the same card. **This engine
is kernel-bandwidth-bound.** Removing the router readback entirely would buy 48 x 6.5 us =
**0.3 ms, under 1%**; it is not a target.

## What did work, and why it was not the reason it looked like

Making the expert staging copy asynchronous (`moearc_copy_h2d_async`, used by `moe::stage`) took
the step **43.19 -> 37.17 ms, 22.87 -> 26.55 tok/s (+16%)**, and the sweep above from 24.04 to
26.97 warm — with every token id, every hit rate and every staged-byte count unchanged.

But **not** by removing a drain. The queue is already empty when staging runs, because the router
readback just emptied it. The blocking copy cost *overlap*: copy n+1 could not be submitted until
copy n landed, so the copy engine never had a queue to stream and the host never ran ahead into
the next block. `moe.stage` fell 17.33 -> 11.07 ms and stayed there under `SYNC_EACH`, which is
what confirms it is real transfer rather than mis-billed device time.

⚠️ **Do not compare the 11.1 ms against `docs/roadmap.md`'s 13.4 GB/s.** That figure comes from
`tools/stream_bench.cpp`, which allocates its host side with `malloc_host` — it is a **pinned**
number. `stage()` copies out of a memory-mapped GGUF: pageable, file-backed pages that Level Zero
must bounce through a driver staging buffer. Measured here per ~930 KiB bank copy: **136 us at a
93% hit rate, 91 us at 0%** as the copies pipeline — 6.7 and 10.5 GB/s respectively.

**Next targets, in order, all of them kernel work:** the expert gate/up matvec (7.4 ms at 91
GB/s), the lm_head (4.0 ms at 63 GB/s), attention (9.8 ms), and a pinned staging ring for
`moe.stage` — whose value is bounded, since an extra mmap->pinned host copy at the measured
22.8 GB/s costs more than it saves unless the pageable path is worse than 10.5 GB/s.

🔴 **And the capacity in the top row is not the one the planner picks.** `memory::plan` chooses
**3157** slots for this card and model, and 3157 does not run — the measured ceiling is between
3050 and 3100, about 85% of the 11.33 GiB the device reports free, where `Headroom::PROVISIONAL`
leaves 88%. Nothing detects it at load: `malloc_device` returns valid pointers past the point
where the memory exists, so the pool reports its size and the first token fails. Measured rows
are in `Headroom::PROVISIONAL` and `Residency::All`.
