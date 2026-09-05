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
path and there is no batched prefill. ⚠️ **This table predates the Q6_K routing at the foot
of this file** — see there for the refreshed throughput; the hit rates and staged bytes are
unaffected by it.

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

---

## The instrument, and what it says once you have one

`MOEARC_SYNC_EACH=1` buys per-phase attribution by waiting after every launch — sound for "how
long is this kernel", and structurally unable to answer "where does the step's time go", because
the waiting is itself what is being measured. After two wrong conclusions in one day from
host-side timers, the engine now carries the instrument that does not have to choose:
**`MOEARC_PROFILE_EVENTS=1`** builds the queue with `enable_profiling` and reads each
submission's own device timestamps, leaving the queue asynchronous.

Two checks before trusting it. **It costs ~5%** (26.6 -> 25.2 tok/s, measured), against
`SYNC_EACH`'s ~10% *plus* the serialisation. And **`calls/step` must come out at 48** on a
48-block model — it read 55.3 at first, because the counters are cumulative from process start
and were folding in the prompt pass and the warm-up.

**The verdict on `SYNC_EACH`: it was broadly right.** Free-running device time agrees with it to
within ~15%, so the kernel-bound conclusion stands and was not an artifact of the instrument.

### Per-kernel device time, queue asynchronous, 68 decode steps at 2952 slots

| kernel | ms/step | bytes/step | achieved | % of 456 GB/s |
| --- | ---: | ---: | ---: | ---: |
| expert down, batched Q6_K/Q4_K (`r2048 m8`) | 3.11 | 418 MB | 134 GB/s | 29% |
| expert gate+up, batched Q4_K (`r768 m16`) | 5.39 | 679 MB | 126 GB/s | 28% |
| `attn_q`, Q4_K (`r4096 m1`) | 1.89 | 226 MB | 120 GB/s | 26% |
| `attn_output`, Q4_K (`r2048 m1`) | 2.00 | 226 MB | 113 GB/s | 25% |
| **lm_head, Q6_K (`r151936 m1`)** | **4.03** | 255 MB | **63 GB/s** | **14%** |
| `attn_k`+`attn_v` (`r512 m1`, x2) | 0.97 | 63 MB | 65 GB/s | 14% |
| **tracked matvec busy** | **17.38** | | | **46% of the 37.66 ms step** |

### 🔴 The `mat_table` finding does not transfer either

An isolated microbenchmark on this card measured our by-value 32-pointer weight table at
**1.66x slower** than a base-pointer-plus-stride kernel (141 vs 235 GB/s), and it reconciled
independently with `moe.expert_down`'s 133 GB/s. It was the best-supported lever we had.

**In the engine the batched kernels — the ones that use `mat_table` — are the *fastest* per byte
we have**: 126 and 134 GB/s, against 113-120 GB/s for unbatched Q4_K matvecs of comparable size
on the same queue in the same step. Whatever the microbenchmark measured, it is not what this
kernel does. `mat_table` is not the bottleneck and restructuring the expert pool to remove it is
not justified.

That is the **second** finding from the same harness to fail in place, after the device-`half`
conversion. Both were well-argued and both were measured — just not on this kernel. The standing
rule that comes out of it: **a microbenchmark of "the same kernel" is not this kernel; port the
change and measure in the engine before believing any of it.**

### What the numbers actually name

1. **Q6_K is half the efficiency of everything else.** The lm_head runs at **63 GB/s** where every
   Q4_K matvec manages 113-134, and it is 4.03 ms — 11% of the step — in a single call. Expert
   down is Q6_K in 24 of 48 blocks and is dragged by the same thing. This is the largest
   *efficiency* gap in the engine and it is one kernel's inner loop.
2. **Staging cannot overlap with compute.** `moe.stage` is ~11 ms and its copies go into the
   **same in-order queue** as the kernels, so a token's transfers and its arithmetic are strictly
   serialised. Tracked matvecs (17.4 ms) plus the untracked kernels (~7.7 ms by `SYNC_EACH`)
   account for ~25 ms of a 37.7 ms step; staging is most of the rest. A second queue with
   explicit event dependencies — the expert matvec waiting on its own staging copy while the
   same block's attention runs — is the shape of the fix, and the in-order safety argument that
   the whole engine rests on would have to be re-made explicitly rather than inherited.
3. **Everything Q4_K sits at 25-29% of peak** against llama.cpp SYCL's ~63% on this card. That
   remains the standing gap, and neither of the two candidate explanations tested today survived.

---

## Q6_K was a red herring, and chasing the cause found a 3.2x kernel bug

The per-kernel table above showed the Q6_K lm_head at **63 GB/s** where every Q4_K matvec managed
113-134, and Q6_K is the obvious suspect: 210 bytes per 256 values, six-bit quants split across a
low-nibble array and a separate high-bits array. But `expert_down` is Q6_K in 24 of its 48 blocks
and ran at 134 GB/s. **If Q6_K were simply expensive, that number should have been dragged down
and it was not.** Resolving that contradiction before optimising anything is what found the real
defect.

**Step 1 — split the profile key by quantisation.** The 134 GB/s was an average over two kernels
that differ by 2x. Per element, in a live decode:

```text
  expert_down  n_cols=768    Q4_K 5.12    Q6_K 5.21   ps/element  -- identical
  attn_v       n_cols=2048   Q4_K 6.87    Q6_K 18.0               -- 2.6x
  lm_head      n_cols=2048                Q6_K 13.0
```

**Step 2 — sweep `n_cols` at fixed total work** (`crates/moearc-kernels/examples/matvec_scaling`,
run through the engine's own `Context::matvec_q`, anchored by reproducing the lm_head's 13.0 to
within 0.4%). Q6_K is **flat at ~13 ps/element from `n_cols` 256 to 4096**. Not `n_cols` either.

**Step 3 — batched against unbatched.** The engine has two matvec entry points with
**token-identical inner loops**, differing only in how the row pointer is formed: a kernel
argument (`base + row * row_bytes`) versus an opaque by-value table (`w.p[mat] + row *
row_bytes`).

```text
  n_cols = 2048        unbatched   batched
  Q4_K                    4.05       4.17    ps/element
  Q5_K                    3.81       3.61
  Q6_K                   13.05       4.13    <- 3.2x
```

Batched-with-one-matrix matches batched-with-eight, so it is the **kernel structure, not the
batching**. And Q5_K — which also reads two byte streams — is untouched, so it is not the split
load either.

### Three hypotheses measured and refuted

| hypothesis | verdict |
| --- | --- |
| Q6_K's unpack is genuinely more work | **No.** Identical to Q4_K at `n_cols = 768`, and fine in the batched kernel at every shape. |
| The `ql`/`qh` split load hurts coalescing | **No.** Q5_K reads two streams too and is unaffected. |
| The in-loop nibble select (Q4_K hoists its equivalent, Q6_K does not) | **No, and worse.** Hoisting it moved unbatched only 13.05 -> 12.53 and regressed *batched* Q6_K 4.13 -> 6.06. Reverted, with a note telling the next person not to repeat it. |

🔴 **The source-level cause is not established.** It is an IGC codegen difference that an opaque
pointer suppresses, on one quantiser in one of two otherwise-identical kernels. What is
established is that it is real, reproducible, and 3.2x.

### The change, and what it bought

`moearc_matvec_q` routes **Q6_K only** through the batched kernel with a single matrix. Q4_K and
Q5_K are left alone: the swap is not free everywhere — at `n_cols = 512` it costs Q4_K
4.98 -> 7.57 ps/element. No shape in this engine uses that, but a future one might.

| | before | after | |
| --- | ---: | ---: | ---: |
| lm_head (`mvq Q6_K r151936`) | 4038.9 us/call | **1240.3** | **3.26x** |
| `attn_v` Q6_K (`mvq Q6_K r512`) | 18.9 us/call | **8.6** | 2.2x |
| tracked matvec busy | 17.41 ms | **14.38 ms** | |
| step | 37.5 ms | **31.8 ms** | |
| **throughput** | **26.6 tok/s** | **29.14 tok/s** | **+9.5%** |

Every greedy token id unchanged on both models. **It is a workaround for a compiler pathology,
not a fix**, and `examples/matvec_scaling` exists so the next person can re-check it after a
driver or compiler upgrade.

### The sweep, re-run after the routing

Same prompt, same 192 tokens, same reference check — every row still reproduces the identical
token ids and the first 64 match llama.cpp.

| slots | LRU warm hit | before | **after** | |
| ---: | ---: | ---: | ---: | ---: |
| 2952 | 93.0% | 26.97 | **29.52** | +9.5% |
| 2056 | 85.3% | 21.73 | **23.10** | +6.3% |
| 1032 | 67.6% | 15.28 | **15.98** | +4.6% |
| 520 | 47.9% | 11.62 | **11.93** | +2.7% |
| 264 | 0.0% | 7.38 | **7.62** | +3.3% |
| `static:23` (2952) | 47.9% | 12.10 | **12.58** | +4.0% |

The gain tapers as capacity falls, which is what it should do: at 264 slots the step is dominated
by staging 1 GiB of experts per token and the lm_head is a smaller share of it.

**Against llama.cpp's 50.13 tok/s the gap is now 1.7x**, from 2.1x this morning. Every point of
that came from measurement, and two of the three changes attempted today were reverted.
