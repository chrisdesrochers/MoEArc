# Baseline: gpt-oss-120B MXFP4 — 59 GiB of weights on an 11.33 GiB card

**5.2x the card.** This is the size target MoEArc exists for, and the first model where the
memory strategy and the host executor are both load-bearing rather than optional.

**Hardware:** Arc B580 12 GB (`level_zero:0`), Core Ultra 7 265K (20 threads), 96 GB DDR5.
**Model:** `gpt-oss-120b-MXFP4.gguf` — 36 blocks, 128 experts, 4 active, `n_embd` 2880,
`n_ff_exp` 2880, 64 query heads over 8 KV heads at `head_dim` 64.
**Split:** 56.73 GiB of experts (96.1%) against 2.29 GiB of dense weights (3.9%).
**Slot:** 12.607 MiB — **4.3x** Qwen3-30B-A3B's 2.92 MiB. 4,608 slots is **56.7 GiB**.
**Date:** 2026-09-05, revised 2026-09-06. **llama.cpp** `e107984bcffcfd701e82738092a2b000b6fda7a2`.

---

## 0. 🔴 Read this before quoting any number below

Everything in sections 1 to 3 was measured **at one context depth — effectively zero** — and
with llama.cpp on **its default thread count**. Both of those turn out to decide the result, so
every figure below now carries its depth and its thread count, and **section 6 is the one to
quote — as corrected by section 7**, which re-measures section 6's own configuration on a quiet
box, finds a better host policy than the one section 6 swept at, and shows that one cell of
section 6.3 is not reproducible at all. Two corrections, in order of how much they matter:

1. **llama.cpp's 15.47 is llama.cpp on 4 of the machine's 20 cores.** `llama-bench`'s default
   `n_threads` is **4** — read out of its own `-o csv` output, not inferred — and `-ncmoe 31`
   puts 31 of 36 blocks' experts on the CPU, so the thread count is not a detail: it is most of
   the incumbent's throughput. Running the *published test verbatim* at `-t 16`, five
   repetitions, three separate invocations, gives **28.39 / 28.42 / 28.58 tok/s** — **about 84%
   above the number in this document.** Nothing was measured wrongly; the tool's default was
   taken for its best.
2. **Both engines were measured with an empty KV cache.** Sliding-window attention (`80c6f7d`)
   removed the `n_ctx > 128` refusal that made a longer context impossible, so the question can
   now be asked — and the two engines answer it in opposite directions. llama.cpp is **flat in
   depth**; MoEArc **falls by 5.9x** between depth 512 and depth 8192.

📌 The result of the two together is that **the headline in `README.md` does not survive**, and
it does not survive at *any* depth, not merely at long ones. Section 6 has the curve.

## 1. The incumbent: llama.cpp — **at depth 0, on default threads**

`llama-bench -m <model> -n 128 -r 2 -ncmoe <N>`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
⚠️ **No `-d`, so the KV cache is empty, and no `-t`, so this is 8 threads of 20.** Section 6
re-measures both. The table stands as a record of what was run; it is not the number to beat.

| `-ncmoe` | prefill pp512 | decode tg128 |
| ---: | ---: | ---: |
| 26 | **crash** | `UR_RESULT_ERROR_DEVICE_LOST` (during tensor init) |
| 28 | **crash** | `ggml_abort` in `ggml_backend_sycl_buffer_init_tensor` |
| 30 | **crash** | `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on `MUL_MAT` |
| **31** | 116.79 ± 0.30 | **15.47 ± 0.00** ← best |
| 32 | 115.47 ± 1.60 | 15.26 ± 0.07 |
| 34 | 115.00 ± 0.71 | 15.03 ± 0.00 |
| 36 | 110.11 ± 0.69 | 14.69 ± 0.00 |

**llama.cpp cannot run this model with fewer than 31 of its 36 MoE layers on the CPU.**
⚠️ *Not re-tested at depth.* Section 6 runs `-ncmoe 31` only. A lower value cannot become viable
at longer context — the KV cache grows and takes device memory from the same pool — so the floor
can only rise, but that is an argument, not a measurement.
Only five blocks' experts fit beside the dense weights and the KV cache. As on Qwen3-30B-A3B,
throughput falls monotonically as more work moves host-side, so 31 is the minimum it must offload
rather than a tuning choice.

⚠️ **A second run three hours later gave 14.43–14.90 across the same settings** (`-r 3`), with
`pp512` down from 116 to 87 and its error bar up from ±0.3 to ±15.7. Nothing about llama.cpp
changed; the box did. **The number to beat is therefore taken as the *higher* of the two
measurements, 15.47** — the conservative choice, since it is the one MoEArc has to clear.

🔴 **Superseded 2026-09-06.** 15.47 is not the number to beat; it is llama.cpp at `tg128 @ d0`
on 8 threads. The same build reaches **24.85 tok/s at `-t 16`, and holds ~28 tok/s out to depth
8192.** See section 6.

---

## 2. MoEArc — **at depth 6**

`examples/hybrid_sweep`, prompt `[976, 9029, 5030, 328, 10128, 382]`
(`The capital city of France is`), 64 generated tokens, `n_ctx = 128`.
🔴 **The prompt is six tokens long, so every row below is throughput at a context depth of
six**, generating into a cache that never exceeds 70 entries. That was not a choice at the time
— `moe.rs` refused anything past the 128-token window — but it is the single most important
qualifier on the table, and section 6 measures what happens without it.
`tok/s` is `(6 prompt + 64 generated) / seconds` — this engine has no batched prefill, so a
prompt token goes through the same decode path and a step is a step.

**Every one of the twenty rows produced ids identical to llama.cpp's, all 64.**

| slots | host | cold tok/s | cold hit | cold staged MiB | **warm tok/s** | vs stream | warm hit | warm staged MiB | cpu/step | busy ms/tok | wait ms/tok |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 144 | `off` | 5.26 | 26.3% | 92257 | 5.35 | — | 26.4% | 92143 | 0.00 | 0.00 | 0.00 |
| 144 | `frac:0.5` | 10.14 | 38.0% | 42661 | 10.71 | +100.1% | 38.6% | 42334 | 63.86 | 35.73 | 2.62 |
| 144 | `frac:0.75` | 13.58 | 71.7% | 10565 | 14.12 | +163.8% | 79.8% | 7829 | 97.96 | 48.18 | 38.67 |
| 144 | `frac:1.0` | 11.34 | 0.0% | 0 | 11.37 | +112.6% | 0.0% | 0 | 141.94 | 65.88 | 67.08 |
| 300 | `off` | 6.03 | 39.8% | 75464 | 6.25 | — | 40.0% | 75149 | 0.00 | 0.00 | 0.00 |
| 300 | `frac:0.5` | 11.61 | 58.5% | 32198 | 12.27 | +96.3% | 60.6% | 30950 | 52.84 | 29.90 | 4.80 |
| 300 | `frac:0.75` | 14.32 | 87.8% | 5244 | 15.02 | +140.3% | 96.7% | 1601 | 87.71 | 44.33 | 41.93 |
| 300 | `frac:1.0` | 11.35 | 0.0% | 0 | 11.27 | +80.4% | 0.0% | 0 | 141.94 | 67.05 | 67.75 |
| 480 | `off` | 6.86 | 49.4% | 63337 | 7.20 | — | 50.3% | 62215 | 0.00 | 0.00 | 0.00 |
| 480 | `frac:0.5` | 14.01 | 79.7% | 18217 | 15.55 | +116.0% | 84.2% | 14649 | 36.93 | 21.17 | 10.31 |
| 480 | `frac:0.75` | 14.05 | 88.0% | 5181 | 15.27 | +112.1% | 100.0% | 0 | 85.81 | 43.12 | 43.34 |
| 480 | `frac:1.0` | 11.40 | 0.0% | 0 | 11.28 | +56.8% | 0.0% | 0 | 141.94 | 65.71 | 67.36 |
| 560 | `off` | 7.10 | 51.8% | 60374 | 7.42 | — | 53.1% | 58760 | 0.00 | 0.00 | 0.00 |
| 560 | `frac:0.5` | 15.47 | 86.2% | 12783 | 17.29 | +133.1% | 90.5% | 9153 | 33.19 | 19.35 | 12.60 |
| 560 | `frac:0.75` | 14.37 | 88.0% | 5181 | 15.42 | +107.9% | 100.0% | 0 | 85.81 | 43.13 | 42.83 |
| 560 | `frac:1.0` | 9.98 | 0.0% | 0 | 11.27 | +51.9% | 0.0% | 0 | 141.94 | 66.43 | 67.07 |
| 600 | `off` | 7.11 | 53.6% | 58180 | 7.50 | — | 55.6% | 55621 | 0.00 | 0.00 | 0.00 |
| **600** | **`frac:0.5`** | 16.03 | 86.9% | 12254 | **17.97** | **+139.5%** | 91.4% | 8270 | 32.40 | 18.46 | 12.43 |
| 600 | `frac:0.75` | 14.26 | 88.0% | 5181 | 14.92 | +98.9% | 100.0% | 0 | 85.81 | 43.39 | 44.79 |
| 600 | `frac:1.0` | 9.70 | 0.0% | 0 | 11.15 | +48.6% | 0.0% | 0 | 141.94 | 67.17 | 68.17 |

**Best: 600 slots at `frac:0.5` — 17.97 tok/s warm, 16.03 cold, at depth 6.**
✅ **Reproduced 2026-09-06 at 17.90 warm** (cold 7.37 — the pool was cold *and* the model file
was not in the page cache, which the original sweep's earlier rows had already warmed). The
engine measurement stands; what does not stand is the comparison it was put against.
600 slots is 7.39 GiB of pool beside 2.29 GiB of dense weights: **13.0% of the expert bank
resident, and the whole model runs.**

---

## 3. Four findings

### 3.1 Streaming alone is PCIe-bound at about 7.5 tok/s, and no pool size fixes it

The `off` rows are the pure-streaming control, and they rise from 5.35 to 7.50 tok/s as the pool
quadruples — a 40% gain for 4x the memory. The reason is in the `staged MiB` column, not the hit
rate: 92 GiB moved over a 70-step run is **1.29 GiB per token**. A step at 5.35 tok/s is 187 ms,
so if staging were the *whole* of it the link would be running at 7.0 GiB/s — it is not the whole
of it, so the achieved rate is somewhat higher and the copy is correspondingly most of the step
rather than all of it. ⚠️ That is an inference from two measured columns, not a measured link
rate. What is measured is that quadrupling the pool removes only 40% of the bytes, and buys
exactly 40% more throughput: **at 600 slots it is still 0.78 GiB per token.**

🔴 **This is a different regime from Qwen3-30B-A3B and the reason is the slot size.** An expert
here is 12.607 MiB against Qwen3's 2.92 MiB, so a miss costs 4.3x more to transfer, while the
step names only 144 experts (36 blocks x 4) against Qwen3's 384 (48 x 8). Fewer, much larger
misses is the worst possible shape for a link and the best possible shape for a CPU.

### 3.2 Host execution is worth +100% to +140% here — the **opposite sign** to the Qwen3 finding

`bench/baselines/qwen3-30b-a3b.md` records that llama.cpp's throughput falls monotonically as
`-ncmoe` moves work to the CPU, and concludes that "computing an expert host-side beats shipping
it over PCIe" is unsupported. **On this model it is supported, and by a wide margin.** Every
`frac:` row beats its `off` control, at every pool size, by at least 48%.

The two are not in conflict. llama.cpp's `-ncmoe` **pins** layers to the CPU permanently, so it
substitutes host compute for device compute. `frac:` splits only the experts that *miss*, and
submits them before the block's device work is queued, so host compute is substituted for **PCIe
transfer** and overlapped with the GPU. The `busy` and `wait` columns are where that shows: at
`600/frac:0.5` the pool is busy 18.46 ms per token and the device thread loses only 12.43 ms of
it, so a third of the CPU's work is hidden. At `frac:1.0`, `busy` and `wait` are equal to two
decimal places at every pool size — nothing is hidden, the CPU is the whole critical path, and
throughput pins at 11.2 tok/s regardless of how much VRAM is available.

### 3.3 `frac:0.75` reaches a 100% warm hit rate and is still slower than `frac:0.5`

From 480 slots up, `frac:0.75` stages **zero bytes** on the warm pass: the CPU absorbs enough
misses that the pool never has to evict. That is the residency thesis working perfectly, and it
is **not** the fastest row. Sending 60.5% of experts to the CPU costs 43 ms of host time per
token where 23.4% costs 19 ms, and the extra 24 ms buys the removal of 9 GiB/run of staging that
was only costing about 12 ms. 🔴 **A perfect hit rate is not the objective; it is a means, and
past the crossover it is bought too dearly.**

### 3.4 ⚠️ The first version of this table said host offload *lost* 60–75%

The sweep was run twice with identical code and identical arguments. The first run reported
`frac:0.5` at 1.41–1.96 tok/s and every host policy **worse** than streaming; the second reported
10.71–17.97 and every host policy better. The difference was the **load average on the box**:
9.50 during the first run against 0.69 during the second, from another agent's build and test
work on the same 20 cores.

📌 **A host-offload measurement is a measurement of the whole machine, not of the engine.** The
tell that the first run was invalid was internal, not external: two rows with identical
`cpu/step`, `cpu share` and `staged MiB` — the same work, by construction — reported 2.07 and
15.02 tok/s. **Identical inputs producing an 8x spread is a broken measurement, whichever number
you prefer.** The `off` rows moved far less (5.12 vs 5.35 at 144 slots), because streaming is
bound by a link the other tenants were not using; the contamination lands almost entirely on the
rows that need the CPU.

---

## 4. What is implemented, and what is not

Six things separate this architecture from `olmoe` and `qwen3moe`, and **every one of them runs
and is wrong if omitted** — the model stays fluent. All six are implemented and gated in
`crates/moearc-engine/tests/gptoss_forward.rs`:

- **MXFP4 experts** (108 tensors, 96% of the file): 4-bit E2M1 codes against a shared E8M0
  power-of-two exponent, 17 bytes per 32 elements. 🔴 Cross-checked **bit-exact** against
  llama.cpp's own `to_float` on 8 million elements from four tensors in four different blocks —
  `max |gpu - llama.cpp| = 0.000e0` — by `moearc-kernels/tests/gguf_crosscheck.rs`.
- **Biases on every projection**: Q, K, V, output; the router, added *before* the top-k so it
  changes which experts run; and every expert of every bank, added *inside* the router's
  weighting.
- **Per-head attention sinks**: one extra logit joining the softmax denominator with no value
  vector, so a head's weights do not sum to one.
- **No QK-norm.**
- **`ggml_swiglu_oai`**: `min(gate, 7)` (clamped above only) through an `alpha = 1.702` sigmoid,
  times `clamp(up, -7, 7) + 1`.
- **A router that softmaxes *after* the top-k**, over the four selected logits.
- **YaRN RoPE**, `freq_base = 150000`, factor 32, `corr_dims = [8, 18]`, effective magnitude
  scale 1.3466. 🔴 llama.cpp's `rope_yarn` has **no position gate**: this applies from token 0,
  and there is no short-sequence regime in which plain RoPE is equivalent.

- ✅ **Sliding-window attention — implemented in `80c6f7d`**, and the `n_ctx > 128` refusal
  is gone with it. The file states `attention.sliding_window = 128`, which llama.cpp applies to
  **alternating** blocks (`set_swa_pattern(2)`: even blocks windowed, odd blocks full causal).
  `attn_decode_ext` now takes a `kv_begin`, so the span is `[kv_begin, n_kv)` — the same
  arithmetic as an additive mask, one reduction cheaper. Windowed blocks get a short ring of
  `ceil(n_swa/32)` pages instead of `ceil(n_ctx/32)`.

  🔴 **The memory consequence is the good news in this document.** The KV cache is *not* what
  makes long context expensive here: measured at load, it is **12 MiB at depth 128, 26 MiB at
  512, 80 MiB at 2048 and 296 MiB at 8192**, against an expert pool of 7.39 GiB. Context costs
  almost nothing in VRAM on this model. Everything section 6 measures happens anyway.

---

## 5. Reproducing sections 1 and 2 (depth 0 and depth 6)

```text
# llama.cpp
ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench \
    -m /zfs/swift/models/gpt-oss-120b-MXFP4.gguf -n 128 -r 2 -ncmoe 31

# MoEArc
MOEARC_TEST_GPU=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
cargo run --release -p moearc-engine --features gpu --example hybrid_sweep -- \
    /zfs/swift/models/gpt-oss-120b-MXFP4.gguf 64 128 \
    144,300,480,560,600 off,frac:0.5,frac:0.75,frac:1.0 \
    bench/references/gpt-oss-120b.capital.ids 976 9029 5030 328 10128 382
```

⚠️ **Run it on an idle machine and check `uptime` first** — see 3.4. And note what these
two commands do *not* pass: no `-d` and no `-t` on the llama.cpp side, and a six-token prompt
on MoEArc's. **For the comparison that matters, use the protocol in 6.1.**

---

## 6. Depth: the comparison at a context anyone would actually use

**Date:** 2026-09-06. This section supersedes sections 1 and 2 as the head-to-head result.

### 6.1 Protocol — identical question to both engines

The thing being measured is **decode throughput with `depth` tokens already in the KV cache**.
Prefill is excluded from the timer on both sides; only the generated tokens are timed. Generated
tokens are held at **64 for every depth**, so the amount of router churn that generation itself
causes is constant and depth is the only thing varying.

```text
# llama.cpp -- llama-bench starts its timer AFTER the -d prefill, so tg is decode-only at depth
ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench \
    -m /zfs/swift/models/gpt-oss-120b-MXFP4.gguf \
    -ncmoe 31 -p 0 -n 64 -t 16 -d 0,128,512,2048,8192 -r 3

# MoEArc -- 64 generated tokens after a `depth`-token real prompt, decode steps only
MOEARC_PROFILE=1 MOEARC_TEST_GPU=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
cargo run --release -p moearc-engine --features gpu --example ctx_curve -- \
    /zfs/swift/models/gpt-oss-120b-MXFP4.gguf 128,512,2048,8192 64 600 frac:0.5 \
    bench/references/gpt-oss-120b.longctx.ids
```

Three details are load-bearing:

- **MoEArc's `ctx_curve` times only the decode steps.** `generate` runs prompt tokens through the
  same path as generated ones, so a stopwatch over the whole call would divide by
  `depth + n` steps and report mostly prefill. The sampling closure is called once per generated
  token; its first call lands the instant prefill finished, and the marks it records fence the
  decode phase exactly.
- **The prompt is real text, not a repeated phrase** — `bench/references/gpt-oss-120b.longctx.ids`
  is 16,384 ids made by tokenising this repository's own documentation, so the file is
  reproducible from the repo. A tiled prompt would revisit the same experts and flatter the hit
  rate. ⚠️ llama-bench's `-d` prefill uses **random** tokens; that is irrelevant to it, because
  `-ncmoe` pins experts to the CPU by layer and its per-token cost does not depend on routing.
  It does mean MoEArc is being asked the harder and more realistic question of the two.
- **Threads.** MoEArc's host pool reports **19 threads**; llama.cpp is given **16**, its best of
  `4,8,12,16,20`. The original baseline let llama.cpp take its default, which is **8**.

### 6.2 llama.cpp: threads first, because the default was leaving 16 cores idle

**`llama-bench`'s default `n_threads` on this box is 4.** That is read out of the tool's own
`-o csv` output (field `n_threads`), not inferred from a timing. Section 1's command line passes
no `-t`, so section 1 is a four-core measurement.

Running **section 1's exact test** — `-ncmoe 31 -n 128` — at `-r 5`, in **three separate
invocations**, because one measurement of this configuration is not enough (see 6.2.1):

| threads | run 1 | run 2 | run 3 |
| ---: | ---: | ---: | ---: |
| **4** *(the default; what section 1 ran)* | 13.58 ± 0.55 | 12.73 ± 0.69 | 14.66 ± 0.21 |
| 8 | 21.07 ± 0.71 | 20.46 ± 0.62 | 22.33 ± 0.28 |
| **16** | **28.39 ± 0.31** | **28.42 ± 0.28** | **28.58 ± 0.25** |

🔴 **The number to beat was never 15.47. It is about 28.5**, and at `-t 16` it is the most stable
figure in this document — three invocations inside 0.2 tok/s of each other. With 31 of 36 blocks'
experts on the CPU, four threads leaves sixteen cores idle; the incumbent was being judged on a
quarter of the machine while MoEArc's host pool used **19 threads**. *(The published 15.47 sits
just above this session's `-t 4` range of 12.73–14.66 — close enough to confirm section 1 ran at
the default, not close enough to call an exact reproduction.)*

#### 6.2.1 ⚠️ Why three invocations, and one run that must not be quoted

**The model is 59.03 GiB, on ZFS, with `arc c_max` set to 16 GiB on a 91 GiB box.** It cannot be
fully cached, and `-ncmoe 31` reads 31 of 36 blocks' experts host-side *every token*. Throughput
therefore depends on what happens to be resident, and that is not constant between processes.

A single `-r 2` sweep run earlier in this session gave **17.59 ± 5.56** at `-t 16` and
**13.67 ± 2.86** at `-t 8` — far below the table above, with error bars 20 times as wide. It was
not discarded for being inconvenient; it is reported here because **its own error bars say it is
not a measurement**, and because the `-r 5` triplicate that replaced it agrees with itself to
0.2 tok/s. This is the same failure mode section 3.4 records, arriving by a third route: not a
busy box and not an unfair thread count, but **a model that does not fit in RAM.**

📌 It applies to *both* engines. MoEArc reads the same 59 GiB file for its staging and its host
experts, so its absolute numbers carry the same dependence. What survives it is the **shape**:
llama.cpp's flatness in depth and MoEArc's 5.9x fall are both far larger than this variance, and
MoEArc's fall is corroborated by the staging counters in 6.4, which are not timings at all.

📌 **This is a measurement-hygiene failure of the same family as section 3.4, not a new one.**
There it was *the box was busy*; here it is *the baseline was given a quarter of the machine and
the challenger nearly all of it*. Both produce a number that is real, reproducible, and answers
the wrong question.

### 6.3 The curve

🔴 **Corrected by section 7.5.** Every row here is `frac:0.5`, which is **not the best host
policy at any depth** — `frac:0.75` beats it by 1.06x to 1.73x — and the depth-2048 row was taken
at a load average of 21 and **understates by 34%**. The depth-8192 cell should not be quoted in
either direction: a clean re-run of it came out 1.6x *lower* while reading 1.48 TiB from disk.

llama.cpp, `-ncmoe 31`, `-r 3`, decode-only at each depth. MoEArc, 600 slots at `frac:0.5`,
64 generated tokens, decode-only, cold and warm reported separately.

| depth | llama.cpp `-t 16` | llama.cpp `-t 8` | **MoEArc warm** | MoEArc cold | MoEArc KV | MoEArc vs best llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / 6 | 28.39–28.58 † | 20.46–22.33 † | **17.90** *(depth 6)* | 7.37 | 12 MiB | **0.63x** |
| 128 | 23.25 ± 1.19 | 16.32 ± 1.03 | **12.12** | 5.62 | 12 MiB | **0.52x** |
| 512 | 26.65 ± 0.33 | 20.78 ± 0.55 | **12.62** | 7.63 | 26 MiB | **0.47x** |
| 2048 | 28.24 ± 0.56 | 22.00 ± 0.38 | **5.69** | 4.74 | 80 MiB | **0.20x** |
| 8192 | 28.15 ± 0.27 | 22.25 ± 0.15 | **2.14** | 1.64 | 296 MiB | **0.076x** |

† **The depth-0 row is the `tg128` triplicate from section 6.2, not `tg64 @ d0` from this
sweep.** The `tg64 @ d0` cells of the depth sweep are warm-up contaminated (12.99 ± 2.64 at
`-t 16`, 6.73 ± 3.14 at `-t 8` — error bars 5 to 20 times every other row's), because
llama-bench's *first* test in a process pays the warm-up and 64 generated tokens amortise it far
less than 128 do. The range quoted is three invocations at `-r 5`, and it is section 1's own
test. ⚠️ It is still the least like-for-like row here: MoEArc's 17.90 is a **warm second pass at
depth 6**, so if anything this row flatters MoEArc. **Depth 128 onwards is the clean
comparison.**

**Answer to the question this was run to settle: the comparison does not hold, does not merely
narrow, and does not merely become context-dependent. It inverts, at every depth measured.**

- **llama.cpp is flat in depth.** 23.25 → 28.15 from depth 128 to 8192 at `-t 16`; it gets
  *faster* with depth and then levels off. Nothing about 8192 tokens troubles it on this card.
- **MoEArc falls 5.9x** over the same span, 12.62 → 2.14.
- The gap goes from **1.9x against us at depth 128 to 13.2x against us at depth 8192.**

### 6.4 Attributing the fall: staging, not attention

🔴 **True in the regime measured here, and it inverts outside it — see section 7.6.** Both rows
below are `frac:0.5` at depths 512 and 2048. At `frac:0.75`, staging is 1–4 ms at *every* depth,
and by depth 8192 the decode step is roughly **69% attention and 1.4% staging**.

Two explanations fit a falling curve, and they call for opposite fixes: attention over more keys
costs more, or a deeper prompt leaves the resident pool holding a more diluted working set so the
decode steps that follow miss more. **They are separable here, and the answer is staging.**

**First, what does not explain it.** Device time in the tracked matvec kernels is *flat*, measured
by `examples/ctx_attrib`, which differences the SYCL event counters between a prefill-only pass
and a full pass so the result is decode steps and nothing else:

| depth | tracked device busy (ms/step) | decode-only hit rate | staged MiB/step | experts/step to CPU |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 35.884 | 72.8% | 339.0 | 45.0 |
| 512 | 38.361 | 87.1% | 181.5 | 32.7 |
| 2048 | 36.348 | 68.9% | 379.2 | 47.2 |

The counters are internally consistent, which is the check that they mean what they say: the
router names 36 blocks x 4 = **144 experts per step**, and at depth 128 that is 45.0 sent to the
CPU plus 99 cache demands; 27.0 of those miss at a 72.8% hit rate, and 27.0 x 12.607 MiB =
**340 MiB**, against 339.0 measured.

🔴 **But the tracked device counters cannot see attention at all** — `moearc_track` is called
only from the matvec paths, so `attn_decode` contributes to no row above. And the ordinary
`profile` phases are *host* wall time around a submit on an asynchronous queue, which charges
attention's device cost to whichever phase later drains the queue. Neither instrument, alone,
can answer the question.

**So it was measured with `MOEARC_SYNC_EACH=1`,** which waits after every launch and therefore
makes each phase's host time equal its device time. Decode steps only, warm pass:

| depth | `decode.total` | `attn.attend` | `attn.qkv` | **`moe.stage`** | `moe.host_sync` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 264.70 | 15.21 | 10.42 | 123.36 | 59.46 |
| 2048 | 795.11 | 49.51 | 22.15 | **547.56** | 87.66 |
| **growth** | **+530.41** | **+34.30** | +11.73 | **+424.20** | +28.20 |

**Of the 530 ms a decode step gains between depth 512 and depth 2048, `moe.stage` is 424 ms —
80%. `attn.attend` is 34 ms — 6.5%.** Attention is a real cost and it is growing, but it is not
what is happening to this engine at depth.

⚠️ **Three limits on that conclusion, stated rather than buried.**

- `MOEARC_SYNC_EACH=1` destroys overlap, so **every absolute number in that table is inflated**
  and its throughput columns are meaningless (the run reported a warm pass *slower* than its cold
  one). The **ratio of growth** is the finding, not the milliseconds.
- Under sync-each, `attn.attend` is 36 launches per step, so its 15.21 ms at depth 512 is mostly
  **launch latency, not work** — which is why the growth term, not the total, is the number
  quoted against `moe.stage`.
- The two depths are 512 and 2048. Depth 8192 was not run under sync-each: three prefills at
  ~2 tok/s is over two hours, and the 512→2048 pair already separates the causes.

**A cross-check from the ordinary asynchronous profile agrees.** There, `attn.attend` never
exceeds **0.11 ms/step** at any depth (0.07 / 0.08 / 0.09 / 0.11 at 128 / 512 / 2048 / 8192) —
submit cost only — while `moe.stage` goes 29.68 → 16.98 → 58.09 → **170.45** and `moe.readback`,
the phase that drains the queue, goes 37.15 → 43.56 → 88.25 → **219.90**. The growth lives in
staging and in the drain that follows it.

### 6.5 Two smaller findings worth keeping

- **The pool degrades *within* a single run at constant depth.** Comparing the first eighth of a
  warm run's decode steps against the last eighth: **+43.8%** at depth 128, +11.2% at 512,
  +18.8% at 2048 and **+78.6%** at 8192. Depth moves by at most 64 positions across those steps,
  so attention is very nearly constant over the comparison — this is the working set losing to
  the pool, measured directly.
- **Cold and warm converge as depth grows**, from 2.2x apart at depth 128 (5.62 → 12.12) to
  1.3x at 8192 (1.64 → 2.14). Where the pool cannot hold a useful fraction of what a deep prompt
  touched, the second pass is nearly as cold as the first — which is the same finding from the
  other side.

### 6.6 Raw output

Every table in this section is derived from these, which are the unedited stdout of the runs:

| file | what |
| --- | --- |
| `bench/results/2026-09-06-moearc-depth-curve.txt` | MoEArc 128/512/2048/8192, the 6.3 curve |
| `bench/results/2026-09-06-llamacpp-threads-tg128.txt` | the 6.2 triplicate, `-r 5` x 3 |
| `bench/results/2026-09-06-llamacpp-depth.txt` | llama.cpp at depth, `-t 16` and `-t 8` |
| `bench/results/2026-09-06-llamacpp-threads-tg64.txt` | ⚠️ the warm-up-contaminated `-n 64` sweep, kept because 6.2.1 cites it |
| `bench/results/2026-09-06-moearc-control-and-attribution.txt` | the 17.90 control, and 6.4's differenced device counters |
| `bench/results/2026-09-06-moearc-sync-each.txt` | 6.4's `MOEARC_SYNC_EACH=1` phase table |

### 6.7 Measurement hygiene for this section

Load average was read before every timed run and printed with it. The 1-minute average was
**0.60 to 1.29 before each llama.cpp sweep** and **1.11 to 3.64 before the MoEArc sweeps**; the
box's idle baseline is ~1.2 with its ordinary daemons (an Incus VM, Forgejo, AdGuard, OpenBao).
No other agent was working on the machine, and the two engines were never run concurrently.

⚠️ **The `load` column inside `ctx_curve`'s own table reads high (14–21) at the later depths, and
that is the sweep measuring itself**: `frac:0.5` drives the host pool across 19 threads, so the
1-minute average carries the *previous* depth's own work. It is reported because a row that
cannot say what else was running is not evidence — but the number that shows the box was quiet
is the pre-run one, not the mid-sweep one.

---

## 7. Host policy against depth — and the two things that changes

**Date:** 2026-09-06, later the same day. Run to test a specific worry: that `frac:0.5` — the
setting section 6.3's whole curve was taken at — is a large **loss** at depth, because the host
executor reads expert weights straight out of a 59.03 GiB mmap that cannot fit a 16 GiB ARC, so a
host-routed expert at depth might be a **disk read** and `moe.host_sync` might be the GPU waiting
on it.

**That worry is wrong, and the opposite is true.** Host routing wins at every depth measured, and
`frac:0.5` is not the best setting — it is the *third* best. Two things do come out of it, and
both matter more than the question that prompted it.

### 7.1 Protocol

Section 6.1 exactly — 600 slots, 64 generated tokens, decode-only timing, cold and warm reported
separately, `bench/references/gpt-oss-120b.longctx.ids` — plus four additions, all of which exist
because the first attempt failed:

- **One pinned binary for every run** (`81c22f09091fc949`). Three other agents were committing to
  this repo during the sweep; rebuilding between arms would have compared engines, not policies.
  *(`residency.rs` changed under the sweep and it did not matter: it is the offline trace
  simulator, and the runtime imports only a plain `ExpertRef` from it.)*
- **A gate that looks for processes, not just loadavg.** Load average is a lagging indicator, and
  `ps comm` truncates at 15 characters — so a name-matching check waves the gcc driver
  (`x86_64-linux-gn`) straight through. The gate sums foreign `%CPU` instead, which needs no list.
  🔴 **`nice` protects CPU shares but not the ZFS ARC:** a niced build next door still evicts the
  model's pages, and this workload is bound by what is resident.
- **Disk and ARC counters around every pass** (`/proc/diskstats` on the pool's partition,
  `arcstats`), bracketed **separately for prefill and decode** — at depth 8192 the prompt is 8,192
  steps against 63 decode steps, so one figure for the pass would describe the prompt.
- **Independent invocations, not repeats inside one.** Process-level variance is the variance that
  bit this project (6.2.1).

Every timed invocation below started at a 1-minute load average between **0.39 and 1.23**.

### 7.2 The sweep

Warm tok/s, mean ± half-range over independent invocations:

| depth | `off` | `frac:0.25` | `frac:0.5` | **`frac:0.75`** | `frac:1.0` | best vs `frac:0.5` | best vs `off` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.78 ±0.02 | 10.47 † | 12.32 ±0.07 | **14.93 ±0.04** | 10.33 † | **1.21x** | 2.20x |
| 512 | 8.66 ±0.00 | 11.98 † | 12.41 ±0.10 | **13.38 ±0.03** | 9.86 † | **1.08x** | 1.55x |
| 2048 | 5.37 ±0.06 | 6.92 † | 7.63 ±0.15 | **8.74 ±0.06** | 7.10 † | **1.15x** | 1.63x |
| 8192 | — | — | 1.31 † | **3.71 †** | — | **2.83x** | — |

† n = 1; every other cell is n = 2. Cold, same runs: 128 `off` 5.39 ±1.26 / `frac:0.5` 12.16 ±0.08
/ `frac:0.75` 11.70 ±2.94; 2048 `off` 5.20 ±0.13 / `frac:0.5` 7.79 ±0.02 / `frac:0.75` 8.55 ±0.33;
8192 `frac:0.5` 1.53, `frac:0.75` 3.96. **Cold spreads are 10–25x wider than warm ones**, because
cold depends on what the previous process left in ARC and warm does not.

**There is no crossover in the `off`-versus-host axis.** `off` is worst at every depth; host
routing is worth 1.55–2.20x. The movement is in the *fraction*: **`frac:0.75` wins at every depth,
and its margin over `frac:0.5` grows from 1.08x at 512 to 2.83x at 8192.** The optimum is
**interior** — `frac:1.0` is worse than `0.75` everywhere — so this is not "more host is always
better"; it is that section 3.3's crossover, found at depth 6, **moves with depth**.

⚠️ **Reproducibility is itself a result.** Warm spreads: `frac:0.75` **±0.2–0.7%**, `off`
**±0.1–1.2%**, `frac:0.5` **±0.5–2.0%**, worst at the deepest row of each. The policy that stages
least is the policy whose numbers hold still.

### 7.3 The mechanism: staging is eliminated, not merely reduced

Milliseconds per decode step, warm, ordinary asynchronous profile. **One round (A), so the phase
and disk columns describe the same invocations** — the second round's phases reproduce these to
within 1.3% (`frac:0.75` at 128: `moe.stage` 1.74 against 1.75, `moe.host_sync` 37.06 against
36.60):

| depth | policy | `decode.total` | `moe.stage` | `moe.host_sync` | `moe.readback` | warm-decode disk |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `off` | 147.52 | 86.41 | 0.00 | 52.46 | 0 MiB |
| 128 | `frac:0.5` | 81.46 | 28.04 | 7.64 | 37.05 | 0 MiB |
| 128 | `frac:0.75` | 66.69 | **1.75** | 36.60 | 20.32 | 0 MiB |
| 2048 | `off` | 183.86 | 79.85 | 0.00 | 95.34 | 0 MiB |
| 2048 | `frac:0.5` | 128.39 | 32.19 | 6.79 | 80.30 | 0 MiB |
| 2048 | `frac:0.75` | 113.48 | **3.96** | 38.34 | 63.06 | 50 MiB |
| 8192 | `frac:0.5` | 760.66 | 159.61 | 48.97 | 370.81 | **4,512 MiB** |
| 8192 | `frac:0.75` | 269.73 | **3.82** | 42.72 | 207.85 | **0 MiB** |

🔴 **`moe.host_sync` is flat in depth** — at `frac:0.75` it is 36.60 / 36.50 / 38.34 / 42.72 ms
across 128 → 8192, a 17% rise over a 64x change in depth. **The host executor does not stall more
at depth, and the worry that prompted this section is answered: no.** What heavy host routing does
is send most misses to the CPU *instead of admitting them*, so `moe.stage` — the thing 6.4 calls
80% of the depth penalty — falls to **1–4 ms and stays there at every depth.**

### 7.4 Where the disk actually bites, and where it does not

**Warm-decode disk reads are 0 MiB in almost every cell of the matrix** — staging at depth is
served from RAM, not NVMe. There is exactly one exception, and it is dramatic:

| @ 8192 | `frac:0.5` | `frac:0.75` |
| --- | ---: | ---: |
| staged, whole pass | 2.71 TiB | 0.30 TiB |
| cold-prefill disk read | **1.48 TiB** | 27 GiB |
| **warm-decode disk read** | **4,512 MiB** (71.6 MiB/step) | **0 MiB** |
| warm tok/s | 1.31 | **3.71** |

`frac:0.5` at depth 8192 faults **~55% of its staged bytes from NVMe**; `frac:0.75` faults 9% and
**nothing at all during warm decode**. 📌 **Staging is disk-bound only in the regime where staging
is large — and the policy that removes staging removes the disk dependence with it.** A pinned
host-memory staging ring would not help here; not staging would.

### 7.5 🔴 Does 6.3's curve understate MoEArc? Yes — and one of its rows is not reproducible

Same policy (`frac:0.5`), same protocol, quiet box:

| depth | 6.3 published | this section | delta | 6.3's own `load` column |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 12.12 | 12.32 ±0.07 | +1.7% | 3.64 |
| 512 | 12.62 | 12.41 ±0.10 | −1.7% | 14.45 |
| 2048 | 5.69 | **7.63 ±0.15** | **+34.1%** | 21.05 |
| 8192 | 2.14 | **1.31** | **−38.8%** | 19.57 |

6.3 reproduces at 128 and 512 and **understates by 34% at 2048**, which is where its own load
column reads 21. Its 8192 row is a different problem: this section's clean-box run of the *same
configuration* came out **worse**, and it read **1.48 TiB** from disk doing it. **Two quiet-box
measurements 1.6x apart mean that cell measures the storage, not the engine** (PROTOCOL §4). It
should not be quoted in either direction.

Against the **best** policy rather than `frac:0.5`, the curve is:

| depth | 6.3 (`frac:0.5`) | best (`frac:0.75`) | ratio |
| ---: | ---: | ---: | ---: |
| 128 | 12.12 | **14.93** | 1.23x |
| 512 | 12.62 | **13.38** | 1.06x |
| 2048 | 5.69 | **8.74** | 1.54x |
| 8192 | 2.14 | **3.71** | 1.73x |

**The published 5.9x falloff from 512 to 8192 is 3.6x at `frac:0.75`.** (At `frac:0.5` this
session measured 9.5x — the same instability as the row above.) MoEArc still falls with depth and
llama.cpp still does not; the fall is **smaller than published and the deep end is understated by
up to 1.73x.**

### 7.6 🔴 "Staging, not attention" inverts by depth 8192

Section 6.4 concluded that staging, not attention, dominates the cost of depth. **That was
measured at depths 512 and 2048 only** — its own text says 8192 was skipped for time — **and at
`frac:0.5`, where staging is large by construction.** At `frac:0.75` and depth 8192, `moe.stage` is
**3.82 ms of a 269.73 ms step: 1.4%.**

`MOEARC_SYNC_EACH=1` at `frac:0.75`, matching 6.4's protocol:

| depth | `attn.attend` — this run (`frac:0.75`) | 6.4's run (`frac:0.5`) |
| ---: | ---: | ---: |
| 512 | 15.24 | 15.21 |
| 2048 | 49.65 | 49.51 |

**The two agree to 0.2%, across different policies and different sessions** — as they must, since
attention does not depend on expert routing. That makes `attn.attend` the one trustworthy quantity
in a sync-each run, and it supports an extrapolation the raw milliseconds do not.

gpt-oss declares `sliding_window = 128` on **alternating** blocks, so **18 of 36 blocks are capped
and 18 are full causal**: attention is `const + slope · depth` by construction, not by curve
fitting. The two points give `attn(d) = 3.77 + 0.02240·d` ms, hence **187.3 ms at depth 8192**.
*(6.4's own `frac:0.5` pair gives 186.7 — a 0.3% difference.)*

Testing that against the **asynchronous** `moe.readback`, the phase that drains the queue:

| depth | async `moe.readback` | predicted `attn.attend` | residual | attention as % of decode step |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 20.32 | 6.64 | 13.68 | 10% |
| 512 | 29.18 | 15.24 | 13.94 | 20% |
| 2048 | 63.06 | 49.65 | 13.41 | 44% |
| **8192** | **207.85** | **187.29** | 20.56 | **69%** |

**The residual is constant at 13.68 ± 0.27 ms across three depths spanning 16x.** `moe.readback` is
attention's device time plus a fixed ~14 ms, which is what a queue-drain phase should look like.

📌 **At depth 8192 the decode step is roughly 69% attention and 1.4% staging.** 6.4's conclusion is
correct in the regime it measured and **inverts outside it** — and the inversion is partly *caused*
by fixing the staging: eliminate staging and attention is what is left. **Attention kernel work,
not a staging optimisation, is what buys the deep end.**

⚠️ **Limits, stated rather than buried.** The 8192 attention figure is an **extrapolation 4x beyond
the measured range**, not a measurement. It rests on three things — the SWA structure that makes
the linear form correct, reproduction of both anchor points across two independent runs, and the
constant residual at three depths — but the residual at 8192 is **20.56 against 13.68**, 50%
larger, so something there is either slightly super-linear or not accounted for. A direct
measurement would need sync-each at 8192, which this session costed at **8–12 hours** from its own
prefill scaling (86 s at depth 512, 1,330 s at 2048) and did not run. ⚠️ The sync-each run's
`moe.stage` and `moe.host_sync` are **discarded**: it faulted 507 GiB during the 2048 prefill and
measured the storage.

### 7.7 A one-line fix that would make this cheap

`moearc_track` is called **only** from the matvec paths in `kernels.cpp`, so `moearc_attn_decode`
appears in no device-counter row and `MOEARC_PROFILE_EVENTS=1` — which attributes per-kernel device
time *without* serialising — cannot see attention either. **That is the only reason section 6.4 and
this section both had to reach for `MOEARC_SYNC_EACH`,** and it is why the number above is an
extrapolation rather than a measurement.

📌 **Adding one `moearc_track` call to `moearc_attn_decode` would make attention attributable from
an ordinary asynchronous run at any depth** — about 45 minutes at depth 8192 instead of 8–12 hours,
and permanently, for every future depth question. *(Not made here: `crates/moearc-kernels/` is not
this section's to edit.)*

### 7.8 Raw output

| file | what |
| --- | --- |
| `bench/results/2026-09-06-moearc-host-policy-depth.txt` | 7.2's sweep, rounds A and B |
| `bench/results/2026-09-06-moearc-host-policy-8192.txt` | the depth-8192 rows |
| `bench/results/2026-09-06-moearc-sync-each-frac075.txt` | 7.6's sync-each pair |
| `bench/results/2026-09-06-moearc-host-policy-DISCARDED-round1.txt` | ⚠️ the discarded first round — see below |

⚠️ **The discarded round is in the tree on purpose.** Its gate was loadavg-only and its
process check matched on truncated names, so it ran against another agent's build. The tell was
internal, not external: **`frac:0.25` at depth 512 reported warm 1.43 against cold 3.63.** A warm
pass slower than the cold pass before it is not a measurement, whichever number you prefer.
Controlled re-runs of those same cells gave **10.47 and 11.98** — an 8–13x swing from box state
alone, which is the fourth route this repo has found to the same class of contamination.
