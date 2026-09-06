# Baseline: gpt-oss-120B MXFP4 — 59 GiB of weights on an 11.33 GiB card

**5.2x the card.** This is the size target MoEArc exists for, and the first model where the
memory strategy and the host executor are both load-bearing rather than optional.

**Hardware:** Arc B580 12 GB (`level_zero:0`), Core Ultra 7 265K (20 threads), 96 GB DDR5.
**Model:** `gpt-oss-120b-MXFP4.gguf` — 36 blocks, 128 experts, 4 active, `n_embd` 2880,
`n_ff_exp` 2880, 64 query heads over 8 KV heads at `head_dim` 64.
**Split:** 56.73 GiB of experts (96.1%) against 2.29 GiB of dense weights (3.9%).
**Slot:** 12.607 MiB — **4.3x** Qwen3-30B-A3B's 2.92 MiB. 4,608 slots is **56.7 GiB**.
**Date:** 2026-09-05. **llama.cpp** `e107984bcffcfd701e82738092a2b000b6fda7a2`.

---

## 1. The incumbent: llama.cpp

`llama-bench -m <model> -n 128 -r 2 -ncmoe <N>`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`.

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
Only five blocks' experts fit beside the dense weights and the KV cache. As on Qwen3-30B-A3B,
throughput falls monotonically as more work moves host-side, so 31 is the minimum it must offload
rather than a tuning choice.

⚠️ **A second run three hours later gave 14.43–14.90 across the same settings** (`-r 3`), with
`pp512` down from 116 to 87 and its error bar up from ±0.3 to ±15.7. Nothing about llama.cpp
changed; the box did. **The number to beat is therefore taken as the *higher* of the two
measurements, 15.47** — the conservative choice, since it is the one MoEArc has to clear.

---

## 2. MoEArc

`examples/hybrid_sweep`, prompt `[976, 9029, 5030, 328, 10128, 382]`
(`The capital city of France is`), 64 generated tokens, `n_ctx = 128`.
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

**Best: 600 slots at `frac:0.5` — 17.97 tok/s warm, 16.03 cold, against llama.cpp's 15.47.**
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

🔴 **Sliding-window attention is declared and NOT implemented.** The file states
`attention.sliding_window = 128`, which llama.cpp applies to **alternating** blocks
(`set_swa_pattern(2)`: even blocks windowed, odd blocks full causal). `moe.rs` **refuses a
context longer than the window by name** rather than attending to keys llama.cpp masks. Below
128 tokens the two masks are identical — `is_masked_swa` masks when `p1 - p0 >= n_swa`, which no
pair of positions inside one window satisfies — so everything above is exact, not approximate.
Implementing it means two masks and two KV caches, not one shorter cache.

---

## 5. Reproducing

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

⚠️ **Run it on an idle machine and check `uptime` first** — see 3.4.
