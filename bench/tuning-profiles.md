# Tuning profiles — llama.cpp + SYCL on Intel Arc B580

**What this is.** The settings MoEArc ships so a user never has to guess a flag. For each model we
measured what each knob is actually worth on this card, where the `-ncmoe` OOM floor sits, how the
winning configuration behaves as context grows, and how much VRAM it leaves free.

**What this is not.** A comparison against anything. There is no MoEArc-vs-llama.cpp number here —
llama.cpp + SYCL *is* the engine. This is its calibration.

---

## Machine and build

| | |
|---|---|
| GPU | Intel Arc B580, `level_zero:0` — 12216 MiB total, **11959 MiB free** at load |
| CPU | Intel Core Ultra 7 265K — **20 cores (8 P + 12 E)**, no SMT |
| RAM | 91 GiB. ZFS `zfs_arc_max` = **16 GiB** |
| llama.cpp | `e107984bc`, build 10788, **SYCL** backend — asserted in the `backends` field of every CSV row |
| Binary | `/zfs/swift/projects/llama.cpp/build/bin/llama-bench` (**not** `build-vulkan/`) |
| Idle baseline | load1 0.65–0.75, CPU busy 1.2–2.6 % of the machine |
| Raw data | `bench/results/tuning/*.txt` — CSV plus per-run guard, `/proc/diskstats` and ZFS ARC counters |
| Harness | `bench/tuning/harness.sh`; drivers `bench/tuning/d1.sh` … `d10.sh` |

Every timed run printed its 1-minute load average **and** a 2-second measured CPU-busy percentage
before starting, and **waited** (up to 300 s) rather than measuring above load 2.5 / busy 12 %.
Busy % is computed from `/proc/stat` deltas, not from `ps` — `ps comm` truncates at 15 characters
and has waved a contaminating process through before.

---

## The noise floor, and the decision rule

Inside a single `llama-bench` process, cells are very stable — stddev is typically **0.0–0.5 %** of
the mean. **That is not the error bar that matters.** Across *independent invocations* of the same
cell the spread is larger:

| cell | independent readings | spread |
|---|---|---|
| gpt-oss-20b, ncmoe 24, t16, d0 | 43.87, 44.02, 44.51, 44.59 | **1.6 %** |
| gpt-oss-120b, ncmoe 36, t16, d0 (warm) | 29.39, 29.51, 29.56, 29.77 | **1.3 %** |
| Qwen3-30B, ncmoe 21, t16, d8192 | 51.74, 52.97, 53.08, 53.11 | **2.6 %** |
| Qwen3.6-35B, ncmoe 22, t16, d0 | 54.04, 55.58 | **2.8 %** |

**Decision rule applied throughout: a setting is better only if it beats the alternative by more
than ~3 %.** Smaller differences are reported as *not separable*, and where two settings tie we
recommend the one with more VRAM headroom. **This rule changes four recommendations below**
(gpt-oss-120b ncmoe, Qwen3-30B threads, Qwen3.6 threads, Scout ncmoe) — in each case away from the
nominally faster setting and towards the safer one.

Throughout, **the first cell of every process is discarded** as warm-up (PROTOCOL §6) — llama-bench
pays process warm-up on its first test regardless of its own internal warmup. Every sweep was
written with its first parameter value duplicated so the discard costs nothing.

---

## Summary — ship these

| model | size | `-ncmoe` | `-t` | KV | tok/s @ d0 | @ d8192 | VRAM @ 8 K | free |
|---|---|---|---|---|---|---|---|---|
| `olmoe-1b-7b-0924-instruct-q4_k_m` | 3.92 GiB | **0** | any | f16 | **282.9** | n/a (ctx 4096) | 3995 MiB @ ctx 256 | 7964 MiB |
| `gpt-oss-20b-MXFP4` | 11.28 GiB | **24** (all) | 16 | f16 | **44.25 ± 0.36** | 43.40 | **1862 MiB** | **10097 MiB** |
| `Qwen3-30B-A3B-Q4_K_M` | 17.28 GiB | **21** (8 K) / 28 (32 K) | 16 | f16 | 69.23 | **52.73 ± 0.65** | 11269 MiB | 690 MiB |
| `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` | 17.28 GiB | **21** (8 K) | 16 | f16 | 69.38 | **53.47** | 11269 MiB | 690 MiB |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | 20.61 GiB | **22** | 16 | f16 | **54.81 ± 1.09** | 52.79 | 11100 MiB | 859 MiB |
| `Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL` | 45.65 GiB | **46** | 16 | f16 | **18.69 ± 0.47** | 11.68 | 9992 MiB | 1967 MiB |
| `gpt-oss-120b-MXFP4` | 59.02 GiB | **36** (all) | 16 | f16 | **29.56 ± 0.16** | 28.52 ± 0.13 | **2543 MiB** | **9416 MiB** |

`tok/s` is `tg64`, warm, pooled over independent invocations. Flash attention is **on** everywhere
(llama.cpp's `auto` resolves to on); do not turn it off. KV cache stays **f16** everywhere.

---

## What each knob was worth

### `-t` — the most valuable flag, but only when experts are on the CPU

`llama-bench` defaults to **4 threads**. On the flagship, measured warm on a quiet box with the two
arms **interleaved** (`-t 16,4,16,4,16`) so that cache state is a common-mode error:

| gpt-oss-120b, ncmoe 36 | readings | mean |
|---|---|---|
| `-t 4` (the default) | 13.88, 14.34 | **14.11 ± 0.33** |
| `-t 16` | 29.51, 29.56 | **29.54 ± 0.04** |

**2.09×.** That is the entire argument for shipping a tuning layer.

Its value scales with how much work is host-side and **collapses to nothing when none is**:

| model | config | measured range | `-t` is worth |
|---|---|---|---|
| olmoe | ncmoe **0** — all on GPU | t4→t20: 282.83, 282.81, 282.13, 282.88, 282.77 | **0.27 % — nothing** |
| Qwen3.6-35B | ncmoe 22 | t8 53.18 → t16 54.04 | **1.6 % — not separable** |
| Qwen3-30B | ncmoe 18 | t8 68.20 → t16 73.04 | **1.07×** |
| Llama-4-Scout | ncmoe 46 | t8 14.27 → t16 18.52 | **1.30×** |
| gpt-oss-120b | ncmoe 36 | t4 14.11 → t16 29.54 | **2.09×** |
| olmoe | ncmoe 16 — all on CPU | t2 49.07 → t12 123.94 | **2.53×** |
| gpt-oss-20b | ncmoe 24 | t2 12.16 → t16 44.55 | **3.66×** |

Note the top and bottom rows are **the same model**. `-t` and `-ncmoe` are not independent knobs and
cannot be tuned separately.

**`-t 16` wins, and `-t 20` (= `nproc`) is worse on five of the six models where threads matter at
all**: 29.39→26.96 (gpt-oss-120b, −8.3 %), 44.55→~39.7 (gpt-oss-20b, −10.8 %), 73.04→68.96
(Qwen3-30B, −5.6 %), 54.04→50.10 (Qwen3.6, −7.3 %), 123.94→110.50 (olmoe at full offload, −10.8 %).
Only Scout is a tie (18.52 vs 18.08, 2.4 %). Cells at t≥20 are also markedly noisier — stddev
1.0–2.2 against 0.0–0.5 below it. **Never set `-t` to `nproc` on a hybrid Intel CPU.**

### `-ncmoe` — the direction depends on the *quantisation*, not the model size

This inverted the prior and is the most useful thing here. Two populations:

**MXFP4 (`gpt-oss`) — offload *everything*.**

The cleanest evidence is an **A/B/A/B run inside a single process** (`-ncmoe 24,0,24,0`), which
controls for drift and read only 22 MiB from disk:

| gpt-oss-20b, t16, d0, interleaved | readings | mean |
|---|---|---|
| `-ncmoe 0` — every expert on the **GPU** | 36.35, 36.36 | **36.36 ± 0.01** |
| `-ncmoe 24` — every expert on the **CPU** | 43.36, 43.47 | **43.42 ± 0.08** |

**1.19×.** The whole 11.28 GiB model *fits* on the card, and putting it there is **19 % slower**
than running every expert on the CPU.

The full sweep agrees, pooled over two independent invocations:

| gpt-oss-20b @ t16, d0 | 0 | 4 | 8 | 12 | 14 | 16 | 18 | 20 | 22 | **24** |
|---|---|---|---|---|---|---|---|---|---|---|
| tok/s | 36.35 | 37.60 | 38.89 | 39.85 | 40.38 | 41.11 | 42.37 | 42.58 | 43.86 | **43.88** |

The trend across the range is large and unambiguous (**+21 %** from 0 to 24), but **adjacent points
within ~1 % are not separable** — the second invocation inverts the 18/20 and 22/24 pairs by ~0.4 %
each. Read the direction, not the individual steps. The conclusion survives at depth (d8192:
ncmoe 24 → 42.97, 16 → 40.52, 8 → 37.75, 0 → OOM).

The plausible reading is that the SYCL MXFP4 matmul path is weak relative to this CPU's. **We did
not profile the kernel, so that is a hypothesis, not a finding** — but it is the obvious thing to
take upstream.

**Q4_K / Q3_K (`Qwen3`, `Qwen3.6`, `Llama-4`) — offload as *little* as VRAM allows.**

| Qwen3-30B @ t16, d0 | 30 | 28 | 26 | 24 | 22 | 20 | 19 | **18** (floor) |
|---|---|---|---|---|---|---|---|---|
| tok/s | 58.58 | 62.40 | 64.59 | 66.15 | 68.83 | 70.60 | 73.22 | **74.08** |

Monotone the other way — 18 beats 30 by **1.26×**. Same shape for Qwen3.6 (22 → 55.58 vs 28 →
50.85, 1.09×).

**The flagship is the exception, and it makes the recommendation easy.** gpt-oss-120b is flat
across its whole feasible range:

| gpt-oss-120b @ t16, d0, warm | **36** | 34 | 33 | 32 | 31 (floor) |
|---|---|---|---|---|---|
| tok/s | **29.56 ± 0.16** | 29.61 ± 0.01 | 28.98 | 29.31 | 29.00 ± 0.33 |

36 vs 31 is **1.9 % — inside the noise floor, so on throughput they tie.** The VRAM does not tie:

| gpt-oss-120b @ 8 K ctx | model on SYCL0 | KV | compute | total | **free** |
|---|---|---|---|---|---|
| `-ncmoe 31` (the previously published config) | 9697.06 | 324.00 | 611.15 | 10632 MiB | 1327 MiB |
| `-ncmoe 36` (recommended) | 1607.61 | 324.00 | 611.15 | **2543 MiB** | **9416 MiB** |

🔴 **The old configuration sat exactly on the OOM floor and gave up 8.1 GiB of VRAM for a difference
we cannot measure.** `-ncmoe 36` is the same speed, cannot OOM from expert placement at any context,
and the freed VRAM is precisely what makes 32 K context reachable (26.83 tok/s, below).

The same logic applies to **Llama-4-Scout at 8 K**, where ncmoe 45/46/47/48 all land in an 11.54–
11.68 band (1.2 %). Ship **46**, not the floor of 45, and take the headroom for free.

### `-fa` — leave it on, but know what it is worth

| Qwen3-30B, ncmoe 21, d8192 | tok/s |
|---|---|
| `-fa on` | 53.11 ± 0.04, 51.74 ± 0.23 |
| `-fa off` | **30.52 ± 0.13** |

**1.72×.** `auto` resolves to on, so the default is already right — but `-fa off` is a trap worth an
explicit guard in the CLI, because it is a flag users copy from other backends' advice.

### KV cache quantisation — a loss on this card, not the usual free win

The hypothesis was that `--cache-type-k/v q8_0` halves KV and buys back offload blocks. Measured on
Qwen3-30B at d8192, the model with the largest KV in the set:

| | tok/s @ ncmoe 21 | `-ncmoe` floor |
|---|---|---|
| `f16` (default) | **52.97** | 21 |
| `q8_0` | **42.87** (−19 %) | 20 |

It costs **19 % of throughput** and buys back exactly **one** block. Worse, at ncmoe 19 it does not
OOM cleanly — it hard-aborts inside `ggml_backend_sycl_synchronize`
(`ggml-sycl/common.hpp:141: SYCL error`, exit 134), which a CLI cannot handle as gracefully as a
returned OOM. **Do not ship quantised KV on Arc.** Untested above 8 K, where its case would be
strongest; the abort makes that hard to explore.

---

## The `-ncmoe` floor, and how it moves with context

Below the floor llama.cpp fails at load / first graph with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY (39)` — usually `Error OP MUL_MAT` at `ggml-sycl.cpp:5617`
or `:5155`, occasionally surfacing through a `memcpy`. It is a clean, detectable failure the CLI
can catch and retry one block higher.

| model | blocks | floor @ d0 | floor @ d8192 | floor @ d32768 |
|---|---|---|---|---|
| olmoe | 16 | 0 | — (ctx max 4096) | — |
| gpt-oss-20b | 24 | 0 | **2** | not measured |
| Qwen3-30B | 48 | **18** | **21** | **28** |
| Qwen3-Coder-30B | 48 | **18** | ≤ 21 (21 verified) | not measured |
| Qwen3.6-35B | 40 | **22** | **22 — unchanged** | not measured |
| Llama-4-Scout | 48 | **44** | **45** | not measured |
| gpt-oss-120b | 36 | **31** | **31** | ≤ 36 (36 verified) |

Two things fall out of this table:

- **The floor is a function of context, not of the model alone.** Qwen3-30B needs 18 blocks
  offloaded at d0 and **28** at 32 K — ten blocks consumed by KV growth. A profile that quotes a
  floor without quoting a context is wrong, and shipping the d0 floor would OOM every long-context
  user. This is the main reason the JSON below is indexed by context tier.
- **Qwen3.6-35B's floor does not move at all.** Its KV at 8 K is **165 MiB** (plus a 63 MiB
  recurrent-state buffer) against Qwen3-30B's **792 MiB**, because only 10 of its 40 blocks are
  attention — the other 30 are SSM. The hybrid architecture is directly visible in the allocator,
  and it is why this model is the cheapest of the set to run long.

---

## Behaviour across depth

Recommended configuration, warm, `tg64`, first cell of each process discarded:

| model (ncmoe) | d0 | d512 | d2048 | d8192 | d32768 | d65536 | d0→d8192 |
|---|---|---|---|---|---|---|---|
| olmoe (0) | 282.90 | 275.78 | 241.41 | — | — | — | 1.29× to d3072 |
| **gpt-oss-20b (24)** | 44.02 | 44.21 | 43.73 | 43.40 | 39.62 | **36.88 ± 0.03** | **1.01× — flat** |
| **gpt-oss-120b (36)** | 29.56 | 26.89¹ | 28.96¹ | 28.52 | **26.83 ± 0.11** | — | **1.04×** |
| Qwen3-30B (21) | 69.23 | 68.81 | 63.73 | 52.73 | ~29 ² | — | 1.31× |
| Qwen3-Coder-30B (21) | 69.38 | 68.62 | 63.56 | 53.47 | — | — | 1.30× |
| Qwen3.6-35B (26) | 51.66 | 53.49 | 52.67 | 51.19 | — | — | **1.01× — flat** |
| Llama-4-Scout (46) | 19.22 | 18.38 | 16.42 | **11.68 ± 0.00** | — | — | **1.65×** |

¹ These two cells come from a run whose d0 cell was cold; they are climbing out of that, not
  reporting a depth effect. Read the warm d0 and the d8192 figure as the reliable pair.
² `-r 1`, single shot, **indicative only.** Reported because the *floor* that run established
  (ncmoe 28) is a binary fact, but the throughput has no error bar and is not a measurement.

**Both gpt-oss models barely notice depth.** Combined with ncmoe 36's 9.4 GiB of free VRAM, that
makes gpt-oss-120b the standout result of this exercise: a **59.02 GiB model on an 11.7 GiB card at
32 K context, 26.83 ± 0.11 tok/s**. gpt-oss-20b reaches **65 K at 36.88 ± 0.03**.

**Llama-4-Scout is the opposite** and users should be warned in the CLI: 192 KiB of KV per token
against 1967 MiB of headroom means it loses **1.65×** by 8 K, and longer context forces `-ncmoe`
higher, which costs again.

---

## Model-by-model notes

### `olmoe-1b-7b-0924-instruct-q4_k_m` — 3.92 GiB · 16 blocks · 64 experts / 8 active · ctx 4096
`-ncmoe 0 -t 16`. The whole model lives on the card with **7964 MiB** spare. `-t` is worth
**nothing** (0.27 % across t4→t20) — the only model where that is true, and a useful control: it
confirms the thread effect elsewhere really is host-side expert compute. Offloading is strictly
harmful — ncmoe 2 costs 11 %, ncmoe 16 costs **55 %**.

### `gpt-oss-20b-MXFP4` — 11.28 GiB · 24 blocks · 32 experts / 4 active · ctx 131072
`-ncmoe 24 -t 16`. **Counter-intuitive and strongly supported: put every expert on the CPU even
though they all fit on the GPU.** Uses **1862 MiB** of VRAM, leaving **10097 MiB** — which is why it
reaches 65 K context at 36.88 tok/s. Depth costs essentially nothing to 8 K. The `-t` sensitivity is
the highest in the set (3.66×), so this model punishes a mistuned thread count hardest.

### `Qwen3-30B-A3B-Q4_K_M` — 17.28 GiB · 48 blocks · 128 experts / 8 active · ctx 40960
`-ncmoe 21 -t 16` for an 8 K default; **raise to 28 for 32 K**. Fastest model in the set at short
context (74.08 at ncmoe 18 / d0) but pays 1.31× by 8 K. Runs with only **690 MiB** free — the
tightest configuration we ship, deliberately, because throughput rises monotonically as `-ncmoe`
falls. `-t 16` beats `-t 12` by 0.9 %, which is **not separable**; either is fine.

### `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` — 17.28 GiB · same architecture
**The profile transfers, and we checked rather than assumed** (PROTOCOL §9: a measurement
transferred from one model is not a measurement of another). Independently measured: floor 18
(identical); ncmoe 18 → **72.77 ± 0.13** against Qwen3-30B's like-for-like **73.04 ± 0.09**
(Δ 0.4 %); and the full depth curve matches at every point within 0.7 %
(69.38 / 68.62 / 63.56 / 53.47 against 69.23 / 68.81 / 63.73 / 52.73). Ship the same flags.

### `Qwen3.6-35B-A3B-UD-Q4_K_M` — 20.61 GiB · 40 blocks (10 attention + 30 SSM) · 256 experts / 8 active
`-ncmoe 22 -t 16`. **Flat with depth (1.01× to 8 K) and its floor does not move with context** —
both consequences of the hybrid stack. `-t` is worth 1.6 % here (not separable), so this is the one
model where getting the thread count wrong is cheap.

⚠️ **A retraction.** An early sweep produced ragged, non-monotone numbers (45.97 ± 4.02 at ncmoe 22,
an 8.7 % stddev) and we nearly wrote *"do not run Qwen3.6 at its floor"*. A clean repeat at `-r 5`
gave a smooth monotone curve with stddev ≤ 0.44 and put the best value **at** the floor. The
original run had passed the guard at busy 5.8 %, the highest of any run in this exercise.
**That conclusion is withdrawn; the discarded run stays in the tree** as
`qwen35-ncmoe.txt` with its error bars visible.

### `Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL` — 45.65 GiB · 48 blocks · 16 experts / **1** active
**llama.cpp supports it** — arch `llama4` loads, generates and benchmarks cleanly, so the "verify
before spending time" question is answered yes. `-ncmoe 46 -t 16`. Slowest of the set and by far
the most depth-sensitive. At 8 K, ncmoe 45–48 are indistinguishable (11.54–11.68), so **ship 46,
not the floor of 45**. Its early d0 numbers (14.5–16.8, stddev 7–16 %) were cold-cache artefacts;
warm they are 18.32 / 18.52 / 19.22 with stddev ≤ 0.27.

### `gpt-oss-120b-MXFP4` — 59.02 GiB · 36 blocks · 128 experts / 4 active · ctx 131072
`-ncmoe 36 -t 16`. **29.56 ± 0.16 tok/s at d0, 28.52 ± 0.13 at 8 K, 26.83 ± 0.11 at 32 K, with
9416 MiB of VRAM free.** Moving `-ncmoe` from the previously published 31 to 36 is free in
throughput and worth **8.1 GiB**.

---

## Things that surprised us

1. **The direction of `-ncmoe` flips with quantisation.** We expected "offload as little as VRAM
   allows" for every model. That holds for Q4_K/Q3_K and is **exactly backwards for MXFP4**, where
   the fastest configuration puts *every* expert on the CPU — including for a model that fits
   entirely in VRAM. A tuner that assumes one direction will be badly wrong on half this catalogue.
2. **`-t` is worth 2–3.7× or literally nothing**, with nothing in between, and which one depends
   entirely on `-ncmoe`. The prior treated them as an ordered list of independent knobs; they are
   not independent.
3. **`nproc` is never the answer** on this hybrid CPU — `-t 20` lost on five of six models and got
   noisier as well as slower.
4. **The published `-ncmoe 31` flagship config was sitting on the OOM floor for no measurable gain**,
   costing 8.1 GiB of VRAM and the 32 K context that VRAM buys.
5. **Quantised KV is a 19 % loss on Arc**, and fails by abort rather than by clean OOM.
6. **The `-ncmoe` floor moves with context** — ten blocks between d0 and 32 K on Qwen3-30B. A
   single-number floor would OOM users.
7. **Qwen3.6's hybrid stack is visible in the allocator** (165 MiB of KV at 8 K where a same-size
   dense-attention MoE needs 792 MiB) and that single fact explains both its flat depth curve and
   its immovable floor.

---

## What we could not measure, and why

- 🔴 **The `-t` optimum between 14 and 18 on gpt-oss-120b.** Two attempts, both page-cache-bound,
  and they **disagreed**. The failure is instructive: the model is **59.02 GiB** against 91 GiB of
  RAM shared with other work, so a multi-cell sweep faults tens of gigabytes *during* the sweep.
  Both runs show throughput **rising monotonically through the cell order** — the signature of a
  cache filling, not of thread scaling — with 42–87 GiB of disk reads and stddev up to 36 % of the
  mean. A one-cell primer run did not fix it. Per PROTOCOL §9 both are **withdrawn, not replaced**;
  they remain in `oss120-t-fine.txt` and `oss120-t-fine2.txt` with their disk counters so the
  discard is auditable. The `-t 4` vs `-t 16` figure survives only because its arms were
  **interleaved**, making cache state a common-mode error. `-t 16` is recommended on the strength
  of the coarse warm sweep (70 MiB of disk reads), not the fine one.
- **`-ncmoe` floors at 32 K for gpt-oss-20b, Qwen3-Coder, Qwen3.6 and Scout**, and any floor at
  64 K or 128 K. Deprioritised: each probe costs a full prefill, and the flagship's 32 K result was
  the one the headroom claim needed.
- **Quantised KV above 8 K**, where its case would be strongest — it aborts before it can be
  characterised.
- **`-ngl`.** Left at `-1` (all layers) throughout. With `-ncmoe` doing the expert placement, `-ngl`
  is the wrong lever for MoE models and we did not spend runs on it.
- **Prompt processing.** Every number here is decode (`tg`). Prefill tuning (`-b` / `-ub`) is a
  separate exercise.
- **Why MXFP4 prefers the CPU.** We measured *that* it does — two models, two depths, thirteen
  monotone points on one of them. We did not profile the kernel, so the cause remains a hypothesis.

### A known limitation of the instrumentation

Disk and ARC counters are recorded **per invocation**, not per cell, so a sweep that reloads the
model between cells cannot separate load-time reads from decode-time reads. Where a cell's stddev
is ≤ 0.5 % we treat decode as cache-resident; where it exceeds ~3 % we treat the run as suspect and
repeat it. That heuristic caught every contaminated run in this exercise, but a per-cell counter
would be strictly better and is worth building into `moearc bench`.

---

## Provenance

Raw CSV, guard readings, disk and ARC counters: `bench/results/tuning/`.
Harness: `bench/tuning/harness.sh`. Drivers: `bench/tuning/d1.sh` … `d10.sh`.
Machine-readable profiles: `bench/tuning-profiles.json`.

Absolute throughput is an artefact of this machine (PROTOCOL §0). What should reproduce on another
Arc box is the **shape**: the MXFP4-vs-Q4_K inversion in `-ncmoe`; the collapse of `-t`'s value as
experts move onto the GPU; `nproc` losing to 16; flash attention's ~1.7× at depth; quantised KV
losing; and the floor moving with context.
