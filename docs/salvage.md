# Salvage — what the retired engine and kernels knew

Written 2026-09-08, ahead of the retirement described in [`pivot-inventory.md`](pivot-inventory.md).

**Why this file exists.** ~19,150 Rust and 2,411 C/C++ lines are about to be deleted. Most of
that is implementation and should go. A minority of it is **knowledge that cost GPU hours and
several wrong turns to obtain**, and it lives in doc comments that will go with the code. This
is that knowledge, lifted out and organised by topic so the question *"did we already test
X?"* is answerable in one read.

**What is here.** Measured numbers with their conditions; hypotheses that were killed by
measurement; invariants where the comment explains a trap rather than the code; and ideas that
survive the pivot conceptually even though the code does not.

**What is not here.** Our RMSNorm. Our attention loop. Anything useful only for rebuilding the
thing being deleted.

**Reading rules, applied throughout.**

- Every item names **the claim, the number, how it was measured, and where it came from**.
- A number whose source states no method is marked **⚠️ unattributed** rather than dressed up.
- A **chosen constant** is never presented as a result. §7 is the register of every one found.
- Where an existing document already covers something properly — [`negative-results.md`](negative-results.md),
  [`calibration.md`](calibration.md), [`hardware-sizing.md`](hardware-sizing.md),
  [`../bench/PROTOCOL.md`](../bench/PROTOCOL.md), [`../bench/policy-sweep.md`](../bench/policy-sweep.md),
  [`../bench/tuning-profiles.md`](../bench/tuning-profiles.md) — this file **cross-references
  and does not restate**.

🔴 **Nothing has been deleted. Do not delete anything on the strength of this file alone** —
§9 is the checklist, and it is gated on `moearc serve` proving the replacement.

---

## Contents

1. [Memory and allocation](#1-memory-and-allocation)
2. [Transfer, PCIe and staging](#2-transfer-pcie-and-staging)
3. [Scheduling, overlap and the token loop](#3-scheduling-overlap-and-the-token-loop)
4. [Kernels — tunings that paid, and hypotheses that died](#4-kernels--tunings-that-paid-and-hypotheses-that-died)
5. [Correctness and comparison methodology](#5-correctness-and-comparison-methodology)
6. [Build, linking and process traps](#6-build-linking-and-process-traps)
7. [Register of chosen constants](#7-register-of-chosen-constants)
8. [Contradictions found while harvesting](#8-contradictions-found-while-harvesting)
9. [Deletion checklist](#9-deletion-checklist)

---

## 1. Memory and allocation

### M1 · `malloc_device` succeeds far past physical VRAM

**Claim.** On the Level Zero runtime on the reference box, allocation success is not evidence
that the memory exists. Pages are not committed until touched.

**Number.** A first probe allocated expert-sized blocks (2,039,808 B, the measured per-expert
size of Qwen3.6-35B-A3B-UD-Q4_K_M) until failure and "succeeded" **20,001 times — 38 GiB on an
11.33 GiB card**.

**How measured.** `tools/vram_probe.cpp` (SYCL, `icpx -fsycl`). The corrected probe writes
`0x5A` over every block with `q.memset(...).wait_and_throw()` before counting it, and stops at
120% of reported free so an over-committing allocator cannot run the driver into a reset.

**Consequences, both of which outlive our engine.**
1. **A cache cannot size itself by allocating until failure on this driver.** It would sail past
   the end of the card and fail later, elsewhere, as a hang or a device reset. Computing the
   budget up front is the only method that works here.
2. **Any future probe must write to what it allocates before counting it** — and the trap applies
   to llama.cpp exactly as it applied to us.

**Where from.** `tools/vram_probe.cpp` header (KEEP); `moearc-engine/src/moe.rs`
`Residency::All`; `moearc-kernels/src/lib.rs` `Context::alloc`. Already recorded in
[`calibration.md`](calibration.md) → *"Measurement: allocator overhead on Arc B580"*.

---

### M2 · The ~85%-of-reported-free allocation cliff, and it is silent

**Claim.** A pool sized above roughly **85% of what the device reports free** loads
successfully and then fails on the *first token*. No allocation return value discovers it.

**Numbers.** Arc B580 reporting 11.33 GiB free, Qwen3-30B-A3B, `n_ctx = 512`, 951 MiB of dense
weights:

```
  3050 slots   8899 MiB pool   9.67 GiB committed   runs
  3100 slots   9045 MiB pool   9.81 GiB committed   "embedding lookup failed"
  3157 slots   9212 MiB pool   9.97 GiB committed   "host-to-device copy failed"
```

**How measured.** `ExpertPool::new` issues one `malloc_device` per slot per bank — 9,300
allocations at 3,100 slots — every one of which returns a valid pointer. The failure surfaces as
a host-to-device copy or a kernel launch. ⚠️ Near the boundary **the driver can spin at 100% of
a core for minutes before it says so.**

🔴 **`Headroom::PROVISIONAL = 0.12` leaves 88%, so `plan()` chooses 3157 slots — the first
setting past the cliff.** The constant was deliberately left alone: changing it moves every plan
on every device and model, and one measurement on one card with one model is not a basis for
that. See §7 and §8·X2.

**Where from.** `moearc-engine/src/moe.rs` `Residency::All`; `moearc-engine/src/memory.rs`
`Headroom::PROVISIONAL`.

---

### M3 · Allocator overhead contributes **zero** to headroom

**Claim.** More memory is committable than the driver reports free, so none of the 12% headroom
is justified by allocator overhead or fragmentation.

**Number.** Reported free 12,168,933,376 B (11.33 GiB); committed before the first write failure
12,418,351,104 B (11.57 GiB) — **+2.05%**. Reproducible to the byte across three consecutive
runs: 6,088 expert-sized blocks every time.

**How measured.** `tools/vram_probe.cpp`, nothing else on the GPU (the B580 is not the display
device on that machine — the Arrow Lake iGPU is, `boot_vga=1`).

🔴 **This is a floor, not the answer.** It measures allocator overhead and fragmentation only —
no activations, no scratch, no kernel working sets, because there were no kernels yet.

**Where from.** [`calibration.md`](calibration.md), already recorded there in full. Reproduced
here only because §8·X2 needs both halves side by side.

---

### M4 · A slot pool is **larger than the bank it holds**, by ~7%

**Claim.** One pool slot holds one *(block, expert)* pair across all three banks, and each
array's slot must be sized to the largest that bank reaches in **any** block — because a GGUF
quantises the same bank differently in different blocks. A budget built from `expert_bytes`
rather than `slot_bytes` over-promises.

**Number.** Qwen3-30B-A3B quantises `ffn_down_exps` at Q6_K in half its blocks and Q4_K in the
rest. 6,144 slots × 2.92 MiB commit **17.51 GiB to store 16.35 GiB of experts** — a 1.16 GiB
difference, **7%**.

**How measured.** Summed from the tensor index at load (`Weights::upload` records what was
actually uploaded, not an estimate).

📌 The difference is not waste to be tuned away — it is the price of a slot that can hold any
block — but it must be counted.

**Where from.** `moearc-engine/src/moe.rs` module header, `ExpertPool`.

---

### M5 · Full residency is not achievable on the models this project targets

| model | slots | slot bytes | full pool | dense (always resident) |
|---|---:|---:|---:|---:|
| OLMoE-1B-7B | 16 × 64 = 1,024 | — | ~3.6 GiB pageable | ~360 MiB |
| Qwen3-30B-A3B | 48 × 128 = 6,144 | 2.92 MiB | **17.51 GiB** | 951 MiB |
| gpt-oss-120B | 36 × 128 = 4,608 | 12.607 MiB | **56.7 GiB** | 2.29 GiB |

Against a B580's 11.33 GiB reported free. `Residency::All` is the default and **fails on these
models, which is the honest outcome** — the alternative is silently choosing a budget the caller
did not ask for.

**Where from.** `moe.rs` `Residency::All`, `Weights::upload`; `tests/qwen3moe_forward.rs`,
`tests/gptoss_forward.rs` headers.

---

### M6 · Slots are not experts — the factor-of-`n_block` error

**Claim.** The file says "64 experts"; the residency planner must be told about
`n_block × n_expert` places to put one. Conflating them understates the model by the block count
while every downstream number still looks reasonable.

**Numbers.** OLMoE: 64 experts → **1,024 slots** (16×). gpt-oss-120B: 128 → **4,608** (36×).

**How guarded.** `memory::llama_split` checks the footprint's slot count against the block
geometry rather than trusting the caller, precisely because this mistake produces a plan for a
thirty-sixth of the memory the model needs. `moearc-cli`'s `footprint` carries a warning about it.

**Where from.** `moe.rs` `Weights::total_slots`; `memory.rs` `llama_split` + its tests.

---

### M7 · The always-resident half must be counted from the file, not from the architecture

**Claim.** `dense_bytes` is what the planner divides the card by, so it must be summed from what
the file actually carries.

**Number.** **gpt-oss's expert biases alone are 159 MiB across 36 blocks** — f32, and therefore
resident rather than streamed. A `dense_bytes` derived from the architecture's switches rather
than from the tensor index would under-report the resident half by that much.

**Where from.** `moe.rs` `Weights::upload`.

---

### M8 · Sliding-window attention is a **memory** finding, not an attention detail

**Claim.** A block that can only ever see the last `n_swa` positions needs `n_swa` of cache in a
ring, not `n_ctx`. On a card where the KV cache and the expert pool compete for the same bytes,
that is bytes an expert slot can have.

**Numbers.**
- gpt-oss-120B: 18 full-attention blocks at the full page count, 18 windowed at **4 pages each**.
- The saving is bounded by `full_blocks / n_block` — at gpt-oss's alternating pattern, **half the
  cache**, approached slowly. **Quote `KvGeometry::saved_bytes`, never the limit.**
- It is exactly **zero** at `n_ctx = n_swa`.
- One gpt-oss expert slot is **12.607 MiB**, which is the exchange rate.

**The related default trap.** gpt-oss declares `context_length = 131072`. Taking it literally
plans a KV cache of **4.51 GiB on an 11.33 GiB card** for a caller who never mentioned context —
**31× the cache** of the 4,096 default, taken straight out of the expert pool. It did not fail;
it succeeded and ran slower.

🔴 Until SWA landed, a `n_ctx > 128` refusal in `moe.rs` stood *accidental* guard over this.
Removing the refusal removed the guard.

**How verified without a GPU.** A test walks a whole 4,096-token sequence asserting that no two
positions inside one window ever resolve to the same physical slot, computing the index exactly
as `State::new`'s table and the kernel do between them.

**Where from.** `moe.rs` `KvGeometry` + its tests; `session.rs` `DEFAULT_N_CTX`.

---

### M9 · The KV cache is f16 on purpose

f16 matches llama.cpp's default `type_k`/`type_v`. f32 would be *more* accurate, which is
exactly why it is not used: a cache that rounds differently puts a difference into every
attention output that has nothing to do with the graph, and this pass existed to be compared
block by block. The same argument is why llama.cpp's own flash-attention path casts K and V to
f16 unconditionally — the precision cost lands on scores that are about to go through a softmax.

**Where from.** `moe.rs` `const KV`; `kernels.cpp` `kv_at`.

---

## 2. Transfer, PCIe and staging

### T1 · MoE's serialisation penalty is **~2%, not 2×** — this is what killed speculative prefetch

**Claim.** The 40 blocks are strictly sequential and block N+1's router runs on block N's output,
so its experts cannot be named — let alone prefetched — until N completes. The concern was that
fetch and compute would **add** rather than overlap. On the transfer side that concern is
unfounded.

**Numbers.** Real geometry (40 blocks, 8 active per block = 320 activations/token, 2,039,808 B
per slot), Arc B580, 20 repetitions after a warm-up transfer:

| hit rate | miss/block | MB/token | one bulk transfer | 40 sequential | penalty |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40.0% | 5 | 408.0 | 30.36 ms | 30.53 ms | **1.01×** |
| 65.9% | 3 | 244.8 | 18.17 ms | 18.45 ms | **1.02×** |
| 80.1% | 2 | 163.2 | 12.12 ms | 12.34 ms | **1.02×** |

Each block's fetch is already ~4–10 MB, comfortably enough to saturate the link, so per-transfer
latency is lost in the noise.

📌 **This lowers the value of speculative expert prefetch considerably.** It was the obvious next
optimisation; there is at most 2% in it *on the transfer path*. ⚠️ Any remaining case has to come
from overlapping fetch with **compute**, which is a different argument — see §8·X4, where
`runtime.rs` states the conclusion without that qualifier.

**How measured.** `tools/stream_bench.cpp`. Full table in [`calibration.md`](calibration.md).

---

### T2 · Asynchronous staging is worth +15%, and the obvious explanation of *why* is wrong

**Claim.** Making the expert-staging upload non-blocking is worth ~15% of decode throughput —
but **not because the blocking copy drained the queue.**

**Numbers.** Qwen3-30B-A3B, 2,952 resident slots, 95 steady-state tokens, changing only this:

```
  moe.stage       17.33 -> 10.11 ms/token
  decode.total    43.19 -> 37.51 ms/token
  throughput      22.87 -> 26.28 tok/s      (+15%)
```

with **every greedy token id, every cache hit rate and every staged-byte count unchanged**.

🔴 **The mechanism.** Trace the in-order queue through one decode block: the router's top-k is
submitted, then its result is **read back**, and that read drains everything. Admission is
host-only. So by the time staging runs *the queue is already empty*, and each blocking upload
waited only for itself. What the wait actually cost was **overlap** — serialised, copy *n+1*
could not be submitted until copy *n* landed, so the copy engine never had a queue to stream and
the host never ran ahead into the next block's kernels.

**Where from.** `moearc-kernels/kernels.cpp` `moearc_copy_h2d_async`;
`moearc-kernels/src/lib.rs` `Context::upload_async`.

---

### T3 · The 13.4 GB/s link figure is a **pinned** number and does not describe staging from an mmap

**Claim.** `docs/roadmap.md`'s 13.4 GB/s comes from `tools/stream_bench.cpp`, which allocates its
host side with `malloc_host`. The engine's staging path copies out of a memory-mapped GGUF —
ordinary pageable, file-backed pages, which Level Zero cannot DMA from directly and must bounce
through a driver staging buffer. **Do not compare the two.**

**Numbers**, measured on the mmap path, per ~930 KiB bank copy:

| condition | per-copy | effective |
| --- | ---: | ---: |
| 93% hit rate (copies isolated) | 136 µs | **6.7 GB/s** |
| 0% hit rate (copies pipeline) | 91 µs | **10.5 GB/s** |

At volume the phase is within ~20% of the pinned link rate.

**The open question, with its arithmetic.** Closing the gap means a pinned staging ring. An extra
mmap→pinned host copy costs the measured **22.8 GB/s** single-core host read rate
(`docs/roadmap.md`), so it pays only if the pageable path is worse than 10.5 GB/s. **The
arithmetic does not obviously favour it. Measure before building it.**

⚠️ §3·S4 is a second, independent argument against a pinned ring: at depth, the policy that
removes staging removes the disk dependence, and *not staging* beats *staging faster*.

**Where from.** `kernels.cpp` `moearc_copy_h2d_async`.

---

### T4 · The router readback costs a **pipeline stall**, not a copy — and the near-zero is the result

**Claim.** Reading the router's choice back to the host once per block per token is one drain of
an asynchronous in-order queue, not 32 bytes of traffic.

**Numbers.** Removing a *second*, redundant download of the router weights, Qwen3-30B-A3B at
2,952 slots, 95 steady-state tokens:

```
  moe.readback    19.01 -> 18.71 ms/token
  throughput      22.87 ->  23.17 tok/s      (+1.3%)
```

🔴 **That +1.3% is the finding.** The 18.7 ms this phase costs is not the copy — 48 calls moving
32 bytes each — it is the **drain**. The first download waits for everything submitted before it;
the second then costs almost nothing because the pipeline is already empty. Removing a second
drain that was never happening buys nothing.

**Scale.** ~390 µs per call = one pipeline stall per block per token. On OLMoE (16 blocks,
resident pool) the same round trip measured **~13 µs**; at 48 blocks with a streaming pool it is
**30× that, and the largest single phase in the step.**

**What would actually fix it.** Not reading the router's choice back at all — driving the expert
gather from the device.

**Where from.** `moe.rs`, the router-readback block in `decode`.

---

### T5 · A matvec is bound by re-reading the **activation**, not by the weights

**Claim.** A matvec reads each weight byte exactly once; what it reads `n_rows` times is the
activation vector — and on these shapes that is the larger number by an order of magnitude.

**Numbers.** An expert's gate matrix is 1.2 MiB against **8 MiB of re-read `x`**. `attn_q`,
`attn_k` and `attn_v` together measured **55 MiB of traffic in 121 µs — 456 GB/s on a card whose
peak is 456 GB/s** — and only 7 MiB of it was weights.

This is what `MATVEC_ROWS = 8` exists for: eight rows to a work-group means one trip through `x`
serves eight of them, with the total work-item count and occupancy unchanged.

**Where from.** `kernels.cpp`, the `MATVEC_ROWS` note.

---

### T6 · Batching did not saturate the card, and the headroom is known

**Numbers.** The expert matvecs move **465 MiB a token**, at **68 GB/s before** batching and
**133 GB/s after**, against a peak of **456 GB/s** — still under a third of what the memory
system will give. A single expert matvec moved **1.18 MB in 12 µs = 98 GB/s**.

**Where from.** `moe.rs` module header; `kernels.cpp` `matvec_q_batched_submit`.

---

### T7 · Warm-decode disk traffic — a claim that needs its qualifier

**Claim as written in-tree.** *"Warm decode on the reference box measured 0 MiB of actual disk
reads at every depth from 128 to 8192 tokens."* This is why `bytes_staged_uncovered` is
documented as **a bound, not a measurement** — the host budget says how many slots fit in RAM,
never which.

⚠️ **The claim is unqualified and the qualifier matters.** `bench/baselines/gpt-oss-120b.md` §7.4
measures **4,512 MiB of warm-decode disk (71.6 MiB/step)** at depth 8192 under `frac:0.5`. See
§8·X3. The reconcilable reading is that 0 MiB holds for the policies that stage little.

**Where from.** `moe.rs` `ResidencyReport::bytes_staged_uncovered`.

---

## 3. Scheduling, overlap and the token loop

### S1 · 🔴 Overlap is not substitution — the idea that outlives the code

**This is the finding most worth preserving in this section, and the inventory is right that it
is genuinely absent from llama.cpp.**

**The distinction.** llama.cpp's `-ncmoe` **pins** whole layers' expert tensors to the CPU at
load time, permanently. It substitutes host compute for device compute, and gets monotonically
slower the more of it there is. `host_experts` splits only the experts that **miss**, and
**submits them before the block's device work is queued**, so host compute is substituted for
**PCIe transfer** and runs *concurrently with* the GPU.

**How that is measured rather than asserted.** Two counters: `busy` (wall time the CPU pool spent
working, per token) and `wait` (wall time the device thread spent blocked in `sync`). **Overlap
is the difference.** If `wait` ≈ `busy`, nothing was hidden and this is substitution.

**Numbers**, gpt-oss-120B on a B580:

| configuration | busy ms/token | wait ms/token | reading |
| --- | ---: | ---: | --- |
| 600 slots, `frac:0.5` | 18.46 | 12.43 | **a third of the CPU's work is hidden** |
| any slot count, `frac:1.0` | — | equal to 2 d.p. | **nothing hidden; CPU is the whole critical path**, throughput pins at 11.2 tok/s regardless of available VRAM |

Every `frac:` row beats its `off` (stream-only) control at every pool size, by at least 48%.

⚠️ **`tok/s` can fall while `wait` is near zero** — that means the CPU is not the cost, the
experts it took are simply not the expensive ones.

🔴 **The sign is model-dependent and reverses.** `bench/baselines/qwen3-30b-a3b.md` finds
llama.cpp's throughput falling monotonically as `-ncmoe` moves work to the CPU. The two are not in
conflict; the mechanisms differ. The shape that makes host execution win here is **fewer, much
larger misses**: a gpt-oss expert is 12.607 MiB against Qwen3's 2.92 MiB (4.3× per miss) while
the step names only 144 experts (36×4) against Qwen3's 384 (48×8). *Fewer, larger misses is the
worst possible shape for a link and the best possible shape for a CPU.*

**Where from.** `moearc-engine/src/host_experts.rs` module header;
`bench/baselines/gpt-oss-120b.md` §3.1, §3.2 (KEEP — read there for the full tables).

---

### S2 · The host/device optimum is **interior**, and it moves with depth

**Claim.** More host is not always better, and a perfect hit rate is not the objective.

**Numbers.** Warm tok/s, gpt-oss-120B, mean ± half-range over independent invocations:

| depth | `off` | `frac:0.5` | **`frac:0.75`** | `frac:1.0` | best vs `frac:0.5` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.78 ±0.02 | 12.32 ±0.07 | **14.93 ±0.04** | 10.33 † | 1.21× |
| 512 | 8.66 ±0.00 | 12.41 ±0.10 | **13.38 ±0.03** | 9.86 † | 1.08× |
| 2048 | 5.37 ±0.06 | 7.63 ±0.15 | **8.74 ±0.06** | 7.10 † | 1.15× |
| 8192 | — | 1.31 † | **3.71 †** | — | 2.83× |

† n = 1. `frac:1.0` is worse than `0.75` everywhere.

🔴 **A 100% hit rate is a means, not the objective.** From 480 slots up, `frac:0.75` stages
**zero bytes** on the warm pass — the residency thesis working perfectly — and at depth 6 it is
**still slower than `frac:0.5`**: sending 60.5% of experts to the CPU costs 43 ms of host time
per token where 23.4% costs 19 ms, and the extra 24 ms buys the removal of ~9 GiB/run of staging
that was only costing about 12 ms. Past the crossover, a perfect hit rate is bought too dearly.
**The crossover moves with depth**, which is why `frac:0.75` wins at 8192 and loses at 6.

📌 **Reproducibility is itself a result.** Warm spreads: `frac:0.75` ±0.2–0.7%, `off` ±0.1–1.2%,
`frac:0.5` ±0.5–2.0%. **The policy that stages least is the policy whose numbers hold still.**

**Where from.** `bench/baselines/gpt-oss-120b.md` §3.3, §7.2 (KEEP).

---

### S3 · The host executor does **not** stall more at depth

**Claim.** The worry that a fork-join host pool would degrade as context grows is answered: no.

**Number.** `moe.host_sync` at `frac:0.75` is **36.60 / 36.50 / 38.34 / 42.72 ms** across depths
128 → 512 → 2048 → 8192 — a **17% rise over a 64× change in depth**.

What heavy host routing does instead is send most misses to the CPU *instead of admitting them*,
so `moe.stage` — the thing measured at 80% of the depth penalty — falls to **1–4 ms and stays
there at every depth**.

**Where from.** `bench/baselines/gpt-oss-120b.md` §7.3 (KEEP).

---

### S4 · Staging is disk-bound only in the regime where staging is large

**Numbers**, gpt-oss-120B at depth 8192:

| | `frac:0.5` | `frac:0.75` |
| --- | ---: | ---: |
| staged, whole pass | 2.71 TiB | 0.30 TiB |
| cold-prefill disk read | **1.48 TiB** | 27 GiB |
| **warm-decode disk read** | **4,512 MiB** (71.6 MiB/step) | **0 MiB** |
| warm tok/s | 1.31 | **3.71** |

`frac:0.5` faults ~55% of its staged bytes from NVMe; `frac:0.75` faults 9% and nothing at all
during warm decode.

📌 **The policy that removes staging removes the disk dependence with it. A pinned host-memory
staging ring would not help here; not staging would.** (Cross-check against T3, which reaches the
same conclusion from the transfer side.)

⚠️ Related, and easy to lose: **`nice` protects CPU shares but not the ZFS ARC.** A niced build
next door still evicts the model's pages, and this workload is bound by what is resident.

**Where from.** `bench/baselines/gpt-oss-120b.md` §7.1, §7.4 (KEEP);
[`../bench/PROTOCOL.md`](../bench/PROTOCOL.md) §4.

---

### S5 · The host collect must be an asynchronous copy — a blocking one is a mid-block fence

**Claim.** Copying the host executor's result back with a blocking copy places a fence in the
middle of every block: it waits for that block's staging and matvecs, which are already
submitted, and gets **billed for them**.

**Number.** Measured at 520 slots and `frac:0.5`: **10.96 ms a token, 20% of the step** — device
time that was going to be paid anyway, charged to the wrong line and serialising the host behind
it.

**The safety argument that makes the async version sound** (worth keeping, because it is the
shape any replacement needs): the source buffer lives in `State` for the session, and the only
thing that writes it is the *next* block's `sync` — which is downstream of that block's router
readback, a blocking download that drains this copy first.

**Where from.** `moe.rs`, the `host_add` call site.

---

### S6 · AVX2/FMA runtime dispatch is worth ~2× on the host expert kernels

The crate is built for a baseline x86-64 target (SSE2, no FMA) and the inner loops are exactly
the shape a 256-bit FMA doubles. **Runtime dispatch rather than a build flag**, so a release
binary runs anywhere and is fast here. **Measured on a Core Ultra 7 265K: roughly 2×.**

**Where from.** `host_experts.rs`, the AVX2 wrapper section.

---

### S7 · Eight independent accumulators, not one — worth ~11× on a host dot product

**Claim.** A float sum is not associative, so a single `s1 +=` is a serial dependency chain LLVM
is **not allowed** to reorder, and it vectorises to nothing. Splitting the row into eight
independent lane chains is a reassociation the *source* performs rather than one the compiler
assumes.

**Number.** The first version of the Q4_K host loop ran at **1.96 GB/s a core** against the
**~22.8 GB/s** a core reads memory at.

The same structure carries a second trick worth knowing: folding the K-quant block minimum out
of the inner loop. `sum((d·sc·q − dmin·m)·x) == d·sc·sum(q·x) − dmin·m·sum(x)`, and `sum(x)`
depends only on the activation, so it is computed **once per job** rather than once per row.
Q6_K has no minimum at all, so its 16-element groups reduce to `d · Σ_groups sc·sum(q·x)`. MXFP4
has no minimum and no sub-block structure, so there is nothing for the trick to hoist — it is
deliberately *not* written like the K-quants.

**Where from.** `host_experts.rs` `dot_q4k` / `dot_q6k` / `dot_mxfp4`.

---

### S8 · Threads = cores − 1, and not out of politeness

The device thread submits every kernel and every staging copy. A host executor that descheduled
it would buy CPU throughput **by delaying the GPU work it is supposed to be hiding behind.**
Core 0 is left for it, and workers are pinned best-effort (a pinning failure costs scheduling
quality, not correctness, so it is not reported).

**Where from.** `host_experts.rs` `default_threads`, `worker`.

---

### S9 · Touch every expert before timing a host bench — or you measure ZFS

**Claim.** A cold `mmap` read is a disk read, so a first pass over the weights measures the
filesystem rather than the CPU.

**Number.** The first version of the host-expert bench reported **410 µs an expert** for exactly
this reason.

The bench's three-number structure is worth reusing: (1) one core, one expert — weight bytes over
wall time, directly comparable to the ~22.8 GB/s single-core host read rate, and the only figure
that says whether the kernel is memory- or arithmetic-bound; (2) the pool, a block's worth at a
time — what the engine actually submits; (3) the implied per-token ceiling.

**Where from.** `moearc-engine/examples/host_expert_bench.rs`.

---

### S10 · A host-offload measurement is a measurement of the whole machine

**Claim.** Load average moved a host-offload result by 8× and **reversed its sign**.

**Numbers.** The same sweep, identical code and identical arguments, run twice: first run
reported `frac:0.5` at **1.41–1.96 tok/s** with every host policy *worse* than streaming; second
reported **10.71–17.97** with every host policy better. The difference was the box's 1-minute
load average: **9.50 against 0.69**, from another agent's build and test work on the same 20 cores.

🔴 **The tell was internal, not external.** Two rows with identical `cpu/step`, `cpu share` and
`staged MiB` — the same work, by construction — reported **2.07 and 15.02 tok/s**. *Identical
inputs producing an 8× spread is a broken measurement, whichever number you prefer.*

📌 The contamination is not uniform: the `off` rows moved far less (5.12 vs 5.35 at 144 slots),
because streaming is bound by a link the other tenants were not using. **It lands almost entirely
on the rows that need the CPU.**

**Where from.** `bench/baselines/gpt-oss-120b.md` §3.4 (KEEP);
[`../bench/PROTOCOL.md`](../bench/PROTOCOL.md) §3, which is the rule this produced.

---

### S11 · The ordering contract: route → admit → stage → compute

**Claim.** Every miss is staged **before** anything computes. A matmul against a slot still being
filled does not fail — **it returns plausible wrong output** — so this ordering is the difference
between a bug that is caught and one that is never found.

**Why blocks cannot overlap.** Block N+1's router runs on block N's output, so its experts cannot
even be *named* until N finishes. That is structural, not an implementation choice.

⚠️ `runtime.rs` justifies writing the loop plainly rather than around a prefetch pipeline by
citing the ~2% figure — see T1 and §8·X4 for the qualifier it omits.

**Where from.** `moearc-engine/src/runtime.rs` module header; asserted directly by a test that
has one object play both the store and the compute role, so it observes staging and compute
interleaved exactly as the runtime issues them.

---

### S12 · The static-split ring must skip victims the same step still needs — and omitting it shipped wrong tokens

**Claim.** In `Policy::StaticSplit`, blocks above the split share a ring one step wide. Plain
round-robin will hand back a ring position holding an expert that was **already reported as a hit
earlier in the same step**; staging then overwrites it before any matmul runs, and two different
experts read the same slot.

**Number / symptom.** The output stays finite and fluent. **`static:15` on Qwen3-30B-A3B diverged
from llama.cpp at token 18 and nowhere earlier.**

🔴 **Why it hid.** It only bites when a ring entry survives from one visit to the next, which
needs the ring at least as wide as the streaming demand *and* the same block visited twice
without an intervening flush. **With two or more streaming blocks the ring is flushed in between,
so every position is stale and round-robin looks correct. One streaming block is the case that
exposes it — and it was the last one swept.**

The existence argument for a victim is worth keeping: reaching the victim search means at least
one demand missed, so at most `streaming − 1` positions can be pinned out of
`ring.len() ≥ streaming`.

**Where from.** `moe.rs` `StaticSplit::next_victim` + its regression test (three lines, no GPU).

---

### S13 · Two implementations of "LRU" that process a trace in different orders are two different policies

**Claim.** The cache walks demands **in the order the router produced them**, advancing recency
once per demand. Every expert the step needs is pinned, so order cannot change which of *them* is
evicted — but it changes **the recency each one ends up holding**, and therefore which of them is
evicted several steps later.

**How found.** An earlier version iterated a sorted, deduplicated list and **disagreed with the
offline simulator on total misses.**

Related and deliberate: three reads of one expert in a step are three demands but **one**
transfer. Charging all three as misses would overstate bus traffic — the number the whole engine
is optimising.

**Where from.** `moearc-engine/src/cache.rs` `admit`; cross-checked against `residency::simulate`,
which is an independent implementation (a resident-vector scan vs a reverse index and a slot map),
so their agreement on miss count was a real check rather than a restatement. ⚠️ **Deleting
`cache.rs` deletes that cross-check.**

---

### S14 · `admit` plans and commits together, on purpose

Splitting plan from commit would let a caller drop a plan and leave the cache **certain of a slot
that was never filled**; the next hit on it reads whatever the slot held before. So a policy that
sends some misses somewhere other than the bus must ask `resident()` first and `admit` only what
it kept. That asymmetry is why `resident()` exists at all.

**Where from.** `cache.rs` `admit` / `resident`; `moe.rs`'s host-split call site, which is the
only caller that needs it.

---

### S15 · A fork-join across a job boundary needs a **full epoch handshake**, not a task-counter drain

**Claim.** `sync` must wait for **every worker to acknowledge the epoch**, not merely for the task
counters to reach zero. The weaker condition is not enough: a worker that had read the epoch but
not yet claimed a task would still be inside the job when the next `submit` overwrote it.

**Symptom if omitted.** An expert computed against the **wrong block's activation** — finite,
fluent, wrong.

Determinism follows from the same structure: every output element is written by exactly one task
and the final weighted sum runs over experts in router order on the calling thread, so a job's
result does not depend on how the work was scheduled.

**Where from.** `host_experts.rs` module header, `submit` / `sync`.

---

### S16 · A worker panic must be caught, or `sync` hangs forever

A worker that dies mid-job never acknowledges its epoch, and `sync` waits for that
acknowledgement indefinitely. Catching the unwind turns a hang into a reported error, and the
dying worker acknowledges anyway so `sync` reaches its `stop` check. ⚠️ **Only in a build that
unwinds** — the release profile sets `panic = "abort"`, where the process dies instead, which is
loud and is the outcome this is second-best to.

The same shape guards the spec check: `BankSpec::check` validates the spec against the bytes it
claims to describe **per bank, per block**, on the submitting thread, because a panic inside a
worker is worse than a wrong answer.

**Where from.** `host_experts.rs` `worker`, `submit`.

---

### S17 · Batching the expert FFN is worth ~2.2× — and the reason is **not** launch count

**Claim.** A submission costs **1.6 µs**, which against a 13 ms token is a rounding error. What
matters is **wave depth**: one expert's gate matvec is 1,024 rows — roughly one pass over a
B580's resident threads — and a kernel one wave deep has no second wave to run while the first
waits on memory.

**Numbers**, measured on OLMoE:

```
  expert FFN     8.39 -> 3.83 ms/token
  decode         63.8 -> 76.7 tok/s
  launches       656  -> 64   per token   (4 per block instead of 41)
```

with **every greedy token id unchanged**.

⚠️ **Not transferable.** Qwen3's experts are **768** rows — narrower still — so the argument
applies harder there and OLMoE's numbers do not describe it.

**Where from.** `moe.rs` module header; `kernels.cpp` `matvec_q_batched_submit`.

---

## 4. Kernels — tunings that paid, and hypotheses that died

> This section is the reason to read this file before trying anything on ggml-sycl's MoE path.
> Several of these were well-argued, and measuring them in place is what killed them.

### K1 · ❌ REFUTED — carrying two or four rows per lane so a lane loads the activation once

**Hypothesis.** Since the matvec is bound by re-reading the activation (T5), have each lane load
`x` once and spend it across several rows.

**Result.** **Neutral at two rows, 4% worse at four.** The lost waves cost exactly what the saved
loads bought.

**Where from.** `moe.rs` module header, *"Two experiments that did not help"*.

---

### K2 · ❌ REFUTED — widening or narrowing the rows a work-group covers

**Result.** **Within 4% across 2, 4, 8 and 16 rows.** `MATVEC_ROWS = 8` was kept; the sweep says
the choice barely matters within that range.

**Where from.** `moe.rs` module header.

---

### K3 · ❌ REFUTED — using the device's native `half` conversion in the matvec hot loop

**Hypothesis, and it was well-supported.** The software integer f16→f32 path looks like the
expensive option — it has a `while` loop for subnormals that runs on every block — and an
isolated microbenchmark of a Q4_K matvec on this card **agreed**, putting
`(float) sycl::bit_cast<sycl::half>(h)` **7% ahead** and claiming it was the gate on a further
**13% from coalescing**.

**Result — it does not transfer.** Swapping only the four hot-loop call sites in `unit_acc`,
measured *in the engine* on Qwen3-30B-A3B at 2,952 resident slots with `MOEARC_SYNC_EACH=1`,
three runs each, spread under 0.2%:

```
                           software f16   device half
  moe.expert_matvec  Q4_K      7.42 ms       7.47 ms    +0.7%
  moe.expert_down    Q6_K      3.29 ms       3.60 ms    +9.4%
  out.matvec         Q6_K      4.02 ms       4.44 ms   +10.4%
  decode.total                40.4  ms      41.0  ms    +1.6%
```

🔴 **Uniformly worse, and worst on the Q6_K kernels.** Likely mechanism *(inference — the numbers
are measured, this sentence is not)*: these kernels are not ALU-bound, so the integer work hides
under memory latency on a pipe the float MACs are not using, and moving it onto the
float/conversion pipe puts it on the contended one.

📌 **The wider lesson, and the reason this entry is here at all: a microbenchmark of "the same
kernel" is not this kernel.** The same 2×2 that produced the 7% also produced the 21% coalescing
figure — **so that figure is unsupported too, until it is measured in place.**

**Correctness was never the obstacle.** Over all 65,536 half bit patterns the two differ on
**1,022**, every one a signalling NaN, which a GGUF scale is never. It is simply slower.

**Where from.** `kernels.cpp` `f16_to_f32`.

---

### K4 · ❌ REFUTED — hoisting the Q6_K in-loop nibble select (to match Q4_K)

**Hypothesis.** Q4_K hoists its nibble-half select out of the loop into a branch on a scalar;
Q6_K leaves the equivalent select inside. Make them consistent.

**Result.** **Measurably worse.** Batched Q6_K went **4.13 → 6.06 ps/element**; the unbatched
path barely moved (13.05 → 12.53). 🔴 **Do not "fix" this to match Q4_K.**

**Where from.** `kernels.cpp` `unit_acc`, the Q6_K arm.

---

### K5 · ⚠️ The Q6_K batched-kernel workaround — 3.2×, cause **not established**, three hypotheses refuted

**Claim.** `moearc_matvec_q` routes Q6_K — **and only Q6_K** — through the *batched* kernel with a
single matrix. That is worth **3.2×**.

**The two kernels have token-identical inner loops.** They differ only in how the row pointer is
formed: `base + row * row_bytes` from a kernel argument (unbatched) against
`w.p[mat] + row * row_bytes` through a by-value table (batched).

**Numbers**, measured at `n_cols = 2048` by `examples/matvec_scaling`, *before* the routing change:

```
             unbatched   batched
   Q4_K         4.05       4.17    ps/element
   Q5_K         3.81       3.61
   Q6_K        13.05       4.13    <- 3.2x, and the reason for the routing
```

Flat across every `n_cols` from 256 to 4096.

🔴 **Three source-level hypotheses were measured and refuted:**
1. **`n_cols`** — the effect is flat across the whole sweep.
2. **The second byte stream** — Q5_K reads two byte streams exactly as Q6_K does and is unaffected.
3. **The in-loop nibble select** — hoisting it made things *worse* (K4).

**What remains is an IGC codegen difference that an opaque pointer suppresses.** It is
characterised and reproducible; **it is a workaround, not a fix.** Re-check it after a compiler or
driver upgrade — exactly the sort of change that could make it unnecessary, or make it necessary
somewhere else.

⚠️ **The swap is not free at every shape.** At `n_cols = 512` the batched kernel costs Q4_K
**4.98 → 7.57 ps/element**. No shape in this engine used it, but a future one might, which is why
only Q6_K is routed.

🔗 **This survives the pivot as an open question about ggml-sycl**, whose MoE GEMV
(`ggml_sycl_mul_mat_vec_q_id`) is also a raw-pointer API. Nobody has checked whether it hits the
same codegen difference.

**Where from.** `kernels.cpp` `moearc_matvec_q`; `moearc-kernels/examples/matvec_scaling.rs`.

---

### K6 · Q6_K-vs-Q4_K cost is a function of **shape**, not quantisation

**Numbers**, measured in a live decode:

| shape | Q6_K | Q4_K | ratio |
| --- | ---: | ---: | ---: |
| `n_cols = 768` (expert `down`) | 5.21 ps/elem | 5.12 ps/elem | **equal** |
| `n_cols = 2048` (`lm_head`, `attn_v`) | 13.0–18.0 | 4.5–6.9 | **2.5–3.5×** |

Both shapes run the same kernel on the same quantiser.

**Where from.** `moearc-kernels/examples/matvec_scaling.rs` header.

---

### K7 · The work split within a row must be over **32-element units**, not blocks

**Claim.** Every supported format has a natural 32-element unit over which the dequantisation
constants are fixed (a K-quant super-block is eight of them; a Q8_0 block is exactly one).
Deriving those constants once per unit rather than per element is worth about **25 instructions
per MAC**, and a decode step spends most of its time here.

**Number.** A block-per-lane split leaves a work-group mostly idle on the shapes this model runs:
an expert's `n_cols` is 2048, which is **eight** Q4_K super-blocks, so **eight of thirty-two lanes
had work and twenty-four sat out**. Splitting by unit gives sixty-four pieces to thirty-two lanes.

🔴 The per-element *expression* is unchanged, term for term and **in the same association**; only
where the constants are computed moves. A lane covering the same elements in the same order gets
a bit-identical answer.

**Where from.** `kernels.cpp`, *"The quantised matvec's inner loop"*.

---

### K8 · RMSNorm lane width is measured, and it is not `WG`

**Claim.** `WG = 32` is one sub-group — the right width for a short row and nowhere near enough
for a long one, because a single work-group has one hardware thread's worth of outstanding loads.

**Number.** On a B580, one RMSNorm over `n_embd = 2048` cost **28 µs at WG = 32 — for 8 KiB of
traffic.**

**Resolution.** `norm_wg(n_cols)`: **256** at `n_cols ≥ 1024`, **64** at `≥ 256`, else 32.
⚠️ The three thresholds themselves are **unattributed** — the 28 µs datum motivates widening, not
these particular breakpoints.

🔴 This changes the *grouping* of the sum of squares, and float addition is not associative, so a
wider group is **not bit-identical** to a narrower one. It is not less accurate either (a wider
tree has a shorter dependence chain and no more error) — but it is a change to the arithmetic, and
the forward-pass tests are what say it is a safe one.

**Where from.** `kernels.cpp` `norm_wg`.

---

### K9 · Zeroing an accumulator must be a kernel, never an upload

A host-to-device copy **synchronises**; a kernel does not. On a queue everything else is free to
run ahead of, the upload of a zero vector was the only thing stopping. Same argument, twice more:
the batched matvec's weight-pointer table and `add_bias_id`'s index array are carried **by value
as kernel arguments** (32 pointers = 256 bytes, well inside what Level Zero takes) rather than
uploaded to a device table — passing them as arguments is *ordered by construction*, with no table
for the kernel to race and no drain to pay.

**Where from.** `kernels.cpp` `moearc_zero`, `mat_table`, `moearc_add_bias_id`.

---

### K10 · Two profiling instruments, and each is wrong for the other's question

| instrument | what it does | correct for | **actively misleading for** |
| --- | --- | --- | --- |
| `MOEARC_SYNC_EACH=1` | waits after every launch | *"how long is this kernel"* | *"where does the step's time go"* — the waiting **is** the thing being measured, and it destroys the overlap you are reasoning about |
| `MOEARC_PROFILE_EVENTS=1` | reads SYCL's own per-submission timestamps | attribution **on a still-asynchronous queue** | nothing, but it **blocks** until outstanding events complete, so it belongs after the work and never inside it |

🔴 **Host wall-clock around an asynchronous queue bills device work to whichever call later drains
it.** Read naively that says attention is free and `moe.stage` is everything — an artefact of
where the queue was drained, not a finding. *(This once made the project retract a **correct**
conclusion; [`../bench/PROTOCOL.md`](../bench/PROTOCOL.md) §7 is the rule it produced.)*

**Two design details of the event counter worth keeping:**

- **The key must carry the kernel's shape *and* its quantisation type.** Without the shape,
  `out.matvec`, `attn.qkv` and `attn.proj` collapse into one number. Without the type, a bank that
  is Q6_K in half its blocks and Q4_K in the other half reports the **average of two kernels that
  differ by 2×** — which is exactly what made Q6_K look fine in `expert_down` and slow in
  `lm_head`.
- **The attention key deliberately omits the key span**, which is the one number driving its cost.
  `n_kv` advances by one every token, so keying on it would mint a distinct key per token —
  **8,192 in a single depth-8192 prefill** — and the event flush resolves keys by linear scan, so
  folding would go **quadratic in the prompt**. The SWA and full-causal blocks therefore share one
  key on purpose: gpt-oss alternates 18 of each, and what a step costs is their **sum**.

**Where from.** `kernels.cpp` `moearc_ctx_create`, `record_event`, `attn_decode`;
`moearc-kernels/src/lib.rs` `profile_events_report`; `moearc-engine/examples/ctx_attrib.rs`.

---

### K11 · 🔴 The instrumentation blind spot that made a wrong answer look confident

**Claim.** MoEArc's device event counters instrument **only matvec paths**, so **attention is
invisible to them**. Neither instrument alone could attribute the depth penalty, and **using
either alone would have given a confident wrong answer.**

This is the single most transferable lesson in §4: *know what your counters cannot see.*

**Where from.** [`../bench/PROTOCOL.md`](../bench/PROTOCOL.md) §7;
`moearc-engine/examples/ctx_attrib.rs`, which exists to answer the depth question by
**differencing** two runs from the same warm pool rather than by modelling it — `prefill` (prompt
+ 1 token) against `full` (prompt + n), on cumulative counters, so `full − prefill` is exactly the
`n − 1` decode steps.

---

### K12 · Launch overhead, measured apart from arithmetic

**Number.** A submission costs **1.6 µs**. The engine issues on the order of a thousand kernels
per decoded token; against a 13 ms token that is a rounding error. `examples/launch_overhead.rs`
measures it with `n = 1` on purpose — a kernel over one element does no work worth measuring, so
whatever time it takes is the cost of asking.

**Where from.** `moe.rs` module header; `moearc-kernels/examples/launch_overhead.rs`.

---

### K13 · Per-layer slot partitioning — already tested and closed

**Do not re-derive this.** [`negative-results.md`](negative-results.md) records it in full:
per-layer miss rate is a **monotonic ramp, not the U-shape the prior predicted**; weighting toward
uniform blocks wins 9 of 9 but is worth only **+0.1 to +0.8 points**, against **+9 to +13** from
simply having 44% more slots.

Together with [`../bench/policy-sweep.md`](../bench/policy-sweep.md) (nine policies against
Belady; the best non-regressing alternative recovers 7–25% of the gap) and
[`hardware-sizing.md`](hardware-sizing.md) (+44% slots beats any policy by ~3×), **three
independent lines of evidence say the same thing: this is a capacity problem, not an allocation
problem. Stop looking for cleverness in how the cache is divided.**

---

## 5. Correctness and comparison methodology

### C1 · 🔴 MoEArc and llama.cpp do not compute the same function — and the difference is llama.cpp's

**Claim.** Every K-quant matmul in `ggml-cpu` has `vec_dot_type = GGML_TYPE_Q8_K`: the f32
**activation** is quantised to 8 bits with one scale per 256 elements before the dot product.
`moearc-kernels` keeps the activation in f32.

**Consequence.** A bit-exact comparison is **not available at any tolerance**, and picking a
tolerance that happens to pass would be picking a number rather than measuring one. So the gates
assert on **the decision the logits encode** — the greedy token id — which fails loudly if any of
the ~200 (OLMoE) or ~600 (Qwen3, 48 blocks) operations per token is wrong.

**Where from.** `tests/olmoe_forward.rs`, `tests/qwen3moe_forward.rs` headers.

---

### C2 · How large that unavoidable difference is — measured, and the calibration is surprising

**Numbers.** OLMoE, single-token prompt `12092`, final logits (`result_output`, 50,304 wide),
`1-cos` being the angle between the two logit vectors:

```
  MoEArc (B580, SYCL)  vs llama.cpp CPU      max|d| 5.25e-1   1-cos 5.68e-3
  llama.cpp Vulkan     vs llama.cpp CPU      max|d| 5.29e-1   1-cos 6.81e-3
  MoEArc (B580, SYCL)  vs llama.cpp Vulkan   max|d| 1.58e-1   1-cos 6.22e-4
```

🔴 **llama.cpp's own two backends disagree with each other slightly *more* than MoEArc disagrees
with either**, on the same file and the same token — and MoEArc is an **order of magnitude
closer** to llama.cpp's GPU backend than that backend is to its own CPU one.

**That is the calibration any tolerance would have to be set against**, and it is why the gates
assert on decisions rather than on floats.

**Where from.** `tests/olmoe_forward.rs` header.

---

### C3 · 🔴 The prompt is part of the gate, and the obvious prompt would have been wrong

**Claim.** A greedy continuation is only a *decision* where the top two logits are further apart
than the arithmetic difference in C1. Below that, you are measuring the tie-break.

**Numbers.** On Qwen3-30B-A3B the difference reaches **max|d| = 1.06** on a 151,936-wide logit
vector, so a step whose top two are within ~0.1 is a coin flip — and once two implementations
branch, they never rejoin.

**Not hypothetical.** On *"The capital of France is"* — the prompt the OLMoE gate uses — step 2 is
a three-way near-tie at a **0.093 margin**, and **llama.cpp's own two backends disagree there**:

```
  step 2, llama.cpp CPU     576 23.5524   15920 23.4591   3555 23.4533   <- picks 576
  step 2, llama.cpp SYCL    (follows 15920)
  step 2, MoEArc          15920 23.8202     576 23.7991   3555 23.3545   <- picks 15920
```

MoEArc then tracks llama.cpp's **SYCL** backend for 35 tokens before the next tie.

**Resolution.** The shipped Qwen3 prompt is chosen for the *opposite* property: llama.cpp's CPU
and SYCL backends produce **identical ids on it for all 64 tokens**. A path both of llama.cpp's
backends agree on is robust to exactly the class of perturbation that separates MoEArc from
either. [`../bench/PROTOCOL.md`](../bench/PROTOCOL.md) §8 records the shipped prompt's minimum
margin of **5.81**, and that at one candidate position **llama.cpp disagreed with itself**
(one-shot prefill chose 279, incremental decode chose 13).

**Where from.** `tests/qwen3moe_forward.rs` header.

---

### C4 · Host and device expert paths are not expected to agree bit for bit

**Claim.** The device dequantises a block and reduces in an f32 tree; the host folds the K-quant
block minimum out of its inner loop (S7) and reduces a row in order. **Same value in exact
arithmetic, different rounding.** It would be suspicious if they agreed.

**Number.** Largest relative disagreement tolerated: **TOL = 1e-5**, scaled by the device output's
own magnitude and **floored at 1.0** so a near-zero channel does not report an enormous relative
error from an absolute one that does not matter. **Measured, not chosen** — each test prints the
error it observed, so a future change that widens it is a test failure rather than a plausible
paragraph of prose.

**The invariant this protects.** The host path must compute the **same function** as the device
path (same activation, same biases, same GLU variant), or the model's output would change with
the *host policy* — and the policy is a performance knob.

**Where from.** `tests/host_experts_gpu.rs`; `host_experts.rs` `Geometry`.

---

### C5 · f32→f16: MoEArc and ggml differ on overflow, and **MoEArc is right**

**Claim.** The two agree bit for bit on every input inside the f16 range — but not outside it.

**Numbers**, measured by walking f32 bit patterns against the local llama.cpp build:

- Every input below **65568.0078** converts identically, **including all 4,096 in
  `[65520, 65536)`** that must saturate to infinity.
- From 65568.0078 upward, ggml stops saturating: **70000 → `0x7c46`, a NaN.**
- Above 2^17 the exponent overflows out of its field **into the sign bit**: **131072 → `0x8000`,
  negative zero**; **645252 → `0x08ec`, an ordinary small number.**

IEEE-754 says an overflowing conversion saturates to infinity, which is what MoEArc does.
**Matching ggml here would be actively dangerous:** a KV cache that silently turned a large
activation into a small one corrupts an answer with no signal at all, where an infinity is at
least loud.

⚠️ **Reported as an environment finding, not a defect.** Observed on one build (llama.cpp compiled
with `icx`); **not checked against a stock gcc build.**

📌 This survives the pivot intact — llama.cpp is now the engine, so this is a property of the
thing we ship on.

**Where from.** `moearc-kernels/tests/f16_crosscheck.rs` header.

---

### C6 · Association order is a correctness property, not a style question

Float addition and multiplication are not associative, and this project asserts greedy output
token for token. Four places where the order is load-bearing:

- **Dequantisation.** `d1 * q - m1` with `d1 = d * sc` is written in exactly the order
  `ggml-quants.c` writes it. `d * (sc * q)` is mathematically identical and numerically
  different — *"the kind of difference that makes a cross-check fail for a reason nobody can
  find."*
- **`moe_combine`** sums `m` ascending from `0.0f`, matching the order of the zero-plus-axpy loop
  it replaced.
- **The router's denominator and weights** stay in **one lane, summed in index order.** Reducing
  them over the work-group would change the weights in the last bits — a change to the model's
  arithmetic, bought for a few microseconds on a 64-element sum.
- **RMSNorm** accumulates the sum of squares in **f32 where ggml uses double**. Deliberate: fp64
  on Arc is emulated where it exists at all. The gap is **bounded by a test against an f64 CPU
  reference**, not assumed away.

**Where from.** `kernels.cpp` throughout; `moearc-kernels/src/lib.rs`.

---

### C7 · The catalogue of silent-wrong-output traps

Every entry below **runs, produces fluent output, and is wrong**. This is the highest-density
knowledge in the retired code and the reason `Config::from_model` allowlists architectures **by
name** rather than deriving a graph from tensor names.

📌 **The framing worth keeping:** none of these is guessable from a GGUF, several are properties
of **llama.cpp's own call site** that no file records, and *fluent nonsense is worse than a
refusal.*

**Geometry read wrongly**
- **`head_dim` is not `n_embd / n_head`.** Qwen3-30B-A3B is 2048 wide with 32 heads and a head
  dimension of **128**; the quotient is 64. Read `attention.key_length`, fall back to the quotient
  only as llama.cpp does. It also makes the Q projection **wider than the residual stream** (4096
  against 2048), so Q-side buffers are `n_head * head_dim`, not `n_embd`.
- **The expert FFN's width** is `expert_feed_forward_length` for `qwen3moe` (**768**), not
  `feed_forward_length` (**6144**) — the latter describes a dense FFN the architecture does not
  have; using it sizes every scratch buffer eight times too large. ⚠️ **gpt-oss states both keys
  and they happen to be equal (2880) on the 120B**, so reading the wrong one would still run — *an
  equality that holds in one checkpoint is not a reason to read the other key.*
- **QK-norm spans a different vector in different architectures.** OLMoE's `attn_q_norm.weight` is
  `n_embd` long and normalises the whole projection before the reshape into heads; Qwen3's is
  `head_dim` long and normalises **each head separately**. Neither raises an error on the other's
  model. gpt-oss has **no QK-norm at all** — a third silent difference, since normalising a
  projection that was not trained normalised rescales every head. *(All three are checkable against
  the tensor's own length, and are checked.)*

**Graph switches no GGUF records**
- **The router softmaxes over all experts before the top-k — and then the three differ.** OLMoE
  (`norm_w = false`): raw probabilities that do not sum to one. Qwen3 (`norm_w = true`): divided
  by their sum, **clamped up to `6.103515625e-5`** (f16's smallest normal). gpt-oss: a softmax
  over **the k selected logits, taken *after* the top-k** — a softmax over 128 renormalised to 4
  is a different vector from a softmax over those 4. Getting it wrong rescales every expert's
  contribution by one factor per block: finite, fluent, wrong.
- **`w_scale`** is `hparams.expert_weights_scale`, which neither of the first two architectures
  sets, so it keeps its `0.0f` default and `build_moe_ffn`'s guard skips the scaling entirely.
- **`swiglu_oai`'s `alpha = 1.702` and `limit = 7`** are `constexpr` at llama.cpp's call site, not
  GGUF keys.

**RoPE**
- **NeoX pairs `(i, i + n_dims/2)`, not `(2i, 2i+1)`.** Both `olmoe` and `qwen3moe` are NeoX.
- 🔴 **YaRN has no position gate.** `rope_yarn` interpolates every frequency and rescales every
  magnitude **from position 0**; the only thing its ramp consults is the channel index. A YaRN
  model run with plain RoPE is wrong on its **first token**, not merely past its trained context.
- 🔴 **`attn_factor` is not the paper's mscale.** llama.cpp computes `cparams.yarn_attn_factor` and
  then **divides it by `1 + 0.1·ln(1/freq_scale)`** precisely so the kernel multiplies it back.
  Passing the paper's mscale in from outside **squares it**. For gpt-oss the value arriving is 1.0
  and the effective mscale is **1.3466** (`freq_base = 150000`, factor 32, `corr_dims = [8, 18]`).
- **A closed-form angle is the right divergence.** ggml's CPU path builds its table by repeated
  multiplication, which a work-item cannot do; the closed form `pos · theta_scale^(i0/2)` is
  exactly what `ggml-sycl/rope.cpp` computes. The two are mathematically equal, differ in the last
  bits, and **the difference grows with position** because a large angle has a large ulp. The CPU
  reference deliberately keeps ggml's iterated form so the test **measures** the gap.
- ⚠️ **A second RoPE base exists and this pass refused it by name.** `rope.freq_base_swa` is used
  only on windowed blocks. gpt-oss's files do not carry it; **Gemma 3's do, and there the two
  differ by two orders of magnitude (10k against 1M).**

**Sliding-window attention**
- 🔴 **The window is not the hard part — the alternation is, and it is invisible in `n_swa`.**
  `set_swa_pattern(n, dense_first = false)` is `is_swa[il] = il % n < n - 1`. At gpt-oss's default
  `n = 2` the **even** blocks are windowed and the odd ones are full causal attention. An
  implementation that windowed **every** block is wrong on half of them; one that windowed **none**
  is wrong on the other half; **both stay fluent.**
- **An absent `attention.sliding_window_pattern` means 2 for gpt-oss's architecture and 6 for
  Gemma 3.** That default is a property of the *architecture*, not of GGUF.
- 🔴 **Below `n_swa`, SWA is a no-op, so a short test proves nothing.** `is_masked_swa` masks when
  `p1 - p0 >= n_swa`, which no pair of positions inside one window satisfies. **A suite that only
  tested short contexts would pass with SWA entirely unimplemented.** The gate therefore runs a
  **158-token prompt and 256 generated tokens, reaching position 413**, so every windowed block
  spends the whole run dropping keys the full blocks keep — ⚠️ and **the prompt is not
  incidental**: llama.cpp's continuation **reproduces the passage**, a copy of 158 tokens that only
  the full-attention blocks can see. An engine that windowed every block could not produce it.
- **The ring needs no ring logic.** `swa_table[p] = p % swa_pages` is the entire ring buffer;
  logical key `j` resolves to `j % ring_tokens`, so a position and the position one ring earlier
  share a slot and no two positions inside one window do. Rounded **up** to a whole page (a
  capacity below `n_swa` would have the oldest key in the window already overwritten) and **capped**
  at the full cache (a window wider than the context can never wrap).

**gpt-oss, six differences, every one silent**
Biases on everything (Q, K, V, output; **the router bias added *before* the top-k, so it changes
which experts run**; and every expert of every bank, added **inside** the router's weighting via
`ggml_add_id`, not after it — folding it into the combined result would scale it by the *sum* of
the weights rather than by each expert's own) · **per-head attention sinks** (one extra logit
joining the softmax denominator with no value vector, so a head's weights **do not sum to one**;
omitting it makes every head's output uniformly **too large, on every block, from the first
token, with nothing to point at**; and the sink is compared against scores that have **already
been scaled** while it is itself raw) · **no QK-norm** · **`ggml_swiglu_oai`** (gate clamped
**above only**; up clamped both ways; **a `+1` on the up branch, so an up projection of zero passes
the gate through instead of annihilating it**; an **alpha-scaled** sigmoid, which is the tanh-free
GELU approximation, not SiLU) · **softmax after the top-k** · **YaRN from position 0**.

**MXFP4** *(not a K-quant)*
- 🔴 **The `kvalues` table holds the true E2M1 values doubled**, and the E8M0 scale byte is
  **halved** (`ggml_e8m0_to_fp32_half`, which is **not** `GGML_E8M0_TO_FP32`). The halving is the
  other half of the doubled table: `half_scale · doubled_value == true_scale · true_value`.
  **Pairing the doubled table with the undoubled scale gives every expert weight twice its value.**
- 🔴 **The two nibbles of a byte are elements `j` and `j + 16` — split halves, not adjacent.**
  Reading them as `2j`/`2j+1` is the same bytes in the wrong order. llama.cpp's own HF converter
  carries a `transform_nibble_layout()` for exactly this.
- `x < 2` needs its own arm in the scale decode because 2^(x−128) for x ∈ {0,1} is a **subnormal**
  f32 that the normalised `(x-1) << 23` form cannot express — it would underflow the exponent field
  and produce a **large** number rather than a tiny one.

**Tensor names and file layout**
- **gpt-oss's pre-FFN norm: the C++ symbol and the GGUF string do not match.**
  `LLM_TENSOR_ATTN_POST_NORM` / `attn_post_norm` in the source; **`post_attention_norm.weight`** in
  the file. A lookup written from the source's spelling **silently finds nothing.**
- **Expert-bank stride.** GGUF stores dimensions fastest-varying first, so the expert index is the
  **last** dimension and each expert's share is contiguous — but the stride is in bytes of a
  quantised block layout, not elements. **Off by one block yields data that is well-formed, the
  right length, and wrong**; the engine runs and answers with a blend of two experts, with no crash
  to debug. Divisibility and partition checks turn every case the arithmetic does not exactly
  account for into an error.
- **A slot sized for the wrong block silently truncates.** `Context::upload` copies `min(src, dst)`
  and reports success either way.
- **Optional tensors must be discovered *and* required consistent across blocks.** A bias present
  in block 0 and absent in block 12 would be applied to a third of the model and skipped for the
  rest.
- **The vocabulary comes from the embedding table's own shape**, not the tokeniser array: the two
  can disagree in a padded model, and the table's row count is what the output matmul produces.
- **The output head is read from the index, not assumed tied to the embedding** — OLMoE carries a
  separate `output.weight`, Q6_K where the embedding is Q4_K. And the **embedding table's format is
  not assumed**: Q4_K in OLMoE, Q8_0 in the Qwen3.6 file.

**Where from.** `moe.rs` module header + `Config::from_model`; `kernels.cpp` throughout;
`tests/gptoss_forward.rs`; `moearc-model/src/tensors.rs`.

---

### C8 · Cross-checks worth building — and the reason they are worth building

📌 **Two implementations written by the same hand from the same source rule out a transcription
slip in one of them, but not a shared misreading of the format.** That is what every check below
exists to rule out.

| check | arbiter | result |
| --- | --- | --- |
| Dequantisers | `ggml_get_type_traits(type)->to_float` — the function pointer llama.cpp's CPU backend itself calls, over bytes lifted from a production GGUF | **MXFP4 bit-exact: `max |gpu − llama.cpp| = 0.000e0`** over 8M elements from four tensors in four different blocks |
| Attention + masked softmax | `ggml_mul_mat` + `ggml_soft_max_ext` on ggml's CPU backend | ✅ — and it gets one thing free: **ggml computed over a flat contiguous run of keys while MoEArc read the same keys through a deliberately scattered page table, so agreement is also evidence the paging is transparent** |
| f32→f16 | `ggml_fp32_to_fp16_row` | ✅ inside the f16 range; see C5 outside it |
| Expert slicing | an independent Python reader seeking to `TensorView::file_offset` | 320/320 and 768/768 |

⚠️ **Every one of these skips *loudly* when its golden directory is absent** — *"a cross-check that
quietly does nothing is worse than no cross-check."*

**Where from.** `moearc-kernels/tests/{gguf_crosscheck,attention_crosscheck,f16_crosscheck}.rs`;
`tools/ggml_dequant_dump.c`, `tools/ggml_attn_ref.c`, `tools/ggml_f16_ref.c`.

---

### C9 · Tolerance discipline — derive from an error model, then print the measurement

**The rule.** No tolerance in this repo was widened until a test passed. Each is **derived** from a
stated error model and then **compared against the error actually measured**, which each test
prints — *"if a tolerance is 100× the observed error, that is visible in the output rather than
hidden in a constant."*

**The model.** f32 unit roundoff `U = 2^-24`, plus three facts:
1. The CPU reference reduces **sequentially in f64**; the GPU reduces in an **f32 tree over 32
   lanes**. For an `n`-term sum that is a relative error of roughly `(n/32 + log2 32) · U`,
   applied to the sum of the **absolute** values of the terms — the only scale that bounds a sum
   with cancellation in it.
2. SYCL's `exp`, `sin`, `cos` are specified to **4 ulp** or better (OpenCL C numerical compliance,
   which SYCL inherits); `pow` to **16**.
3. `icpx` compiles at its default **`-ffp-model=fast`**, so it may contract `a*b - c` into an FMA
   and drop one rounding — worth about **1 ulp per expression**.

Where a bound needs slack beyond that model, it gets a small integer factor **and the factor is
named**.

📌 `reference.rs` accumulates in **f64 where ggml uses `ggml_float`**, so it is **the
higher-precision side of every comparison, not a mirror** — which is exactly why the inventory
marks it RE-PURPOSE rather than RETIRE.

**Where from.** `moearc-kernels/tests/kernels_gpu.rs` header;
`moearc-kernels/tests/forward_pass_gpu.rs`.

---

### C10 · Tie-breaks are part of the contract

- **Greedy sampling: first index wins.** `llama_sampler_greedy_apply` keeps its running best on a
  strict `>`, so it keeps the **lowest** index. A `>=` picks the highest, and the two disagree on
  any exact tie.
- **Router selection: lower expert index wins**, both within a lane (strict `>`, ascending scan)
  and across lanes. That matters more than it looks: **expert choice drives which weights get paged
  in, and a nondeterministic router makes a residency cache impossible to reason about.**
- **Selection is by logit, not by probability.** Softmax is strictly monotonic so the orderings are
  identical, and comparing logits avoids the case where several experts' probabilities round to the
  same f32 and the ordering becomes an artefact of the `exp`.

**Where from.** `session.rs` `argmax`; `kernels.cpp` `moearc_topk_router`.

---

### C11 · Residency must never change what is computed — and that is a free bug detector

**Claim.** Residency decides what has to **move**, never what is **computed**. So **every row of a
residency sweep must emit identical token ids**, whatever its budget. A row that disagrees with
the others has a paging bug — a slot read before it was filled, or two experts sharing one — and
a sweep reporting only throughput would present that bug as a result.

**The check needs no reference implementation at all.** ⚠️ Its analogue under llama.cpp is sweeping
`n_cpu_moe` with the same invariance assertion, and it is worth keeping as a shape.

⚠️ The *host-policy* axis is different: the host path is deliberately not bit-identical to the
device (C4), so a **late** divergence on a near-tied logit is expected and is reported as the step
it happened at. A divergence at step 0, or a row whose cold and warm runs disagree with **each
other**, is a bug.

**Where from.** `moearc-engine/examples/residency_sweep.rs`, `examples/hybrid_sweep.rs`.

---

### C12 · Fully-masked rows and empty spans must be handled explicitly

A fully-masked softmax row divides by zero. The sum is clamped to the smallest normal f32 first,
turning that row into **zeros rather than NaNs** — *"one NaN row would propagate through every
later matmul and destroy the whole batch, not just the row that was empty."* And an **empty key
span** is refused rather than returned as NaNs: it is not "attend to nothing", it is a caller that
computed the window wrong.

Same family: `dot_row` for a bank with no host kernel returns **NaN rather than a plausible
number** — unreachable because `BankSpec::check` refuses it at construction, and *if it ever
becomes reachable, a NaN in the logits is loud where a zero would be silent.*

**Where from.** `kernels.cpp` `moearc_softmax`, `attn_decode`; `host_experts.rs` `dot_row`.

---

## 6. Build, linking and process traps

### B1 · The `_intel_fast_memcpy` static-link wall

**Claim.** Building the SYCL object as a static `.a` with `ar` and letting cargo link it with `cc`
fails on **`undefined symbol: _intel_fast_memcpy`** — a symbol from `libintlc`, one of several
Intel runtime libraries `icpx` links automatically and `cc` knows nothing about.

🔴 **Chasing them one at a time (`intlc`, `irc`, `imf`, `svml`, `irng`, …) is the trap: the list is
a property of the compiler version, not of our code.**

**Resolution.** Let `icpx` perform the link. The `.so` records its own dependencies in `DT_NEEDED`
and cargo links one library. It is also the shape we ship anyway — the SYCL runtime cannot be
statically linked.

📌 **This survives the pivot verbatim.** `libggml-sycl.so`'s own `DT_NEEDED` names
`libsvml` / `libirng` / `libimf` / `libintlc.so.5`, which is why the inventory flags
`launcher.sh`'s `LD_LIBRARY_PATH` ordering as newly load-bearing. `moearc-llama/build.rs` already
cites and inherits this.

**Where from.** `moearc-kernels/build.rs` module header.

---

### B2 · 🔴 The rpath that reached only the tests — 309 green tests over a binary that could not start

**Claim.** A build script's `cargo:rustc-link-arg` applies **only to the crate that emits it** and
is not inherited by anything downstream.

**Symptom.** An `-Wl,-rpath,$OUT_DIR` emitted from `moearc-kernels/build.rs` reached that crate's
own **tests** — the one target class that *does* inherit it — and nothing else. **`moearc-server`
linked and then died in the dynamic loader with `libmoearc_kernels.so: cannot open shared object
file`, while every test was green.**

**Why the obvious fixes don't work.** `cargo:rustc-link-search` and `rustc-link-lib` **do**
propagate (which is why downstream crates link at all) but `-L` is a *link*-time path and leaves
no trace in the executable. Emitting more link args (`-bins`, `-tests`, …) cannot help — they are
all scoped to the emitting crate too.

**Resolution: the soname carries the path.** `ld` copies a library's `DT_SONAME` **verbatim** into
the `DT_NEEDED` of everything that links it, and glibc's loader treats a `DT_NEEDED` string
containing a slash as a **path** rather than a name to search for. Setting the soname to the
object's absolute location reaches every consumer — binaries, tests, examples, benches, in this
crate and any other — with no cooperation and no environment variable.

**Two consequences.** It is **not relocatable** (the path is a build tree's `OUT_DIR`, so it is a
development build only — the packaged `dlopen` route never consults `DT_NEEDED` and is
unaffected). It is **stale-proof by construction** (`OUT_DIR` is stable for a given crate, profile
and feature set; cargo rewrites in place; nothing is copied, so there is no cache to invalidate
and no copy to race).

The object also carries its own **`DT_RUNPATH`** (not `DT_RPATH` — it resolves this object's
dependencies and is not inherited by theirs, which is correct, since `libsycl.so.9` carries
`$ORIGIN` and finds its Unified Runtime adapters itself).

🔴 **The same trap exists verbatim for `libllama.so` / `libggml-sycl.so`.**

**The test that catches it.** A binary whose only job is to be started, run with an **emptied**
environment (`env_clear` — no `PATH`, no `HOME`), and separately under a **hostile**
`LD_LIBRARY_PATH` (passing that means the resolution is not merely surviving without help, it is
not *asking* for help). ⚠️ **It must call a real symbol** — the linker drops an unused `DT_NEEDED`
under `--as-needed`, and a binary that does not use the library is not evidence about one that
does. ⚠️ And it is a fair proxy **only because the build script now emits no link args at all**; a
test asserts that equivalence, so the day someone reintroduces a local rpath the proxy stops being
fair *and says so*. `packaging/bundle.sh:73` already ships it under the name `moearc-selftest`.

**Where from.** `moearc-kernels/build.rs`; `tests/clean_env_binary.rs`;
`src/bin/moearc-kernels-smoke.rs`.

---

### B3 · Drop order is load-bearing, in two different places, and both are invisible failures

**(a) `session.rs` — the device thread's locals.** `ctx` and `mapped` are **two separate `let`s in
that order**, and it became load-bearing the moment staging started uploading **asynchronously out
of the mapping**. Locals drop in reverse declaration order, so the SYCL queue is destroyed first
and its destructor waits for everything submitted to it — including an expert copy whose *source*
is a page of `mapped` — before the file is unmapped. **The other order is a use-after-free the host
cannot see: pages unmapped while the device is still DMA-ing out of them, with nothing anywhere
reporting an error.** ⚠️ It was a **single tuple binding** before — exactly the shape that makes
this invisible.

**(b) `moe.rs` — `Model::drop`.** `host_add` submits an asynchronous copy whose source is
`State::cpu_out`, a plain host `Vec`. `State` lives **inside** `Model`, declared *after* the
`Context`, so it drops **before** it. Without an explicit sync in `Drop`, a copy still in flight
would read a freed buffer and nothing would report it.

📌 **In both cases the incidental property is deliberately not relied on.** Every `decode` ends
with a blocking logit readback, which on an in-order queue drains everything — so in practice
nothing is outstanding. That is a property of the current graph; a future decode that returns
without a readback would silently remove it.

**Where from.** `session.rs` `device_thread`; `moe.rs` `impl Drop for Model`.

---

### B4 · Every failure path of a load must **join** the device thread

**Claim.** A load that got as far as uploading weights has gigabytes of USM on the card. When it
then fails, the worker returns and runs the whole teardown — **thousands of `moearc_free_device`
calls and a SYCL queue destructor** — on a thread the caller was about to **detach** by dropping
its `JoinHandle`. The caller meanwhile prints the error and, in a one-shot tool, returns from
`main`; **process exit runs the SYCL and Level Zero atexit handlers concurrently with that
teardown.**

**Symptom.** The crash lands **after** the error message, so it reads as *"this project is
broken"* rather than *"my driver is too old"*.

⚠️ **It is a race**, so it does not reproduce on a machine whose load succeeds, or whose main
thread happens to be slower than the teardown. **Joining removes the race rather than shortening
it**, and the wait is bounded — the worker has already sent its reply.

**Where from.** `session.rs` `load_with`.

---

### B5 · Drain the queue before **reporting** a device error

**Claim.** Every kernel wrapper funnels its return code through one place, so that is the one place
that knows a device operation failed — **and the caller's next act, on every error path in the
engine, is to unwind and free the buffers that operation was using.** The queue is asynchronous, so
work submitted before the failure may still be reading them.

**Symptom if omitted.** Freeing USM out from under a live kernel is **a segfault in the driver,
arriving after the error message and looking like a crash in whatever ran next.**

The drain's own verdict is discarded on purpose — the queue is already failed and the original
return code carries the diagnosis. What matters is that nothing is in flight when the error
returns.

**Where from.** `moearc-kernels/src/lib.rs` `Context::check`.

---

### B6 · `upload_async`'s contract, and the two things it costs the caller

1. **`src` must stay allocated and unmodified until the copy completes**, and the caller cannot
   observe when that is short of a full `sync`. The only sound sources are ones that outlive the
   `Context` — a memory-mapped file, or a buffer owned for the engine's life. 🔴 **Passing a
   temporary, a stack array, or a buffer about to be reused is UB that will not fail loudly:** it
   produces a slot filled with whatever the memory became, which is finite, plausible, wrong.
   ⚠️ **If staging ever sources from anything but the mapping — a decompressed buffer, a reordered
   scratch, a temporary — it must go back to the blocking upload.**
2. **Failures surface later, at the next synchronisation.** An over-committed pool used to fail on
   the copy itself with a legible message; it now fails on whichever kernel or readback comes next
   (see M2, where that is exactly what happens).

**Ordering is preserved by the queue, not by the wait** — the memcpy is submitted before the matvec
that reads the slot, so the matvec runs after it. That is why `stage`-before-compute remains a
correctness property rather than an accident of every call waiting.

**Where from.** `kernels.cpp` `moearc_copy_h2d_async`; `moearc-kernels/src/lib.rs`; `moe.rs`
`stage`.

---

### B7 · Submission is not completion — the API-level version of the same trap

Every kernel method **submits work and returns**. `Ok(())` means the device accepted the launch,
not that it ran. The queue is **in-order**, so ordering needs no expressing — but a host that wants
to read a result, or reuse a host buffer a copy is sourcing from, must reach a synchronisation
point, and **there are exactly two, and they are the same two that report a failed kernel**:
`sync()` and `download()`.

🔴 **The in-order queue is load-bearing, not incidental.** An out-of-order queue with these
submissions would produce output that is finite, fluent and wrong.

**Where from.** `moearc-kernels/src/lib.rs` `Context`; `kernels.cpp` `moearc_ctx_create`.

---

## 7. Register of chosen constants

🔴 **This project has been badly burned by a guessed value being reported as a result.** Everything
below is an **input**, not an output. Nothing here should ever be quoted as a measurement.

| constant | value | where | status |
|---|---|---|---|
| `Headroom::PROVISIONAL` | `Fraction(0.12)` | `memory.rs` | 🔴 **Chosen, explicitly not measured**, and not borrowed from another runtime. Withholds 1.36 GiB on an 11.33 GiB card — **~700 expert slots**. Measurement (M2) says the real ceiling is ~85% of reported free while this leaves 88%; left alone deliberately (see M2). Printed in the plan's rationale. |
| `Policy::page_tokens` | `256` | `memory.rs` | 🔴 **Chosen, not measured.** Trades allocator bookkeeping against internal fragmentation. ⚠️ **Disagrees with the engine — see §8·X1.** |
| `Policy::min_context_tokens` | `2048` | `memory.rs` | 🔴 **A product judgement, not a measurement** — a guess at the shortest context worth starting a server for. **Load-bearing in a way that is easy to miss:** with `Bias::Experts` the planned context lands on this number *exactly* whenever the leftover bytes are worth less than one KV page, which is **the common case** (reference model: 1.95 MiB experts against a 5 MiB page). `Reason::ContextAtPolicyFloor` exists so the output says *"this is the policy talking back"* rather than reporting a capacity. `moe.rs` overrides it with the session's actual `n_ctx` at load — leaving the default makes the planner refuse any session shorter than 2048. |
| `Bias::Experts` as `#[default]` | — | `memory.rs` | 🔴 **Inherited from FreeToken and contradicted by MoEArc's own measurement.** [`freetoken-priors.md`](freetoken-priors.md) P8: on a real routing trace **not one expert slot in 6,144 is touched every token** (hottest p = 0.979; 78% below p = 0.10) while **every KV byte is**, so at 64K **KV-first costs 0.08 GB/token against experts-first's 5.83**. *KV residency is all-or-nothing; expert residency degrades gracefully.* Should be justified by finding the crossover `L*` or the default should move. **Right now it is neither.** |
| `Reserve::DEFAULT` | `Fraction(1/5)` | `host_budget.rs` | **A product judgement, and explicitly not a physical quantity** — there is no correct value to measure, only a policy about how much of someone's machine a local inference tool may claim. Settled, do not re-litigate: a **50%-of-RAM cap on the model** was proposed and refused, because half of the 91 GiB reference box is 45.5 GiB and **gpt-oss-120B is 59.0 GiB** — and because the weights are `mmap`ped, so a cap would not protect anything. |
| `DEFAULT_N_CTX` | `4096` | `session.rs` | **Chosen, with its cost stated:** 148.5 MiB on gpt-oss-120B, about **twelve expert slots**, against **4.51 GiB** for the trained 131,072. Bounds the *default*, never the knob. |
| `PAGE_TOKENS` | `32` | `moe.rs` | ⚠️ **Unattributed** — *"it only has to be small enough that a sequence's unfilled tail is cheap."* |
| `WG` | `32` | `kernels.cpp` | A native sub-group width on Intel Xe. **A tuning constant, not a correctness one** — any power of two gives the same answer up to the order of the reduction. |
| `MATVEC_ROWS` | `8` | `kernels.cpp` | **Reasoned** from the measured activation re-read (T5) and **swept** (K2: within 4% across 2/4/8/16). Not arbitrary, not strongly determined either. |
| `norm_wg` thresholds | `≥1024 → 256`, `≥256 → 64`, else `32` | `kernels.cpp` | The **28 µs at WG=32** datum (K8) is measured and motivates widening. ⚠️ **The three breakpoints themselves are unattributed.** |
| `MAX_TOPK` | `32` | `kernels.cpp` | Bounds a per-work-item array (kernels cannot allocate). Real MoE models use ≤8; **chosen with headroom.** |
| `MAX_BATCHED_MATS` | `32` | `kernels.cpp` / `lib.rs` | Bounds a **kernel argument** — 32 pointers = 256 bytes, well inside what Level Zero takes. A caller with more matrices is **refused, not truncated**, which is also what bounds a model's active expert count at load. |
| `spin_budget` | `2_000_000` `spin_loop()` iterations (`MOEARC_HOST_SPIN`) | `host_experts.rs` | **Explicitly tunable** — *"the right value is a property of the machine and the model."* Reasoning: during generation a block arrives every few hundred µs, so a spinning worker sees the next job in tens of ns and never pays a futex round trip. |
| `default_threads` | cores − 1 (`MOEARC_HOST_THREADS`) | `host_experts.rs` | **Reasoned, not swept** — protect the device thread (S8). |
| `chunk_for` | ≥16 rows, "a couple of chunks per thread per phase" | `host_experts.rs` | ⚠️ **Unattributed.** Rationale given (a straggler costs a fraction of a chunk rather than a whole one), no measurement. |
| `TOL` | `1e-5` relative, floored at magnitude 1.0 | `tests/host_experts_gpu.rs` | ✅ **Measured**, and each test prints the error it observed. Included here only to be explicit that it is *not* in the guessed category. |
| `swiglu_oai` `alpha` / `limit` | `1.702` / `7` | `kernels.cpp`, `moe.rs` | **Transcribed from llama.cpp's call site** (`constexpr` at `LLM_FFN_SWIGLU_OAI_MOE`), not GGUF keys and not ours. Passed as arguments so the kernel states its arithmetic rather than hiding two magic numbers. |
| `VERIFIED_RUNTIME_BUILD` | `37_020` | `moearc-device/src/fitness.rs` (**KEEP**) | Not retired, listed because `bench/guard.rs:597` gates published results on it and **it must now also cover llama.cpp's SYCL requirements.** |

---

## 8. Contradictions found while harvesting

Each of these is a place where two parts of the tree currently say different things. **None is
resolved here** — they are reported so somebody can decide.

### X1 · `page_tokens` — the planner and the engine disagree by 8×

`memory.rs` `Policy::default()` sets **`page_tokens: 256`**; `moe.rs` sets **`const PAGE_TOKENS:
usize = 32`**. The planner reserves KV at 256-token page granularity and models internal
fragmentation as up to 255 wasted tokens per sequence; the engine allocates 32-token pages and
wastes at most 31. **The plan's page count does not describe the allocation the engine makes.**
Both are documented as chosen constants; **neither cites the other.**

### X2 · Two "ceilings" for device memory, a page apart, and they point opposite ways

- `memory.rs` / `moe.rs`: *"the usable ceiling is about **85%** of what the device reports free"* —
  a **write/kernel** failure at 9.81 GiB against 11.33 GiB reported free, **with a model loaded**.
- [`calibration.md`](calibration.md): **12,418,351,104 B committed = 102.05% of reported free**,
  reproducible to the byte, **with nothing loaded but `memset` blocks.**

They are not the same experiment — the engine measurement carries 951 MiB of dense weights, a KV
cache, activations, scratch and live kernels; the probe carries none of that. **The reconcilable
reading is that 85% is a ceiling for a loaded engine's *pool*, not for the allocator** — which is
also exactly the "activation memory is not included" caveat calibration.md states. ⚠️ **But no
document states the reconciliation**, and a reader meets *"85%"* and *"102%"* within a page of each
other.

### X3 · Warm-decode disk reads: 0 MiB or 4,512 MiB

`moe.rs` `ResidencyReport::bytes_staged_uncovered`: *"Warm decode on the reference box measured
**0 MiB** of actual disk reads at every depth from 128 to 8192 tokens."*
`bench/baselines/gpt-oss-120b.md` §7.4: **4,512 MiB** of warm-decode disk (71.6 MiB/step) at depth
8192 under `frac:0.5`. The `moe.rs` claim is **unqualified** and appears to hold only for the
policies that stage little (S4). It is load-bearing there, because it is the justification for
describing `bytes_staged_uncovered` as a bound expected to disagree with reality on this box.

### X4 · `runtime.rs` states the prefetch conclusion without its qualifier

`runtime.rs`'s module header justifies the plain token loop with *"Measurement on an Arc B580
showed this costs about 2% versus one bulk transfer — see docs/calibration.md — because each
block's fetch already saturates the link."* [`calibration.md`](calibration.md) itself is careful to
add: *"Any remaining case for it has to come from overlapping fetch with **compute**, which is a
different argument and needs compute numbers we do not have."* Anyone reading `runtime.rs` alone
comes away thinking prefetch is closed. **It is closed on the transfer path only.**

### X5 · The inventory's stated harvest target is not where the risk actually is

[`pivot-inventory.md`](pivot-inventory.md) flags `moe.rs` for *"the ~85%-of-reported-free
allocation cliff and `malloc_device` succeeding past physical VRAM"*. **Both of those are also in
`memory.rs`, `moearc-kernels/src/lib.rs`, `tools/vram_probe.cpp` and `docs/calibration.md`** — four
other places, three of which are KEEP. **Neither finding was ever at risk from deleting `moe.rs`.**

The genuinely single-sourced findings in `moe.rs` are different ones: the **slot-vs-bank 7%
over-promise** (M4), the **router-readback drain** (T4), the **batched-FFN measurement** (S17), the
**`static:15` ring-victim divergence at token 18** (S12), and the **SWA/KV geometry arithmetic**
(M8). Those are what §9 is gated on.

### X6 · Two line counts in the inventory are stale

`moearc-engine/src/memory.rs` is listed at **797** lines and is **1,102** (the `llama_split`
section post-dates the count). `host_budget.rs` is listed at **894** and is **935**. Everything
else spot-checked matches. Minor, but the checklist below uses measured counts.

---

## 9. Deletion checklist

**Preconditions for every row.** `moearc serve` proves the replacement end to end; this document
is merged; and for anything under `moearc-server`'s reach, `src/engine.rs` is rewired first.

Legend: ✅ harvested, safe to delete · ⚠️ harvested but **gated** on something else moving first ·
🔴 do **not** delete.

### 🔴 How the verdicts below were checked — 2026-09-09

**This list has been wrong twice.** It marked `session.rs` do-not-delete while marking three of
its own imports safe, and it marked `hybrid_sweep.rs` deletable while `packaging/bundle.sh` was
staging that exact binary into every release tarball as `moearc-bench`. Both errors have the
same cause: the verdicts were reached by reading the tree.

So they were re-derived by **deleting the candidates and compiling**. The whole set below was
removed from a working copy, the module declarations and `[features]` entries that named them
were removed with them, and `cargo check --workspace --all-targets` was run on the result. The
tree was then restored from a backup taken before the experiment; nothing was committed.

**Result: the workspace compiles with all of it gone**, on default features, with one
pre-existing error in `crates/moearc-cli/src/bench/guard.rs` that is unrelated in-flight work by
another author, and **nine warnings, all of one kind** — `unexpected cfg condition value: gpu`,
at five sites in `bench/probe.rs` and four in `bench/timed.rs`. Those nine are the complete
remaining surface. Everything else in the workspace is already independent of the retired
engine.

⚠️ **What the experiment did not prove.** It ran with `moearc-cli`'s `gpu` feature *removed*,
which is what cfg's out `bench/timed.rs`'s `measure()` GPU arm — the last caller of `Session`,
`SessionOptions`, `StopConditions`, `moe::Residency`, `host_experts::HostPolicy` and `profile`.
Deleting these files without that feature also going is a compile error, not a clean removal.
That file belongs to the benchmark work and is not touched here.

### `crates/moearc-engine/src/`

| file | lines | harvested as | verdict |
|---|---:|---|---|
| `moe.rs` | 2,911 | M2, M4, M5, M6, M7, M8, M9, T4, T6, T7, S5, S12, S17, K1, K2, K12, C6, C7, B3(b), B6 | ✅ |
| `host_experts.rs` | 1,466 | **S1** (the idea most likely to return in phase 3), S6, S7, S8, S15, S16, C4, C12; the `off`/`frac:<f>`/`over:<n>`/`all` spelling; the compute-placement-vs-data-placement coupling hazard (`HostPolicy::Fraction(1.0)` against a budget that backs nothing sends **every** host-executed expert to the drive) | ✅ |
| `cache.rs` | 381 | S13, S14, the same-step pinning invariant | ⚠️ **One loss is real:** `cache.rs` and `residency::simulate` are two independent LRU implementations whose agreement on miss count was a genuine cross-check. Deleting this deletes the check. Note it, then delete. |
| `runtime.rs` | 411 | S11, plus §8·X4 | ✅ |
| `kv.rs` | 353 | The paged-vs-contiguous rationale; `KvUsage` reporting *"token slots allocated but not yet written"* so the waste-bounding claim stays honest; take-the-page-before-mutating so an exhausted pool leaves the sequence unchanged | ⚠️ **Gated.** `KvUsage::utilisation()` feeds `ServeSample::kv_utilisation` in the TUI and **has no llama.cpp source** (inventory, *"five gaps"*). Delete only after that dial is re-sourced or deliberately removed. |
| `session.rs` | 599 | B3(a), B4, C10, `DEFAULT_N_CTX`'s rationale (§7), the error-string unwrapping that stops *"unsupported model: unsupported model: …"* | ✅ **UNBLOCKED 2026-09-09.** The contract below was not copied into a comment — it was copied into **`moearc_server::generate::drive`**, the one token loop both the stub and the llama.cpp generator are written against, with six tests that run on default features. The one that matters asserts the last clause directly: a scripted model records every token it is fed, and a stop token, a cancelled token and a budget-exhausting token are each shown to be absent from that record. `moearc-server/src/engine.rs` now drives `moearc_llama::Context` and names nothing in this crate. **The contract, verbatim:** blocking · takes `&self` · `on_token` called **once per accepted token, in order** · `false` stops promptly and returns stats so far · **stop tokens** enforced engine-side and **not** emitted through `on_token`, **stop strings** are the caller's · sampling is a caller-supplied closure · a token that was never emitted must not reach the KV cache. `moearc-server`'s `Generator` was written against exactly this shape. |
| `profile.rs` | 101 | Zero-cost-when-off (one relaxed atomic load, no clock read); **phases do not nest, so the residue between their sum and the wall clock is host work nobody attributed**; `reset()` after warm-up; dropping on an early `?` is deliberate | ✅ — `llama_perf_context` may cover it; 101 lines either way. |

### `crates/moearc-engine/examples/` and `tests/`

| file | lines | harvested as | verdict |
|---|---:|---|---|
| `hybrid_sweep.rs` | 254 | S1's **`busy` vs `wait` is the overlap measurement**; *"tok/s can fall while `wait` is near zero"*; C11's host-axis caveat | ⚠️ **This row said ✅ and was wrong: it is a shipped release binary.** `packaging/bundle.sh` staged it as `moearc-bench` and `bench/reproduce.sh` execs `./moearc-bench` from an installed tarball. bundle.sh no longer stages it (2026-09-09) — it cannot, without putting the retired engine back in the payload. ⬜ `bench/reproduce.sh` still names it and now reports that it cannot find its binary; re-point it at `moearc bench` before deleting this. |
| `profile_decode.rs` | 231 | Throw away the first tokens (a cold pool would attribute staging to whichever phase ran first); **residency must be required, not defaulted** — on a model that does not fit, the defaults cannot be allocated and *"a profile of a configuration the caller did not choose is worse than no profile"* | ✅ |
| `host_expert_bench.rs` | 221 | **S9** (410 µs artefact) + the three-numbers structure | ✅ |
| `olmoe_generate.rs` | 80 | nothing — superseded exactly by `moearc-llama/examples/generate.rs` | ✅ |
| `olmoe_forward.rs` | 357 | **C1, C2** (the three-way logit table) | ✅ |
| `qwen3moe_forward.rs` | 382 | **C1, C3**; the two-sessions-alive VRAM arithmetic (`static:12` at 1,544 slots ≈ 7.8 GiB total works, `static:20` does not) | ✅ |
| `gptoss_forward.rs` | 509 | **C7**'s gpt-oss list and MXFP4 notes; the SWA test-design finding (below `n_swa` a short test proves nothing; 158+256 tokens to position 413; the passage-reproducing prompt); the `#![cfg(feature = "gpu")]`-not-`required-features` reasoning | ✅ |
| `host_experts_gpu.rs` | 300 | **C4** + `TOL` | ✅ |

### `crates/moearc-kernels/`

| file | lines | harvested as | verdict |
|---|---:|---|---|
| `kernels.cpp` | 1,922 | **K3, K4, K5, K7, K8, K9, K10**, T2, T3, T5, T6, C6, C7 (MXFP4 / RoPE / YaRN / sinks / `swiglu_oai` / router), C12, B6, B7 | ✅ |
| `src/lib.rs` | 1,253 | M1, T2, K10, C6, C10, **B5, B6, B7** | ✅ |
| `src/ffi.rs` | 266 | The return-code convention (`0` ok, `-1` device failure or null, `-2` argument out of range) and **why not bindgen** (parsing `<sycl/sycl.hpp>` would put oneAPI headers back in every consumer's build) | ✅ |
| `build.rs` | 144 | **B1, B2** | ✅ — **checked 2026-09-08:** `moearc-llama/build.rs` carries six references to `_intel_fast_memcpy` / soname / `DT_SONAME`, so the knowledge has already moved. |
| `tests/*` (7 `.rs` files + `common/`) | 2,291 | **C8, C9, C5, B2**'s clean-env harness | ⚠️ **`f16_crosscheck.rs` carries C5, which is a finding about the local llama.cpp build, not about our kernels.** It survives the pivot and is now a property of the engine we ship on. Make sure C5 is landed here before the file goes. |
| `examples/matvec_scaling.rs` | 139 | **K5, K6** | ⚠️ **Consider keeping.** It is the re-check harness for a **workaround whose cause is unknown**, and ggml-sycl's MoE GEMV is also a raw-pointer API that nobody has checked for the same codegen difference. |
| `examples/launch_overhead.rs` | 130 | **K12** (the 1.6 µs) | ✅ |
| `tools/*.c` | 489 | golden generators for C8's cross-checks | ⚠️ **Delete only with the tests they feed.** If `reference.rs` is kept as an oracle (it is RE-PURPOSE), some of these become the way to regenerate its goldens. |
| `src/reference.rs` | 931 | — | 🔴 **RE-PURPOSE, do not delete.** Per the inventory: no `unsafe`, no FFI, no SYCL, no device; accumulates in f64 where ggml uses `ggml_float`, so it is **the higher-precision side of every comparison, not a mirror** (C9). ⚠️ Its `QuantType` block-geometry table duplicates `moearc-model/src/quant.rs` — resolve deliberately. |
| `src/bin/moearc-kernels-smoke.rs` | 24 | **B2** | 🔴 **RE-PURPOSE.** The trap it catches exists verbatim for `libllama.so`/`libggml-sycl.so`. |

### `crates/moearc-model/`

| file | lines | harvested as | verdict |
|---|---:|---|---|
| `src/tensors.rs` | 929 | **C7**'s expert-bank stride trap; the mmap-not-read rationale (a 20.6 GiB file read into a `Vec<u8>` is 20.6 GiB of RSS before a single token, on a machine whose GPU has 12); the tensor-name conventions incl. gpt-oss's `post_attention_norm` mismatch | ⚠️ **Land the tensor-name table somewhere first** if `moearc-llama` ever needs to reach for a tensor by name. Otherwise ✅ — **compiler-verified 2026-09-09**: only `moe.rs` reached it. `moearc-cli` uses `moearc_model::{gguf, quant, pull, ModelInfo}`, none of which is this file; the crate itself stays. Delete the `pub mod tensors;` line in `src/lib.rs` with it. |
| `examples/map.rs` | 177 | The `VmRSS`-either-side-of-the-map measurement that **backs** the zero-copy claim | ⚠️ Cite the measurement before deleting the evidence for it. |
| `examples/expert_probe.rs` | 94 | The independent-reader check on slice arithmetic (seek to `file_offset`, read `len`, compare) — *"the only way an off-by-one stride is ever going to be caught"* | ✅ after `tensors.rs` |
| `tests/mapped_model.rs` | 158 | offsets only | ✅ after `tensors.rs` |

### Top level

| path | harvested as | verdict |
|---|---|---|
| `registry/`, `site/`, `src/{cli,engine,kernels,server}/`, `third_party/ipex-llm/` | nothing — all contain only `.gitkeep` | ✅ |
| `dist/` (built tarball) | nothing | ✅ delete the artifact, keep the directory. **It packages the old architecture and must not be published.** |
| `tools/vram_probe.cpp` | M1, M3 | 🔴 **KEEP** — the trap applies to llama.cpp too. |
| `tools/stream_bench.cpp` | T1, T3 | 🔴 **KEEP or archive** — it is the only measurement of the serialisation penalty, and T3's "measure before building a pinned ring" is unresolved. |
| `bench/traces/`, `bench/references/`, `bench/baselines/`, `bench/results/`, `bench/policy-sweep.md` | S1–S4, S10 cite them | 🔴 **KEEP.** Irreplaceable data, not code. ⚠️ **Do not quote llama.cpp comparisons out of `baselines/`** — that is where the withdrawn numbers live. The MoEArc-against-itself host-policy results cited in §3 are not withdrawn. |
