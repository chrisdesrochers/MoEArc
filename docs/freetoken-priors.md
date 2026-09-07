# FreeToken priors — what we can believe before we measure

**Written 2026-09-06.** Sources: a read-only study clone of `FlashML-org/FreeToken` (Apache-2.0) at
`/zfs/swift/projects/FreeToken-study`; the paper, [arXiv:2608.16157v1](https://arxiv.org/abs/2608.16157)
(Yang et al., 17 Aug 2026, 7 sections, **no appendix**); and the repo's `docs/`.

## Why this document exists

FreeToken paid for discovering *which knobs matter on a CPU+GPU MoE split and how they interact*.
That knowledge transfers. **Their constants do not** — they are fitted to NVIDIA hardware, and
several to a specific driver version on a specific card.

The point is to start from an informed position: *"our optimizations should push us towards better"*,
not "try every combination to see what sticks".

🔴 **This is not a port and no FreeToken code is copied into MoEArc.** What follows are findings and
structure. FreeToken text is quoted only as *evidence for a finding*, briefly and attributed.
MoEArc's implementation is clean-room Rust. See [`archaeology.md`](archaeology.md) for the
three-source split and [`calibration.md`](calibration.md) for the constants audit this document does
not repeat.

## How to read the tags

| Tag | Meaning |
|---|---|
| **TRANSFERS** | Structural; should hold on any CPU+GPU split. Believe it before measuring. |
| **NVIDIA-SPECIFIC** | True, but tied to their hardware. The Arc equivalent question is named. |
| **MUST RE-MEASURE** | The shape transfers, the value cannot. The settling experiment is named. |

Every prior states **what it predicts**. A prior that cannot be refuted is not worth holding.

---

## Part 1 — the mechanism, extracted

Five of the priors are corollaries of this, so it is worth stating in full first. This is the answer
to *"what is their scheduling decision, exactly"*.

### The three-way decision

Per **MoE layer, per decode step**, the routed experts are split into hits `H` (already resident) and
`m` unique misses. The misses are then partitioned (paper §3.2, Eq. 1): `M = F ∪̇ C`.

1. **Resident (`H`)** — in the GPU slot cache. Runs on the GPU.
2. **Fetched (`F`, size `q`)** — copied host→device into evicted slots, runs on the GPU, and **stays
   resident** for future reuse.
3. **Overflow (`C`)** — left non-resident, slot id rewritten to `-1`, and **computed on the CPU** from
   the host-resident pool. Residency is unchanged. The two partial results are summed.

The CPU branch is launched *first*, so it overlaps the GPU fetch + GEMM. Exposed latency is the slower
branch, and that is exactly what the split is chosen to balance. The output is exact — *"without
algorithmic approximation"*.

### Where the crossover comes from — it is solved, not tuned

**Paper form (§3.2, Eqs. 2–4).** With `S` = bytes of one expert, `B_P` = pinned host→device transfer
bandwidth, `B_H` = host-side expert-processing bandwidth, and the two paths contending for the same
DRAM:

```
B_R = max(B_H − B_P, 0)                                  (Eq. 2)
T_fill(q) ≈ qS/B_P     T_cpu(m−q) ≈ (m−q)S/(B_H − B_P)   (Eq. 3)
                    q* ≈ m · B_P / B_H                    (Eq. 4)
```

**Shipped form**, which supersedes it. `moe/bench_profile.py::load_hybrid_fetch_fraction` prefers

```
f* = pcie_gather_overlap_gbs / (pcie_gather_overlap_gbs + cpu_moe_overlap_gbs)
```

and falls back to the paper's `B_P/B_H` only for older profiles. The difference is the whole point,
and their own comment (`moe/benchbw.py`) states it better than I could:

> *"The standalone numbers cannot predict this split. Assuming full DRAM contention (CPU keeps
> `cpu_bw - pcie_bw` under DMA) over-penalizes a CPU kernel that never saturated DRAM to begin with…
> assuming no contention ignores it entirely. Measuring the contended pair directly gives the hybrid
> backend its bandwidth-matched fetch split."*

`q*` is recomputed **every layer, every step**, because `m` changes every step; the bandwidths are
per-machine constants. The whole computation is device-resident inside a captured CUDA graph — they
note the closed form was chosen partly *because* it is cheap enough to stay there.

It takes exactly **two bandwidths and the current miss count**. It does not use router scores, expert
popularity, batch size, or layer index.

### Calibration is a real, cached, per-machine measurement

`ft bench bw` is a manual one-shot command. It measures the *production* kernels — the real
`CpuMoeExecutor` GEMV and the real `copy_missing` PCIe gather — standalone and then **concurrently**,
and writes `~/.cache/freetoken/benchbw/<gpu-uuid>.json`. A profile from a different GPU is **discarded
with a warning rather than borrowed**. Without a profile, hybrid never activates and the fetch cap
falls back to a fixed 1 expert per layer per step.

The join key is the **expert format**, not the model: *"the CPU-MoE-vs-PCIe-gather bandwidth ratio the
choice rides on is dominated by (format, hardware), not by the exact model"*.

### Which misses get fetched

`q*` is rounded to an integer, and *which* experts fill `F` is delegated to the replacement policy —
in the shipped code, **expert recency** (an LRU over experts, not slots), tie-broken to the lower id.
The comment names the alternative it replaced:

> *"this prioritizes *recurring* misses for caching, lowering the steady miss rate. Otherwise the
> lowest expert ids are fetched … the original routing-blind heuristic."*

It always retains **at least one fill**, so the cache keeps warming even when the CPU handles most
misses.

### And it refuses to engage when unsure

`load_backend_recommendation` returns `"hybrid"` only when **every** benched workload sharing the
expert format recommends it; a mixed verdict resolves to plain offload. The rule is
`B_H > 2.0 × B_P`. No profile → `None` → stay on offload.

📌 **The shape to copy: measure two bandwidths under contention, solve for the balance point, and
refuse to engage the mechanism when the measurement is missing or ambiguous.**

---

## Part 2 — the priors

### P1 — The CPU/GPU crossover is a closed form over two measured bandwidths. And our own number already matches it
**TRANSFERS — and this is the one I would bet on hardest.**

The derivation contains no NVIDIA term. Any CPU+GPU split has a host compute rate and a host→device
transfer rate.

🔴 **The striking part.** MoEArc found empirically, by sweep, that **`frac:0.75` host routing wins at
every depth**. FreeToken's closed form says the host fraction should be `1 − B_P/B_H`. Our Arc
bring-up measured a B580 GPU kernel reading pinned host memory at **13.47 GB/s** — *"the same
bandwidth as the dedicated copy engine"*, with random 8 KiB scatter at 13.37 GB/s — so `B_P ≈ 13.5`.
`B_H` on CHARA is **not measured**, but across the plausible DDR range FreeToken reports for real edge
hosts (47.5–63 GB/s):

| `B_H` | predicted host fraction `1 − B_P/B_H` |
|---|---|
| 47.5 | 0.72 |
| 53.8 | 0.75 |
| 63.2 | 0.79 |

**Our swept optimum is 0.75.** ⚠️ Stated carefully: `B_H` is unmeasured, `B_P` is a host-read rather
than a copy-engine gather, and MoEArc's `frac` and FreeToken's `1 − f*` are only *believed* to be the
same quantity. This is a bracket that wants confirming, not a result.

**Predicts:** measuring `B_H` and `B_P` on CHARA *under contention* will place `1 − f*` within a few
points of 0.75, with **no sweep**. **Refuted if** the measured prediction lands far from the swept
optimum — in which case the first suspect is a standalone rather than contended measurement, since
both paths read the same DRAM and standalone numbers over-count both.

**Also predicts** the threshold call: `B_H/B_P ≈ 3.5–4.7 > 2.0`, so hybrid should be strongly
recommended on this box, not marginal.

**Value if it holds:** the entire host-routing fraction becomes a two-measurement calculation instead
of a sweep, per machine, forever — and it generalises to Arc cards we do not own.

---

### P2 — Their q* is incomplete, and our own data says how
**TRANSFERS — and this is the delta we own.**

`q*` balances *bandwidth only*. There is no term anywhere for cache pollution. But MoEArc measured a
second, independent benefit: **a miss routed host-side is never admitted, so it never evicts.** The
margin of `frac:0.75` *grows* with depth — 1.08× at 512 to 2.83× at 8192 — which bandwidth balance
cannot explain, because neither bandwidth changes with prompt depth.

FreeToken's own design partly concedes this: `C` is defined as leaving residency unchanged, and they
force at least one fill so the cache still warms. They noticed the residency side-effect. They did not
put it in the objective.

**Predicts:** the empirically optimal host fraction on Arc sits **above** `1 − f*`, and the gap
**widens with prompt depth and cache pressure** (small pool, long context, high miss rate).

**Refuted if:** the optimum sits at or below the bandwidth prediction, or the gap is
depth-independent. Either would mean the depth margin has some other cause and P2 is a story rather
than a finding.

**Experiment:** sweep host fraction at 512 / 2048 / 8192 on a fixed pool and plot the optimum against
`1 − f*`. The harness exists. This converts a corroborated observation into a claim FreeToken does not
have. Note the two-sided value: **P1 holding and P2 holding are not in tension** — P1 says the
bandwidth term predicts the bulk, P2 says a depth-dependent correction sits on top.

---

### P3 — Which misses you admit matters; expert-recency beats routing-blind selection
**TRANSFERS.**

Recency-ordered admission costs nothing in bandwidth — the same *number* of experts moves — and
lowers steady-state miss rate by preferentially caching experts that recur. FreeToken shipped it as
the default and kept the routing-blind version alive behind `FREETOKEN_HYBRID_FETCH=lowest_id`, which
is how we know it was an A/B they resolved rather than a first design.

**Predicts:** recency-ordered admission beats arbitrary/first-k on Arc, for free. **Refuted if**
indistinguishable — which would itself matter, because it would say routing is near-memoryless at step
granularity and would undercut every recency mechanism at once, LRU eviction included.

---

### P4 — Hit rate is bounded by routing skew, not policy sophistication
**TRANSFERS. Corroborates our Belady result, and strongly.**

Three pieces of evidence from a project with ~11.4k stars and a published paper:

1. It exposes `--moe-cache-policy` through a CLI flag, a config field, the TUI and the API — and the
   implementation is `policy_ids = {"lru": 0}`. **One policy, ever.**
2. Its own routing analysis computes an `oracle_hit` ceiling from stationary top-C frequency mass,
   described as *"an upper bound on hit rate that depends purely on how skewed routing is, independent
   of any LRU/LFU dynamics."*
3. **LFU, ARC, S3-FIFO and Belady appear nowhere in the paper.** Their one policy comparison (Fig. 4b)
   is *cross-engine placement*, not a replacement-policy sweep.

MoEArc scored nine policies against Belady: the best non-regressing one recovers 7–25% of the gap,
while **+44% slots beats any of them by ~3×**.

**Predicts:** LFU, ARC, 2Q or a learned policy on Arc will produce gains inside the noise of a
few-percent slot-count change. **Refuted if** any policy moves hit rate more than a slot-count change
of equal engineering cost.

**Action:** stop spending on replacement policy. Spend on **pool size** (P7) and **non-admission**
(P2).

---

### P5 — Per-layer miss rate is U-shaped; the ends are cheapest to move off the GPU
**TRANSFERS — and the cheapest experiment in this document.**

Not a passing remark: it is the load-bearing justification for their automatic layer-selection rule
(`engine/engine.py::_auto_cpu_layers`), which moves `ceil(L · (1 − budget/bank_bytes))` layers to the
CPU, split head and tail:

> *"Locks just enough head+tail layers: per-layer decode miss rates are U-shaped, so the ends are the
> cheapest to move off the slot cache."*

The claim is about MoE routing across depth, not about CUDA. It is stated without quantification
anywhere in paper or code.

**Predicts:** on the gpt-oss traces MoEArc already has, per-MoE-block miss rate is highest at the
first and last blocks. Therefore **per-layer host routing beats uniform host routing at the same
average fraction**, and the ends should be routed host-side first.

**Refuted if:** the curve is flat or inverted on gpt-oss.

**Experiment:** re-reduce existing trace data by block index. No GPU time, no new measurement. It
either hands us a free structural win or kills a plausible idea before it costs anything.

---

### P6 — Their model assumes copy and compute are independent resources. Arc weakens that; an Arc iGPU breaks it
**NVIDIA-SPECIFIC → two sharp Arc questions.**

Every one of the paper's six systems is a **discrete GPU**. Integrated GPUs, unified memory, Apple
Silicon, ROCm and Intel are never mentioned. The `B_P`-vs-`B_H` distinction presupposes a link
separating two distinct memory pools.

**(a) On the discrete B580, the copy may be unnecessary.** MoEArc's own Arc bring-up of FreeToken
(study branch, 2026-09-04) established that on XPU `torch.empty(pin_memory=True)` *is*
`sycl::malloc_host`, UVA identity holds, and a GPU kernel reads that host memory at **13.47 GB/s, the
same as the copy engine, with random 8 KiB scatter at 13.37 GB/s**.

That is structurally different from CUDA. FreeToken's slot cache exists to avoid a host read that is
slow and latency-bound under scatter. If Arc's in-place host read matches the copy *and* tolerates
scatter, fetching stops being a bandwidth win — **it becomes purely a bet on reuse.**

> **Predicts:** a direct GPU-reads-pinned-host expert GEMV on the B580 lands within ~1× of
> fetch-then-compute for a *cold* expert. **Refuted if** the in-place read degrades once a real GEMV's
> access pattern replaces the microbenchmark's. **Experiment:** time one expert's GEMV both ways,
> cold, at real routing scatter. If it holds, a whole class of transfer optimisation stops being worth
> building.

**(b) On an Arc iGPU there is no second resource.** It reports host RAM as device memory
(91,890,372,608 bytes). Eq. 2 gives `B_R = max(B_H − B_P, 0) → 0` and the model degenerates — which
the paper itself half-anticipates: *"As B_H approaches B_P, q\* approaches the total miss count m, and
the system degenerates into pure on-demand cache fill."*

> **Predicts:** on an Arc iGPU the two contended bandwidths are close, `f*` degenerates, and the split
> delivers no speedup over either pure path. **Action:** detect unified memory and **disable the
> split** rather than computing a meaningless `q*`. FreeToken already has the right instinct — it
> refuses hybrid without a confident profile — it simply never met hardware where the question was
> degenerate.

⚠️ **Two places their code hands a non-CUDA device a fabricated hardware parameter**, both worth
avoiding by construction: `attention/fi.py` falls back to a hardcoded `sm_count = 128`, and
`attention/triton/attention.py` treats `smem_optin == 0` as "unknown device" (that second one handles
it *gracefully*, by selecting conservative tiles, and is the pattern worth copying).

---

### P7 — `memory_ratio = 0.9` is unmeasured by them, unmeasured by us, and worth more than any cache policy
**MUST RE-MEASURE.**

`engine/config.py` carries `memory_ratio: float = 0.9`; the `(1 − memory_ratio)` remainder is
CUDA-graph and activation headroom. It appears in the paper **not at all** — §3.3 describes the
elastic split qualitatively and gives no budget formula. MoEArc ported the surrounding arithmetic into
`crates/moearc-engine/src/memory.rs`, whose header flags its 12% `Headroom` as *"a placeholder chosen
to be conservative — not a measured value"*. **Neither project has measured it**; ours is at least
honest about that in the code.

It is not small. `memory.rs` records that 12% on an 11.33 GiB card withholds 1.36 GiB — roughly **700
expert slots**. Against MoEArc's measured coverage curve (13% of the bank resident → 53–79% of touches
covered; 35% → 85–98%), 700 slots is a materially different hit rate.

**Predicts:** the correct Arc headroom is not 10%, and measuring it moves hit rate further than any
replacement policy will (P4). **Experiment:** allocate, capture, run a long-context decode, read peak
free VRAM. One afternoon, and it gates every other memory decision.

📌 One design detail worth taking: their budget arithmetic is **isolated, device-free and asserts**
— *"Reject here so `--moe-cache-auto` fails in arithmetic instead of OOMing in a later CUDA
allocation."* MoEArc's `memory.rs` already goes further with checked `u64`.

---

### P8 — 🔴 Their paper describes why KV-first is right at depth. Their engine ships MoE-first anyway
**MUST RE-MEASURE. The most interesting disagreement here.**

`engine/cache_budget.py::plan_cache_budget` reserves a KV floor, lets experts greedily consume
everything up to full residency, and gives KV the remainder. The config states it as policy:

> `kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)`

And yet the paper (§3.3) states the mechanism that argues against its own default:

> *"agentic sessions accumulate context across turns, so KV cache demand grows while the expert
> working set stays roughly fixed, and a split chosen on the first turn is wrong many turns later."*

They resolve it with an **escape hatch, not a policy**: `POST /v1/cache/rebuild` reconfigures the split
at a scheduler safe point without reloading weights. Elastic memory in FreeToken is operator-driven,
not autonomous. They also cite prior work that went the other way — WiSP splits by marginal latency
value, FluxMoE prioritises KV capacity — so KV-first exists in their own related work and they did not
take it.

**FreeToken publishes no residency-tradeoff measurement.** MoEArc's does: on a real routing trace,
**not one expert slot in 6,144 is touched every token** (hottest p = 0.979; 78% below p = 0.10) while
**every KV byte is**, so at 64K, KV-first costs 0.08 GB/token against experts-first's 5.83. KV
residency is all-or-nothing; expert residency degrades gracefully.

**My reading, offered as hypothesis not fact:** these are two ends of one curve. Their 8192-token KV
floor is a short-context regime where KV is cheap. We measured at 64K where KV dominates. There is
almost certainly a context length `L*` computable from `kv_bytes_per_token`, `per_expert_bytes` and the
coverage curve.

**Predicts:** `L*` exists and is finite. **Refuted if** one bias wins from 512 to 64K — a cleaner
result than the hypothesis, and equally worth having.

🔴 **Flagging an inherited default our own data contradicts:** `memory.rs` defines
`enum Bias { Experts, Context }` with **`Experts` as `#[default]`**. That is FreeToken's choice carried
into MoEArc, against a MoEArc measurement arguing the other way at long context. It should either be
justified by finding `L*` and switching on it, or the default should move. Right now it is neither.

---

### P9 — FreeToken has nothing to say about decode at depth. It does not publish the curve at all
**NVIDIA-SPECIFIC in the weak sense: not applicable rather than wrong.** And this is a finding about
*their* evidence, not just their design.

MoEArc's depth problem: throughput falls with prompt depth, and the cause is measured. FreeToken
offers no prior on this, and it is important to say so plainly rather than manufacture one.

- **Prefill vs prompt length is published** (Fig. 4a, RTX 5090, Qwen3.6 BF16) and it *rises* — 6.7k
  tok/s at 16k tokens, with each 8,192-token chunk taking 1.19–1.22 s, exactly the time to stream the
  64.4 GB expert pool once at 52.7 GB/s. **Prefill is transfer-bound at the PCIe ceiling and compute is
  fully hidden.**
- **Decode vs context depth is not published.** No throughput-vs-position curve, no tok/s at 4k/16k/64k.
  The only depth-adjacent decode claim is aggregate stability — *"stays within 12% of the single-turn
  W1 value"* across agent workloads reaching 56–65k tokens. That is a stability claim over workloads,
  not a curve over positions.
- **No decomposition of decode cost into attention vs expert movement anywhere.**

Their answer to depth is architectural and orthogonal: support model families whose attention is
already sub-quadratic (dedicated pools for sliding-window, linear/GDN recurrent state, DeepSeek-V4's
compressed + indexer tiers, block-sparse), plus radix prefix caching and "semantic anchor" checkpoints
so agentic context edits do not force a re-prefill. Every one of those reduces *recompute* or *KV
footprint*. **None makes a full-attention block cheaper.**

**Predicts:** nothing about our attention kernel. Recorded so nobody hunts for a trick that is not
there — and so that the absence is not mistaken for their having solved it.

---

### P9b — But they ship one lever we parse and do not use, and it is the highest-value item I found
**TRANSFERS. Actionable today. It came not from their policy but from noticing what they have that we
already know about and ignore.**

FreeToken ships a dedicated sliding-window KV pool (`kvcache/hybrid_swa_pool.py`,
`kvcache/swa_radix_cache.py`) and a `--swa-full-tokens-ratio` knob. MoEArc **parses** the window and
its pattern — `crates/moearc-engine/src/moe.rs` documents `n_swa`, `swa_pattern` and `is_swa_block`
carefully and correctly, including that gpt-oss declares `attention.sliding_window = 128` on
**alternating** blocks — and then:

- `is_swa_block` and `n_swa` have **zero call sites anywhere else in `moearc-engine`**; and
- `crates/moearc-model/src/lib.rs::kv_bytes_per_token` multiplies **every** KV block by full width with
  no window term.

On gpt-oss at 8192 tokens with half the blocks windowed to 128, MoEArc is reserving roughly **twice the
KV it needs**, and — if the attention kernel likewise attends full history on windowed blocks — doing
roughly **twice the attention work at depth**.

**Predicts:** honouring the window on the windowed half cuts KV bytes at 8192 by ~45–49% and cuts
attention time at depth by close to half. That attacks the depth curve directly, *and* frees bytes that
become expert slots — feeding P7 and relieving the cache pressure P2 is about. One change, three
benefits.

**Refuted if:** the attention kernel already skips masked keys efficiently (in which case only the KV
over-reservation is real — still worth fixing), or if the declared window does not survive into the
GGUF we load.

**Verify first, and cheaply — this is a read, not a benchmark:** confirm from the loaded GGUF that
`attention.sliding_window = 128` and `sliding_window_pattern = 2` are present, and read the attention
kernel to see whether masked keys are skipped or merely masked.

---

### P10 — Some of their constants are driver-bug workarounds; and some do not transfer across NVIDIA's own generations
**NVIDIA-SPECIFIC. Listed to be ignored — with one structural lesson that may not be.**

- `_SMALL_BANK_FEAT_BYTES = 256 * 1024` records that `cudaMemcpyBatchAsync` *"silently degrades to a
  SYNCHRONOUS copy when a batch mixes large entries with sub-~256KB entries… (H100 + CUDA 13.0,
  empirically bisected)"*, costing **−22% end-to-end on gpt-oss at 2048 tokens**.
- `MARLIN_MAX_CACHE_SIZE = 992` is an artifact of vLLM's `moe_align_block_size`.

Both are already marked **DELETE** in [`calibration.md`](calibration.md).

🔴 **The strongest single argument for MoEArc's measure-don't-inherit stance is in their own tree.**
`kernel/triton/mxfp8_linear.py` documents a tile choice worth ~30% at M=1 on RTX PRO 6000 / 5090 —
and then: *"but on H100 the dot kernel is ~26% faster even at M=1. If sm_90 becomes a serving target,
re-benchmark and dispatch by arch (or drop this kernel)."* **A constant that inverts between two NVIDIA
generations will not survive a vendor change.** Their other crossovers each name the sweeping hardware
in-comment: `_b12x_min_intermediate = 1024` (RTX 5090), `_GROUPED_MIN_ROUTES = 768` (H100),
`_DECODE_MARLIN_DEEPK_THRESHOLD = 2048`, `blocks_per_bank = 64` (*"~22 GB/s per 1024-thread block on
H100"*). Those comments are a re-sweep list, not a table of values.

**One structural lesson may transfer — MUST RE-MEASURE:** *heterogeneous entry sizes in a batched
transfer are a hazard.* Does a SYCL queue submitting mixed-size copies show an analogous cliff on Arc?
A short microbenchmark settles it, and a null result removes something from the design space.

---

### P11 — Do not pin a CPU worker pool with a spin-wait on a laptop
**TRANSFERS, and it is a trap we would otherwise walk into.** MoEArc targets laptops.

`moe/cpu_executor.py`, on their first implementation:

> *"(The first cut used a spin-wait kernel; that pinned reported utilization at 99% and laptop CPU/GPU
> dynamic power schedulers responded by clamping the CPU frequency — **a net decode regression on
> power-coupled edge devices**.)"*

They replaced it with stream memory-op waits. The related default is also load-bearing and reasoned
twice in-source: **one thread per *physical* core, pinned**, because *"MoE decode is
memory-bandwidth-bound, so SMT siblings only contend for the same core's load ports without adding
bandwidth"* — and *"the spin-barrier degrades badly when oversubscribed."*

**Predicts:** a hybrid CPU pool that busy-waits will measure *faster* on a desktop and *slower* on a
laptop, and the regression will look like thermal noise rather than a bug. It also predicts host
threads should default to physical cores, not logical — directly relevant given
`MOEARC_HOST_THREADS` and the 20-core reference box. **Refuted if** an A/B of spin vs. event-wait shows
no laptop penalty.

---

## Part 3 — negative results and abandoned paths

The highest-value question in the brief, and the answer is mixed.

🔴 **Their development history is not minable.** The public repo opens with a squashed
`feat: initial open-source release`. Everything tried and rejected before that is not in the git log.
That is the biggest gap in this document and it is not recoverable from the source.

**The paper has exactly one true ablation.** Prefill double-buffering disabled (§5.3) costs **19% of
throughput at 4k tokens, 25% at 8k, 26% at 16k** — *"the penalty growing with prompt length as the
share of hidden computation rises."*

**Their strongest negative result is an argument, not an experiment: they deliberately did not build
expert prediction.** §6, having surveyed Mixtral-offloading, MoE-Infinity, ProMoE, ExpertFlow and
FineMoE:

> *"These systems differ in how well they predict misses, but not in how they serve them: **every miss
> is ultimately a PCIe transfer, so decode latency remains bounded by the link no matter how accurate
> prediction becomes, while host compute capacity sits idle.**"*

> *"FreeToken keeps the routed computation exact and the model unmodified; it changes how residual
> misses are served rather than how well they are predicted."*

⚠️ This is a claim, not a measurement — they never built a predictor and showed it underperforming.
But it **converges from a third direction** with two MoEArc findings: that MoE's serialisation penalty
is ~2% not 2× (which lowered prefetch's value), and that hit rate is pool-bound not policy-bound (P4).
Three independent arguments against speculative expert prefetch is enough to stop considering it.

**Rejected by design, stated in the paper:** fidelity-relaxing methods — HOBBIT's reduced-precision
replicas, SiDA/SMoE's expert skipping, Pre-gated MoE's router retraining — *"trading accuracy for
bandwidth"*. FreeToken keeps output exact. Also: host-side scheduling heuristics are rejected on
CUDA-graph grounds, and warmup is eliminated by construction — *"the first request is served with a
cold cache."*

**Recorded in code — each env-var A/B is a place they were not fully confident:**

| Switch | Rejected alternative | What it tells us |
|---|---|---|
| `FREETOKEN_HYBRID_FETCH=lowest_id` | routing-blind fetch selection | recency admission was a measured improvement (P3) |
| `FREETOKEN_FUSED_COPY=0` | per-bank copies, one launch each | fusing won, but the old path stayed *"for A/B profiling"* |
| `FREETOKEN_BANK_CUDA_ALLOC` | `cudaHostAlloc` vs mmap + register-after-fill | off by default: *"registered mmaps already read at the PCIe roofline"* — 📌 **inverted on Arc**, where our bring-up found allocate-then-fill is *required* |

**Optimisations that regressed, verbatim from their comments** — a shortlist worth reading before
attempting any of these on Arc: flattening a padded tile is *"~40% slower"* at decode sizes; loading
weights earlier helps decode but *"regresses larger batches"*; larger tiles / more warps *"regress
(register pressure)"* on H100/B200; a narrow-N deep-K split is 13% faster at K>2048 but *"short-K
shapes regress under it"*; GLM-5 FP8 requantisation gains 36→45.5 tok/s but *"the win needs CUDA
graphs — launch-bound eager decode gets slower"*.

**Two absences that are themselves results:** one replacement policy behind a policy flag (P4), and no
multi-GPU anywhere in the paper — every configuration is a single GPU.

---

## Part 4 — their measured results, and how far to trust them

**Baselines:** llama.cpp, Ollama, KTransformers, MoE-Infinity. Not vLLM.
**Models:** DeepSeek-V4-Flash (284B/13B active, MXFP4), Qwen3.6-35B-A3B (BF16, *"exact precision parity
across engines"*), GLM-5.2 (753B/40B, NVFP4). **Six discrete-GPU systems**, `B_P`/`B_H` measured on
deployed tensor shapes, not spec sheets — 3090/4090 at `B_P ≈ 25`, 5090 at 49–53, and a 4060 laptop on
PCIe 4.0 **×8** at `B_P = 11.8` against `B_H = 47.5`.

**Headline decode (RTX 5090):** Qwen3.6 BF16 **77–83 tok/s, 1.8–2.3×** the strongest baseline per
workload; DSV4-Flash MXFP4 22–25 tok/s, 1.5–1.9×. Across hardware on a coding-agent workload:
**1.3× on 3090/4090, 1.9–2.1× on 5090, 1.8× on the 4060 laptop** (39.3 tok/s on 8 GB over ×8 — *"92%
of the RTX 4090 rate"*). GLM-5.2 on one PRO 6000: **14.9 vs llama.cpp's 7.3**.
**Hit rate (Fig. 4b), replayed on identical traces at equal capacity:** at the 5090's real serving
capacity, FreeToken's global LRU misses **16%** on Qwen3.6 (37% of the pool resident) and **39%** on
DSV4-Flash (11% resident), against llama.cpp's routing-blind static split at **62%** and **89%**.

### How rigorous is it? Better than ours was, and still not reproducible

**What they did right, and we should copy:** thread counts are reported (*"Qwen3.6 at 6 CPU threads,
DSV4-Flash at 8"*), and the rented dual-socket servers are **capped at 6 CPU threads and pinned to the
GPU's NUMA node** because their CPUs *"far exceed any edge host"* — then validated against two real
edge machines at natural thread counts. Weight formats are aligned bit-exactly across engines. Baseline
failure modes are documented rather than hidden. **The cap binds every engine including their own**,
which is the opposite of the failure that sank our numbers.

🔴 **What is missing, and it is substantial:**

- **No command lines anywhere.** No `-ngl`, no `-ncmoe`/`--n-cpu-moe`, no `-ot`, no `-fa`, no
  batch/ubatch — and no `ft serve` flags for their own runs either.
- **No engine versions or commit hashes** for any baseline.
- 🔴 **No offload-split disclosure.** They characterise llama.cpp's policy as a *"routing-blind static
  split"* and never say **what split they gave it** — the single most performance-determining llama.cpp
  knob for an oversized MoE, and the exact subject of their thesis.
- **No batch size reported** for any measurement.
- **No variance, error bars, or repeat counts.** Fig. 4b bands are workload min–max, not run-to-run
  noise.
- **No independent reproduction.** All numbers are author-produced.

**Verdict: their thread-count handling is defensible and explicitly reported; their
offload-configuration handling is not reported at all.** A reader cannot reconstruct the llama.cpp
baseline from the paper, so whether it was well-tuned is unfalsifiable from the text. We withdrew every
one of our own llama.cpp comparisons on 2026-09-06 for a neighbouring failure. **We are not entitled to
assume another project avoided it, and the correct posture is: cite FreeToken's mechanism, not
FreeToken's multipliers.** No performance ratio of theirs appears in this document as evidence for
anything.

---

## Part 5 — does their model predict our awkward regime?

The brief's sharpest question: **llama.cpp's `-ncmoe` static CPU dumping currently beats dynamic
caching on our 12 GB card, because a 20-core CPU outruns a small cache. Does their model predict that?**

**Yes — and it says `-ncmoe` is a crude approximation of the correct policy in our bandwidth regime.**

Read `-ncmoe N` in their terms: it is `q = 0` for the dumped blocks, chosen statically at load time.
Never fetch, always compute host-side. Now apply Eq. 4 with our numbers: `B_P ≈ 13.5`, `B_H` plausibly
~50, so `q*/m ≈ 0.27` — **73% of every step's misses should go to the CPU.** Their own 4060 laptop case
is the same shape: `B_P/B_H = 11.8/47.5 = 0.248`, so **75% of misses never cross the bus**, and that is
the configuration where they report 1.8× and *"92% of the RTX 4090 rate"* on an 8 GB card.

So the regime is not anomalous — it is the regime their design is *aimed* at. What `-ncmoe` gets right
is that most misses belong on the CPU. What it gets wrong is that the split is static, per-tensor and
routing-blind, so it also refuses the ~27% of misses that *should* be fetched and cached, and it cannot
respond to `m` changing per step. Fig. 4b measures that cost as hit rate: 62–89% miss for the static
split against 16–39% for global LRU.

**Predicts:** on our box, a correctly-parameterised hybrid should beat `-ncmoe` — but a *dynamic cache
that fetches every miss* should lose to it, which is what we observed. **The failure was never
"dynamic caching doesn't work here"; it was fetching too high a fraction of misses on a machine whose
`B_P/B_H` is ~0.27.** That is directly testable, and it reframes a result we had read as a defeat.

⚠️ **One contrary datapoint, and it is worth keeping.** Community issue #122 (ROCm port, RX 9060 XT
16 GB) reports FreeToken **losing** to llama.cpp on a model that fits entirely in VRAM — gpt-oss-20b
MXFP4 at ~61 tok/s against llama.cpp's 83.5. That is outside FreeToken's design target (their `fused`
backend is *never* auto-selected, precisely because guessing that experts fit is a load-time OOM rather
than a slow run). But it says plainly: **when the model fits, this whole architecture is overhead.**
Third-party, uncontrolled, and the llama.cpp side again shows no `-t`/`-ncmoe`/`-ot` — low confidence,
recorded for honesty.

---

## Part 6 — what I could not determine

Stated plainly, because this project has been burned by exactly the failure of not stating it.

1. **Their benchmark methodology is not published in enough detail to judge.** Specifically, the
   llama.cpp offload split is never disclosed. This is a valid finding, and it is the reason no
   FreeToken speed multiplier is used as evidence anywhere above.
2. **Whether their hybrid backend is on by default in practice.** It engages only behind a manual
   per-GPU benchmark and a unanimous verdict. How often that fires on real user hardware is not
   determinable.
3. **The `2.0` recommendation threshold's provenance.** *"CPU MoE bandwidth > 2× PCIe gather
   bandwidth"* is a measured NVIDIA crossover with no derivation in code or paper. Already marked
   **Measure** in `calibration.md`; it should be *derived* from the two bandwidths, not inherited.
4. **Whether `L*` (P8) exists.** A hypothesis from two datapoints at opposite ends of one curve.
   Neither project has measured it.
5. **P1's numerical match is a bracket, not a result.** `B_H` on CHARA is unmeasured; `B_P ≈ 13.5` is a
   host-read rather than a copy-engine gather; and that MoEArc's `frac` and FreeToken's `1 − f*` denote
   the same quantity is believed, not verified. **Verify all three before this is quoted anywhere.**
6. **P9b's magnitude.** The ~2× argument is arithmetic from the declared window and pattern, not a
   measurement. It predicts a direction and an order of magnitude; the verification step is named
   because the arithmetic can be right while the kernel is already doing the right thing.

---

## Part 7 — the honest assessment

**The policy transfers almost completely. The performance does not.**

What transfers is the **shape**: a three-way per-layer decision; a crossover *solved* from two
contention-measured bandwidths rather than swept; admission ordered by expert recency; layer selection
informed by a U-shaped miss profile; a refusal to engage the mechanism without a confident measurement;
and a set of documented dead ends (prefetch, spin-waiting, fidelity relaxation) we now do not have to
walk down. None of that contains an NVIDIA term. It is roughly the whole of what we wanted from them,
and it turns an open search space into a directed one.

What does **not** transfer is everything downstream of a kernel. Their numbers rest on NVIDIA-tuned
kernels, Marlin/flashinfer GEMM paths, CUDA graphs as an architectural assumption, and a slot cache
designed around CUDA's cost of a scattered host read. MoEArc's kernels run at **25–29% of the card's
peak bandwidth against llama.cpp's 63%**, and no scheduling prior closes a 2.3× kernel gap. That is the
strongest argument I found for the pivot to vendoring llama.cpp+SYCL as the engine and keeping
scheduling and tuning as ours: **the priors in this document are exactly the layer we would be keeping,
and they are the layer that survives a hardware change.** Their own tree proves the point — a tile
choice that inverts between an H100 and a 5090 was never going to survive the trip to Xe.

Three places we are ahead rather than behind, and all three should be treated as ours:

- **P2** — their `q*` has no cache-pollution term; our depth-margin measurement says one belongs there.
- **P8** — their paper states why KV demand grows with agentic depth and their engine ships MoE-first
  anyway, resolved by a manual REST call. Our measurement quantifies what they described qualitatively,
  and nobody has found `L*`.
- **Depth** — they publish no decode-versus-depth curve and no attention/staging decomposition. We have
  both. That is a genuinely open area, not a solved one we are late to.

And the highest-value single finding was not theirs at all: **P9b**, which came from noticing that
FreeToken ships a sliding-window KV pool while MoEArc parses `n_swa`, documents it carefully, and then
uses it nowhere. If the arithmetic holds it is the largest win available, and it is a
correctness-adjacent fix rather than an optimisation.

**If I could run only three things:** P5 (free — re-reduce existing traces by block index), P1 (one
`bench bw`-equivalent measurement, which either validates the whole closed form or exposes a wrong
bandwidth), and P9b's verification read. None of the three needs a benchmark campaign, and between them
they either confirm or kill most of this document.

---

## Attribution

FreeToken is Apache-2.0 (`FlashML-org/FreeToken`); its paper is
[arXiv:2608.16157](https://arxiv.org/abs/2608.16157). Quotations above are from its source comments and
paper text, reproduced as evidence for findings. MoEArc is **inspired by** FreeToken's architecture; no
FreeToken code is copied into this project. See [`../NOTICE`](../NOTICE) and
[`archaeology.md`](archaeology.md).
