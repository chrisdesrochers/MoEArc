# Negative results

Things that were tried and did not pay. Each one is here so it is not rebuilt, and because a
refuted hypothesis is cheaper to read than to re-derive.

---

## Per-layer slot partitioning — real, and negligible

**Date:** 2026-09-07 · **Cost:** ~15 minutes, offline simulation over committed traces, no GPU.

### The prior

`docs/freetoken-priors.md` P5 records FreeToken's claim that per-layer miss rate is **U-shaped**
— *"the ends are the cheapest to move off the slot cache"* — unquantified anywhere in their work.

### What the traces actually show

Measured over the three committed gpt-oss-120B decode traces (36 blocks × 128 experts, 4 active),
scoring each block's routing concentration as `1 − H/H_max`:

| trace | first 7 blocks | middle | last 7 |
| --- | ---: | ---: | ---: |
| prose | 0.188 | 0.208 | **0.262** |
| code | 0.107 | 0.130 | **0.146** |
| reasoning | 0.129 | 0.139 | **0.147** |

🔴 **Not a U. A monotonic ramp** — early blocks route most uniformly, late blocks most
concentrated, consistently across all three prompt types. On the code trace **block 0 touches 125
of 128 experts with a top-1 probability of 0.037**; on prose, block 24 concentrates **76.5% of its
traffic into 8 experts**.

### The hypothesis, and its refutation

*Give more cache to blocks that cache well.* **Wrong, and wrong in direction** — LRU simulated at
600 / 1200 / 2300 slots:

| weighting | prose | code | reasoning |
| --- | ---: | ---: | ---: |
| toward **concentrated** blocks | −1.4 to −3.5 | −0.8 to −3.1 | +0.4 to −2.4 |
| toward **uniform** blocks | **+0.4 to +0.8** | **+0.2 to +0.4** | **+0.1 to +0.4** |

The inverse wins **9 of 9**, so the direction is real. The magnitude is not.

📌 **Why the intuition was backwards:** a concentrated block captures most of its traffic in a
handful of slots, so extra slots there buy nothing. A uniform block needs many more slots to
capture the same fraction. **But LRU within each block already exploits the skew** — the partition
has almost nothing left to correct.

### Verdict

**Not worth building.** +0.8 points at best, against **+9 to +13 points** from simply having 44%
more slots. This is the third axis tested and closed:

| lever | worth |
| --- | --- |
| cache policy (9 policies scored against Belady) | 7–25% of the gap to optimal |
| **per-layer slot partition** | **+0.1 to +0.8 points** |
| **pool size** | **+9 to +13 points for +44% slots** |

**Three independent lines of evidence now say the same thing: this is a capacity problem, not an
allocation problem.** Stop looking for cleverness in how the cache is divided.

### Reproduce

Offline over `bench/traces/gptoss120b-{prose,code,reasoning}.decode.ndjson`; per-block routing
entropy, then per-block LRU at a partitioned budget. No GPU, no engine, seconds to run.
