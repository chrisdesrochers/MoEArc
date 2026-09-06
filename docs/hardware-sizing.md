# Choosing a card for MoEArc

**The deciding number is not how much VRAM you have. It is what fraction of expert *touches*
the card covers.** Those differ a lot, and the difference is the whole reason this project
exists.

## Why capacity is the wrong metric

MoE routing is heavily skewed: a few experts are picked constantly and most are picked rarely.
So a cache holding a modest fraction of the expert bank intercepts a much larger fraction of the
accesses.

Measured on the **gpt-oss-120B** traces in `bench/traces/` — real `ffn_moe_topk` values from
running decodes, at that model's actual geometry (36 blocks × 128 experts, 4 active):

| bank resident | slots | prose | code | reasoning | **worst-case miss** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 230 | 50.8% | 31.2% | 32.4% | 68.8% |
| 10% | 460 | 71.0% | 46.3% | 49.1% | 53.7% |
| **13%** | 599 | 78.8% | 53.3% | 56.9% | **46.7%** |
| 20% | 921 | 89.6% | 66.2% | 70.8% | 33.8% |
| 25% | 1152 | 93.7% | 73.4% | 78.0% | 26.6% |
| **35%** | 1612 | 97.8% | 84.2% | 87.9% | **15.8%** |
| 40% | 1843 | 98.9% | 88.2% | 91.3% | 11.8% |
| **50%** | 2304 | 99.8% | 93.9% | 95.9% | **6.1%** |
| 60% | 2764 | 100.0% | 97.4% | 98.4% | 2.6% |

🔴 **What you run matters as much as which card you buy.** At 13% residency, prose covers
**78.8%** of touches and code covers **53.3%** — a 25-point spread on the same hardware and the
same model. Code and reasoning workloads revisit experts less, so they punish a small pool much
harder. **Size for the workload you actually have.**

## What that means per card

For **gpt-oss-120B** (59 GiB of weights; experts are ~96% of the file), assuming ~3.9 GiB of the
card goes to dense weights and KV:

| card | expert pool | bank resident | coverage (prose → code) | worst-case miss |
| --- | ---: | ---: | ---: | ---: |
| Arc B580 (11.3 GB) | 7.4 GiB | 13.1% | 79.0% → 53.6% | **46.4%** |
| Arc Pro B60 (24 GB) | 20.1 GiB | 35.5% | 98.0% → 84.6% | **15.4%** |
| Arc Pro B70 (32 GB) | 28.1 GiB | 49.6% | 99.8% → 93.7% | **6.3%** |

📌 **Buy the step from ~12 GB to ~24 GB first.** It removes **31 points** of miss traffic; the
further step to 32 GB removes **9 more**. Both are worth having — the first is about 3.4× the
second, measured in the bytes that actually cross the bus.

**Why this is the bottleneck and not something else:** `bench/baselines/gpt-oss-120b.md` §6.4
measures expert staging at **80%** of the throughput lost to prompt depth, against **6.5%** for
attention. Staged bytes are a direct function of miss rate, so miss rate is the lever.

**And it cannot be fixed in software.** `bench/policy-sweep.md` scores nine cache policies against
the Belady optimum on these same traces: LRU sits 12.8–20.4 points below Belady, and the best
non-regressing alternative recovers only **7–25%** of that. In slot-equivalents, the best policy
is worth **+4% to +12%** more pool while simply *having* +44% more slots beats it by ~3×. **These
are capacity misses, not bad decisions.**

## RAM sizes the model. VRAM sizes the experience.

They are not interchangeable, and adding the wrong one makes things worse:

- **RAM sets what you can hold.** Weights are memory-mapped; past what fits in page cache they
  page from disk. On a 96 GB machine the practical ceiling is roughly 80–90 GB of weights.
- **VRAM sets your miss rate**, per the table above.

🔴 **Adding RAM alone can make throughput worse.** A 12 GB card holding a 104 GB model sits at
**~7% residency** — below the 13% that already degrades badly. You would be able to load a model
you cannot serve well.

## Applying this to a model you are considering

```
bank_resident  ≈ (VRAM_GB − dense_and_kv_GB) / (model_GB × 0.96)
```

Look that fraction up in the first table to get your expected miss rate — using the **code**
column unless you only ever run prose. Aim for **35% or better**; below ~15% expect throughput to
fall sharply as prompt depth grows.

`moearc info <model>` reports the plan for your actual hardware, including resident slots.

## ⚠️ Limits of this analysis

- **Coverage is not throughput.** It predicts **staged bytes**, which §6.4 shows is the dominant
  term at depth — not tok/s, for which this project has no validated model and publishes none.
- Slots are ranked by access frequency, which is an offline oracle, so real cache behaviour will
  differ. At the one point where both exist, the engine measured a **68.9%** decode hit rate at
  depth 2048 against this curve's 53.3–78.8% range for that residency — inside the range.
- Routing skew is **model-specific**. These numbers are gpt-oss-120B. Qwen3-30B, with 8 of 128
  active instead of 4, reuses considerably more per step (35.5–47.1% step-to-step against
  22.7–34.3%) and is a materially easier cache. **Do not transfer this table to another model** —
  regenerate it from that model's own trace.

🔴 **Correction, 2026-09-06.** The first version of this page derived its curve from a
**Qwen3-30B** trace and applied it to gpt-oss, and claimed the result was conservative. It was
**optimistic**: it gave 40.6% / 7.3% / 1.2% miss for the three cards above, against the measured
46.4% / 15.4% / 6.3%. The advice to prefer the 12→24 GB step survived; the magnitudes did not.
That is what the caveat about transferring between models is doing here, and it is why the table
above is now measured on the model it describes.
