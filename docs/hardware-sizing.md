# Choosing a card for MoEArc

**The deciding number is not how much VRAM you have. It is what fraction of expert *touches*
the card covers.** Those differ a lot, and the difference is the whole reason this project
exists.

## Why capacity is the wrong metric

MoE routing is heavily skewed: a few experts are picked constantly and most are picked rarely.
So a cache holding a modest fraction of the expert bank intercepts a much larger fraction of the
accesses. Measured on `bench/traces/qwen3-30b-prose.decode.ndjson` — real `ffn_moe_topk` values
from a running decode, not a simulation:

| bank resident | touch coverage | miss rate | amplification |
| ---: | ---: | ---: | ---: |
| 5% | 34.0% | 66.0% | 6.80× |
| 10% | 50.9% | 49.1% | 5.09× |
| **13%** | **59.1%** | **40.9%** | 4.54× |
| 20% | 74.2% | 25.8% | 3.71× |
| 25% | 82.1% | 17.9% | 3.28× |
| **35%** | **92.3%** | **7.7%** | 2.64× |
| 40% | 95.4% | 4.6% | 2.39× |
| 50% | 98.9% | 1.1% | 1.98× |
| 60% | 100.0% | 0.0% | 1.67× |

🔴 **The curve saturates near 60% residency.** Past that, more VRAM buys nothing on this axis.

## What that means per card

For **gpt-oss-120B** (59 GiB of weights; experts are ~96% of the file), assuming ~3.9 GiB of the
card goes to dense weights and KV:

| card | expert pool | bank resident | miss rate |
| --- | ---: | ---: | ---: |
| Arc B580 (11.3 GB) | 7.4 GiB | 13.1% | **40.6%** |
| Arc Pro B60 (24 GB) | 20.1 GiB | 35.5% | **7.3%** |
| Arc Pro B70 (32 GB) | 28.1 GiB | 49.6% | **1.2%** |

📌 **Buy the step from ~12 GB to ~24 GB. The step from 24 to 32 is diminishing returns.**
Cutting misses 5.6× removes most of the bottleneck; the further 6× buys much less, because the
curve has already flattened.

**Why this is the bottleneck and not something else:** `bench/baselines/gpt-oss-120b.md` §6.4
measures expert staging at **80%** of the throughput lost to prompt depth, against **6.5%** for
attention. Staged bytes are a direct function of miss rate, so miss rate is the lever.

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

Look that fraction up in the first table to get your expected miss rate. Aim for **35% or
better**; below ~15% expect throughput to fall sharply as prompt depth grows.

`moearc info <model>` reports the plan for your actual hardware, including resident slots.

## ⚠️ Limits of this analysis

- The coverage curve is measured on a **Qwen3-30B** trace (48 blocks × 128 experts, 8 active) and
  applied to **gpt-oss** geometry (36 × 128, **4 active**). Different models have different
  routing skew.
- It is **conservative**: it ranks slots by frequency, an offline oracle-ish proxy, yet the engine
  measured a **68.9%** decode hit rate at depth 2048 where this curve predicts ~59%. Reality was
  better than the projection at the one point where both exist.
- Coverage is not throughput. It predicts **staged bytes**, which is the dominant term at depth —
  not tok/s, for which this project has no validated model and therefore publishes none.
