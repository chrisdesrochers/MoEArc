# What is LRU costing the expert cache?

An offline replacement-policy study over captured routing traces. No GPU, no throughput model —
hit rate and staged bytes only, every candidate scored against Belady's optimal.

**Question.** MoEArc's expert cache holds 600 slots while the router names 144 experts per decode
step. That is 4.17 steps of history. `bench/baselines/gpt-oss-120b.md` §6.5 shows the pool
degrading *within a single run at constant depth*, which is the signature of a working set losing
to a cache. Is that a policy problem or a pool-size problem?

**Answer, up front.** It is a **pool-size problem.** At 600 slots on real gpt-oss-120B decode
traces, LRU sits **12.8 to 20.4 points below Belady**. The best candidate that does not regress on
any content type recovers **7% to 25%** of that gap — worth **+4% to +12% more pool** — where
Belady is worth **+61% to +97%**.
No policy in the LRU / LFU / SLRU / 2Q / LRU-K / TinyLFU / pinned-hot family recovers materially
more, and several are worse than LRU on at least one content type. Replacing LRU is a real but
small win; the pool is simply too small, and that is where the work is.

---

## 🔴 Geometry — read this before quoting any number

Three model shapes appear below and **they are not interchangeable**. The engine's numbers are
gpt-oss-120B; the traces that already existed in `bench/traces/` were Qwen.

| Model | blocks | experts / block | active | activations / step | total experts | MiB / expert |
|---|---:|---:|---:|---:|---:|---:|
| **gpt-oss-120B** (MXFP4) — *the engine's model* | 36 | 128 | **4** | **144** | 4608 | **12.607** |
| gpt-oss-20B (MXFP4) | 24 | 32 | 4 | 96 | 768 | — |
| Qwen3.5-MoE 35B-A3B (Q4_K_M) | 40 | 256 | **8** | 320 | 10240 | 1.95 |
| Qwen3-30B-A3B (Q4_K_M) | 48 | 128 | **8** | 384 | 6144 | — |

The engine's operating point — **600 slots** — is **4.17 steps of history**, **13.0% of the
model's 4608 experts**, and **7.39 GiB** of VRAM at 12.607 MiB apiece.

**Six new traces were captured for this study** so the headline is first-party rather than
transposed: three gpt-oss-120B (512 decode steps each) and three gpt-oss-20B (1024 each), prose /
code / reasoning, same capture patch and conventions as the existing set. Provenance is in
`bench/traces/README.md`. **The gpt-oss-120B traces are the ones the recommendation rests on.**
The Qwen tables are kept because a policy that wins only on one architecture has not won —
but they are a different working-set shape and are labelled as such throughout.

Capacity is reported as **steps of history** (`slots / activations-per-step`) as well as in slots,
because that is the only axis on which models with different `n_expert_used` can be compared.

### What the traces look like

| trace | steps | working set | as % of model | top 10% of experts hold | step-to-step reuse |
|---|---:|---:|---:|---:|---:|
| gptoss120b-prose | 512 | 2442 / 4608 | 53.0% | 52.4% | 34.3% |
| gptoss120b-code | 512 | 3685 / 4608 | 80.0% | 40.9% | 26.0% |
| gptoss120b-reasoning | 512 | 3486 / 4608 | 75.7% | 41.7% | 22.7% |
| gptoss20b-prose | 1024 | 630 / 768 | 82.0% | 34.0% | 46.8% |
| gptoss20b-code | 1024 | 746 / 768 | 97.1% | 37.8% | 45.8% |
| gptoss20b-reasoning | 1024 | 710 / 768 | 92.4% | 36.8% | 43.4% |
| qwen35moe-prose | 1024 | 6725 / 10240 | 65.7% | 50.7% | 42.3% |
| qwen35moe-code | 1024 | 8713 / 10240 | 85.1% | 50.7% | 38.7% |
| qwen35moe-reasoning | 1024 | 8721 / 10240 | 85.2% | 44.5% | 35.5% |
| qwen3-30b-prose | 192 | 3702 / 6144 | 60.3% | 37.9% | 45.4% |
| qwen3-30b-fibonacci | 192 | 4419 / 6144 | 71.9% | 43.1% | 47.1% |

🔴 **gpt-oss-120B is the hardest of the three shapes, and by a wide margin.** Its pool is 13% of
the model where Qwen3.5's 3976 slots are 39%, and its step-to-step reuse is **22.7–34.3%** against
Qwen's 35.5–47.1%. Four active experts per block instead of eight means each step's demand overlaps
the previous step's less. **A residency conclusion drawn on Qwen would have been drawn in a much
friendlier regime than the one the engine runs in**, which is why the traces were captured.

---

## The operating point: gpt-oss-120B at 600 slots

Every candidate, scored against Belady. Δ is against LRU; "% of gap" is the fraction of the
LRU→Belady headroom the policy recovers.

| policy | prose | Δ / % of gap | code | Δ / % of gap | reasoning | Δ / % of gap | **worst Δ** |
|---|---:|---|---:|---|---:|---|---:|
| static split (widest that fits) | 16.7 | −56.1 | 13.9 | −42.2 | 13.9 | −40.7 | −56.1 |
| **LRU** (incumbent) | **72.8** | — | **56.1** | — | **54.6** | — | — |
| LFU | 78.4 | +5.6 / 44% | 55.9 | −0.2 / −1% | 57.9 | +3.3 / 16% | −0.2 |
| SLRU protected=50% | 74.4 | +1.6 / 13% | 57.2 | +1.1 / 6% | 58.1 | +3.5 / 17% | +1.1 |
| SLRU protected=70% | 75.5 | +2.7 / 21% | 57.2 | +1.1 / 6% | 58.4 | +3.8 / 19% | **+1.1** |
| SLRU protected=80% | 76.0 | +3.2 / 25% | 56.0 | −0.1 / −1% | 57.9 | +3.3 / 16% | −0.1 |
| 2Q kin=25% kout=200% | 73.0 | +0.2 / 2% | 55.7 | −0.4 / −2% | 55.9 | +1.3 / 6% | −0.4 |
| **LRU-2** | 76.0 | +3.2 / 25% | **57.0** | +0.9 / 5% | 58.3 | +3.7 / 18% | **+0.9** |
| pinned-hot pin=50% | 74.8 | +2.0 / 16% | 55.7 | −0.4 / −2% | 56.1 | +1.5 / 7% | −0.4 |
| TinyLFU, no window | 76.2 | +3.4 / 27% | 51.2 | −4.9 / −28% | 56.7 | +2.1 / 10% | −4.9 |
| W-TinyLFU window=20% | 77.5 | +4.7 / 37% | 56.1 | +0.0 / 0% | 58.9 | +4.3 / 21% | +0.0 |
| **W-TinyLFU window=40%** | 76.0 | +3.2 / 25% | **57.4** | +1.3 / 7% | **58.8** | +4.2 / 21% | **+1.3** |
| phase-LRU | 72.6 | −0.2 / −2% | 55.9 | −0.2 / −1% | 54.5 | −0.1 / −1% | −0.2 |
| **Belady (optimal)** | **85.6** | +12.8 / 100% | **73.7** | +17.6 / 100% | **75.0** | +20.4 / 100% | — |

**Only three candidates beat LRU on all three content types**: W-TinyLFU with a 40% window
(+1.3 worst case), SLRU with 70% protected (+1.1), and LRU-2 (+0.9). They are within a point of
each other and none exceeds 25% of the Belady headroom on any trace.

🔴 **`code` is the trace that kills candidates.** It has the largest working set (80% of the model),
the lowest routing skew (top 10% hold 40.9%), and the largest Belady gap (17.6 points) — and it is
where LFU, SLRU-80, 2Q, pinned-hot and windowless TinyLFU all fall *below* LRU. Averaging the
three content types would have declared LFU the winner (+2.9 mean) while it is a regression on
code. **Reporting only the mean would have shipped a regression.**

### The same numbers as staged bytes

144 experts × 12.607 MiB = **1815 MiB demanded per decode step**. Staged MiB/step = miss rate × 1815.

| policy | prose | code | reasoning |
|---|---:|---:|---:|
| static split | 1512 | 1563 | 1563 |
| **LRU** | **494** | **797** | **824** |
| LRU-2 | 436 | 781 | 757 |
| SLRU-70 | 445 | 777 | 755 |
| W-TinyLFU w=40% | 436 | 773 | 748 |
| **Belady** | **261** | **477** | **454** |

The best policy on each trace saves **58 MiB/step on prose**, **24 on code**, **76 on reasoning**.
Belady saves **233**, **320** and **370**.

*No conversion to tok/s is offered. There is no validated model from staged bytes to throughput in
this project, and inventing one is the specific failure this codebase has already been bitten by.*

### And the same numbers as pool size — the framing that settles the question

For each policy: how many slots plain LRU would need to reach the same hit rate (interpolated on
the measured LRU capacity curve).

| | prose | code | reasoning |
|---|---:|---:|---:|
| best online policy | +73…+134 slots (+12…+22%) | **+22 slots (+4%)** | +65…+76 slots (+11…+13%) |
| **Belady** | **+367 slots (+61%)** | **+581 slots (+97%)** | **+509 slots (+85%)** |

**Perfect knowledge is worth roughly a doubling of the pool. Every online policy tested is worth
under a quarter of one, and on the hardest trace under a twentieth.** That is the whole result.

---

## Capacity sweeps

Hit rate %, gpt-oss-120B. `static` is `Trace::widest_static_split` — the most generous static
split that fits the same budget, charged no compulsory misses (see the caveat in
`bench/traces/README.md`; it is modelled in the incumbent's favour).

### gptoss120b-prose (working set 2442)

| slots | steps of history | static | lru | lfu | slru80 | 2q25 | lru-2 | pin50 | tinylfu | w-tinylfu20 | **belady** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 180 | 1.25 | 2.8 | 38.7 | 42.9 | 40.9 | 39.9 | 40.7 | 44.4 | 43.8 | 43.3 | **49.2** |
| 216 | 1.50 | 2.8 | 42.5 | 48.5 | 46.3 | 44.7 | 45.8 | 46.9 | 48.2 | 48.0 | **57.3** |
| 288 | 2.00 | 5.6 | 50.1 | 57.3 | 54.6 | 52.4 | 53.9 | 53.2 | 55.1 | 56.3 | **67.4** |
| 432 | 3.00 | 11.1 | 61.8 | 69.4 | 66.3 | 63.4 | 66.2 | 64.8 | 66.6 | 68.4 | **78.6** |
| **600** | **4.17** | 16.7 | **72.8** | 78.4 | 76.0 | 73.0 | 76.0 | 74.8 | 76.2 | 77.5 | **85.6** |
| 864 | 6.00 | 27.8 | 82.9 | 87.1 | 85.6 | 82.6 | 85.7 | 84.1 | 85.5 | 86.2 | **91.6** |
| 1152 | 8.00 | 38.9 | 89.8 | 91.9 | 90.9 | 88.8 | 91.0 | 90.3 | 90.7 | 91.2 | **94.6** |
| 1790 | 12.43 | 66.7 | 95.5 | 95.9 | 95.7 | 94.0 | 95.9 | 95.6 | 95.7 | 95.8 | **96.6** |
| 2880 | 20.00 | 100.0 | 96.7 | 96.7 | 96.7 | 96.7 | 96.7 | 96.7 | 96.7 | 96.7 | **96.7** |

### gptoss120b-code (working set 3685)

| slots | steps of history | static | lru | lfu | slru80 | 2q25 | lru-2 | pin50 | tinylfu | w-tinylfu20 | **belady** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 180 | 1.25 | 2.8 | 29.4 | 31.1 | 31.1 | 30.6 | 30.9 | 32.9 | 28.8 | 29.0 | **39.8** |
| 216 | 1.50 | 2.8 | 33.0 | 34.3 | 34.8 | 34.1 | 34.6 | 34.5 | 31.3 | 32.5 | **47.1** |
| 288 | 2.00 | 5.6 | 39.2 | 39.7 | 40.5 | 39.8 | 40.6 | 38.3 | 36.3 | 38.4 | **55.8** |
| 432 | 3.00 | 8.3 | 47.5 | 48.3 | 47.8 | 48.2 | 49.4 | 47.7 | 44.2 | 47.9 | **66.3** |
| **600** | **4.17** | 13.9 | **56.1** | 55.9 | 56.0 | 55.7 | 57.0 | 55.7 | 51.2 | 56.1 | **73.7** |
| 864 | 6.00 | 19.4 | 65.2 | 65.1 | 66.9 | 65.0 | 66.5 | 65.6 | 60.4 | 65.6 | **81.4** |
| 1152 | 8.00 | 27.8 | 73.0 | 72.9 | 74.6 | 73.1 | 74.3 | 73.8 | 68.8 | 73.5 | **86.2** |
| 1790 | 12.43 | 47.2 | 85.3 | 84.6 | 85.4 | 83.6 | 84.9 | 85.8 | 81.8 | 85.5 | **91.8** |
| 2880 | 20.00 | 77.8 | 93.4 | 93.6 | 93.6 | 91.5 | 93.5 | 93.5 | 93.0 | 93.6 | **95.0** |

### gptoss120b-reasoning (working set 3486)

| slots | steps of history | static | lru | lfu | slru80 | 2q25 | lru-2 | pin50 | tinylfu | w-tinylfu20 | **belady** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 180 | 1.25 | 2.8 | 26.4 | 29.1 | 27.8 | 27.5 | 27.8 | 31.0 | 29.3 | 28.5 | **37.3** |
| 216 | 1.50 | 2.8 | 30.1 | 33.0 | 31.7 | 31.3 | 31.5 | 33.1 | 32.4 | 32.2 | **45.1** |
| 288 | 2.00 | 5.6 | 36.5 | 39.2 | 38.0 | 37.5 | 38.1 | 36.8 | 38.1 | 38.6 | **54.8** |
| 432 | 3.00 | 8.3 | 45.6 | 48.9 | 47.9 | 47.1 | 49.5 | 46.9 | 47.9 | 49.7 | **66.4** |
| **600** | **4.17** | 13.9 | **54.6** | 57.9 | 57.9 | 55.9 | 58.3 | 56.1 | 56.7 | 58.9 | **75.0** |
| 864 | 6.00 | 22.2 | 67.7 | 68.6 | 68.7 | 66.3 | 69.1 | 67.7 | 67.0 | 69.2 | **82.8** |
| 1152 | 8.00 | 30.6 | 76.1 | 76.8 | 77.6 | 75.6 | 77.4 | 77.4 | 75.0 | 77.9 | **87.9** |
| 1790 | 12.43 | 50.0 | 87.6 | 87.8 | 88.8 | 86.1 | 88.3 | 88.1 | 85.6 | 88.3 | **93.2** |
| 2880 | 20.00 | 83.3 | 94.7 | 94.7 | 94.8 | 93.2 | 94.7 | 94.7 | 94.4 | 94.6 | **95.3** |

**The shape of these three tables is the finding.** Between 600 and 864 slots — a 44% larger pool —
LRU alone gains **+10.1 (prose), +9.1 (code), +13.1 (reasoning)** points. No policy change at 600
slots gains more than **+4.3** on any of them.
Capacity dominates policy by roughly an order of magnitude at the pool size the engine has.

### Other geometries, for the transfer check

Same policies, same method. Full tables were produced for `gptoss20b-{prose,code,reasoning}`,
`qwen35moe-{prose,code,reasoning}` and `qwen3-30b-{prose,fibonacci}` at nine capacities each —
95 (trace, capacity) points in total. Summarised:

| policy | mean Δ vs LRU, all 11 traces × all capacities | worst single point | regressions |
|---|---:|---:|---:|
| SLRU-80 | **+1.25 pts** | **−0.1** | **1 / 95** |
| LRU-2 | +1.17 | −0.4 | 7 / 95 |
| LFU | +1.13 | −2.6 | 3 / 95 |
| W-TinyLFU w=20% | +1.00 | −3.3 | 8 / 95 |
| pinned-hot 50% | +0.46 | −6.5 | 23 / 95 |
| 2Q kin=25% | +0.01 | −1.9 | 38 / 95 |
| TinyLFU, no window | −1.03 | −6.5 | 3 / 95 |

Restricted to the band that matters (gpt-oss-120B, 3–8 steps of history):

| policy | mean Δ vs LRU | worst | mean % of Belady gap |
|---|---:|---:|---:|
| W-TinyLFU w=20% | +2.42 | +0.0 | 18.1% |
| LFU | +2.34 | −0.2 | 18.9% |
| **LRU-2** | +2.27 | **+0.9** | 16.6% |
| SLRU-80 | +1.92 | −0.1 | 14.7% |
| pinned-hot 50% | +0.98 | −0.4 | 7.5% |
| 2Q kin=25% | +0.13 | −1.4 | −1.0% |
| TinyLFU, no window | −0.24 | −4.9 | 0.4% |

The *ordering* is broadly stable across geometries — SLRU and LRU-2 at the top, 2Q and windowless
TinyLFU at the bottom, on Qwen and gpt-oss alike — so the earlier Qwen-only evidence was not
misleading about which policies are worth considering. ⚠️ Individual policies still flip sign
between content types *within* a geometry (LFU is +3.4 on qwen35moe-prose at 1333 slots and −0.2 on
gptoss120b-code at 600), which is why the per-trace tables above are the ones to read and the mean
column is not. What the Qwen-only evidence *was* misleading about is **magnitude**: at Qwen3.5's real
operating point (3976 slots = 12.4 steps of history) LRU is already at 89–95% and the whole
question is nearly moot, while at gpt-oss-120B's 600 slots LRU is at 54.6–72.8% and it is not.

---

## Per-policy verdicts

**LFU** — the strongest single number in the study (+5.6 on gpt-oss prose) and **disqualified
anyway**: −0.2 on gpt-oss code, −2.6 on qwen35moe-reasoning at 2560 slots. Pure frequency cannot
forget, so an expert that was hot in the first 200 tokens holds a slot for the next 800. It wins
where the working set is small and concentrated and loses where content shifts.

**SLRU** — the most reliable of the family: one regression in 95 points, and that one is −0.1. Best
at `protected = 70%` (+1.1 worst-case on gpt-oss-120B); the 80% setting used in the main sweep is
slightly over-tuned to prose and gives back the code trace. A probationary segment is exactly the
right *mechanism* for the §6.5 failure mode — it just is not worth many points here.

**2Q** — essentially LRU, with 38 regressions in 95 points. Its ghost list is what should have
saved it, and does not: with the pool at 4.17 steps of history, an expert evicted from the
admission queue is re-referenced so rarely within the ghost's lifetime that the promotion path
almost never fires. Sized generously (`kout = 200%` of capacity, keys only, no VRAM cost) and it
still does not help.

**LRU-K (K=2)** — **the pragmatic winner.** Never worse than LRU by more than 0.4 points anywhere,
positive on all three gpt-oss-120B traces at 600 slots, and its entire implementation is *one extra
`u64` per expert* — the penultimate reference stamp — with the identical eviction scan. Every other
candidate needs segments, ghost lists, or a frequency sketch to reach the same place.

**Pinned hot set** — the idea the trace skew suggested, and it does not survive contact. 23
regressions in 95 points, worst −6.5. The pinned set is learned from a warm-up prefix (pinning a
top-K taken from the whole trace would be Belady in disguise), and a prefix-learned hot set goes
stale: 34.0–52.4% of activations land in the top 10% of experts, but *which* experts those are
drifts with content. Pinning freezes a decision that wants to keep moving.

**TinyLFU without a window** — 🔴 **the clearest negative result, and a predicted one.** −4.9 points
on gpt-oss code, worse than LRU on average across all 95 points. A brand-new expert arrives with a
frequency of 1 and loses every admission contest against an incumbent, so the cache calcifies around
whatever it saw first. This is precisely the failure W-TinyLFU's window exists to fix, and the
window fixes it: window=0% scores 51.2 on code, window=40% scores 57.4. **An admission filter with
no grace period is worse than no admission filter.**

**W-TinyLFU** — the best mean gain in the 3–8-step band, and the window size is doing all the work:
on code, 51.2 → 52.4 → 54.4 → 56.1 → 57.4 as the window goes 0 → 5 → 10 → 20 → 40%. But note where
that ends: **a 40% window means 40% of the cache is plain LRU**, and the policy is converging
towards LRU as it improves. That is a strong hint that the frequency signal is not where the
remaining headroom lives.

**phase-LRU** — 🔴 **my own hypothesis, measured and falsified.** A decode step walks blocks in
order, so a global reference clock makes every block-35 expert look more recent than every block-0
expert *from the same step*, while the block-0 expert is needed sooner on the next pass. The bias is
real. Correcting it — recency quantised to the step, ties broken by distance round the block cycle —
is worth **nothing**: within 0.2 points of LRU at every capacity on all three gpt-oss-120B traces,
and slightly *worse* at the tightest ones. The reason is visible in the geometry: at 4.17 steps of
history almost every eviction candidate was last used in a different step, so the tie-break the
policy exists for almost never fires. It is kept in `Policy`, with its numbers in its doc comment,
so the idea is not re-derived and re-tried.

---

## So: policy problem, or pool problem?

**Pool problem.** Three independent readings of the same data say so:

1. **The gap is large but not reachable.** LRU is 12.8–20.4 points below Belady at 600 slots and no
   online policy closes more than a quarter of it, on any trace, at any setting tried.
2. **Capacity moves it an order of magnitude harder.** +44% slots (600→864) is worth +9.1 to +13.1
   points to LRU; the best policy change at 600 is worth +0.9 to +4.3.
3. **Expressed in the same unit, the best policy buys +4% pool on the hardest trace where Belady
   buys +97%.**

And the reason is structural, not incidental: the pool is **13% of the model's experts** and
step-to-step reuse is **22.7–34.3%**. A cache holding an eighth of the working set, refilled a
quarter over every step, is in the regime where replacement policy stops mattering — the misses are
capacity misses, not bad-decision misses.

**This does not weaken the project's central claim.** Dynamic residency still beats the static split
by an enormous margin at every capacity measured (56.1% vs 13.9% at 600 slots on gpt-oss code —
and the static baseline is modelled *generously*, charged no compulsory misses). The finding is
narrower: having chosen dynamic residency, the choice *among* dynamic policies is nearly free.

---

## Recommendation

**Ship LRU-2 (`Policy::LruK { k: 2 }`), or ship nothing, and put the effort into slots.**

LRU-2 is the only candidate that is positive on all three gpt-oss-120B traces at the operating
point and never loses more than 0.4 points anywhere in 95 measured points, and it costs one extra
`u64` per expert with no change to the eviction scan or the cache's structure. It buys 16–67
MiB/step. If the engine's LRU is a linked list, this is an afternoon; if adopting it costs more
than that, **the data supports doing nothing**, because the honest size of the prize is a few
points of hit rate.

SLRU with `protected = 70%` is the runner-up and is marginally more robust across all geometries
(1 regression in 95 vs 7); prefer it if segments are already in the design. W-TinyLFU w=40% is the
largest gain but its best configuration is 40% plain LRU by volume, which is a poor trade for a
frequency sketch plus aging plus three segments.

**Where the effort should go instead**, in rough order of expected value:

1. **More slots per GiB.** 12.607 MiB/expert is the denominator of everything above. Anything that
   shrinks the resident copy — a lower-precision VRAM format than the MXFP4 on disk, or caching only
   the hot matrices of an expert — converts directly into slots, and slots are worth 5–10× what
   policy is worth.
2. **Measure whether staging is actually bandwidth-bound.** §6.5's degradation is attributed 80% to
   staging. If some of that is per-transfer overhead rather than bytes, batching miss fetches would
   move the number without needing either more slots or a better policy — and this study cannot see
   it, because hit rate is blind to how the misses are transferred.
3. **Only then, policy.** LRU-2 as above.

---

## Caveats

- **Six new traces, one seed, one sampler, 512 (120B) or 1024 (20B) decode steps.** Not a claim
  about MoE routing in general.
- **gpt-oss-120B traces are 512 steps, not 1024**, to bound the capture's cost on a shared machine.
  Shorter traces weight compulsory misses more heavily, which is *unfavourable* to every dynamic
  policy and favourable to none, so the comparison is not skewed by it — but the absolute hit rates
  would be a little higher on a longer run.
- **The prose prompt had to be modified for gpt-oss-120B.** With the unmodified prompt the model
  emitted `... (the rest of the essay... )` and hit EOG after 10 and 21 tokens on two separate
  attempts, and a first capture at 512 steps produced degenerate `... ... ...` output. That
  degenerate trace was **discarded, not used.** The committed trace prefills the assistant turn with
  an opening clause and generates real continuous prose; the exact prompt is in the trace header.
- **W-TinyLFU is charged its scratch slot.** A rejected candidate is still fetched and used, which
  physically needs a buffer, so the policy is simulated with `capacity - 1` retained slots. This is
  conservative by one slot in 600.
- **`Policy::PinnedHot` learns its hot set from a trace prefix**, never from the whole trace. A
  top-K taken from the full trace would be Belady with extra steps.
- **The static-split baseline is modelled in the incumbent's favour** — no compulsory misses, and
  sized by experts actually touched rather than the full 128 per resident block. See
  `bench/traces/README.md`. Where it appears to beat a dynamic policy (only at capacities at or
  above the whole working set) that is the modelling artifact, not a result.
- **Hit rate is not throughput.** Nothing here is converted to tok/s and nothing here should be.

---

## Reproducing

```sh
export PATH=/zfs/swift/projects/rust/cargo/bin:$PATH

# the capacity sweep — every policy against Belady, every trace, nine capacities (~15 min)
cargo test --release -p moearc-engine policy_sweep -- --ignored --nocapture

# the dial sweep at one capacity (~1 min)
cargo test --release -p moearc-engine policy_tuning -- --ignored --nocapture

# narrow either one
MOEARC_SWEEP_TRACES=/abs/path/a.decode.ndjson,/abs/path/b.decode.ndjson \
MOEARC_SWEEP_MULTS=3,4.167,6 \
cargo test --release -p moearc-engine policy_sweep -- --ignored --nocapture
```

Both live in `crates/moearc-engine/src/residency.rs` and are `#[ignore]`d — they are minutes of CPU,
not seconds. The simulator has no runtime dependencies and its RNG is hand-rolled on purpose, so
every number above is reproducible byte-for-byte on any machine, forever.
