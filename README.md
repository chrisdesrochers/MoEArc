# MoEArc

**Run 30B–120B mixture-of-experts models on an ordinary Intel Arc gaming PC.**
One command to install, one command to run, OpenAI- and Anthropic-compatible
out of the box.

> **Status: it beats llama.cpp on a 59 GiB model, on a 12 GB card.**
>
> ✅ **gpt-oss-120B — 59.0 GiB of weights — runs on an 11.33 GiB Arc B580 at 17.2 tok/s**,
> with **13% of the expert bank resident** and greedy output matching llama.cpp for all
> 64 tokens. Independently re-verified.
>
> ✅ **llama.cpp does 15.47 tok/s on the same model and card — and cannot run it at all
> below `--n-cpu-moe 31`.** Below that it crashes with `OUT_OF_DEVICE_MEMORY`. Only 5 of
> its 36 blocks fit in VRAM.
>
> ✅ **On Qwen3-30B, dynamic residency beats a static split 45.08 vs 13.44 tok/s at matched
> capacity**, with 24× less data crossing the bus and identical output.
>
> 🔴 **Not yet:** no batched prefill, so there is no counterpart to llama.cpp's prefill
> throughput. Matvecs run at 25–29% of the card's peak bandwidth against llama.cpp's 63%.
> No adaptive policy — the residency fractions above were found by sweeping, not chosen by
> the engine. Sliding-window models are refused above 128 tokens of context rather than
> silently approximated.

```
╭ What will fit ──────────────────────────────────────────────────────────────────╮
│  ✓ gpt-oss-120b              mxfp4  59.0 GiB   612 / 4,608 experts · 2,048 ctx  │
│  ✓ qwen3.6-35b-a3b-ud        q4_K   20.6 GiB  3,976 / 10,240 experts            │
│  ✓ qwen3-30b-a3b             q4_K   17.3 GiB  3,108 / 6,144 experts             │
│  ✓ olmoe-1b-7b-0924-instruct q4_K    3.9 GiB  1,024 / 1,024 experts             │
╰──────────────────────────────────────────────── moearc info <model> ────────────╯
```

*`moearc` on an Arc B580 — real devices, real models, real plans. Every number above is
read from this machine.*

## Why

MoE models are the reason a consumer box can punch above its VRAM: only a
fraction of the parameters are active per token, so attention and the KV cache
can live in VRAM while routed experts stream from system RAM.

That approach is proven — on NVIDIA. [FreeToken](https://github.com/FlashML-org/FreeToken)
does it well: bandwidth-adaptive CPU–GPU co-execution, an LRU expert cache, elastic VRAM.
It supports **RTX 30/40/50 and nothing else.**

**Arc owners have no equivalent, and Arc now ships the cheapest VRAM per dollar on the
market.** Intel's own `ipex-llm` got there first and was archived on 2026-01-28 with no
community fork. Ollama on Arc falls back to Vulkan — which we measured at **4.8× slower
than SYCL on the same card**. llama.cpp SYCL is genuinely good, but its CPU/GPU split is
*static*: you pick `--n-cpu-moe` by hand, and picking it wrong costs you 20% or fails to
load at all.

**MoEArc is FreeToken for Intel GPUs** — the same co-execution ideas, SYCL-first, with an
install experience that does not require you to know what oneAPI is.

**Inspired by, not derived from.** FreeToken proved the shape of the answer on NVIDIA and we
are grateful for it. MoEArc is an independent implementation in Rust: no FreeToken code is
vendored, linked, or ported here, and where their design carries NVIDIA-specific constants we
deliberately do not inherit them. We intend to benchmark honestly against them — see
`bench/README.md` for what that comparison can and cannot show.

This is an open-source project because the gap is a community problem, not a private one:
anyone who bought an Arc card to run local models is currently choosing between a slow
Vulkan path and hand-tuning flags. Contributions and hardware reports are welcome.

## Design in one paragraph

VRAM holds attention weights, the KV cache, shared experts and an LRU cache of
routed experts. System RAM holds the full expert store in pinned memory. Per
expert, per step, the scheduler picks the cheapest of three options: run it from
the VRAM cache, fetch it over PCIe and run it on the GPU, or run it on the CPU
and ship only the activation. The crossover is calibrated on first launch and
refined online, so the same binary behaves sensibly on a B580 over PCIe 4.0 x8
and a B70 over 5.0 x16 — with no flags.

See [`docs/architecture.md`](docs/architecture.md).

## Hardware

| Tier | Hardware |
|---|---|
| Reference | Core Ultra 7 265K, 96 GB DDR5, Arc B580 12 GB |
| Target at launch | Arc A-series, B570/B580, Arc Pro B50/B60/B65/B70 |
| Works but slow | Core Ultra iGPUs (Lunar Lake / Arrow Lake) |

PCIe bandwidth is the bottleneck for expert fetch, so cache hit rate is the whole
game. That is the thesis this project is testing.

## Benchmarks

Numbers here are reported against llama.cpp on the identical box and commit hash.
Protocol: [`bench/README.md`](bench/README.md). Results: `bench/results/`.

⚠️ `moearc bench` does not exist yet; today's figures come from the named tools and examples in
this repo, each cited in the table below.

### 🔴 The baseline in an earlier version of this file was contaminated

That version quoted **46.9 tok/s** as llama.cpp's tuned SYCL decode. The run it came from is
marked `CONTAMINATED-DISCARDED` — GPU contention — and its files are deliberately not
committed. **It should not be quoted, including by us, and it is retracted here.** A clean
re-run at the swept `-ncmoe 22` is outstanding.

The one comparable figure we currently hold is **46.48 tok/s** from `llama-bench` on the
Vulkan backend, single run, `-r 1`, device not pinned. That is indicative, not a baseline.

### What is measured

All figures below are **Qwen3-30B-A3B Q4_K_M (17.3 GiB) on an Arc B580 (11.33 GiB)** unless
stated. Every row was re-verified independently of the run that produced it, and every
configuration generates **identical token ids**.

| | result | how |
|---|---|---|
| **Correctness** | greedy output identical to llama.cpp, **64/64** token ids | patched `llama-eval-callback`, both its CPU and SYCL backends |
| **MoEArc decode** | **45.08 tok/s** at 2952 resident slots with `frac:0.75` host routing | `examples/hybrid_sweep` |
| **vs the static split** | **45.08 vs 13.44 tok/s** at matched capacity, **24× less staged** | same sweep, same slots, same pool |
| **vs llama.cpp** | 45.08 against its **50.13** — which needs `-ncmoe 24` to run at all | `bench/baselines/qwen3-30b-a3b.md` |
| **The floor** | stream-only falls to **7.76 tok/s** at 4.3% residency; with host routing, **29.74** | `examples/hybrid_sweep` |
| Overlap | **94%** of host arithmetic runs while the device is busy | `MOEARC_PROFILE_EVENTS=1` |
| Numerical agreement | 1−cos **2.37e-3** vs llama.cpp CPU | its *own* Vulkan backend differs from its CPU by more |

### The result that was not predicted

Routing misses to the CPU raises the cache hit rate from **92.2% to 99.6%** and cuts expert
traffic **24×** — and *not* because of the CPU's arithmetic. **A miss routed host-side is never
admitted, so it never evicts a resident expert.** The host path works as a pressure-release
valve on the cache. Above ~1000 slots that is most of the gain, and it is a bigger effect than
the overlap it was built for.

📌 Bit-exactness against llama.cpp is **unavailable by construction**: `ggml-cpu` quantises the
f32 activation to 8 bits before every K-quant matmul, while MoEArc keeps f32. So the spread was
measured rather than a tolerance chosen — and llama.cpp's two backends disagree with each other
more than MoEArc disagrees with either.

### What is not measured

- **Prefill.** There is none. llama.cpp's 3218 tok/s has no counterpart here, and every number
  above is decode.
- **Kernel efficiency.** Every matvec sits at 25–29% of the card's 456 GB/s peak where
  llama.cpp reaches 63%. Two well-argued explanations for that gap were tested this session and
  **both died when measured in the engine** — see `docs/roadmap.md`.
- **An adaptive policy.** `frac:0.75` and `frac:1.0` were found by sweeping. The engine does not
  yet choose for itself, and the mechanism costs 1–4% when it routes almost nothing.
- **Anything above 17.3 GiB.** A 63.4 GB model is staged for the next test; nothing is claimed
  about it yet.

📌 **Three findings were retracted during this work** — a fabricated driver-version string, a
profile that mis-attributed device time under an async queue, and two microbenchmark results
that did not survive measurement in place. They are documented rather than deleted, because a
retraction is a claim like any other and this repo has been wrong confidently before.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
