# MoEArc

**Run 30B–120B mixture-of-experts models on an ordinary Intel Arc gaming PC.**
One command to install, one command to run, OpenAI- and Anthropic-compatible
out of the box.

> **Status: it generates text, and it is slow.**
>
> ✅ MoEArc runs a full forward pass on an Arc B580 and its greedy output matches
> llama.cpp **token id for token id** — 40/40 on one prompt, 23/23 on another including
> the end-of-generation token. That is a correctness result, independently re-verified.
>
> 🔴 **6.7 tok/s, unoptimised.** llama.cpp does roughly 46 on the same card. The gap is
> understood, not mysterious: no batched prefill, the router's choice is read back to the
> host 16 times per token, and the whole model is held resident.
>
> 🔴 **That last point matters most: because the model is fully resident, the expert cache
> is not exercised at all — so this project's central claim is still untested in the
> engine.** Everything measured about dynamic residency so far comes from simulation over
> recorded routing traces. Until a run streams experts under a real VRAM budget and beats a
> tuned static split on the same box, **there is no performance claim here.**

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

| | result | how |
|---|---|---|
| **Correctness** | greedy output identical to llama.cpp, **40/40** and **23/23** token ids | re-verified independently against a patched `llama-eval-callback` |
| **MoEArc decode** | **6.7–6.9 tok/s**, unoptimised, model fully resident | `examples/olmoe_generate` |
| Numerical agreement | 1−cos **5.68e-3** vs llama.cpp CPU | llama.cpp's *own* Vulkan backend differs from its CPU by **6.81e-3** |
| Expert miss path | 32.9 / 55.0 / 82.5 tok/s ceilings at 40 / 65.9 / 80.1% hit | `tools/stream_bench.cpp` |
| Residency, **simulated** | LRU **88.9–95.2%** vs static split 42.5–55.0% | recorded routing traces, `bench/traces/` |

📌 Bit-exactness against llama.cpp is **unavailable by construction**: `ggml-cpu` quantises the
f32 activation to 8 bits before every K-quant matmul, while MoEArc keeps f32. So the spread was
measured rather than a tolerance chosen — and llama.cpp's two backends disagree with each other
more than MoEArc disagrees with either.

### What is not measured

**MoEArc's thesis is that a dynamic, bandwidth-aware policy beats a hand-tuned static one.**
The simulated hit rates above say it should. **No engine run has demonstrated it**, because
nothing yet forces experts to stream. That is the next milestone and the only one that matters.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
