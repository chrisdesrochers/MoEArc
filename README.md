# MoEArc

**Run 30B–120B mixture-of-experts models on an ordinary Intel Arc gaming PC.**
One command to install, one command to run, OpenAI- and Anthropic-compatible
out of the box.

> **Status: pre-M0. The engine is unwritten.** What exists is a measured baseline:
> the reference Arc B580 is installed, and tuned llama.cpp SYCL does **46.9 tok/s**
> decode on Qwen3.6-35B-A3B Q4_K_M. That is the number MoEArc has to beat, and until
> `bench/results/` contains something faster, there is no claim here.

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

Every number in this README comes from `moearc bench` and nothing else, always
reported against llama.cpp SYCL and Vulkan on the identical box and commit hash.
Protocol: [`bench/README.md`](bench/README.md). Results: `bench/results/`.

**The baseline exists; MoEArc's numbers do not.** Measured on the reference box,
llama.cpp SYCL `e107984bc`, Qwen3.6-35B-A3B UD-Q4_K_M (20.6 GiB), Arc B580 12 GiB:

| backend | best `-ncmoe` | prefill tok/s | decode tok/s |
|---|---|---|---|
| **SYCL** | 22 | 403.9 | **46.9** |
| Vulkan | 22 | 174.9 | 9.5 |

🔴 `-ncmoe` was **swept, not guessed** — below 22 the model will not fit 12 GiB, and above it
throughput falls monotonically to ~37. A baseline taken at an arbitrary split would make any
later MoEArc "win" meaningless, so the sweep is committed alongside the number.

These are the tuned static-split numbers. **MoEArc's thesis is that a dynamic,
bandwidth-aware policy beats a hand-tuned static one.** That is what M2 must demonstrate.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
