# MoEArc

**Run 30B–120B mixture-of-experts models on an ordinary Intel Arc gaming PC.**
One command to install, one command to run, OpenAI- and Anthropic-compatible
out of the box.

> **Status: pre-M0.** Nothing works yet. The engine is unwritten and the
> reference GPU is not yet installed. Follow `bench/results/` — if there is no
> number there, there is no claim here.

## Why

MoE models are the reason a consumer box can punch above its VRAM: only a
fraction of the parameters are active per token, so attention and the KV cache
can live in VRAM while routed experts stream from system RAM.

That approach is proven — on NVIDIA. On Intel Arc it isn't, despite Arc now
shipping the cheapest VRAM per dollar on the market. Intel's own `ipex-llm` got
there first and was archived on 2026-01-28 with no community fork. Ollama on Arc
falls back to Vulkan; llama.cpp SYCL works but leaves a lot on the floor for MoE,
and getting it running is a multi-hour ordeal.

MoEArc is an Arc-native, SYCL-first MoE server with a real expert cache, wrapped
in an install experience that does not require you to know what oneAPI is.

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

There are no numbers yet. That is not modesty, it is the actual state.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
