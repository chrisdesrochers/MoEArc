# Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  moearcd daemon (systemd service)                            │
│   • HTTP: OpenAI + Anthropic-compatible, streaming, tools    │
│   • model manager (pull / list / rm, auto-download)          │
│   • hardware auto-detect + auto-tune profile                 │
├──────────────────────────────────────────────────────────────┤
│  Scheduler                                                   │
│   • continuous batching                                      │
│   • bandwidth-adaptive split policy (q*)                     │
│   • semantic-anchor KV checkpoints                           │
├──────────────────────────┬───────────────────────────────────┤
│  GPU (SYCL / Xe)         │  CPU (AVX2 / AVX-VNNI / AVX-512)  │
│   • attention (FA, XMX)  │   • routed expert FFN (cold)      │
│   • KV cache (q8)        │   • expert weight store (pinned)  │
│   • shared experts       │   • prefill pre-staging           │
│   • LRU expert cache     │                                   │
│   • dense/embed/lm_head  │                                   │
└──────────────────────────┴───────────────────────────────────┘
          ▲   PCIe expert fetch (double-buffered, async)   ▲
```

## Memory layout

VRAM is a single SYCL USM device allocation, arena-carved. Attention weights,
shared experts and the dense/embedding/lm_head tensors are fixed; the KV cache
and the expert cache are elastic and the scheduler moves the boundary between
them at runtime without reloading weights. Expert weights sit in pinned host
memory so device copies are DMA.

## Bandwidth-adaptive execution (q*)

Per expert, per step, choose:

1. **GPU-resident** — already in the VRAM LRU, run on Xe.
2. **Fetch-then-GPU** — PCIe copy, run, insert into cache. Cost ≈ `bytes / pcie_bw`.
3. **CPU-execute** — run the FFN on CPU from host RAM, ship only the activation.
   Cost ≈ `flops / cpu_rate + bytes / ddr_bw`.

Calibrated by a short micro-benchmark on first launch, persisted, then refined
online with an EMA.

## Expert cache

Global LRU keyed by `(layer, expert_id)`. Router logits for layer `L+1` are known
before attention for `L+1` runs, so its top-k can be prefetched asynchronously.
Under continuous batching the hot set is pinned batch-aware.

## Prefill

Full-layer double-buffered streaming: the GPU processes layer `L` while DMA loads
layer `L+1`'s experts.

## Weight format

Flat, mmap-able, one contiguous block per `(layer, expert)`, converted from GGUF
or safetensors at `moearc pull` time.
