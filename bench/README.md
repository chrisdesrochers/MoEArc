# Benchmark protocol

Every number in the README comes from `moearc bench` and nothing else.

| Metric | How |
|---|---|
| Decode tok/s | 512 tokens at KV depths 0, 4k, 16k |
| Prefill tok/s | 4k and 16k prompts |
| TTFT | 4k prompt |
| VRAM / RAM | peak resident |
| Power | wall watts (smart plug) + Level Zero GPU power → tok/s/W |
| Correctness | wikitext-2 perplexity within 0.5% of llama.cpp, same quant |

Always reported against llama.cpp SYCL **and** Vulkan on the identical box and
commit hash.

## Rules

- A result without the box, the commit hash and the quant is not a result.
- `baselines/` holds llama.cpp numbers. `results/` holds ours. Both are committed.
- No "should be faster" without a number. Measure, change, measure.
- If a change makes nothing faster, that is a finding — commit it anyway.

## M0 baseline

The number to beat. Recorded **before** any new code is written, and **not** on
the integrated GPU — the iGPU is a compile target, not a baseline.

```bash
llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -ngl 999 -ncmoe 8 \
  -fa 1 -ctk q8_0 -ctv q8_0 -p 4096 -n 512 -d 0,4096,16384 -ub 1024
```

If M2 cannot beat this, we stop and say so.
