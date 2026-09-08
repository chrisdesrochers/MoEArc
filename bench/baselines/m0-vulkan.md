> # 🔴 SUPERSEDED — do not quote the comparisons in this file
>
> **Every MoEArc-versus-llama.cpp figure below is withdrawn.** They were produced with
> `llama-bench` at its **4-thread default on a 20-core box** while MoEArc's host pool used 19
> threads — so they measure an unfair configuration, not the two engines. No corrected
> head-to-head is published: two re-measurement attempts disagree by 2× *and about which engine
> wins*. See [`bench/PROTOCOL.md`](../PROTOCOL.md) §1.
>
> **MoEArc no longer has an engine to compare.** It now uses llama.cpp+SYCL and ships measured
> *tuning* — see [`docs/strategy.md`](../../docs/strategy.md).
>
> This file is kept because a retraction is a claim like any other, and because its
> non-comparative measurements (kernel timings, cache counters, traces) remain valid.

ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Graphics (ARL) (Intel open-source Mesa driver) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = Intel(R) Arc(tm) B580 Graphics (BMG G21) (Intel open-source Mesa driver) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |  n_cpu_moe | n_ubatch | type_k | type_v |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | -------: | -----: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | Vulkan     | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1      |          pp4096 |        252.00 ± 9.59 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | Vulkan     | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1      |         pp16384 |        220.36 ± 5.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | Vulkan     | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1      |           tg512 |          9.17 ± 0.55 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | Vulkan     | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1      |  pp4096 @ d4096 |        236.93 ± 9.66 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | Vulkan     | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1      | pp16384 @ d4096 |        201.28 ± 6.53 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | Vulkan     | 999 |         22 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1      |   tg512 @ d4096 |          8.92 ± 0.92 |
