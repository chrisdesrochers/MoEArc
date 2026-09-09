# moearc bench

**VERDICT: qualified — every number below carries a caveat** · 0.0.0 · generated 2026-09-09T08:10:00Z UTC

## Result

- **This machine, depth 60.** 110.98 ± 1.24 tok/s warm, 97.07 ± 13.65 cold, decode-only — an artefact of this box, not a portable number.

> **What reproduces and what does not.** Absolute throughput depends on CPU, memory bandwidth, PCIe generation, filesystem and whether the model fits in page cache; it is reported below as an artefact of *this* machine and should not be compared across machines. The **shape** results are a deterministic replay of committed routing traces — no clock is read and no device is touched — so they should come out identical on yours, to the last digit. Those are the result.

## Checks

Thresholds in force: refuse above load **2.50** (12% of 20 logical CPUs, floor 2.0), warn above **1.50**; a model above **80%** of the cache ceiling is flagged; a result needs **3** independent invocations and is refused a headline at a stddev of **20%** of its mean (warned at 10%).

| | rule | check |
|---|---|---|
| ⚠️ warn | §2 | built from a dirty working tree |
| ok | §3 | box quiet — load 1.28 |
| ok | §4 | the model fits in cache — 0.25x the ceiling |
| ok | §1 | moearc ran GPU-only — no host pool to pin |
| ok | §1 | llama-bench pinned to 19 threads |
| ok | §2 | Intel(R) Arc(TM) B580 Graphics on level_zero (xe / L0 build 37020) |
| ok | §2 | Level Zero build 37020 |
| ok | §2 | /zfs/swift/projects/llama.cpp/build/bin/llama-bench is a `SYCL` build — the `level_zero` counterpart |
| ok | §3 | card idle — 11.3 GiB free of 11.3 GiB |
| ⚠️ warn | §0 | no routing trace was replayed — 22 capture(s) were found and all were skipped |
| ok | §4 | model warmed into cache — 3.9 GiB in 1.5s |
| ⚠️ warn | §5 | moearc decode, cold pool, depth 60: stddev is 14% of the mean |
| ok | §5 | moearc decode, warm pool, depth 60: 110.98 ± 1.24 tok/s |
| ok | §5 | llama.cpp decode, -t 19, depth 60: 280.01 ± 0.08 tok/s |

**§2 — built from a dirty working tree**

moearc release / x86_64-unknown-linux-gnu / features gpu / commit 8f352e88c / working tree dirty. The commit named here does not describe what was compiled.

**§0 — no routing trace was replayed — 22 capture(s) were found and all were skipped**

§0 makes the shape results the headline and the absolutes an artefact of one machine, so a run that replayed no trace has not produced the thing it calls its result. Each skipped file is named with its reason in the Shape section — most often that it was captured from a different model, which §9 forbids transferring, or that it is a prefill capture. Pass --all-traces to replay them regardless, --trace <FILE> to name one, or drop --model to compare hit rates without the byte columns.

**§5 — moearc decode, cold pool, depth 60: stddev is 14% of the mean**

97.07 ± 13.65 over 3 independent invocations, stddev 14.1% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 81.31, 104.76, 105.15.

## Machine

| | |
|---|---|
| device | Intel(R) Arc(TM) B580 Graphics (level_zero) |
| driver | xe / L0 build 37020 |
| Level Zero build | 37020 |
| device memory | 11.3 GiB free of 11.3 GiB (measured free VRAM) |
| logical CPUs | 20 |
| load average (1 min) | 1.28 |
| memory | 70.1 GiB available of 91.5 GiB |
| ZFS ARC | 16.0 GiB in use, cap 16.0 GiB (`zfs_arc_max`) |
| model | /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf — 3.9 GiB on zfs |
| moearc host threads | 19 requested, 0 reported by the engine |
| build | release · x86_64-unknown-linux-gnu · features [gpu] · commit 8f352e88c (dirty tree) |

## Shape — the result that should reproduce on your machine

Policy under test: **lru**, against the widest static split that fits the *same* capacity. Ladder: 9 geometric points from the trace's peak single-step demand (the least capacity at which it is servable) to its working set (where every policy ties) — both ends are properties of the trace, so the ladder means the same thing on a model with 144 activations per step and one with 320. Knee: the rung farthest from the straight line joining the first and last rungs, in normalised log2(slots) x hit-rate space — the standard elbow of a saturating curve, and the capacity past which each further doubling buys visibly less.

**no trace was replayed**

Byte columns and the card marker are attached only to captures whose own header names `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a slot size belongs to one model (gpt-oss-120B's is 12.607 MiB, Qwen3-30B-A3B's is 2.92 MiB) and carrying one onto another model's trace is PROTOCOL §9's last failure.

Not replayed:

- `gptoss120b-code.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss120b-code.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss120b-prose.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss120b-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss120b-reasoning.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss120b-reasoning.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss20b-code.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss20b-code.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss20b-prose.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss20b-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss20b-reasoning.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss20b-reasoning.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen3-30b-fibonacci.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen3-30b-fibonacci.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen3-30b-prose.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen3-30b-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen35moe-code.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen35moe-code.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen35moe-prose.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen35moe-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen35moe-reasoning.decode.ndjson` — not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen35moe-reasoning.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)

🔴 A hit rate predicts **staged bytes** and nothing else. It does not predict tok/s: there is no validated model between them in this project, so none is published here and no throughput figure anywhere in this section is derived from one.

## Absolutes — this machine only

Decode-only throughput: the timer starts after prefill, so these are steady-state decode figures at the stated depth and not an average over the prompt. Cold and warm are separate questions and are never averaged together. Each figure is the mean ± sample stddev over **independent invocations of this binary**, not iterations inside one process.

| depth | tokens | residency | host | threads | cold tok/s | warm tok/s | warm/cold | cold hit | warm hit | disk read | ARC miss |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 64 | all | off | 0 | 97.07 ± 13.65 | 110.98 ± 1.24 | 1.14x | 94.3% | 100.0% | 212.0 MiB | 0.0% |

A large `disk read` column means the run faulted the model off the drive and measured the storage rather than the engine (PROTOCOL §4). `ARC miss` is the ZFS cache's own miss rate over the same window, machine-wide — the second, independent reading §4 asks for. Both coming back near zero is a result, not an absence: it says staging read from RAM, and that this run measured the engine.

Every individual invocation, so the spread is visible:

- depth 60: cold 81.31, 104.76, 105.15 · warm 112.41, 110.23, 110.29 · load before each child 1.34, 1.47, 1.52

## Incumbent — llama.cpp

`/zfs/swift/projects/llama.cpp/build/bin/llama-bench` · build `e107984bc` · backends `SYCL` · model `/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf`

🔴 The thread count in the `threads` column is read out of `llama-bench -o csv`'s own `n_threads` field, never inferred and never assumed to have followed `-t`. The binary's path was given explicitly; it was not found by glob.

| depth | -t asked | n_threads reported | decode tok/s |
|---:|---:|---:|---:|
| 60 | 19 | 19 | 280.01 ± 0.08 |

Quoted at its best configuration, **-t 19** — PROTOCOL §1 requires the baseline be swept and quoted at its best, not its first.

<details>
<summary>every llama-bench invocation, verbatim</summary>

```
$ /zfs/swift/projects/llama.cpp/build/bin/llama-bench -m /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf -p 0 -n 64 -t 19 -r 3 -o csv -d 60,60
[invocation 1/3]
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"e107984bc","10788","Intel(R) Core(TM) Ultra 7 265K","Intel(R) Arc(TM) B580 Graphics","SYCL","/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf","olmoe A1.7B Q4_K - Medium","4211730432","6919161856","2048","512","19","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","auto","auto","0","0","0","0","0","0","64","60","2026-09-09T08:09:46Z","228365836","276314","280.252349","0.338305"
"e107984bc","10788","Intel(R) Core(TM) Ultra 7 265K","Intel(R) Arc(TM) B580 Graphics","SYCL","/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf","olmoe A1.7B Q4_K - Medium","4211730432","6919161856","2048","512","19","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","auto","auto","0","0","0","0","0","0","64","60","2026-09-09T08:09:49Z","228531441","177274","280.049105","0.216536"

$ /zfs/swift/projects/llama.cpp/build/bin/llama-bench -m /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf -p 0 -n 64 -t 19 -r 3 -o csv -d 60,60
[invocation 2/3]
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"e107984bc","10788","Intel(R) Core(TM) Ultra 7 265K","Intel(R) Arc(TM) B580 Graphics","SYCL","/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf","olmoe A1.7B Q4_K - Medium","4211730432","6919161856","2048","512","19","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","auto","auto","0","0","0","0","0","0","64","60","2026-09-09T08:09:52Z","228438975","228998","280.162538","0.281011"
"e107984bc","10788","Intel(R) Core(TM) Ultra 7 265K","Intel(R) Arc(TM) B580 Graphics","SYCL","/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf","olmoe A1.7B Q4_K - Medium","4211730432","6919161856","2048","512","19","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","auto","auto","0","0","0","0","0","0","64","60","2026-09-09T08:09:54Z","228513190","186718","280.071485","0.228954"

$ /zfs/swift/projects/llama.cpp/build/bin/llama-bench -m /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf -p 0 -n 64 -t 19 -r 3 -o csv -d 60,60
[invocation 3/3]
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
"e107984bc","10788","Intel(R) Core(TM) Ultra 7 265K","Intel(R) Arc(TM) B580 Graphics","SYCL","/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf","olmoe A1.7B Q4_K - Medium","4211730432","6919161856","2048","512","19","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","auto","auto","0","0","0","0","0","0","64","60","2026-09-09T08:09:57Z","228482696","184470","280.108861","0.226256"
"e107984bc","10788","Intel(R) Core(TM) Ultra 7 265K","Intel(R) Arc(TM) B580 Graphics","SYCL","/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf","olmoe A1.7B Q4_K - Medium","4211730432","6919161856","2048","512","19","0x0","0","50","f16","f16","-1","0","layer","0","0","-1","auto","0.00","none","auto","auto","0","0","0","0","0","0","64","60","2026-09-09T08:09:59Z","228636176","315383","279.921061","0.385989"
```

</details>

## Not measured

Named so a gap reads as a gap rather than as a zero.

- staging-versus-attention attribution with prompt depth (PROTOCOL §0 claim 2). It needs a synchronous device profile (`MOEARC_SYNC_EACH=1`) at two depths on a model large enough for staging to bind, which is a far longer run than this command takes; `bench/baselines/gpt-oss-120b.md` §6.4 carries the measurement.

## Reproduce

```sh
/tmp/moearc-fixed bench --all --model /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf --prompt-ids bench/references/olmoe-1b-7b.capital.ids --depths 60 --tokens 64 --repeats 3 --llama-bench /zfs/swift/projects/llama.cpp/build/bin/llama-bench --llama-bench-inner-repeats 3 --out /tmp/art-final.md
```

<details>
<summary>machine-readable</summary>

```json
{
  "tool": "moearc bench",
  "tool_version": "0.0.0",
  "generated_utc": "2026-09-09T08:10:00Z",
  "command": [
    "/tmp/moearc-fixed",
    "bench",
    "--all",
    "--model",
    "/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
    "--prompt-ids",
    "bench/references/olmoe-1b-7b.capital.ids",
    "--depths",
    "60",
    "--tokens",
    "64",
    "--repeats",
    "3",
    "--llama-bench",
    "/zfs/swift/projects/llama.cpp/build/bin/llama-bench",
    "--llama-bench-inner-repeats",
    "3",
    "--out",
    "/tmp/art-final.md"
  ],
  "verdict": "qualified",
  "headline": [
    "**This machine, depth 60.** 110.98 ± 1.24 tok/s warm, 97.07 ± 13.65 cold, decode-only — an artefact of this box, not a portable number."
  ],
  "reading": {
    "logical_cpus": 20,
    "load1": 1.28,
    "mem_total_bytes": 98257694720,
    "mem_available_bytes": 75250085888,
    "zfs_arc": {
      "c_max_bytes": 17179869184,
      "size_bytes": 17158423920,
      "hits": 10799468818,
      "misses": 33555367
    },
    "model": {
      "path": "/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
      "bytes": 4213512672,
      "filesystem": "zfs"
    },
    "engine_threads": {
      "requested": 19,
      "reported": 0
    },
    "host_policy": "off",
    "incumbent": {
      "binary": "/zfs/swift/projects/llama.cpp/build/bin/llama-bench",
      "build_commit": "e107984bc",
      "backends": "SYCL",
      "threads": {
        "requested": 19,
        "reported": 19
      },
      "model_filename": "/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
      "device_selector": "level_zero:0"
    },
    "build": {
      "commit": "8f352e88c",
      "dirty": true,
      "profile": "release",
      "target": "x86_64-unknown-linux-gnu",
      "features": [
        "gpu"
      ]
    },
    "device": {
      "name": "Intel(R) Arc(TM) B580 Graphics",
      "backend": "level_zero",
      "driver": "xe / L0 build 37020",
      "driver_build": 37020,
      "budget_source": "measured free VRAM",
      "total_bytes": 12168933376,
      "free_bytes": 12168933376
    },
    "gpu_compiled_in": true,
    "expected_backend": "level_zero"
  },
  "thresholds": {
    "load_fraction": 0.125,
    "load_floor": 2.0,
    "load_warn_ratio": 0.6,
    "cache_tight_ratio": 0.8,
    "cv_warn": 0.1,
    "cv_refuse": 0.2,
    "min_invocations": 3,
    "vram_occupied_refuse": 0.1
  },
  "findings": [
    {
      "level": "warn",
      "code": "build-commit",
      "headline": "built from a dirty working tree",
      "detail": "moearc release / x86_64-unknown-linux-gnu / features gpu / commit 8f352e88c / working tree dirty. The commit named here does not describe what was compiled.",
      "rule": "§2"
    },
    {
      "level": "pass",
      "code": "load",
      "headline": "box quiet — load 1.28",
      "detail": "1-minute load average 1.28 on 20 logical CPUs. Refuse above 2.50 (one eighth of the machine, floor 2.0); warn above 1.50. PROTOCOL §3 records a sweep at load 9.50 on this 20-thread box that reported the opposite of the truth, reproducibly.",
      "rule": "§3"
    },
    {
      "level": "pass",
      "code": "page-cache",
      "headline": "the model fits in cache — 0.25x the ceiling",
      "detail": "/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf is 3.9 GiB against a cache ceiling of 16.0 GiB (zfs_arc_max on zfs) — a ratio of 0.25x. PROTOCOL §4: a `-r 2` sweep of a model 3.7x its ARC gave 17.59 ± 5.56 where an `-r 5` triplicate gave 28.5 ± 0.2. The disk-read counters bracketing each run are the check.",
      "rule": "§4"
    },
    {
      "level": "pass",
      "code": "threads-moearc",
      "headline": "moearc ran GPU-only — no host pool to pin",
      "detail": "`--host-policy off` builds no host expert pool, so the engine reports 0 host threads by construction and the 19 that were pinned were never drawn on. §1's failure was *accepting that a flag took*; here there is nothing for it to take. What this configuration compares is a GPU-only MoEArc against whatever the incumbent was given, and the host policy is printed beside every row so that asymmetry travels with the number.",
      "rule": "§1"
    },
    {
      "level": "pass",
      "code": "threads-incumbent",
      "headline": "llama-bench pinned to 19 threads",
      "detail": "Asked for 19, and llama-bench reported 19 in its own output. PROTOCOL §1: `llama-bench`'s default is 4 threads on a 20-core box and every published comparison that accepted it had to be withdrawn.",
      "rule": "§1"
    },
    {
      "level": "pass",
      "code": "device",
      "headline": "Intel(R) Arc(TM) B580 Graphics on level_zero (xe / L0 build 37020)",
      "detail": "Expected backend `level_zero`, found `level_zero`. PROTOCOL §2: selecting a binary by glob order once picked a Vulkan build 4.8x slower than SYCL — it produced real CSV, plausible numbers and exit 0, and only the backend field revealed it.",
      "rule": "§2"
    },
    {
      "level": "pass",
      "code": "runtime-build",
      "headline": "Level Zero build 37020",
      "detail": "Level Zero compute-runtime build 37020. Measured on this project's B580, same card, same free VRAM: build 27642 (Ubuntu 24.04 stock `libze-intel-gpu1`) does not enumerate the card at all and reports the integrated GPU's host RAM as VRAM; build 33578 (Intel's client repo for noble) detects the card and then fails at model load with `host-to-device copy failed on the device`; build 37020 (Ubuntu 26.04, `26.05.37020.3`) loads and decodes, matching llama.cpp's token ids. That makes 33578 a distribution constraint rather than a memory problem — proven by the 26.04 pass on identical hardware. These are measurements on one machine, not a published minimum, so an older build is a caution here and never a refusal.",
      "rule": "§2"
    },
    {
      "level": "pass",
      "code": "incumbent-backend",
      "headline": "/zfs/swift/projects/llama.cpp/build/bin/llama-bench is a `SYCL` build — the `level_zero` counterpart",
      "detail": "Read out of the tool's own `-o csv`, build e107984bc. Never chosen by glob: the binary path was given explicitly. llama.cpp calls this backend `sycl` and MoEArc calls it `level_zero`; the names differ because they come from two different registries, not because the builds differ. ⚠️ This field does not say which SYCL backend the runtime then selected — that is decided at run time and llama-bench does not print it, so the value in force is recorded instead of inferred: `ONEAPI_DEVICE_SELECTOR=level_zero:0`.",
      "rule": "§2"
    },
    {
      "level": "pass",
      "code": "device-busy",
      "headline": "card idle — 11.3 GiB free of 11.3 GiB",
      "detail": "0 B of 11.3 GiB on Intel(R) Arc(TM) B580 Graphics was already allocated when this run started — 0% of the card. Refuse above 10%. 🔴 Measured failure: a run on this box passed `box quiet — load 1.29` and measured for sixteen minutes while another agent's `llama-server` held 5.0 GiB of the same card with the same model; a second engine waits on a GPU queue rather than on CPUs, so the load average never moved. PROTOCOL §3: never run two engines, or two agents' benchmarks, concurrently.",
      "rule": "§3"
    },
    {
      "level": "warn",
      "code": "shape-empty",
      "headline": "no routing trace was replayed — 22 capture(s) were found and all were skipped",
      "detail": "§0 makes the shape results the headline and the absolutes an artefact of one machine, so a run that replayed no trace has not produced the thing it calls its result. Each skipped file is named with its reason in the Shape section — most often that it was captured from a different model, which §9 forbids transferring, or that it is a prefill capture. Pass --all-traces to replay them regardless, --trace <FILE> to name one, or drop --model to compare hit rates without the byte columns.",
      "rule": "§0"
    },
    {
      "level": "pass",
      "code": "warm-cache",
      "headline": "model warmed into cache — 3.9 GiB in 1.5s",
      "detail": "Read sequentially before the first timed child so that no invocation pays for faulting it off the drive, and so the cold/warm split describes the *expert pool* rather than the page cache. Ceiling 16.0 GiB (zfs_arc_max). Suppress with --no-warm-cache.",
      "rule": "§4"
    },
    {
      "level": "warn",
      "code": "dispersion",
      "headline": "moearc decode, cold pool, depth 60: stddev is 14% of the mean",
      "detail": "97.07 ± 13.65 over 3 independent invocations, stddev 14.1% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 81.31, 104.76, 105.15.",
      "rule": "§5"
    },
    {
      "level": "pass",
      "code": "dispersion",
      "headline": "moearc decode, warm pool, depth 60: 110.98 ± 1.24 tok/s",
      "detail": "110.98 ± 1.24 over 3 independent invocations, stddev 1.1% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 112.41, 110.23, 110.29.",
      "rule": "§5"
    },
    {
      "level": "pass",
      "code": "dispersion",
      "headline": "llama.cpp decode, -t 19, depth 60: 280.01 ± 0.08 tok/s",
      "detail": "280.01 ± 0.08 over 3 independent invocations, stddev 0.0% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 280.05, 280.07, 279.92.",
      "rule": "§5"
    }
  ],
  "shape": {
    "policy": "lru",
    "knee_definition": "the rung farthest from the straight line joining the first and last rungs, in normalised log2(slots) x hit-rate space — the standard elbow of a saturating curve, and the capacity past which each further doubling buys visibly less",
    "ladder_definition": "9 geometric points from the trace's peak single-step demand (the least capacity at which it is servable) to its working set (where every policy ties) — both ends are properties of the trace, so the ladder means the same thing on a model with 144 activations per step and one with 320",
    "model_context": {
      "file": "olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
      "slot_bytes": 4079616,
      "card_slots": 1024
    },
    "traces": [],
    "skipped": [
      [
        "gptoss120b-code.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss120b-code.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss120b-prose.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss120b-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss120b-reasoning.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss120b-reasoning.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss20b-code.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss20b-code.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss20b-prose.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss20b-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss20b-reasoning.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss20b-reasoning.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen3-30b-fibonacci.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen3-30b-fibonacci.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen3-30b-prose.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen3-30b-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen35moe-code.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen35moe-code.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen35moe-prose.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen35moe-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen35moe-reasoning.decode.ndjson",
        "not captured from `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen35moe-reasoning.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ]
    ]
  },
  "absolutes": [
    {
      "depth": 60,
      "generated_tokens": 64,
      "residency": "all",
      "host_policy": "off",
      "threads_requested": 19,
      "cold": {
        "label": "moearc decode, cold pool, depth 60",
        "unit": "tok/s",
        "values": [
          81.30711947749268,
          104.76404517924506,
          105.14697075557818
        ]
      },
      "warm": {
        "label": "moearc decode, warm pool, depth 60",
        "unit": "tok/s",
        "values": [
          112.40824235613871,
          110.22770082589524,
          110.29434684767106
        ]
      },
      "cold_hit_rate": {
        "label": "cold hit rate, depth 60",
        "unit": "fraction",
        "values": [
          0.9431529471544716,
          0.9431529471544716,
          0.9431529471544716
        ]
      },
      "warm_hit_rate": {
        "label": "warm hit rate, depth 60",
        "unit": "fraction",
        "values": [
          1.0,
          1.0,
          1.0
        ]
      },
      "warm_over_cold": 1.1432333688914382,
      "invocations": [
        {
          "load1_before": 1.34,
          "load1_after": 1.47,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 0,
          "resident_slots": 1024,
          "total_slots": 1024,
          "slot_bytes": 4079616,
          "n_ctx": 4096,
          "depth": 60,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 0.774839896,
            "decode_tok_s": 81.30711947749268,
            "prefill_seconds": 4.016575984,
            "hit_rate": 0.9431529471544716,
            "demands": 15744,
            "staged_bytes": 3415523328,
            "token_ids": [
              7908,
              11253,
              273,
              1264,
              9118,
              36298,
              273,
              4797,
              13,
              3168,
              13,
              285,
              2502,
              15,
              187,
              187,
              510,
              5347,
              2846,
              273,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              6253,
              2846,
              275,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              4585,
              11129,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              15,
              187,
              187,
              510,
              4585,
              1127,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              13,
              534,
              310,
              577,
              13,
              23576,
              17249
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 0.560457122,
            "decode_tok_s": 112.40824235613871,
            "prefill_seconds": 0.443204728,
            "hit_rate": 1.0,
            "demands": 15744,
            "staged_bytes": 0,
            "token_ids": [
              7908,
              11253,
              273,
              1264,
              9118,
              36298,
              273,
              4797,
              13,
              3168,
              13,
              285,
              2502,
              15,
              187,
              187,
              510,
              5347,
              2846,
              273,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              6253,
              2846,
              275,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              4585,
              11129,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              15,
              187,
              187,
              510,
              4585,
              1127,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              13,
              534,
              310,
              577,
              13,
              23576,
              17249
            ]
          },
          "io": {
            "disk_read_bytes": 222298112,
            "arc_hits": 1826851,
            "arc_misses": 212
          },
          "phases": []
        },
        {
          "load1_before": 1.47,
          "load1_after": 1.52,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 0,
          "resident_slots": 1024,
          "total_slots": 1024,
          "slot_bytes": 4079616,
          "n_ctx": 4096,
          "depth": 60,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 0.60135135,
            "decode_tok_s": 104.76404517924506,
            "prefill_seconds": 0.818492471,
            "hit_rate": 0.9431529471544716,
            "demands": 15744,
            "staged_bytes": 3415523328,
            "token_ids": [
              7908,
              11253,
              273,
              1264,
              9118,
              36298,
              273,
              4797,
              13,
              3168,
              13,
              285,
              2502,
              15,
              187,
              187,
              510,
              5347,
              2846,
              273,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              6253,
              2846,
              275,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              4585,
              11129,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              15,
              187,
              187,
              510,
              4585,
              1127,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              13,
              534,
              310,
              577,
              13,
              23576,
              17249
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 0.571544172,
            "decode_tok_s": 110.22770082589524,
            "prefill_seconds": 0.453508578,
            "hit_rate": 1.0,
            "demands": 15744,
            "staged_bytes": 0,
            "token_ids": [
              7908,
              11253,
              273,
              1264,
              9118,
              36298,
              273,
              4797,
              13,
              3168,
              13,
              285,
              2502,
              15,
              187,
              187,
              510,
              5347,
              2846,
              273,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              6253,
              2846,
              275,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              4585,
              11129,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              15,
              187,
              187,
              510,
              4585,
              1127,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              13,
              534,
              310,
              577,
              13,
              23576,
              17249
            ]
          },
          "io": {
            "disk_read_bytes": 36864,
            "arc_hits": 276,
            "arc_misses": 6
          },
          "phases": []
        },
        {
          "load1_before": 1.52,
          "load1_after": 1.52,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 0,
          "resident_slots": 1024,
          "total_slots": 1024,
          "slot_bytes": 4079616,
          "n_ctx": 4096,
          "depth": 60,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 0.599161341,
            "decode_tok_s": 105.14697075557818,
            "prefill_seconds": 0.821774622,
            "hit_rate": 0.9431529471544716,
            "demands": 15744,
            "staged_bytes": 3415523328,
            "token_ids": [
              7908,
              11253,
              273,
              1264,
              9118,
              36298,
              273,
              4797,
              13,
              3168,
              13,
              285,
              2502,
              15,
              187,
              187,
              510,
              5347,
              2846,
              273,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              6253,
              2846,
              275,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              4585,
              11129,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              15,
              187,
              187,
              510,
              4585,
              1127,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              13,
              534,
              310,
              577,
              13,
              23576,
              17249
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 0.571198813,
            "decode_tok_s": 110.29434684767106,
            "prefill_seconds": 0.453576633,
            "hit_rate": 1.0,
            "demands": 15744,
            "staged_bytes": 0,
            "token_ids": [
              7908,
              11253,
              273,
              1264,
              9118,
              36298,
              273,
              4797,
              13,
              3168,
              13,
              285,
              2502,
              15,
              187,
              187,
              510,
              5347,
              2846,
              273,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              6253,
              2846,
              275,
              6181,
              310,
              7785,
              15,
              187,
              187,
              510,
              4585,
              11129,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              15,
              187,
              187,
              510,
              4585,
              1127,
              275,
              6181,
              310,
              7812,
              2071,
              1377,
              13,
              534,
              310,
              577,
              13,
              23576,
              17249
            ]
          },
          "io": {
            "disk_read_bytes": 0,
            "arc_hits": 334,
            "arc_misses": 0
          },
          "phases": []
        }
      ]
    }
  ],
  "incumbent": {
    "binary": "/zfs/swift/projects/llama.cpp/build/bin/llama-bench",
    "command": [
      "-m",
      "/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
      "-p",
      "0",
      "-n",
      "64",
      "-t",
      "19",
      "-r",
      "3",
      "-o",
      "csv",
      "-d",
      "60,60"
    ],
    "facts": {
      "binary": "/zfs/swift/projects/llama.cpp/build/bin/llama-bench",
      "build_commit": "e107984bc",
      "backends": "SYCL",
      "threads": {
        "requested": 19,
        "reported": 19
      },
      "model_filename": "/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
      "device_selector": "level_zero:0"
    },
    "points": [
      {
        "threads_requested": 19,
        "threads_reported": [
          19
        ],
        "depth": 60,
        "generated_tokens": 64,
        "decode": {
          "label": "llama.cpp decode, -t 19, depth 60",
          "unit": "tok/s",
          "values": [
            280.049105,
            280.071485,
            279.921061
          ]
        },
        "warmup_discarded": true
      }
    ],
    "best_threads": 19,
    "raw_output": "\n$ /zfs/swift/projects/llama.cpp/build/bin/llama-bench -m /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf -p 0 -n 64 -t 19 -r 3 -o csv -d 60,60\n[invocation 1/3]\nbuild_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n\"e107984bc\",\"10788\",\"Intel(R) Core(TM) Ultra 7 265K\",\"Intel(R) Arc(TM) B580 Graphics\",\"SYCL\",\"/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf\",\"olmoe A1.7B Q4_K - Medium\",\"4211730432\",\"6919161856\",\"2048\",\"512\",\"19\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"-1\",\"0\",\"layer\",\"0\",\"0\",\"-1\",\"auto\",\"0.00\",\"none\",\"auto\",\"auto\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"64\",\"60\",\"2026-09-09T08:09:46Z\",\"228365836\",\"276314\",\"280.252349\",\"0.338305\"\n\"e107984bc\",\"10788\",\"Intel(R) Core(TM) Ultra 7 265K\",\"Intel(R) Arc(TM) B580 Graphics\",\"SYCL\",\"/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf\",\"olmoe A1.7B Q4_K - Medium\",\"4211730432\",\"6919161856\",\"2048\",\"512\",\"19\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"-1\",\"0\",\"layer\",\"0\",\"0\",\"-1\",\"auto\",\"0.00\",\"none\",\"auto\",\"auto\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"64\",\"60\",\"2026-09-09T08:09:49Z\",\"228531441\",\"177274\",\"280.049105\",\"0.216536\"\n\n$ /zfs/swift/projects/llama.cpp/build/bin/llama-bench -m /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf -p 0 -n 64 -t 19 -r 3 -o csv -d 60,60\n[invocation 2/3]\nbuild_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n\"e107984bc\",\"10788\",\"Intel(R) Core(TM) Ultra 7 265K\",\"Intel(R) Arc(TM) B580 Graphics\",\"SYCL\",\"/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf\",\"olmoe A1.7B Q4_K - Medium\",\"4211730432\",\"6919161856\",\"2048\",\"512\",\"19\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"-1\",\"0\",\"layer\",\"0\",\"0\",\"-1\",\"auto\",\"0.00\",\"none\",\"auto\",\"auto\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"64\",\"60\",\"2026-09-09T08:09:52Z\",\"228438975\",\"228998\",\"280.162538\",\"0.281011\"\n\"e107984bc\",\"10788\",\"Intel(R) Core(TM) Ultra 7 265K\",\"Intel(R) Arc(TM) B580 Graphics\",\"SYCL\",\"/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf\",\"olmoe A1.7B Q4_K - Medium\",\"4211730432\",\"6919161856\",\"2048\",\"512\",\"19\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"-1\",\"0\",\"layer\",\"0\",\"0\",\"-1\",\"auto\",\"0.00\",\"none\",\"auto\",\"auto\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"64\",\"60\",\"2026-09-09T08:09:54Z\",\"228513190\",\"186718\",\"280.071485\",\"0.228954\"\n\n$ /zfs/swift/projects/llama.cpp/build/bin/llama-bench -m /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf -p 0 -n 64 -t 19 -r 3 -o csv -d 60,60\n[invocation 3/3]\nbuild_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n\"e107984bc\",\"10788\",\"Intel(R) Core(TM) Ultra 7 265K\",\"Intel(R) Arc(TM) B580 Graphics\",\"SYCL\",\"/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf\",\"olmoe A1.7B Q4_K - Medium\",\"4211730432\",\"6919161856\",\"2048\",\"512\",\"19\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"-1\",\"0\",\"layer\",\"0\",\"0\",\"-1\",\"auto\",\"0.00\",\"none\",\"auto\",\"auto\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"64\",\"60\",\"2026-09-09T08:09:57Z\",\"228482696\",\"184470\",\"280.108861\",\"0.226256\"\n\"e107984bc\",\"10788\",\"Intel(R) Core(TM) Ultra 7 265K\",\"Intel(R) Arc(TM) B580 Graphics\",\"SYCL\",\"/zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf\",\"olmoe A1.7B Q4_K - Medium\",\"4211730432\",\"6919161856\",\"2048\",\"512\",\"19\",\"0x0\",\"0\",\"50\",\"f16\",\"f16\",\"-1\",\"0\",\"layer\",\"0\",\"0\",\"-1\",\"auto\",\"0.00\",\"none\",\"auto\",\"auto\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"64\",\"60\",\"2026-09-09T08:09:59Z\",\"228636176\",\"315383\",\"279.921061\",\"0.385989\"\n"
  },
  "not_measured": [
    "staging-versus-attention attribution with prompt depth (PROTOCOL §0 claim 2). It needs a synchronous device profile (`MOEARC_SYNC_EACH=1`) at two depths on a model large enough for staging to bind, which is a far longer run than this command takes; `bench/baselines/gpt-oss-120b.md` §6.4 carries the measurement."
  ]
}
```

</details>
