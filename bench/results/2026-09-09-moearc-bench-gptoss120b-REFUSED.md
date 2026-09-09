# moearc bench

**VERDICT: REFUSED — no number below is a measurement** · 0.0.0 · generated 2026-09-09T08:00:31Z UTC

## Result

**There is none.** At least one check below refused, so anything this run produced describes the state of the machine rather than the engine. The figures are kept in place, unheadlined, so the refusal is auditable rather than convenient — see `bench/PROTOCOL.md` §5 and §9.

> **What reproduces and what does not.** Absolute throughput depends on CPU, memory bandwidth, PCIe generation, filesystem and whether the model fits in page cache; it is reported below as an artefact of *this* machine and should not be compared across machines. The **shape** results are a deterministic replay of committed routing traces — no clock is read and no device is touched — so they should come out identical on yours, to the last digit. Those are the result.

## Checks

Thresholds in force: refuse above load **2.50** (12% of 20 logical CPUs, floor 2.0), warn above **1.50**; a model above **80%** of the cache ceiling is flagged; a result needs **3** independent invocations and is refused a headline at a stddev of **20%** of its mean (warned at 10%).

| | rule | check |
|---|---|---|
| ok | §2 | build d184743b2 |
| ok | §3 | box quiet — load 0.73 |
| ⚠️ warn | §4 | the model CANNOT be cached — 3.69x the cache ceiling |
| ok | §1 | moearc host pool pinned to 19 threads |
| ok | §2 | Intel(R) Arc(TM) B580 Graphics on level_zero (xe / L0 build 37020) |
| ok | §2 | Level Zero build 37020 |
| ok | §3 | card idle — 11.3 GiB free of 11.3 GiB |
| ⚠️ warn | §4 | the model cannot be warmed into cache — 59.0 GiB vs a 16.0 GiB ceiling |
| ok | §3 | waited 564s in total for the box to go quiet, before 5 of the invocations |
| ⚠️ warn | §5 | moearc decode, cold pool, depth 128: stddev is 13% of the mean |
| ok | §5 | moearc decode, warm pool, depth 128: 12.14 ± 0.12 tok/s |
| 🔴 **REFUSE** | §5 | moearc decode, cold pool, depth 512: stddev is 34% of the mean — this is not a measurement |
| ok | §5 | moearc decode, warm pool, depth 512: 12.67 ± 0.05 tok/s |

**§4 — the model CANNOT be cached — 3.69x the cache ceiling**

/zfs/swift/models/gpt-oss-120b-MXFP4.gguf is 59.0 GiB against a cache ceiling of 16.0 GiB (zfs_arc_max on zfs) — a ratio of 3.69x. PROTOCOL §4: a `-r 2` sweep of a model 3.7x its ARC gave 17.59 ± 5.56 where an `-r 5` triplicate gave 28.5 ± 0.2. The disk-read counters bracketing each run are the check.

**§4 — the model cannot be warmed into cache — 59.0 GiB vs a 16.0 GiB ceiling**

Reading it in would evict as much as it caches, so it was not attempted (zfs_arc_max). Every invocation will fault some of the model off the drive and the disk-read column beside each row is what says how much. PROTOCOL §4: this is stated rather than fixed, because the model this project exists for is 3.7x its ARC and refusing it would make the tool useless for its own headline case.

**§5 — moearc decode, cold pool, depth 128: stddev is 13% of the mean**

11.25 ± 1.42 over 3 independent invocations, stddev 12.6% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 9.62, 12.19, 11.94.

**§5 — moearc decode, cold pool, depth 512: stddev is 34% of the mean — this is not a measurement**

10.54 ± 3.53 over 3 independent invocations, stddev 33.5% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 6.46, 12.65, 12.51.

## Machine

| | |
|---|---|
| device | Intel(R) Arc(TM) B580 Graphics (level_zero) |
| driver | xe / L0 build 37020 |
| Level Zero build | 37020 |
| device memory | 11.3 GiB free of 11.3 GiB (measured free VRAM) |
| logical CPUs | 20 |
| load average (1 min) | 0.73 |
| memory | 70.5 GiB available of 91.5 GiB |
| ZFS ARC | 16.1 GiB in use, cap 16.0 GiB (`zfs_arc_max`) |
| model | /zfs/swift/models/gpt-oss-120b-MXFP4.gguf — 59.0 GiB on zfs |
| moearc host threads | 19 requested, 19 reported by the engine |
| build | release · x86_64-unknown-linux-gnu · features [gpu] · commit d184743b2 (clean tree) |

## Shape — the result that should reproduce on your machine

Policy under test: **lru**, against the widest static split that fits the *same* capacity. Ladder: 9 geometric points from the trace's peak single-step demand (the least capacity at which it is servable) to its working set (where every policy ties) — both ends are properties of the trace, so the ladder means the same thing on a model with 144 activations per step and one with 320. Knee: the rung farthest from the straight line joining the first and last rungs, in normalised log2(slots) x hit-rate space — the standard elbow of a saturating curve, and the capacity past which each further doubling buys visibly less.

**dynamic residency beat the widest matched-capacity static split on 3/3 traces, by 19.9 to 57.3 points**

Byte columns and the card marker are attached only to captures whose own header names `gpt-oss-120b-MXFP4.gguf` — a slot size belongs to one model (gpt-oss-120B's is 12.607 MiB, Qwen3-30B-A3B's is 2.92 MiB) and carrying one onto another model's trace is PROTOCOL §9's last failure.

### gptoss120b-code.decode.ndjson

512 decode steps · 73,728 activations · working set 3,685 experts · peak 144 per step · knee at **1,638** slots

| slots | dynamic hit | static hit (blocks) | gap (points) | dynamic staged | static staged | pts/doubling |
|---:|---:|---:|---:|---:|---:|---:|
| 144 | 25.9% | 2.8% (1) | +23.2 | 672.3 GiB | 882.5 GiB | — |
| 216 | 33.0% | 2.8% (1) | +30.2 | 608.1 GiB | 882.5 GiB | +12.1 |
| 324 | 41.9% | 5.6% (2) | +36.3 | 527.4 GiB | 857.3 GiB | +15.2 |
| 486 | 50.4% | 11.1% (4) | +39.3 | 450.1 GiB | 806.8 GiB | +14.6 |
| 728 | 61.2% | 16.7% (6) | +44.5 | 352.2 GiB | 756.4 GiB | +18.5 |
| 1,092 | 71.4% | 25.0% (9) | +46.4 | 259.3 GiB | 680.8 GiB | +17.5 |
| 1,638 | 83.4% | 41.7% (15) | +41.7 | 151.0 GiB | 529.5 GiB | +20.4 |
| 2,457 | 91.1% | 63.9% (23) | +27.2 | 80.8 GiB | 327.8 GiB | +13.2 |
| 3,685 | 95.0% | 100.0% (36) | -5.0 | 45.4 GiB | 0 B | +6.7 | *(whole working set resident: the dynamic policy never evicts, so its misses here are all compulsory, while the static split is modelled as resident from step zero and pays no warm-up. Not a comparison — excluded from the claim.)*

⚠️ **612 slots** is where *this* card lands on the curve, which is a fact about this machine and not part of the shape.

Provenance of the capture, verbatim:

```json
{"format":"moearc-trace-v1","phase":"decode","n_layer":36,"n_layers_routed":36,"n_expert_used":4,"n_prompt_tokens":113,"n_prefill_steps":113,"n_decode_steps":512,"hit_eog":false,"model_file":"gpt-oss-120b-MXFP4.gguf","quantisation":"MXFP4","llama_cpp_commit":"e107984bcffcfd701e82738092a2b000b6fda7a2","llama_cpp_patched":true,"captured_utc":"2026-09-06T15:22:55Z","prompt_name":"gptoss120b-code","prompt":"<|start|>system<|message|>You are a helpful assistant. Reasoning: low<|end|><|start|>user<|message|>Write a complete, heavily commented Rust implementation of a red-black tree in a single file. Include the node representation, insertion with full rebalancing, deletion with full rebalancing, an in-order iterator implementing Iterator, a Debug impl, and a module of unit tests covering insertion order, deletion of leaves and internal nodes, and the red-black invariants. Output only Rust source code with comments; do not write any explanatory prose outside the code.<|end|><|start|>assistant<|channel|>final<|message|>\n","backend":"CPU (-ngl 0)","sampling":"temp 0.7, top-k 20, top-p 0.8, seed 20260904","capture_threads":6}
```

### gptoss120b-prose.decode.ndjson

512 decode steps · 73,728 activations · working set 2,442 experts · peak 144 per step · knee at **1,203** slots

| slots | dynamic hit | static hit (blocks) | gap (points) | dynamic staged | static staged | pts/doubling |
|---:|---:|---:|---:|---:|---:|---:|
| 144 | 34.2% | 2.8% (1) | +31.4 | 597.1 GiB | 882.5 GiB | — |
| 205 | 41.3% | 2.8% (1) | +38.5 | 532.7 GiB | 882.5 GiB | +13.9 |
| 292 | 50.5% | 5.6% (2) | +45.0 | 449.1 GiB | 857.3 GiB | +18.0 |
| 416 | 60.6% | 11.1% (4) | +49.5 | 357.2 GiB | 806.8 GiB | +19.8 |
| 593 | 72.3% | 16.7% (6) | +55.6 | 251.5 GiB | 756.4 GiB | +22.8 |
| 845 | 82.3% | 25.0% (9) | +57.3 | 160.7 GiB | 680.8 GiB | +19.6 |
| 1,203 | 90.6% | 41.7% (15) | +49.0 | 84.9 GiB | 529.5 GiB | +16.4 |
| 1,714 | 95.2% | 63.9% (23) | +31.3 | 43.9 GiB | 327.8 GiB | +8.9 |
| 2,442 | 96.7% | 100.0% (36) | -3.3 | 30.1 GiB | 0 B | +3.0 | *(whole working set resident: the dynamic policy never evicts, so its misses here are all compulsory, while the static split is modelled as resident from step zero and pays no warm-up. Not a comparison — excluded from the claim.)*

⚠️ **612 slots** is where *this* card lands on the curve, which is a fact about this machine and not part of the shape.

Provenance of the capture, verbatim:

```json
{"format":"moearc-trace-v1","phase":"decode","n_layer":36,"n_layers_routed":36,"n_expert_used":4,"n_prompt_tokens":161,"n_prefill_steps":161,"n_decode_steps":512,"hit_eog":false,"model_file":"gpt-oss-120b-MXFP4.gguf","quantisation":"MXFP4","llama_cpp_commit":"e107984bcffcfd701e82738092a2b000b6fda7a2","llama_cpp_patched":true,"captured_utc":"2026-09-06T15:35:05Z","prompt_name":"gptoss120b-prose","prompt":"<|start|>system<|message|>You are a helpful assistant. Reasoning: low<|end|><|start|>user<|message|>Write the opening 2000 words of a continuous narrative history of lighthouse construction around the North Atlantic. Cover the early wooden and stone towers, the Eddystone rebuilds, the Stevenson family in Scotland, the shift from oil to electric illumination, the invention and spread of the Fresnel lens, and how improvements in cement, cast iron and marine engineering changed what could be built on an exposed rock. Write flowing prose only: no headings, no bullet points, no lists. Never abbreviate, never summarise, and never write a placeholder such as \"(the rest of the essay)\" or \"...\" \u2014 write every sentence out in full.<|end|><|start|>assistant<|channel|>final<|message|>The story of lighthouse building around the North Atlantic begins","backend":"CPU (-ngl 0)","sampling":"temp 0.7, top-k 20, top-p 0.8, seed 20260906","capture_threads":6}
```

### gptoss120b-reasoning.decode.ndjson

512 decode steps · 73,728 activations · working set 3,486 experts · peak 144 per step · knee at **1,572** slots

| slots | dynamic hit | static hit (blocks) | gap (points) | dynamic staged | static staged | pts/doubling |
|---:|---:|---:|---:|---:|---:|---:|
| 144 | 22.7% | 2.8% (1) | +19.9 | 701.6 GiB | 882.5 GiB | — |
| 214 | 29.9% | 2.8% (1) | +27.2 | 635.8 GiB | 882.5 GiB | +12.7 |
| 319 | 38.7% | 5.6% (2) | +33.2 | 556.1 GiB | 857.3 GiB | +15.3 |
| 476 | 48.0% | 11.1% (4) | +36.9 | 471.8 GiB | 806.8 GiB | +16.1 |
| 709 | 60.2% | 16.7% (6) | +43.5 | 361.4 GiB | 756.4 GiB | +21.1 |
| 1,055 | 74.0% | 27.8% (10) | +46.2 | 236.1 GiB | 655.6 GiB | +24.1 |
| 1,572 | 84.8% | 44.4% (16) | +40.3 | 138.1 GiB | 504.3 GiB | +18.8 |
| 2,341 | 92.6% | 66.7% (24) | +25.9 | 67.6 GiB | 302.6 GiB | +13.5 |
| 3,486 | 95.3% | 100.0% (36) | -4.7 | 42.9 GiB | 0 B | +4.7 | *(whole working set resident: the dynamic policy never evicts, so its misses here are all compulsory, while the static split is modelled as resident from step zero and pays no warm-up. Not a comparison — excluded from the claim.)*

⚠️ **612 slots** is where *this* card lands on the curve, which is a fact about this machine and not part of the shape.

Provenance of the capture, verbatim:

```json
{"format":"moearc-trace-v1","phase":"decode","n_layer":36,"n_layers_routed":36,"n_expert_used":4,"n_prompt_tokens":151,"n_prefill_steps":151,"n_decode_steps":512,"hit_eog":false,"model_file":"gpt-oss-120b-MXFP4.gguf","quantisation":"MXFP4","llama_cpp_commit":"e107984bcffcfd701e82738092a2b000b6fda7a2","llama_cpp_patched":true,"captured_utc":"2026-09-06T15:25:22Z","prompt_name":"gptoss120b-reasoning","prompt":"<|start|>system<|message|>You are a helpful assistant. Reasoning: high<|end|><|start|>user<|message|>A freight train leaves station A at 6:00 travelling at 40 km/h. A passenger train leaves station B, 300 km away on the same line, at 7:30 travelling towards A at 90 km/h. There is a single siding 180 km from A that can hold one train. Work out, step by step, whether the trains can pass without a collision, at what time and place they would meet on open track, and what the latest departure time from B would be for the passenger train to reach A without the freight ever having to wait more than twenty minutes in the siding. Show all reasoning.<|end|><|start|>assistant<|channel|>analysis<|message|>\n","backend":"CPU (-ngl 0)","sampling":"temp 0.7, top-k 20, top-p 0.8, seed 20260904","capture_threads":6}
```

Not replayed:

- `gptoss120b-code.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss120b-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss120b-reasoning.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss20b-code.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss20b-code.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss20b-prose.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss20b-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `gptoss20b-reasoning.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `gptoss20b-reasoning.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen3-30b-fibonacci.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen3-30b-fibonacci.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen3-30b-prose.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen3-30b-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen35moe-code.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen35moe-code.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen35moe-prose.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen35moe-prose.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)
- `qwen35moe-reasoning.decode.ndjson` — not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway.
- `qwen35moe-reasoning.prefill.ndjson` — prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)

🔴 A hit rate predicts **staged bytes** and nothing else. It does not predict tok/s: there is no validated model between them in this project, so none is published here and no throughput figure anywhere in this section is derived from one.

## Absolutes — this machine only

Decode-only throughput: the timer starts after prefill, so these are steady-state decode figures at the stated depth and not an average over the prompt. Cold and warm are separate questions and are never averaged together. Each figure is the mean ± sample stddev over **independent invocations of this binary**, not iterations inside one process.

| depth | tokens | residency | host | threads | cold tok/s | warm tok/s | warm/cold | cold hit | warm hit | disk read | ARC miss |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 600 | frac:0.5 | 19 | 11.25 ± 1.42 | 12.14 ± 0.12 | 1.08x | 71.8% | 72.7% | 2.1 GiB | 0.2% |
| 512 | 64 | 600 | frac:0.5 | 19 | 10.54 ± 3.53 | 12.67 ± 0.05 | 1.20x | 75.7% | 75.9% | 12.9 GiB | 0.3% |

A large `disk read` column means the run faulted the model off the drive and measured the storage rather than the engine (PROTOCOL §4). `ARC miss` is the ZFS cache's own miss rate over the same window, machine-wide — the second, independent reading §4 asks for. Both coming back near zero is a result, not an absence: it says staging read from RAM, and that this run measured the engine.

Every individual invocation, so the spread is visible:

- depth 128: cold 9.62, 12.19, 11.94 · warm 12.16, 12.25, 12.01 · load before each child 0.83, 2.41, 2.44
- depth 512: cold 6.46, 12.65, 12.51 · warm 12.73, 12.63, 12.66 · load before each child 2.44, 2.40, 2.40

## Not measured

Named so a gap reads as a gap rather than as a zero.

- llama.cpp as a baseline — pass --llama-bench <PATH>. It is never searched for: PROTOCOL §2 records a glob-ordered pick that silently selected a Vulkan build 4.8x slower than SYCL.
- staging-versus-attention attribution with prompt depth (PROTOCOL §0 claim 2). It needs a synchronous device profile (`MOEARC_SYNC_EACH=1`) at two depths on a model large enough for staging to bind, which is a far longer run than this command takes; `bench/baselines/gpt-oss-120b.md` §6.4 carries the measurement.

## Reproduce

```sh
/tmp/moearc-fixed bench --all --model /zfs/swift/models/gpt-oss-120b-MXFP4.gguf --prompt-ids bench/references/gpt-oss-120b.longctx.ids --depths 128,512 --tokens 64 --repeats 3 --residency 600 --host-policy frac:0.5 --out /tmp/art-gptoss.md
```

<details>
<summary>machine-readable</summary>

```json
{
  "tool": "moearc bench",
  "tool_version": "0.0.0",
  "generated_utc": "2026-09-09T08:00:31Z",
  "command": [
    "/tmp/moearc-fixed",
    "bench",
    "--all",
    "--model",
    "/zfs/swift/models/gpt-oss-120b-MXFP4.gguf",
    "--prompt-ids",
    "bench/references/gpt-oss-120b.longctx.ids",
    "--depths",
    "128,512",
    "--tokens",
    "64",
    "--repeats",
    "3",
    "--residency",
    "600",
    "--host-policy",
    "frac:0.5",
    "--out",
    "/tmp/art-gptoss.md"
  ],
  "verdict": "refused",
  "headline": null,
  "reading": {
    "logical_cpus": 20,
    "load1": 0.73,
    "mem_total_bytes": 98257694720,
    "mem_available_bytes": 75731968000,
    "zfs_arc": {
      "c_max_bytes": 17179869184,
      "size_bytes": 17320113816,
      "hits": 10792928488,
      "misses": 33534327
    },
    "model": {
      "path": "/zfs/swift/models/gpt-oss-120b-MXFP4.gguf",
      "bytes": 63387346208,
      "filesystem": "zfs"
    },
    "engine_threads": {
      "requested": 19,
      "reported": 19
    },
    "host_policy": "frac:0.5",
    "incumbent": null,
    "build": {
      "commit": "d184743b2",
      "dirty": false,
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
      "level": "pass",
      "code": "build-commit",
      "headline": "build d184743b2",
      "detail": "moearc release / x86_64-unknown-linux-gnu / features gpu / commit d184743b2 / working tree clean",
      "rule": "§2"
    },
    {
      "level": "pass",
      "code": "load",
      "headline": "box quiet — load 0.73",
      "detail": "1-minute load average 0.73 on 20 logical CPUs. Refuse above 2.50 (one eighth of the machine, floor 2.0); warn above 1.50. PROTOCOL §3 records a sweep at load 9.50 on this 20-thread box that reported the opposite of the truth, reproducibly.",
      "rule": "§3"
    },
    {
      "level": "warn",
      "code": "page-cache",
      "headline": "the model CANNOT be cached — 3.69x the cache ceiling",
      "detail": "/zfs/swift/models/gpt-oss-120b-MXFP4.gguf is 59.0 GiB against a cache ceiling of 16.0 GiB (zfs_arc_max on zfs) — a ratio of 3.69x. PROTOCOL §4: a `-r 2` sweep of a model 3.7x its ARC gave 17.59 ± 5.56 where an `-r 5` triplicate gave 28.5 ± 0.2. The disk-read counters bracketing each run are the check.",
      "rule": "§4"
    },
    {
      "level": "pass",
      "code": "threads-moearc",
      "headline": "moearc host pool pinned to 19 threads",
      "detail": "Asked for 19, and moearc host pool reported 19 in its own output. PROTOCOL §1: `llama-bench`'s default is 4 threads on a 20-core box and every published comparison that accepted it had to be withdrawn.",
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
      "code": "device-busy",
      "headline": "card idle — 11.3 GiB free of 11.3 GiB",
      "detail": "0 B of 11.3 GiB on Intel(R) Arc(TM) B580 Graphics was already allocated when this run started — 0% of the card. Refuse above 10%. 🔴 Measured failure: a run on this box passed `box quiet — load 1.29` and measured for sixteen minutes while another agent's `llama-server` held 5.0 GiB of the same card with the same model; a second engine waits on a GPU queue rather than on CPUs, so the load average never moved. PROTOCOL §3: never run two engines, or two agents' benchmarks, concurrently.",
      "rule": "§3"
    },
    {
      "level": "warn",
      "code": "warm-cache",
      "headline": "the model cannot be warmed into cache — 59.0 GiB vs a 16.0 GiB ceiling",
      "detail": "Reading it in would evict as much as it caches, so it was not attempted (zfs_arc_max). Every invocation will fault some of the model off the drive and the disk-read column beside each row is what says how much. PROTOCOL §4: this is stated rather than fixed, because the model this project exists for is 3.7x its ARC and refusing it would make the tool useless for its own headline case.",
      "rule": "§4"
    },
    {
      "level": "pass",
      "code": "load-settle",
      "headline": "waited 564s in total for the box to go quiet, before 5 of the invocations",
      "detail": "🔴 The wait is almost always this benchmark waiting for itself: a 1-minute load average does not forget the 19-thread invocation that just finished, and on this project's own box one host-offload run takes a quiet 0.99 to 2.51 — over the 2.50 refusal. §3's threshold is there to detect *another* tenant, so the tool waits for its own contribution to decay rather than refusing itself. Set --load-settle 0 to refuse at once instead.",
      "rule": "§3"
    },
    {
      "level": "warn",
      "code": "dispersion",
      "headline": "moearc decode, cold pool, depth 128: stddev is 13% of the mean",
      "detail": "11.25 ± 1.42 over 3 independent invocations, stddev 12.6% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 9.62, 12.19, 11.94.",
      "rule": "§5"
    },
    {
      "level": "pass",
      "code": "dispersion",
      "headline": "moearc decode, warm pool, depth 128: 12.14 ± 0.12 tok/s",
      "detail": "12.14 ± 0.12 over 3 independent invocations, stddev 1.0% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 12.16, 12.25, 12.01.",
      "rule": "§5"
    },
    {
      "level": "refuse",
      "code": "dispersion",
      "headline": "moearc decode, cold pool, depth 512: stddev is 34% of the mean — this is not a measurement",
      "detail": "10.54 ± 3.53 over 3 independent invocations, stddev 33.5% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 6.46, 12.65, 12.51.",
      "rule": "§5"
    },
    {
      "level": "pass",
      "code": "dispersion",
      "headline": "moearc decode, warm pool, depth 512: 12.67 ± 0.05 tok/s",
      "detail": "12.67 ± 0.05 over 3 independent invocations, stddev 0.4% of the mean. Warn at 10%, refuse to headline at 20% — §5 states that a run whose stddev is 20–30% of its mean is not a measurement, and the good triplicate it is contrasted against sat at 0.7%. Individual values: 12.73, 12.63, 12.66.",
      "rule": "§5"
    }
  ],
  "shape": {
    "policy": "lru",
    "knee_definition": "the rung farthest from the straight line joining the first and last rungs, in normalised log2(slots) x hit-rate space — the standard elbow of a saturating curve, and the capacity past which each further doubling buys visibly less",
    "ladder_definition": "9 geometric points from the trace's peak single-step demand (the least capacity at which it is servable) to its working set (where every policy ties) — both ends are properties of the trace, so the ladder means the same thing on a model with 144 activations per step and one with 320",
    "model_context": {
      "file": "gpt-oss-120b-MXFP4.gguf",
      "slot_bytes": 13219200,
      "card_slots": 612
    },
    "traces": [
      {
        "name": "gptoss120b-code.decode.ndjson",
        "provenance": "{\"format\":\"moearc-trace-v1\",\"phase\":\"decode\",\"n_layer\":36,\"n_layers_routed\":36,\"n_expert_used\":4,\"n_prompt_tokens\":113,\"n_prefill_steps\":113,\"n_decode_steps\":512,\"hit_eog\":false,\"model_file\":\"gpt-oss-120b-MXFP4.gguf\",\"quantisation\":\"MXFP4\",\"llama_cpp_commit\":\"e107984bcffcfd701e82738092a2b000b6fda7a2\",\"llama_cpp_patched\":true,\"captured_utc\":\"2026-09-06T15:22:55Z\",\"prompt_name\":\"gptoss120b-code\",\"prompt\":\"<|start|>system<|message|>You are a helpful assistant. Reasoning: low<|end|><|start|>user<|message|>Write a complete, heavily commented Rust implementation of a red-black tree in a single file. Include the node representation, insertion with full rebalancing, deletion with full rebalancing, an in-order iterator implementing Iterator, a Debug impl, and a module of unit tests covering insertion order, deletion of leaves and internal nodes, and the red-black invariants. Output only Rust source code with comments; do not write any explanatory prose outside the code.<|end|><|start|>assistant<|channel|>final<|message|>\\n\",\"backend\":\"CPU (-ngl 0)\",\"sampling\":\"temp 0.7, top-k 20, top-p 0.8, seed 20260904\",\"capture_threads\":6}",
        "steps": 512,
        "demands": 73728,
        "working_set": 3685,
        "peak_step_demand": 144,
        "rows": [
          {
            "slots": 144,
            "dynamic_hit": 0.2593587239583333,
            "static_hit": 0.027777777777777776,
            "static_layers": 1,
            "gap_points": 23.158094618055554,
            "optimal_hit": null,
            "dynamic_staged_bytes": 721847635200,
            "static_staged_bytes": 947552256000,
            "points_per_doubling": null,
            "trivial": false
          },
          {
            "slots": 216,
            "dynamic_hit": 0.3300645616319444,
            "static_hit": 0.027777777777777776,
            "static_layers": 1,
            "gap_points": 30.228678385416664,
            "optimal_hit": null,
            "dynamic_staged_bytes": 652935945600,
            "static_staged_bytes": 947552256000,
            "points_per_doubling": 12.087242786750126,
            "trivial": false
          },
          {
            "slots": 324,
            "dynamic_hit": 0.4189588758680556,
            "static_hit": 0.05555555555555555,
            "static_layers": 2,
            "gap_points": 36.34033203125,
            "optimal_hit": null,
            "dynamic_staged_bytes": 566297308800,
            "static_staged_bytes": 920479334400,
            "points_per_doubling": 15.19658339235764,
            "trivial": false
          },
          {
            "slots": 486,
            "dynamic_hit": 0.5041639539930556,
            "static_hit": 0.1111111111111111,
            "static_layers": 4,
            "gap_points": 39.30528428819445,
            "optimal_hit": null,
            "dynamic_staged_bytes": 483254294400,
            "static_staged_bytes": 866333491200,
            "points_per_doubling": 14.565904313517034,
            "trivial": false
          },
          {
            "slots": 728,
            "dynamic_hit": 0.6119384765625,
            "static_hit": 0.16666666666666666,
            "static_layers": 6,
            "gap_points": 44.527180989583336,
            "optimal_hit": null,
            "dynamic_staged_bytes": 378214531200,
            "static_staged_bytes": 812187648000,
            "points_per_doubling": 18.48676242454084,
            "trivial": false
          },
          {
            "slots": 1092,
            "dynamic_hit": 0.7143825954861112,
            "static_hit": 0.25,
            "static_layers": 9,
            "gap_points": 46.438259548611114,
            "optimal_hit": null,
            "dynamic_staged_bytes": 278369913600,
            "static_staged_bytes": 730968883200,
            "points_per_doubling": 17.51293780324645,
            "trivial": false
          },
          {
            "slots": 1638,
            "dynamic_hit": 0.8336046006944444,
            "static_hit": 0.4166666666666667,
            "static_layers": 15,
            "gap_points": 41.69379340277777,
            "optimal_hit": null,
            "dynamic_staged_bytes": 162173145600,
            "static_staged_bytes": 568531353600,
            "points_per_doubling": 20.381136408120764,
            "trivial": false
          },
          {
            "slots": 2457,
            "dynamic_hit": 0.9109429253472222,
            "static_hit": 0.6388888888888888,
            "static_layers": 23,
            "gap_points": 27.205403645833336,
            "optimal_hit": null,
            "dynamic_staged_bytes": 86797267200,
            "static_staged_bytes": 351947980800,
            "points_per_doubling": 13.22107392481282,
            "trivial": false
          },
          {
            "slots": 3685,
            "dynamic_hit": 0.9500189887152778,
            "static_hit": 1.0,
            "static_layers": 36,
            "gap_points": -4.998101128472221,
            "optimal_hit": null,
            "dynamic_staged_bytes": 48712752000,
            "static_staged_bytes": 0,
            "points_per_doubling": 6.682333185248751,
            "trivial": true
          }
        ],
        "knee_slots": 1638,
        "card_slots": 612,
        "dynamic_beats_static": true
      },
      {
        "name": "gptoss120b-prose.decode.ndjson",
        "provenance": "{\"format\":\"moearc-trace-v1\",\"phase\":\"decode\",\"n_layer\":36,\"n_layers_routed\":36,\"n_expert_used\":4,\"n_prompt_tokens\":161,\"n_prefill_steps\":161,\"n_decode_steps\":512,\"hit_eog\":false,\"model_file\":\"gpt-oss-120b-MXFP4.gguf\",\"quantisation\":\"MXFP4\",\"llama_cpp_commit\":\"e107984bcffcfd701e82738092a2b000b6fda7a2\",\"llama_cpp_patched\":true,\"captured_utc\":\"2026-09-06T15:35:05Z\",\"prompt_name\":\"gptoss120b-prose\",\"prompt\":\"<|start|>system<|message|>You are a helpful assistant. Reasoning: low<|end|><|start|>user<|message|>Write the opening 2000 words of a continuous narrative history of lighthouse construction around the North Atlantic. Cover the early wooden and stone towers, the Eddystone rebuilds, the Stevenson family in Scotland, the shift from oil to electric illumination, the invention and spread of the Fresnel lens, and how improvements in cement, cast iron and marine engineering changed what could be built on an exposed rock. Write flowing prose only: no headings, no bullet points, no lists. Never abbreviate, never summarise, and never write a placeholder such as \\\"(the rest of the essay)\\\" or \\\"...\\\" \\u2014 write every sentence out in full.<|end|><|start|>assistant<|channel|>final<|message|>The story of lighthouse building around the North Atlantic begins\",\"backend\":\"CPU (-ngl 0)\",\"sampling\":\"temp 0.7, top-k 20, top-p 0.8, seed 20260906\",\"capture_threads\":6}",
        "steps": 512,
        "demands": 73728,
        "working_set": 2442,
        "peak_step_demand": 144,
        "rows": [
          {
            "slots": 144,
            "dynamic_hit": 0.3422037760416667,
            "static_hit": 0.027777777777777776,
            "static_layers": 1,
            "gap_points": 31.44259982638889,
            "optimal_hit": null,
            "dynamic_staged_bytes": 641104761600,
            "static_staged_bytes": 947552256000,
            "points_per_doubling": null,
            "trivial": false
          },
          {
            "slots": 205,
            "dynamic_hit": 0.4131673177083333,
            "static_hit": 0.027777777777777776,
            "static_layers": 1,
            "gap_points": 38.53895399305555,
            "optimal_hit": null,
            "dynamic_staged_bytes": 571941907200,
            "static_staged_bytes": 947552256000,
            "points_per_doubling": 13.926568870845497,
            "trivial": false
          },
          {
            "slots": 292,
            "dynamic_hit": 0.5052083333333334,
            "static_hit": 0.05555555555555555,
            "static_layers": 2,
            "gap_points": 44.96527777777778,
            "optimal_hit": null,
            "dynamic_staged_bytes": 482236416000,
            "static_staged_bytes": 920479334400,
            "points_per_doubling": 18.03507688469796,
            "trivial": false
          },
          {
            "slots": 416,
            "dynamic_hit": 0.6064995659722222,
            "static_hit": 0.1111111111111111,
            "static_layers": 4,
            "gap_points": 49.53884548611111,
            "optimal_hit": null,
            "dynamic_staged_bytes": 383515430400,
            "static_staged_bytes": 866333491200,
            "points_per_doubling": 19.8370986058209,
            "trivial": false
          },
          {
            "slots": 593,
            "dynamic_hit": 0.7229546440972222,
            "static_hit": 0.16666666666666666,
            "static_layers": 6,
            "gap_points": 55.62879774305556,
            "optimal_hit": null,
            "dynamic_staged_bytes": 270015379200,
            "static_staged_bytes": 812187648000,
            "points_per_doubling": 22.76965534709067,
            "trivial": false
          },
          {
            "slots": 845,
            "dynamic_hit": 0.8229437934027778,
            "static_hit": 0.25,
            "static_layers": 9,
            "gap_points": 57.29437934027778,
            "optimal_hit": null,
            "dynamic_staged_bytes": 172563436800,
            "static_staged_bytes": 730968883200,
            "points_per_doubling": 19.570441302297226,
            "trivial": false
          },
          {
            "slots": 1203,
            "dynamic_hit": 0.9064127604166666,
            "static_hit": 0.4166666666666667,
            "static_layers": 15,
            "gap_points": 48.97460937499999,
            "optimal_hit": null,
            "dynamic_staged_bytes": 91212480000,
            "static_staged_bytes": 568531353600,
            "points_per_doubling": 16.37888008203366,
            "trivial": false
          },
          {
            "slots": 1714,
            "dynamic_hit": 0.9516872829861112,
            "static_hit": 0.6388888888888888,
            "static_layers": 23,
            "gap_points": 31.279839409722232,
            "optimal_hit": null,
            "dynamic_staged_bytes": 47086790400,
            "static_staged_bytes": 351947980800,
            "points_per_doubling": 8.864660618549,
            "trivial": false
          },
          {
            "slots": 2442,
            "dynamic_hit": 0.9668782552083334,
            "static_hit": 1.0,
            "static_layers": 36,
            "gap_points": -3.312174479166663,
            "optimal_hit": null,
            "dynamic_staged_bytes": 32281286400,
            "static_staged_bytes": 0,
            "points_per_doubling": 2.9745620719746784,
            "trivial": true
          }
        ],
        "knee_slots": 1203,
        "card_slots": 612,
        "dynamic_beats_static": true
      },
      {
        "name": "gptoss120b-reasoning.decode.ndjson",
        "provenance": "{\"format\":\"moearc-trace-v1\",\"phase\":\"decode\",\"n_layer\":36,\"n_layers_routed\":36,\"n_expert_used\":4,\"n_prompt_tokens\":151,\"n_prefill_steps\":151,\"n_decode_steps\":512,\"hit_eog\":false,\"model_file\":\"gpt-oss-120b-MXFP4.gguf\",\"quantisation\":\"MXFP4\",\"llama_cpp_commit\":\"e107984bcffcfd701e82738092a2b000b6fda7a2\",\"llama_cpp_patched\":true,\"captured_utc\":\"2026-09-06T15:25:22Z\",\"prompt_name\":\"gptoss120b-reasoning\",\"prompt\":\"<|start|>system<|message|>You are a helpful assistant. Reasoning: high<|end|><|start|>user<|message|>A freight train leaves station A at 6:00 travelling at 40 km/h. A passenger train leaves station B, 300 km away on the same line, at 7:30 travelling towards A at 90 km/h. There is a single siding 180 km from A that can hold one train. Work out, step by step, whether the trains can pass without a collision, at what time and place they would meet on open track, and what the latest departure time from B would be for the passenger train to reach A without the freight ever having to wait more than twenty minutes in the siding. Show all reasoning.<|end|><|start|>assistant<|channel|>analysis<|message|>\\n\",\"backend\":\"CPU (-ngl 0)\",\"sampling\":\"temp 0.7, top-k 20, top-p 0.8, seed 20260904\",\"capture_threads\":6}",
        "steps": 512,
        "demands": 73728,
        "working_set": 3486,
        "peak_step_demand": 144,
        "rows": [
          {
            "slots": 144,
            "dynamic_hit": 0.2269965277777778,
            "static_hit": 0.027777777777777776,
            "static_layers": 1,
            "gap_points": 19.921875,
            "optimal_hit": null,
            "dynamic_staged_bytes": 753388646400,
            "static_staged_bytes": 947552256000,
            "points_per_doubling": null,
            "trivial": false
          },
          {
            "slots": 214,
            "dynamic_hit": 0.2994927300347222,
            "static_hit": 0.027777777777777776,
            "static_layers": 1,
            "gap_points": 27.171495225694443,
            "optimal_hit": null,
            "dynamic_staged_bytes": 682732022400,
            "static_staged_bytes": 947552256000,
            "points_per_doubling": 12.684317891740877,
            "trivial": false
          },
          {
            "slots": 319,
            "dynamic_hit": 0.3873969184027778,
            "static_hit": 0.05555555555555555,
            "static_layers": 2,
            "gap_points": 33.18413628472222,
            "optimal_hit": null,
            "dynamic_staged_bytes": 597058387200,
            "static_staged_bytes": 920479334400,
            "points_per_doubling": 15.262584555146228,
            "trivial": false
          },
          {
            "slots": 476,
            "dynamic_hit": 0.480224609375,
            "static_hit": 0.1111111111111111,
            "static_layers": 4,
            "gap_points": 36.91134982638889,
            "optimal_hit": null,
            "dynamic_staged_bytes": 506586182400,
            "static_staged_bytes": 866333491200,
            "points_per_doubling": 16.076699531634034,
            "trivial": false
          },
          {
            "slots": 709,
            "dynamic_hit": 0.6017930772569444,
            "static_hit": 0.16666666666666666,
            "static_layers": 6,
            "gap_points": 43.51264105902778,
            "optimal_hit": null,
            "dynamic_staged_bytes": 388102492800,
            "static_staged_bytes": 812187648000,
            "points_per_doubling": 21.148813632704382,
            "trivial": false
          },
          {
            "slots": 1055,
            "dynamic_hit": 0.7398681640625,
            "static_hit": 0.2777777777777778,
            "static_layers": 10,
            "gap_points": 46.20903862847222,
            "optimal_hit": null,
            "dynamic_staged_bytes": 253531036800,
            "static_staged_bytes": 703895961600,
            "points_per_doubling": 24.080674329485298,
            "trivial": false
          },
          {
            "slots": 1572,
            "dynamic_hit": 0.8478868272569444,
            "static_hit": 0.4444444444444444,
            "static_layers": 16,
            "gap_points": 40.34423828125,
            "optimal_hit": null,
            "dynamic_staged_bytes": 148253328000,
            "static_staged_bytes": 541458432000,
            "points_per_doubling": 18.7741583748045,
            "trivial": false
          },
          {
            "slots": 2341,
            "dynamic_hit": 0.9255235460069444,
            "static_hit": 0.6666666666666666,
            "static_layers": 24,
            "gap_points": 25.88568793402778,
            "optimal_hit": null,
            "dynamic_staged_bytes": 72586627200,
            "static_staged_bytes": 324875059200,
            "points_per_doubling": 13.513231303175173,
            "trivial": false
          },
          {
            "slots": 3486,
            "dynamic_hit": 0.9527180989583334,
            "static_hit": 1.0,
            "static_layers": 36,
            "gap_points": -4.7281901041666625,
            "optimal_hit": null,
            "dynamic_staged_bytes": 46082131200,
            "static_staged_bytes": 0,
            "points_per_doubling": 4.734035148519969,
            "trivial": true
          }
        ],
        "knee_slots": 1572,
        "card_slots": 612,
        "dynamic_beats_static": true
      }
    ],
    "skipped": [
      [
        "gptoss120b-code.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss120b-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss120b-reasoning.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss20b-code.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss20b-code.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss20b-prose.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss20b-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "gptoss20b-reasoning.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "gptoss20b-reasoning.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen3-30b-fibonacci.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen3-30b-fibonacci.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen3-30b-prose.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen3-30b-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen35moe-code.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen35moe-code.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen35moe-prose.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen35moe-prose.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ],
      [
        "qwen35moe-reasoning.decode.ndjson",
        "not captured from `gpt-oss-120b-MXFP4.gguf` — a hit rate describes one model's routing and does not transfer to another (PROTOCOL §9). Pass --all-traces to replay it anyway."
      ],
      [
        "qwen35moe-reasoning.prefill.ndjson",
        "prefill capture — residency is a decode-time question, and a prefill trace is short enough to be dominated by compulsory misses (bench/traces/README.md)"
      ]
    ]
  },
  "absolutes": [
    {
      "depth": 128,
      "generated_tokens": 64,
      "residency": "600",
      "host_policy": "frac:0.5",
      "threads_requested": 19,
      "cold": {
        "label": "moearc decode, cold pool, depth 128",
        "unit": "tok/s",
        "values": [
          9.61727304004435,
          12.186579072007142,
          11.940950763439313
        ]
      },
      "warm": {
        "label": "moearc decode, warm pool, depth 128",
        "unit": "tok/s",
        "values": [
          12.157191806945407,
          12.246566791342431,
          12.006839960533997
        ]
      },
      "cold_hit_rate": {
        "label": "cold hit rate, depth 128",
        "unit": "fraction",
        "values": [
          0.7183023590180521,
          0.7183023590180521,
          0.7183023590180521
        ]
      },
      "warm_hit_rate": {
        "label": "warm hit rate, depth 128",
        "unit": "fraction",
        "values": [
          0.7269595633048916,
          0.7269595633048916,
          0.7269595633048916
        ]
      },
      "warm_over_cold": 1.0789987036868194,
      "invocations": [
        {
          "load1_before": 0.83,
          "load1_after": 9.31,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 19,
          "resident_slots": 600,
          "total_slots": 4608,
          "slot_bytes": 13219200,
          "n_ctx": 4096,
          "depth": 128,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 6.550713465,
            "decode_tok_s": 9.61727304004435,
            "prefill_seconds": 13.595156383,
            "hit_rate": 0.7183023590180521,
            "demands": 18779,
            "staged_bytes": 69929568000,
            "token_ids": [
              6240,
              17,
              13,
              20,
              35971,
              4,
              328,
              290,
              3609,
              2359,
              410,
              558,
              29,
              350,
              976,
              220,
              6106,
              33,
              2359,
              382,
              220,
              6106,
              35971,
              65,
              27032,
              9621,
              11,
              220,
              4621,
              35971,
              49130,
              33,
              306,
              220,
              1125,
              50005,
              6516,
              6011,
              32616,
              523,
              29,
              91349,
              6240,
              70,
              555,
              12,
              2907,
              12,
              1130,
              33,
              2733,
              220,
              1055,
              13,
              15,
              15507,
              33,
              328,
              28460,
              2733,
              13719,
              402,
              448,
              220
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 5.182117795,
            "decode_tok_s": 12.157191806945407,
            "prefill_seconds": 9.762772551,
            "hit_rate": 0.7269595633048916,
            "demands": 18869,
            "staged_bytes": 68105318400,
            "token_ids": [
              6240,
              17,
              13,
              20,
              35971,
              4,
              328,
              290,
              3609,
              2359,
              410,
              558,
              29,
              350,
              976,
              220,
              6106,
              33,
              2359,
              382,
              220,
              6106,
              35971,
              65,
              27032,
              9621,
              11,
              220,
              4621,
              35971,
              49130,
              33,
              306,
              220,
              1125,
              50005,
              6516,
              6011,
              32616,
              523,
              29,
              91349,
              6240,
              70,
              555,
              12,
              2907,
              12,
              1130,
              33,
              2733,
              220,
              1055,
              13,
              15,
              15507,
              33,
              328,
              28460,
              2733,
              13719,
              402,
              448,
              220
            ]
          },
          "io": {
            "disk_read_bytes": 1622355968,
            "arc_hits": 664558,
            "arc_misses": 1567
          },
          "phases": []
        },
        {
          "load1_before": 2.41,
          "load1_after": 10.19,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 19,
          "resident_slots": 600,
          "total_slots": 4608,
          "slot_bytes": 13219200,
          "n_ctx": 4096,
          "depth": 128,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 5.169621403,
            "decode_tok_s": 12.186579072007142,
            "prefill_seconds": 12.312049006,
            "hit_rate": 0.7183023590180521,
            "demands": 18779,
            "staged_bytes": 69929568000,
            "token_ids": [
              6240,
              17,
              13,
              20,
              35971,
              4,
              328,
              290,
              3609,
              2359,
              410,
              558,
              29,
              350,
              976,
              220,
              6106,
              33,
              2359,
              382,
              220,
              6106,
              35971,
              65,
              27032,
              9621,
              11,
              220,
              4621,
              35971,
              49130,
              33,
              306,
              220,
              1125,
              50005,
              6516,
              6011,
              32616,
              523,
              29,
              91349,
              6240,
              70,
              555,
              12,
              2907,
              12,
              1130,
              33,
              2733,
              220,
              1055,
              13,
              15,
              15507,
              33,
              328,
              28460,
              2733,
              13719,
              402,
              448,
              220
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 5.144298894,
            "decode_tok_s": 12.246566791342431,
            "prefill_seconds": 9.847234804,
            "hit_rate": 0.7269595633048916,
            "demands": 18869,
            "staged_bytes": 68105318400,
            "token_ids": [
              6240,
              17,
              13,
              20,
              35971,
              4,
              328,
              290,
              3609,
              2359,
              410,
              558,
              29,
              350,
              976,
              220,
              6106,
              33,
              2359,
              382,
              220,
              6106,
              35971,
              65,
              27032,
              9621,
              11,
              220,
              4621,
              35971,
              49130,
              33,
              306,
              220,
              1125,
              50005,
              6516,
              6011,
              32616,
              523,
              29,
              91349,
              6240,
              70,
              555,
              12,
              2907,
              12,
              1130,
              33,
              2733,
              220,
              1055,
              13,
              15,
              15507,
              33,
              328,
              28460,
              2733,
              13719,
              402,
              448,
              220
            ]
          },
          "io": {
            "disk_read_bytes": 278691840,
            "arc_hits": 151378,
            "arc_misses": 279
          },
          "phases": []
        },
        {
          "load1_before": 2.44,
          "load1_after": 10.0,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 19,
          "resident_slots": 600,
          "total_slots": 4608,
          "slot_bytes": 13219200,
          "n_ctx": 4096,
          "depth": 128,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 5.2759617930000005,
            "decode_tok_s": 11.940950763439313,
            "prefill_seconds": 10.279501243,
            "hit_rate": 0.7183023590180521,
            "demands": 18779,
            "staged_bytes": 69929568000,
            "token_ids": [
              6240,
              17,
              13,
              20,
              35971,
              4,
              328,
              290,
              3609,
              2359,
              410,
              558,
              29,
              350,
              976,
              220,
              6106,
              33,
              2359,
              382,
              220,
              6106,
              35971,
              65,
              27032,
              9621,
              11,
              220,
              4621,
              35971,
              49130,
              33,
              306,
              220,
              1125,
              50005,
              6516,
              6011,
              32616,
              523,
              29,
              91349,
              6240,
              70,
              555,
              12,
              2907,
              12,
              1130,
              33,
              2733,
              220,
              1055,
              13,
              15,
              15507,
              33,
              328,
              28460,
              2733,
              13719,
              402,
              448,
              220
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 5.247009222,
            "decode_tok_s": 12.006839960533997,
            "prefill_seconds": 9.80441344,
            "hit_rate": 0.7269595633048916,
            "demands": 18869,
            "staged_bytes": 68105318400,
            "token_ids": [
              6240,
              17,
              13,
              20,
              35971,
              4,
              328,
              290,
              3609,
              2359,
              410,
              558,
              29,
              350,
              976,
              220,
              6106,
              33,
              2359,
              382,
              220,
              6106,
              35971,
              65,
              27032,
              9621,
              11,
              220,
              4621,
              35971,
              49130,
              33,
              306,
              220,
              1125,
              50005,
              6516,
              6011,
              32616,
              523,
              29,
              91349,
              6240,
              70,
              555,
              12,
              2907,
              12,
              1130,
              33,
              2733,
              220,
              1055,
              13,
              15,
              15507,
              33,
              328,
              28460,
              2733,
              13719,
              402,
              448,
              220
            ]
          },
          "io": {
            "disk_read_bytes": 317046784,
            "arc_hits": 238802,
            "arc_misses": 326
          },
          "phases": []
        }
      ]
    },
    {
      "depth": 512,
      "generated_tokens": 64,
      "residency": "600",
      "host_policy": "frac:0.5",
      "threads_requested": 19,
      "cold": {
        "label": "moearc decode, cold pool, depth 512",
        "unit": "tok/s",
        "values": [
          6.4647826127817165,
          12.650260669705135,
          12.513072230283557
        ]
      },
      "warm": {
        "label": "moearc decode, warm pool, depth 512",
        "unit": "tok/s",
        "values": [
          12.731608892065736,
          12.628293281941488,
          12.662626936800194
        ]
      },
      "cold_hit_rate": {
        "label": "cold hit rate, depth 512",
        "unit": "fraction",
        "values": [
          0.7567218867729782,
          0.7567218867729782,
          0.7567218867729782
        ]
      },
      "warm_hit_rate": {
        "label": "warm hit rate, depth 512",
        "unit": "fraction",
        "values": [
          0.7594240747740627,
          0.7594240747740627,
          0.7594240747740627
        ]
      },
      "warm_over_cold": 1.2021749792666323,
      "invocations": [
        {
          "load1_before": 2.44,
          "load1_after": 18.13,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 19,
          "resident_slots": 600,
          "total_slots": 4608,
          "slot_bytes": 13219200,
          "n_ctx": 4096,
          "depth": 512,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 9.745107264,
            "decode_tok_s": 6.4647826127817165,
            "prefill_seconds": 55.629806201,
            "hit_rate": 0.7567218867729782,
            "demands": 58131,
            "staged_bytes": 186945926400,
            "token_ids": [
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220,
              17,
              11,
              45404,
              13071,
              1698,
              79795,
              198,
              74924,
              220,
              81033,
              3465,
              3690,
              68,
              12,
              16,
              65,
              12,
              22,
              65,
              12,
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 4.948314116,
            "decode_tok_s": 12.731608892065736,
            "prefill_seconds": 46.213301839,
            "hit_rate": 0.7594240747740627,
            "demands": 58202,
            "staged_bytes": 185095238400,
            "token_ids": [
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220,
              17,
              11,
              45404,
              13071,
              1698,
              79795,
              198,
              74924,
              220,
              81033,
              3465,
              3690,
              68,
              12,
              16,
              65,
              12,
              22,
              65,
              12,
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220
            ]
          },
          "io": {
            "disk_read_bytes": 11493232640,
            "arc_hits": 3964732,
            "arc_misses": 11047
          },
          "phases": []
        },
        {
          "load1_before": 2.4,
          "load1_after": 16.93,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 19,
          "resident_slots": 600,
          "total_slots": 4608,
          "slot_bytes": 13219200,
          "n_ctx": 4096,
          "depth": 512,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 4.980134532,
            "decode_tok_s": 12.650260669705135,
            "prefill_seconds": 46.344481368,
            "hit_rate": 0.7567218867729782,
            "demands": 58131,
            "staged_bytes": 186945926400,
            "token_ids": [
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220,
              17,
              11,
              45404,
              13071,
              1698,
              79795,
              198,
              74924,
              220,
              81033,
              3465,
              3690,
              68,
              12,
              16,
              65,
              12,
              22,
              65,
              12,
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 4.988797662,
            "decode_tok_s": 12.628293281941488,
            "prefill_seconds": 41.305593673,
            "hit_rate": 0.7594240747740627,
            "demands": 58202,
            "staged_bytes": 185095238400,
            "token_ids": [
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220,
              17,
              11,
              45404,
              13071,
              1698,
              79795,
              198,
              74924,
              220,
              81033,
              3465,
              3690,
              68,
              12,
              16,
              65,
              12,
              22,
              65,
              12,
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220
            ]
          },
          "io": {
            "disk_read_bytes": 488955904,
            "arc_hits": 119585,
            "arc_misses": 679
          },
          "phases": []
        },
        {
          "load1_before": 2.4,
          "load1_after": 16.86,
          "device": "Intel(R) Arc(TM) B580 Graphics",
          "host_threads": 19,
          "resident_slots": 600,
          "total_slots": 4608,
          "slot_bytes": 13219200,
          "n_ctx": 4096,
          "depth": 512,
          "cold": {
            "decode_steps": 63,
            "decode_seconds": 5.034734783,
            "decode_tok_s": 12.513072230283557,
            "prefill_seconds": 50.856827392,
            "hit_rate": 0.7567218867729782,
            "demands": 58131,
            "staged_bytes": 186945926400,
            "token_ids": [
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220,
              17,
              11,
              45404,
              13071,
              1698,
              79795,
              198,
              74924,
              220,
              81033,
              3465,
              3690,
              68,
              12,
              16,
              65,
              12,
              22,
              65,
              12,
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220
            ]
          },
          "warm": {
            "decode_steps": 63,
            "decode_seconds": 4.975270954,
            "decode_tok_s": 12.662626936800194,
            "prefill_seconds": 41.806185853,
            "hit_rate": 0.7594240747740627,
            "demands": 58202,
            "staged_bytes": 185095238400,
            "token_ids": [
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220,
              17,
              11,
              45404,
              13071,
              1698,
              79795,
              198,
              74924,
              220,
              81033,
              3465,
              3690,
              68,
              12,
              16,
              65,
              12,
              22,
              65,
              12,
              17,
              13,
              20,
              83,
              983,
              2335,
              19,
              20555,
              256,
              220,
              17,
              13,
              20,
              15507,
              33,
              256,
              220,
              16,
              11,
              35367,
              820,
              220
            ]
          },
          "io": {
            "disk_read_bytes": 1828323328,
            "arc_hits": 1045246,
            "arc_misses": 1852
          },
          "phases": []
        }
      ]
    }
  ],
  "incumbent": null,
  "not_measured": [
    "llama.cpp as a baseline — pass --llama-bench <PATH>. It is never searched for: PROTOCOL §2 records a glob-ordered pick that silently selected a Vulkan build 4.8x slower than SYCL.",
    "staging-versus-attention attribution with prompt depth (PROTOCOL §0 claim 2). It needs a synchronous device profile (`MOEARC_SYNC_EACH=1`) at two depths on a model large enough for staging to bind, which is a far longer run than this command takes; `bench/baselines/gpt-oss-120b.md` §6.4 carries the measurement."
  ]
}
```

</details>
