# Benchmark protocol

Every number in the README comes from `moearc bench` and nothing else.

## Hardware of record

M0 is measured on this box and no other. A number from different hardware is a different number.

| | |
|---|---|
| Host | Reference box — Core Ultra 7 265K, 20 cores, 91 GB RAM (81 GB free at bench time) |
| GPU | **Intel Arc B580, 12 GiB VRAM** (11.93 GiB usable, stolen excluded), `xe` driver, `level_zero:0`, `renderD129` |
| OS | Ubuntu 26.04.1, kernel 7.0.0-30, oneAPI DPC++ 2026.1.1 |
| Not used | The Arrow Lake iGPU on `renderD128`. It has 4 Xe-cores and **no XMX**. It is a compile target, never a baseline. |

🔴 **12 GiB of VRAM is the governing constraint of this entire project.** Every candidate model
below is 18–21 GB at Q4 — all of them overflow the card. MoE experts therefore stream from system
RAM on every token. That is precisely the bottleneck MoEArc exists to attack, and it is why the
CPU/GPU split must be **measured, not assumed** (see M0 procedure).

## Metrics

| Metric | How |
|---|---|
| Decode tok/s | 512 tokens at KV depths 0, 4k, 16k |
| Prefill tok/s | 4k **and 16k** prompts |
| TTFT | 4k prompt |
| VRAM / RAM | peak resident |
| Power | wall watts (smart plug) + Level Zero GPU power → tok/s/W |
| Correctness | wikitext-2 perplexity within 0.5% of llama.cpp, same quant |

Always reported against llama.cpp **SYCL** and **Vulkan** on the identical box and commit hash.

## Rules

- A result without the box, the commit hash and the quant is not a result.
- `baselines/` holds llama.cpp numbers. `results/` holds ours. Both are committed.
- No "should be faster" without a number. Measure, change, measure.
- If a change makes nothing faster, that is a finding — commit it anyway.
- **No tuning parameter gets a value without a measurement or a stated reason.** See the flag
  table below. This rule exists because the first version of this file failed it — read on.

## Models

Two models are baselined, because they answer different questions.

| Role | Model | Q4_K_M | Why |
|---|---|---|---|
| **Control** | Qwen3-30B-A3B-Instruct-2507 (30.5B / 3.3B active, Apache 2.0) | 18.6 GB | The most widely published MoE benchmark target in existence. If our numbers are far off public ones, **our harness is wrong** — not the hardware. Nothing else gives that calibration. |
| **Target** | Qwen3.6-35B-A3B (36B / 3B active, Apache 2.0) | ~21 GB | The current intelligence leader in the ~3B-active class and what people will actually run. At 21 GB against 12 GiB it stresses expert streaming *harder* — baselining only on the smaller model would understate the problem we claim to solve. |

Optional third, for coding-specific comparison only: **GLM-4.7-Flash** (30B/3B, MIT, 18.3 GB).
It needs `--jinja` and the January 2026 llama.cpp sigmoid-scoring fix; our reference build is at
today's HEAD so the fix is present.

## 🔴 Device selection — the highest-risk step in this protocol

**The B580 is not the default device on either backend, and the index differs between them.**

| Backend | B580 is | Default (device 0) is |
|---|---|---|
| SYCL | `level_zero:0` | the B580 |
| **Vulkan** | **`Vulkan1`** | **the Arrow Lake iGPU** |

```
Vulkan0: Intel(R) Graphics (ARL)   70279 MiB   matrix cores: none
Vulkan1: Intel(R) Arc(tm) B580     12216 MiB   matrix cores: KHR_coopmat
```

⚠️ **A wrong-device Vulkan run does not fail — it succeeds and lies.** The iGPU is UMA, so it
advertises **70 GB** of memory and will load an 18–21 GB model without ever OOMing. It has no
matrix cores. The result looks like a baseline and is worthless, measured on precisely the
hardware this protocol forbids ("the iGPU is a compile target, not a baseline").

**Therefore every Vulkan invocation must pin the device explicitly**, and every recorded result
must state which device it ran on. A result that does not name the device is not a result.

### SYCL on Battlemage — RESOLVED 2026-09-03. It was our packaging, not upstream.

⚠️ **This section previously declared the SYCL backend broken on Battlemage. That was wrong**, and
the claim survived six ruled-out hypotheses before the real cause turned up. Recorded because the
wrong diagnosis briefly became an argument about the project's direction.

**Symptom:** `llama-bench` aborted inside `llama_model_load_from_file`:
`ggml-sycl.cpp [ggml_backend_sycl_device_get_memory] failed to get device memory size`.

**Real cause:** `libze-dev` was not installed. `libze_loader.so.1` (runtime, from `libze1`) was
present, but with no `.so` symlink or headers, cmake's `find_library(ZE_LOADER_LIB ze_loader)`
**failed silently** — so the build printed `GGML_SYCL_SUPPORT_LEVEL_ZERO_API ON` while linking
nothing. Confirmed with `ldd | grep ze_loader` on the built library, which is the check that
should have been run first.

📌 **A cmake option reporting `ON` does not mean the feature is present.** `ldd` the artifact.

**Fix:** `sudo apt install -y libze-dev`, then build with `-DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON`.

**Ruled out first, all wrong** — kept so nobody re-runs them: `GGML_SYCL_GET_MEM_API=1`,
`ZES_ENABLE_SYSMAN=1`, both together, `ONEAPI_DEVICE_SELECTOR=level_zero:0`,
`ONEAPI_DEVICE_SELECTOR=opencl:gpu`, and rebuilding with the flag but without `libze-dev`.

### Reference builds are PINNED

Both backends must be built at the same commit. The SYCL script originally did
`reset --hard origin/master` on every run, which silently moved the tree and **overwrote the pin
file**, leaving the two references on different commits. It now checks out
the pinned-commit file (`llama.cpp-COMMIT`, beside the checkout) and only writes that file when absent. Move the pin
deliberately: `LLAMACPP_COMMIT=<sha> ./build-llamacpp-sycl.sh`.

### Smoke-test numbers (NOT M0 — 0.5B model, tiny params, for plumbing only)

| Backend | device | pp128 | tg32 |
|---|---|---|---|
| SYCL | B580 (`level_zero:0`) | 5266.78 | 184.42 |
| Vulkan | B580 (`Vulkan1`) | 4377.58 | 160.24 |
| Vulkan | iGPU (`Vulkan0`) | 1195.95 | 29.62 |

🔴 These are **not results** — wrong model, wrong parameters, no sweep. They exist only to prove
the harness runs and to quantify the wrong-device trap (**5.4× on decode**).

## M0 baseline

The number to beat. Recorded **before** any new code is written.

**M0 is llama.cpp at its best, not llama.cpp at an arbitrary setting.** The M2 gate says that if
we cannot beat this, we stop and say so — which is only meaningful against a baseline that was
actually tuned. A hobbled baseline makes any later win fake, and fake is worse than slow.

### Procedure

**1. Sweep the CPU/GPU MoE split.** This is the single parameter that decides the result on a
12 GiB card, and it differs per model:

```bash
for N in 0 4 8 12 16 20 24 28 32 36; do
  llama-bench -m "$MODEL" -ngl 999 -ncmoe "$N" \
    -fa 1 -ctk q8_0 -ctv q8_0 -p 4096 -n 512 -ub 1024 \
    -o json >> "sweep-$(basename "$MODEL" .gguf).json"
done
```

Low `-ncmoe` will OOM on a 12 GiB card; that is expected and is itself data. Record the failures.

**2. Take the winning `-ncmoe` and run the full protocol:**

```bash
llama-bench -m "$MODEL" -ngl 999 -ncmoe "$BEST" \
  -fa 1 -ctk q8_0 -ctv q8_0 \
  -p 4096,16384 -n 512 -d 0,4096,16384 -ub 1024
```

**3. Repeat both steps for the Vulkan backend**, per the metrics table.

**4. Commit** the sweep JSON alongside the headline number. The sweep is the evidence that the
baseline is honest.

### Every flag, justified

| Flag | Value | Why |
|---|---|---|
| `-ngl 999` | all layers | Offload everything to GPU, then pull MoE layers back with `-ncmoe`. Standard llama.cpp idiom. |
| `-ncmoe` | **swept** | Decides the CPU/GPU split. Governs the result on a 12 GiB card. Never guessed. |
| `-fa 1` | on | Flash attention; strictly faster and lower memory where supported. |
| `-ctk`/`-ctv` | `q8_0` | Quantized KV cache. Chosen because 16k-depth decode is a reported metric and an f16 cache costs VRAM we do not have. ⬜ Costs some quality — confirm against the perplexity gate before treating as settled. |
| `-ub 1024` | 1024 | ⬜ **Not yet justified.** Physical batch size; affects prefill throughput. Sweep it once alongside `-ncmoe`, then fix it with a reason or drop it. |
| `-p 4096,16384` | both | The metrics table promises prefill at 4k **and** 16k. |
| `-n 512` | 512 | Matches the decode metric. |
| `-d 0,4096,16384` | three depths | Matches the decode-at-depth metric. |

## ⚠️ Provenance of the previous version of this file

The original M0 command in this file specified **`-ncmoe 8`**. That value was:

- written in commit `bf61c74`, **2026-09-02 15:50 UTC**
- while the Arc B580 was first bound by the `xe` driver at **2026-09-03 01:08 UTC** — **9h18m later**
- present exactly **once** in the whole repository, justified **nowhere**

It was authored by the AI agent working on this repo (committed under the owner's git identity),
and it was a plausible-looking integer chosen with no hardware to choose it against. It was
presented in the same voice as measured values.

The same version also promised a 16k prefill number in the metrics table that its command
**never generated** (`-p 4096` only).

📌 This note stays until M0 is measured. It is the reason for the "no value without a measurement
or a stated reason" rule above — an unjustified constant written down in a confident voice is
indistinguishable from a result, and this project's credibility rests entirely on that difference.

## 🔴 MTP — a second way to understate the baseline

Unsloth ships **MTP (Multi-Token Prediction)** builds of Qwen3.6-35B-A3B, and llama.cpp drives them
with `--spec-type draft-mtp --spec-draft-n-max N`. Upstream explicitly warns *"do not assume 2 is
optimal"* — i.e. **N must be swept, exactly like `-ncmoe`.**

MTP can materially raise decode throughput. **If M0 is measured without it while MoEArc is later
measured with anything comparable, the win is manufactured.** This is the same defect as the
invented `-ncmoe 8`, in a second parameter.

⬜ **Decide before M0 is recorded**, and state the decision in the result:
- **Option A — baseline both.** Record plain and MTP separately. M2 must then beat the *better* one.
  Honest, and roughly doubles the sweep.
- **Option B — plain only.** Defensible only if MoEArc will never use speculative decoding, and the
  result must say so explicitly so nobody compares across the line later.

📌 The rule this falls out of: *M0 is llama.cpp at its best, not llama.cpp at a convenient setting.*

## Cross-engine comparison — including FreeToken

We want to be measured against the prior art, not just against ourselves. But the obvious
comparison is **not available**, and pretending otherwise would produce a dishonest chart.

🔴 **FreeToken and MoEArc cannot be run on the same hardware.** FreeToken is CUDA-only; it
supports RTX 30/40/50 and nothing else. An attempt to run it on Arc got as far as loading a
23 GB model and serving `/v1/models` before hanging in `causal_conv1d_varlen` during warmup.
So there is no same-card head-to-head to be had, and any number putting the two engines
side by side is comparing two different pieces of silicon.

That leaves three comparisons, in descending order of how much they prove:

1. **MoEArc vs llama.cpp SYCL, same Arc card, same model, same quantisation.** This is the
   real gate and the only true apples-to-apples number. llama.cpp's CPU/GPU split is static;
   ours is planned. If we cannot beat a hand-tuned `--n-cpu-moe` on the same hardware, we have
   not earned our existence.
2. **MoEArc vs Ollama/Vulkan on the same Arc card.** The path most Arc owners are actually on
   today, so it is the number that describes what a user gains by switching.
3. **MoEArc on Arc vs FreeToken's published RTX figures.** Cross-vendor and cross-silicon —
   informative about whether the *approach* transfers, worthless as a claim about which engine
   is faster. Any use of it must say so in the same breath, and must state both cards' price
   and memory bandwidth, since those explain most of any gap.

📌 Rule: **never publish (3) without (1).** A cross-vendor chart with no same-card baseline
next to it reads as a performance claim regardless of the caveat printed under it.

## Open gaps

- ⬜ **No Vulkan build yet.** The metrics table requires llama.cpp SYCL **and** Vulkan; only the
  SYCL reference build exists so far. M0 is incomplete until both are measured.
- ⬜ **Power measurement has no instrument.** `tok/s/W` needs a smart plug that is not yet in place;
  Level Zero GPU power alone is not wall power.
- ⬜ **`-ub` unjustified** (see flag table).
- ⬜ **KV-cache `q8_0` quality cost unquantified** — must pass the perplexity gate.
