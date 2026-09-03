# Archaeology — mining the Intel Arc prior art

**Milestone M0.5. Survey updated 2026-09-03 — FreeToken added as the primary source.**

## What MoEArc is

**MoEArc is FreeToken for Intel GPUs.** [FreeToken](https://github.com/FlashML-org/FreeToken)
is an Apache-2.0 edge-native MoE serving engine that makes frontier-scale MoE models run on
consumer hardware through bandwidth-adaptive CPU–GPU co-execution. It targets **NVIDIA RTX
30/40/50 only**. Arc owners have no equivalent. That gap is the entire project.

This reframes the archaeology. There are **three sources with three distinct roles**, and
conflating them is what sent the original plan to the wrong repository:

| Source | Role | What we take |
|---|---|---|
| **FreeToken** | **The design** | Scheduling policy, expert cache, elastic memory — the architecture |
| **llama.cpp `ggml-sycl`** | **The kernels** | Working SYCL quant/MoE/attention kernels, today |
| **IPEX `xpu-main`** | **The substrate** | Xe-specific technique, XMX paths, torch-XPU device handling |

📌 The original plan's error was asking **one** repository to be all three. `ipex-llm` was
none of them.

---

## 0. `FlashML-org/FreeToken` — the reference implementation ✅ PRIMARY

The thing we are porting. Measured 2026-09-03 on a full clone (not `--depth 1` — that
shortcut is what produced a false negative on IPEX):

| | |
|---|---|
| License | **Apache-2.0** — clean to study and adapt |
| Commits | 51 on `main`, **last commit 2026-09-03** |
| Branches | 9 active feature branches |
| Traction | ~11.4k stars, ~1.1k forks |

**Code inventory — the number that matters:**

```
Python     116,709 lines / 499 files    ← the policies. Backend-agnostic.
CUDA/C++    13,050 lines /  24 files    ← the only part that needs porting
131 of 499 Python files (26%) reference CUDA/NVIDIA
```

**The 13k lines of CUDA are not one problem, they are four — and only one is hard:**

| Pile | Lines | Porting reality |
|---|---|---|
| GGUF quant kernels — `mmq.cuh`, `mmvq.cuh`, `vecdotq.cuh`, `moe.cuh`, `moe_vec.cuh`, `dequantize.cuh`, `ggml-common.h`, `gguf_kernel.cu` | **~7,500** | 🔴 **These are llama.cpp's kernel family, by name.** `ggml-sycl` already ships `mmq`, `mmvq`, `topk-moe`. **Adapt existing SYCL — do not port CUDA.** |
| CPU MoE — `cpu_moe_ext.cpp` | 2,160 | CPU-side. Largely portable as-is. |
| NCCL multi-GPU — `nccl227.h`, `pynccl.cu` | 759 | **Skip.** Single-card target. |
| JIT / infra — `fast_index_copy.cuh`, `index.cu`, `store.cu`, `batch_memcpy.cuh`, `pinned_tensor.cpp`, `ple_store_ext.cpp`, `tensor.h` | ~2,600 | Needs SYCL equivalents. Memcpy/indexing helpers, not novel math. |

**The genuinely novel GPU math to author is a small fraction of 13k lines**, and its largest
component has a SYCL counterpart already building on the reference box.

⚠️ **Two real risks, stated plainly:**

1. **26% of the Python touches CUDA** — ~131 files of device management to migrate to
   `torch.xpu`. Mechanical rather than novel, but it is not free.
2. **FreeToken is a fast-moving target.** 51 commits, last one *today*, 9 open branches.
   Cloning a live upstream means permanent rebase pressure. Decide early whether this is a
   **hard fork** or a **tracked port**, and record the decision — it governs everything.

⬜ **Unverified:** that `ggml-sycl` covers *every* kernel FreeToken needs. That is the next
concrete check, and it is cheap.

---

---

## Survey results

### 1. `intel/ipex-llm` — documentation only. No kernels, ever.

| | |
|---|---|
| HEAD | `de6bce27133ab250f13fd5d549c197519ce16d30` |
| Archived | 2026-01-28 |
| Commits | 4,113 |
| License | Apache-2.0 |

- **668 `.py`, 409 `.md`, 122 `.txt`, 104 `.sh` — and 2 C/C++ files, neither SYCL.**
- **Zero C/C++/SYCL files have ever existed in the repository**, verified across all
  branches and all history (`git log --all --diff-filter=A -- "*.cpp" "*.hpp" "*.cu"`).
- No `.gitmodules`, so nothing was hidden behind an uninitialised submodule.
- FlashMoE appears in exactly **7 files, every one of them a README or quickstart
  `.md`**. There is no FlashMoE implementation here.
- The 6 `.patch` files are oneCCL / vLLM / torch / Docker serving patches — **not**
  the llama.cpp SYCL backend patches the plan hoped for.

FlashMoE shipped as **prebuilt binaries** in the "portable zip" and PyPI wheels.
The source was never published in this repo.

**Value retained:** `docs/mddocs/Quickstart/flashmoe_quickstart.md` documents which
models ran on which cards with which flags. That is offload-*policy* evidence, and
it still informs q\*. It is documentation-grade insight, not a kernel head start.

### 2. `intel/intel-extension-for-pytorch` @ `xpu-main` — the actual quarry ✅

A **different repository** from ipex-llm, and the one that carries Intel's SYCL.
The default branch is CPU-only; the GPU code lives on separate branches (82 in
total, including `xpu-main` and 17 `release/xpu/*`).

| | |
|---|---|
| Branch | `xpu-main` |
| HEAD | `d0f992ff0ce79fb78581828a285e2cb56caab4e8` (2025-12-18) |
| License | Apache-2.0 — clean lift into this project |

- **983 C/C++ files, 203 using `sycl::`**
- **15 files using `joint_matrix` / XMX / DPAS** — the systolic-array path
- **125 files referencing `sym_int4` / `woq` / `dequant`** — the low-bit weight paths

Layout of the SYCL sources:

```
93  csrc/gpu/aten          ← the bulk: operators and kernels
17  csrc/gpu/deepspeed
 4  csrc/gpu/runtime       ← device/queue/USM management
 2  csrc/gpu/oneDNN
 2  csrc/gpu/mamba
 2  csrc/gpu/bitsandbytes
 1  csrc/gpu/vllm
```

This is what §M0.5 was actually describing. Frozen December 2025, readable,
Apache-2.0, and targeted at our hardware.

### 3. `ggml-org/llama.cpp` @ `ggml/src/ggml-sycl/` — alive, and further along than assumed ⚠️

| | |
|---|---|
| HEAD at survey | `7798007` (2026-09-02 — *today*) |
| License | MIT (permissive; compatible with Apache-2.0, attribution required) |
| SYCL files | 117 of 1,015 C/C++ sources |

Kernels already present:

- **`topk-moe.cpp` / `.hpp`** — MoE top-k routing, in SYCL, today
- **`mmvq.cpp`** — quantised mat-vec, the decode-path kernel
- **`mmq.cpp`**, `dmmv.cpp`, `dequantize.hpp`, `quants.hpp`, `convert.cpp`
- **`fattn.cpp`, `fattn-tile.cpp`, `fattn-mkl.cpp`, `fattn-onednn.cpp`,
  `fattn-buffers.cpp`** — flash attention, several backends
- `fusion.cpp`

**This cuts both ways and the second half is the uncomfortable one.** Less to write
than the plan assumed — but the M2 gate is *"≥1.5× llama.cpp SYCL decode, ≥3×
prefill"*, and the thing we must beat already ships a MoE top-k kernel and
quantised mat-vec, and gains commits daily. **The baseline is stronger and it is a
moving target.** Pin the comparison commit hash in every benchmark or the gate
means nothing.

---

## What this does to the plan

| Plan v4 said | Reality |
|---|---|
| Mine `ipex-llm` for kernels | No kernels there. Mine `intel-extension-for-pytorch@xpu-main` instead |
| "Frozen, readable, Apache-2.0 code" | True — of IPEX xpu-main, not ipex-llm |
| "2–4 months saved, if the code builds" | Unproven. Nothing has been built yet. Treat as zero until a kernel compiles |
| Start from llama.cpp SYCL FA | Still right — and llama.cpp already has more than FA |
| (FreeToken not mentioned) | **It is the design we are porting.** Primary source, Apache-2.0, 116k lines of portable policy |
| "We take the kernels, not the architecture" (of IPEX) | **Inverted for FreeToken: take the architecture, not the kernels** |

The estimated saving should be carried at **zero** until something actually
compiles against oneAPI 2026.1. It is a hypothesis, not a credit.

---

## The five questions — still open

### 1. Offload policy — now answerable from a working implementation
FreeToken states its approach: **bandwidth-adaptive CPU–GPU co-execution (`q*` policy),
global LRU expert caching, elastic VRAM reallocation between expert cache and KV, full-layer
double-buffered prefill streaming.** Read those four mechanisms in the Python and write down
the actual decision rule, thresholds and calibration procedure for each.
**Source of record:** FreeToken's Python. The ipex-llm quickstart docs remain a weak
secondary (policy as *documented* rather than implemented).
📌 This question was previously unanswerable from the listed sources. It now has a
reference implementation, which is the single biggest change to this milestone.

### 2. Kernels to lift
Catalog IPEX `csrc/gpu/aten` and llama.cpp `ggml-sycl` for quantised GEMV/GEMM,
attention, and elementwise fusions. For each: builds against oneAPI 2026.1? uses
XMX? perf vs llama.cpp master on the reference box? → **lift / adapt / rewrite**.

### 3. Xe-specific tricks
Sub-group sizes, work-group tiling, USM vs buffer, Alchemist → Battlemage handling,
driver workarounds. Concentrated in the 15 `joint_matrix` files.

### 4. Weight format
IPEX's `sym_int4` / `woq` layouts vs GGUF vs our flat format.

### 5. What to avoid
IPEX is a PyTorch extension — heavy runtime, Python-first. **From IPEX we take the kernels,
not the architecture.** From FreeToken it is the **exact inverse**: take the architecture,
leave the CUDA.
⚠️ And do not inherit FreeToken's NVIDIA-only assumptions wholesale — NCCL multi-GPU (759
lines) is dead weight on a single-card Arc box and should never be ported.

### 6. Fork or track? 🆕
FreeToken gained 51 commits and 9 branches and committed again *the day we surveyed it*.
**Hard fork** (freeze, diverge, own it) or **tracked port** (rebase onto upstream forever)?
This is a governance decision, not a technical one, and it must be made before M1 — it
determines whether upstream's velocity is an asset or a tax.

## Gate

This document merged with a shortlist of **≥3 kernels or patterns** to lift into
M1/M2, **plus** a written answer to Q1 (the four FreeToken mechanisms) and a fork-or-track
decision (Q6). Time-box porting to one week per kernel. Lift the *ideas* — tiling,
sub-group layout, offload policy — even where the code will not build.

## Attribution

Anything vendored into `third_party/` keeps its upstream header and cites the
commit recorded above. **FreeToken is Apache-2.0**; IPEX is Apache-2.0; llama.cpp is MIT. All require
attribution — see [../NOTICE](../NOTICE).
