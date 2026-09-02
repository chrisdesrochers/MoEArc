# Archaeology — mining the Intel Arc prior art

**Milestone M0.5. Survey complete 2026-09-02. Deep reading not started.**

The plan (v4 §M0.5) directed us to mine `intel/ipex-llm` for its low-bit SYCL
kernels before writing any of our own, estimating a **2–4 month saving**.

**That instruction was aimed at the wrong repository.** The survey below corrects
it. The prior art is real and it is substantial — it is simply not where the plan
said it was, and one of the three sources is not frozen at all.

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

The estimated saving should be carried at **zero** until something actually
compiles against oneAPI 2026.1. It is a hypothesis, not a credit.

---

## The five questions — still open

### 1. FlashMoE offload policy
Static per-layer split or dynamic? Caching or prefetch? Prefill vs decode?
**Sources:** the ipex-llm quickstart docs (policy as *documented*), and
`csrc/gpu/aten` in IPEX xpu-main (policy as *implemented*).

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
IPEX is a PyTorch extension — heavy runtime, Python-first. **We take the kernels,
not the architecture.**

## Gate

This document merged with a shortlist of **≥3 kernels or patterns** to lift into
M1/M2. Time-box porting to one week per kernel. Lift the *ideas* — tiling,
sub-group layout, offload policy — even where the code will not build.

## Attribution

Anything vendored into `third_party/` keeps its upstream header and cites the
commit recorded above. IPEX is Apache-2.0; llama.cpp is MIT. Both require
attribution — see [../NOTICE](../NOTICE).
