# Kernel strategy: should MoEArc stop writing kernels?

Written 2026-09-06. **This is an assessment and a plan. Nothing here has been implemented, and
no benchmark was run to produce it** — every number is quoted from a file already in this repo
or read out of llama.cpp's source. Where a number could not be verified, it says so.

The question: MoEArc hand-wrote 1,922 lines of SYCL. llama.cpp's `ggml-sycl` is 41,215 lines of
SYCL that already does most of it, is MIT-licensed, and sits in a sibling checkout. FreeToken —
the project MoEArc is modelled on — did not write kernels; it took llama.cpp's. Should we?

**Short answer: yes for the matvec family, no for most of the rest, and probably not before two
llama.cpp measurements that nobody has made.** The detail is below, and so is the case for not
doing it — which is stronger than it looks, because the repo's own profile names a *different*
fix with a larger expected return.

---

## 0. The headline, before the detail

| | finding |
| --- | --- |
| **Can ggml-sycl's kernels be called directly?** | **Yes.** The op layer is `(ctx, ggml_tensor*)`, and both are constructible without ggml's graph, scheduler, allocator or backend. The MoE GEMV is better than that — it is already a **raw-pointer API**. |
| **Minimal vendoring unit** | **The whole `ggml-sycl/` directory plus `ggml-base`** (~65k lines). Not individual kernel files. Trimmable to ~35k, which matters for build time and for keeping oneMKL out of the link. |
| **Which shape** | **(a) vendor sources, call directly** — same choice FreeToken made, for different and weaker reasons. |
| **Migration order** | **Matvec first, not attention.** The attention claim does not survive contact with the committed profiles, and ggml's oneDNN/MKL attention paths are **prefill-only**, which MoEArc does not have. |
| **Where the win comes from** | **The kernel body only** — llama.cpp's integer `vec_dot_*_q8_1` against Q8_1 activations, versus our dequantise-to-f32-and-MAC. Every other candidate explanation is either already spent or already retracted in this repo. |
| **Honest expectation** | Roughly **half to two-thirds** of the 29%→63% matvec gap, landing near 45–55% of peak — worth about **1.25× decode throughput at 2 K context and near nothing at 8 K**. **And the two numbers being compared were not measured the same way on the same model** — see §8. |
| **🔴 The finding that should decide it** | Two llama.cpp measurements, an hour each, would tell you whether this is worth 35,000 vendored lines. Neither has been made. See §8. |

---

## 1. Licence: MIT into Apache-2.0

**Verified on the checkout at `/zfs/swift/projects/llama.cpp`.** `LICENSE` reads
`MIT License / Copyright (c) 2023-2026 The ggml authors`. Every `ggml-sycl/*.cpp|hpp` file
carries a two-part header:

```
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

### The compatibility question, answered precisely

MIT is a permissive licence with one obligation: *"The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software."*
There is no copyleft, no reciprocity, no restriction on the licence of the combined work.

So: **MoEArc may distribute a combined work under Apache-2.0.** What it may **not** do is
*relicense* the MIT files. The distinction is the whole of the compliance work:

| ✅ correct | ❌ wrong |
| --- | --- |
| Vendored files keep their original headers, byte for byte | Replacing headers with Apache-2.0 ones |
| llama.cpp's full `LICENSE` text ships as a file in the tarball | Referencing it by URL only |
| `NOTICE` records upstream URL + **exact commit SHA** + date | "derived from llama.cpp" with no pin |
| Modified vendored files gain a change note *below* the original header | Silent edits to vendored source |
| Top-level `LICENSE` stays Apache-2.0 and `README` says the tree is mixed | Claiming the whole tree is Apache-2.0 |

The dual header is not ours to interpret. **Preserve both lines verbatim** and do not attempt to
decide which one governs; either grant permits what we are doing, and reproducing both is
compliant under either reading.

### Concrete mechanics

```
third_party/llama.cpp/
  LICENSE                 # verbatim copy of llama.cpp's MIT LICENSE
  UPSTREAM                # repo URL, commit SHA, date, `git describe` output
  vendor.sh               # re-extracts the trimmed file set from a sibling checkout
  ggml/…                  # the vendored sources, headers untouched
```

`NOTICE` gains an entry in the shape the existing `third_party/ipex-llm` entry already
establishes — that convention is already correct, it just has no occupant yet:

```
  third_party/llama.cpp/
      llama.cpp / ggml, MIT.
      Copyright (c) 2023-2026 The ggml authors.
      Portions Copyright (C) 2024-2025 Intel Corporation (ggml-sycl), MIT and
      Apache-2.0 WITH LLVM-exception, as stated in each file's header.
      Upstream: https://github.com/ggml-org/llama.cpp
      Commit:   <40-hex SHA>  (<date>)
      Modified: third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl-rt.cpp
                — extracted from ggml-sycl.cpp; see the note in that file.
      Full MIT text: third_party/llama.cpp/LICENSE
```

**One asymmetry worth stating in the README, not just here.** Apache-2.0 carries an express
patent grant; MIT does not. A downstream user of MoEArc receives our patent grant over our own
contributions and **no patent grant at all** over the vendored portion. That is not an
incompatibility and it is how every Apache project that vendors MIT code stands — but a public
repo that claims "Apache-2.0" without qualification is making a slightly stronger implicit
promise than it can keep. One sentence fixes it.

⚠️ `bundle.sh --with-runtime` is already documented as **not Apache-2.0**. That stays true and
gets worse if oneDNN/oneMKL are adopted (§4).

---

## 2. Are the kernels entangled with ggml's abstractions?

This is the crux, and the answer is much better than the file names suggest. Three tiers.

### Tier 1 — already raw pointers. No ggml types at all.

`mmvq.hpp` exposes, as public non-static functions:

```cpp
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type src0_type,
    const void *   vx_base,               // start of stacked expert weights
    const void *   vy,                    // pre-quantized src1 (Q8_1)
    const int32_t *ids_dev,               // device-side int32, length n_experts_used
    float *        dst_base,
    int ncols, int nrows, int n_experts_used,
    size_t expert_weight_stride,          // bytes between experts in vx_base
    size_t dst_row_stride,
    size_t src1_row_stride,               // 0 = shared src1, else per-expert
    dpct::queue_ptr stream);              // == sycl::queue*
```

plus `…_q_id_reorder` (SoA variant) and `…_q_glu_reorder` (fused gate/up + GLU). The kernel body
(`mul_mat_vec_q_moe`, `mmvq.cpp:2681`) does exactly:

```cpp
const int    i02 = ids_dev[item.get_group(1)];
const char * vx  = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
```

**This is MoEArc's expert-GEMV shape, written out.** A base pointer, a device-side index list,
a uniform stride. No `ggml_tensor`, no context, no graph. Type coverage:
`Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_0 Q2_K Q3_K Q4_K Q5_K Q6_K MXFP4 NVFP4` — which covers OLMoE
(Q4_K), Qwen3-30B (Q4_K/Q6_K) and gpt-oss (MXFP4). Unhandled types `return false` so a caller
can fall back rather than abort.

`convert.hpp` is nearly as clean: `ggml_get_to_fp32_sycl(type, dst)` **returns a function
pointer** of type `void(*)(const void*, float*, int64_t k, queue*)`. Only the *selector* takes a
tensor, and only to read `dst->src[0]->extra->optimized_feature.reorder`. Pass a tensor with
`extra == nullptr` and it selects the standard-layout path.

### Tier 2 — `(ctx, ggml_tensor*)`, and both are cheap to build

`fattn.hpp`, `norm.hpp`, `rope.hpp`, `softmax.hpp`, `getrows.hpp`, `mmq.hpp`, `dmmv.hpp` all take
`ggml_backend_sycl_context &` and one or more `ggml_tensor *`. What that actually costs:

**`ggml_tensor` is a POD C struct** (`ggml.h:674`) — `type`, `buffer`, `ne[4]`, `nb[4]`, `op`,
`op_params[16]`, `flags`, `src[10]`, `view_src`, `view_offs`, `data`, `name[64]`, `extra`,
`padding[8]`. It can be stack-allocated and filled by hand. `buffer` and `extra` may be null.

**`ggml_backend_sycl_context` is constructible with one integer** (`common.hpp:338`):

```cpp
explicit ggml_backend_sycl_context(int device);
```

It holds queues, a memory pool, oneDNN engine/stream maps and flash-attention KV scratch
buffers. It has **no** reference to a graph, a scheduler, a backend, or a buffer type.

I read `ggml_sycl_flash_attn_ext` end to end (`fattn.cpp:275`). It reads only tensor *metadata*
to dispatch — `ne[]`, `nb[]`, `type`, `op_params`, `src[0..4]` — and then the chosen kernel reads
`->data`. A grep for `->buffer` / `ggml_backend_buffer` across every kernel file returns **zero
hits**. The only `->extra` uses are the reorder flag in `mmvq.cpp` and `convert.cpp`.

The ggml-core surface those files call is 15 functions, all metadata:
`ggml_element_size`, `ggml_nelements`, `ggml_type_size`, `ggml_type_name`, `ggml_nrows`,
`ggml_is_contiguous{,_1}`, `ggml_is_contiguously_allocated`, `ggml_blck_size`,
`ggml_is_quantized`, `ggml_are_same_shape`, `ggml_nbytes`, `ggml_is_permuted`,
`ggml_get_op_params_{f32,i32}`, `ggml_rope_yarn_corr_dims`, `ggml_can_fuse`.

**One real contract the caller must honour.** `ggml_sycl_fattn_get_extra(dst)` carves flash
attention's F16 K/V/Q staging buffers *off the end of `dst->data`*:

```cpp
extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);   // then reserves upward
```

In llama.cpp the backend buffer type over-allocates for this. As a direct caller we must do it
ourselves — and the API for it is public: `ggml_sycl_flash_attn_ext_get_alloc_size(dst)`.
Allocate that many bytes for the output tensor, not `ggml_nbytes(dst)`. Get this wrong and
attention writes past the output.

### Tier 3 — genuinely entangled, and it is the one file the design notes named

`topk-moe.hpp` exposes only:

```cpp
int ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);
```

It is a **graph pattern-matcher**: it inspects nodes `i..i+n` of a `ggml_cgraph`, decides whether
they form a fusable top-k/softmax/MoE subgraph, and returns how many nodes it consumed. There is
no callable kernel behind it without constructing a graph.

📌 `docs/archaeology.md:50` says *"`ggml-sycl` already ships `mmq`, `mmvq`, `topk-moe`"* — and of
those three, **`topk-moe` is the one that cannot be lifted.** Our `moearc_topk_router` stays.

---

## 3. The minimal vendoring unit

**It is not individual kernel files.** `mmvq.cpp` compiles only against `common.hpp`, which pulls
`base.hpp`, `dpct/helper.hpp` (3,782 lines), `ggml.h`, `ggml-impl.h`, `ggml-sycl.h`,
`presets.hpp`, `type.hpp`, `sycl_hw.hpp`, `fattn-buffers.hpp`. And it *links* against symbols
that live in `ggml-sycl.cpp`:

| symbol | defined in | needed by |
| --- | --- | --- |
| `ggml_sycl_info()` | `ggml-sycl.cpp:216` | everything (device table, `opt_feature`, arch) |
| `ggml_sycl_init()` | `ggml-sycl.cpp:114` | " |
| `g_ggml_sycl_enable_flash_attention` | `ggml-sycl.cpp:106` | fattn dispatch |
| `ggml_backend_sycl_context::new_pool_for_device` | `ggml-sycl.cpp:1924` | any op that uses scratch |
| `…::new_pool_for_host` | `ggml-sycl.cpp:1919` | " |
| `…::new_fattn_kv_buffers` | `ggml-sycl.cpp:1934` | attention |
| `ggml_sycl_pool_leg` / `_vmm` / `_host` | `ggml-sycl.cpp:1593/1732/1845` | " |

And `ggml-sycl.cpp` in turn calls `ggml_backend_buffer_init`, `ggml_backend_buft_alloc_buffer`,
`ggml_backend_cpu_buffer_type`, `ggml_backend_reg_dev_get` — i.e. it needs `ggml-backend.cpp`.

### The closure, concretely

```
third_party/llama.cpp/ggml/
  include/    ggml.h  ggml-alloc.h  ggml-backend.h  ggml-cpp.h  ggml-opt.h  ggml-sycl.h  gguf.h
  src/        ggml.c  ggml.cpp  ggml-alloc.c  ggml-backend.cpp  ggml-backend-meta.cpp
              ggml-threading.cpp  ggml-quants.c  ggml-impl.h  ggml-common.h
  src/ggml-sycl/   the directory
```

`ggml-base` as CMake defines it is ~24k lines; `ggml-sycl/` is 41,215. **~65k lines total.**

### Trimming, and why it is worth doing

Two reasons: build time (this is the owner's homelab and `icpx -fsycl` over 100+ TUs is loud),
and **keeping oneMKL out of the link** (§4).

**Keep:** `common.{cpp,hpp}` `base.hpp` `presets.hpp` `type.hpp` `quants.hpp` `quantize.hpp`
`dequantize.hpp` `vecdotq.hpp` `sycl_hw.{cpp,hpp}` `mem.{cpp,hpp}` `dpct/helper.hpp`
`mmvq.cpp` `dmmv.cpp` `convert.cpp` `getrows.cpp` `norm.cpp` `softmax.cpp` `rope.cpp`
`element_wise.cpp` — and, only if attention is adopted, `fattn*.{cpp,hpp}` +
`template-instances/fattn-{tile,vec}*.cpp`.

**Drop:** `conv*.cpp` `conv2d*.cpp` `conv3d.cpp` `col2im-1d.cpp` `im2col.cpp` `pool.cpp`
`upscale.cpp` `pad*.cpp` `roll.cpp` `diag.cpp` `concat.cpp` `count-equal.cpp` `cumsum.cpp`
`set*.cpp` `repeat_back.cpp` `fill.cpp` `fwht.cpp` `tsembd.cpp` `binbcast.cpp` `add-id.cpp`
`outprod.cpp` `solve_tri.cpp` `opt-step.cpp` `cross_entropy_loss.cpp` `wkv.cpp` `gla.cpp`
`ssm_conv.cpp` `ssm_scan.cpp` `gated_delta_net.cpp` `lightning-indexer.cpp` `dsv4-hc.cpp`
`cpy.cpp` `fusion.cpp` `topk-moe.cpp` `mmq.cpp` `esimd.hpp`.

`mmq.cpp` (3,030 lines) is the **batched** quantised matmul — it is a prefill kernel and MoEArc
has no batched prefill. Drop it until §6 stage 5.

**`ggml-sycl.cpp` (7,013 lines) is the awkward one.** We need ~400 lines of it (the device-info
init at 106–262 and the three pool classes plus their factories at 1593–1960) and nothing else —
the other 6,600 lines are backend registration, buffer types, graph-op dispatch, split-tensor
handling. Three options:

1. Compile it whole and accept `ggml-backend.cpp` + the MKL link.
2. **Extract the needed ~400 lines** into `third_party/llama.cpp/ggml/src/ggml-sycl/ggml-sycl-rt.cpp`
   as a *modified vendored file*, original header intact plus a change note, recorded in `NOTICE`.
3. Guard the rest behind a `#ifdef MOEARC_SYCL_RUNTIME_ONLY` in a patched copy.

**Recommend (2).** It is legal under MIT with a change note, it removes `ggml-backend.cpp` and
`ggml-alloc.c` from the closure, and it removes `ggml-sycl.cpp` from the MKL user list.
`vendor.sh` must re-derive it on every re-vendor, and the file header must say so loudly, because
this is the one piece that will silently rot.

Closure after trimming: **~35k lines**, ~30 SYCL TUs, `ggml.c` + `ggml.cpp` + `ggml-quants.c` +
`ggml-threading.cpp` from core.

---

## 4. Build integration

### The technique the owner pointed at

The blog's pattern (vendor C/C++ into the repo, build it from `build.rs` via the `cc`/`cmake`
crates into a static lib, link it, wrap it in a `-sys` crate) transfers **in shape but not in
mechanism**, and the reason is already recorded in this repo.

`docs/packaging.md:37-49`: a static `.a` was tried and **failed** on
`undefined symbol: _intel_fast_memcpy`, from `libintlc` — one of several Intel runtime libraries
`icpx` links automatically, and *that list belongs to the compiler version, not to our code.* The
settled answer is a **shared object linked by `icpx` itself**. The `cc` crate cannot do that: it
manages compiler invocation and produces an archive.

**So: extend the existing `build.rs`, do not adopt `cc`.** Compile the trimmed ggml-sycl source
set into **the same `libmoearc_kernels.so`**, with the same `icpx -fsycl -shared` link. One
library, one `DT_NEEDED`, and the soname-as-path trick (`build.rs`, and
`packaging/elf-relocatable.py`) is untouched. What transfers from the blog is the *discipline*:
vendor into the repo, pin the upstream commit, one crate owns the build, a script re-vendors.

Flags that must be matched to upstream or the kernels misbehave:

| flag | why |
| --- | --- |
| `-DGGML_SYCL_WARP_SIZE=16` | kernels carry `[[sycl::reqd_sub_group_size(WARP_SIZE)]]`; wrong value = compile error or wrong reduction |
| `-Xs -ze-intel-greater-than-4GB-buffer-required` | upstream link option; without it >4 GB allocations behave differently |
| `-DGGML_SYCL_DNNL=0` | oneDNN off (see below) |
| `-Wno-narrowing` | upstream sets it; some instances need it |
| `-I third_party/llama.cpp/ggml/include -I .../ggml/src -I .../ggml/src/ggml-sycl` | |

### 🔴 Build time is a first-class design constraint here

~30 SYCL TUs through `icpx -fsycl`, each generating SPIR-V, is minutes to tens of minutes and
loud. `cargo:rerun-if-changed` over a vendored tree is necessary but not sufficient — `OUT_DIR`
is wiped by `cargo clean` and varies per feature set.

**Add a content-addressed cache.** `build.rs` hashes (vendored source bytes + flag string +
`icpx --version`) into `target/moearc-ggml-cache/<sha256>/libggml_sycl_objs.a`, checks it first,
and only compiles on a miss. This is the single most important build decision for anyone running
this on a machine they can hear.

### Packaging and the Intel EULA — does vendoring change it?

`docs/packaging.md:311` establishes the position: nothing of Intel's is redistributed;
`packaging/fetch-runtime.py` downloads Intel's runtime **from Intel's own channel**, SHA-pinned
in `packaging/runtime.lock.json`, so the user accepts Intel's terms from Intel. Default tarball
4.8 MB, installed runtime **73 MB**.

**If the trimmed source set above is used: nothing changes.** No new Intel library enters the
link. That is not an accident — three of the five ggml-sycl files that call oneMKL
(`conv3d.cpp`, `outprod.cpp`, `solve_tri.cpp`) are on the drop list, and the fourth
(`ggml-sycl.cpp`) is replaced by the extracted runtime file. The fifth is `fattn-mkl.cpp`.

⚠️ **One qualification.** `dpct/helper.hpp:22` is `#include <oneapi/mkl.hpp>`, unconditional, and
`common.hpp` includes it — so **every** ggml-sycl TU needs the oneMKL *headers* at build time.
Only the *link* is avoidable, and only because `dpct::gemm` is a template that nothing in the
trimmed set instantiates. The build machine needs the full oneAPI Base Toolkit; the user's
machine does not. That is the same asymmetry MoEArc already lives with.

**If `fattn-mkl.cpp` and/or `fattn-onednn.cpp` are adopted, this changes materially.** Measured
on this box:

| library | file | size |
| --- | --- | ---: |
| oneDNN | `libdnnl.so.3.11` | **108 MB** |
| oneMKL | `libmkl_sycl_blas.so.6` | 75 MB |
| | `libmkl_core.so.3` | 76 MB |
| | `libmkl_intel_ilp64.so.3` | 17 MB |
| | `libmkl_tbb_thread.so.3` + `libtbb.so.12` | ~25 MB |

Runtime fetch goes **73 MB → ~370 MB**; `bundle.sh --with-runtime` goes 29 MB → ~250 MB+. The
*mechanism* extends unchanged — Intel publishes `mkl` and oneDNN on PyPI, which is the channel
`fetch-runtime.py` already uses, so this is a size and audit-surface cost, not a licence problem.
But the "files" allowlist in `runtime.lock.json` exists specifically to keep the runtime small
and auditable, and this would be a 5× regression against that goal.

**§6 argues these two paths buy MoEArc nothing today anyway.**

---

## 5. What survives and what dies, file by file

### `crates/moearc-kernels/kernels.cpp` — 1,922 lines

| region | lines (approx) | verdict |
| --- | ---: | --- |
| Quant block structs, `q45k_scale_min`, `elem_at`, `unit_acc`, dequant formulas | ~1–730 | **DIES** with the matvec family — this is the largest single chunk and it is a re-transcription of `ggml-quants.c` that we are maintaining ourselves |
| `moearc_matvec_q_batched` | 1146–1206 | **DIES** → `ggml_sycl_mul_mat_vec_q_id`. Exact structural match. |
| `moearc_matvec_q` | 1046–1111 | **DIES** → `ggml_sycl_mul_mat_vec_q_id` with `n_experts_used = 1`, `ids_dev = [0]`, strides 0. *(The per-type `mul_mat_vec_*_q8_1_sycl` entry points are `static` in `mmvq.cpp`; the `_id` trick avoids having to un-static them, i.e. avoids a vendored modification.)* |
| `moearc_dequant` | 1009–1026 | **DIES** → `ggml_get_to_fp32_sycl(type, nullptr-extra tensor)` |
| `moearc_gather_experts` | 968–998 | 🔴 **DEAD CODE — delete it today, independent of this migration.** See below. |
| `moearc_attn_decode` | 1837–1919 | **CONDITIONAL** → `ggml_sycl_flash_attn_ext`, blocked on paged KV (§8) |
| `moearc_matvec_f32` | 1116–1135 | **KEEP** — `_id` has no F16/F32 case; ggml's `dmmv` is tensor-shaped and this is not hot |
| `moearc_topk_router` | 1495–1588 | **KEEP** — ggml's is graph-shaped (§2 tier 3) |
| `moearc_kv_append` | 1782–1799 | **KEEP** — ours is paged; ggml's `set_rows` is not |
| `moearc_rmsnorm` `moearc_softmax` `moearc_rope` | 1246–1472 | **KEEP** — gate-verified, not hot, and swapping costs the tensor shim for no measured gain |
| `moearc_silu` `swiglu*` `swiglu_oai*` `add` `mul` `zero` `axpy` `quantize_f16` `moe_combine` `add_bias_id` `embed_rows` | 1220–1770 | **KEEP** — all have ggml equivalents, all tensor-shaped, all negligible |
| `moearc_ctx_*`, `alloc/free_device`, `copy_h2d{,_async}`, `copy_d2h`, `sync`, `device_name`, the event-profiling subsystem | 804–966, 839–872 | **KEEP — this is the residency/scheduling substrate and it is ours** |

**Net: roughly 970 of 1,922 lines die** (the quant helpers plus the matvec family plus dequant),
~120 more are conditional, and ~830 stay. That is about half the file — a real reduction, and a
smaller one than "we should never have written kernels" implies.

### `crates/moearc-kernels/src/reference.rs` — 931 lines

**Survives, and grows.** It is the CPU oracle, not a kernel. It must gain a `q8_1` arm (§7).

### 🔴 `gather_experts` is dead code, and the migration costs an engine refactor

Two findings here, and the first one corrects an assumption this document started with.

**`moearc_gather_experts` has no callers in the engine.** `grep -rn gather_experts --include=*.rs
crates/` returns only its own FFI declaration, its own safe wrapper, and its own two tests. It is
described in the crate as "the residency hot path" and it has never been on any path. The engine's
expert GEMV (`moe.rs:2289`) builds a `Vec<&DeviceBuffer>` of slot pointers and passes it to
`matvec_q_batched`, which takes a **by-value 32-pointer table** (`mat_table`). **There is no
per-token gather copy to delete.** Delete the kernel as dead code; it is not a migration win.

**And the ggml kernel's ABI turns the pool layout into a *cost*, not a win.**
`crates/moearc-engine/src/moe.rs:1145` — `ExpertPool` is three `Vec<DeviceBuffer>`, one
`malloc_device` per slot per bank, with the reason stated in the source:

> *Three parallel arrays rather than one buffer per slot, because `matvec_q` reads a weight
> matrix from the start of a `DeviceBuffer` and there is no way to offset into one.*

`ggml_sycl_mul_mat_vec_q_id` requires the opposite: **one base pointer, one uniform stride, a
device-side id array.** So adopting it forces reshaping `ExpertPool` into three contiguous
per-bank allocations and rewriting the slot→pointer path in `residency.rs`/`moe.rs` to emit a
`Vec<i32>` of slot indices. That is a real refactor of the engine's most load-bearing structure,
**forced by the vendored kernel's ABI**, and it must be counted against the migration.

⚠️ **And this repo has already measured that the layout change itself buys nothing.**
`bench/baselines/qwen3-30b-a3b.md:219` — an isolated microbenchmark measured the by-value pointer
table at **1.66× slower** than base-pointer-plus-stride (141 vs 235 GB/s), and that finding was
**retracted**: in the engine the `mat_table` kernels are *the fastest per byte we have*
(126 and 134 GB/s, against 113–120 for unbatched matvecs on the same queue in the same step).
The standing rule from that retraction applies directly here: *"a microbenchmark of 'the same
kernel' is not this kernel."*

**So stage 1's win is the kernel body alone — the int8 vec-dot — and the indirection change is
overhead we pay to get at it.** Two consolations, neither of them performance: `moe.rs:1067`
documents `ExpertPool::new` making **9,300 `malloc_device` calls** on Qwen3-30B with a failure
mode where the pool reports success and dies on the first token; three allocations removes that.
And the contiguous pool is a prerequisite for the `_reorder` path if it is ever pursued (§8).

---

## 6. Which shape, and the migration order

### Shape (a) vs shape (b)

**FreeToken chose (a)** — verified by reading a local clone at
`/zfs/swift/projects/FreeToken-study/`. Its kernels live in
`python/freetoken/kernel/csrc/gguf/` and carry explicit provenance headers:

```
// copied from vllm .../csrc/quantization/gguf/mmq.cuh
// copied from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmq.cu
```

`mmq.cuh` 881 + `mmvq.cuh` 352 + `vecdotq.cuh` 2037 + `moe.cuh` 1379 + `moe_vec.cuh` 413 +
`dequantize.cuh` 583 + `ggml-common.h` 1029 + `gguf_kernel.cu` 848 = **7,522 lines of 13,050**.
A grep for `ggml_tensor|ggml_backend|ggml_context` across the whole `csrc/` tree returns **zero
hits**. It is a PyTorch CUDA extension that takes `torch::Tensor`, extracts raw device pointers,
and launches vendored kernel templates. Not a llama.cpp fork (52 commits, its own history).

🔴 **But the reason (a) was cheap for FreeToken does not transfer.** The chain is
llama.cpp → **vLLM** → FreeToken. vLLM had already done the de-tensorisation work: it ported
ggml's CUDA kernels into standalone raw-pointer templates for its own GGUF support. FreeToken
copied vLLM's copy. **Nobody has done that for SYCL.** On the SYCL side, upstream has
de-tensorised exactly one thing — the MoE GEMV — and everything else sits behind a
`(ctx, ggml_tensor*)` façade.

So MoEArc's (a) is a *hybrid*: **vendor the whole backend as a build unit, but call the op
functions directly — never the `ggml_backend` interface.** Raw pointers where upstream offers
them (`mmvq`, `convert`); a stack-tensor shim where it does not (attention, if adopted).

**Shape (b)** — build a `ggml_cgraph` per token and express residency as tensor `data` pointers —
is genuinely possible: `ggml_mul_mat_id` over a stacked expert tensor is exactly our shape, and
setting `src[0]->data` to a slot pointer is how you'd inject a residency decision. What it costs:
ggml's graph allocator wants to own intermediate buffers; our slot pool becomes a custom
`ggml_backend_buffer`; the async second-queue design (`docs/roadmap.md` names this as the fix for
staging/compute serialisation) is not expressible through ggml's scheduler; and the top-k/MoE
fusion path would then be available but only by matching ggml's exact node ordering.

And `docs/dependencies.md` already ruled on the principle, about `llama-cpp-rs`:

> **Not for the engine** — building on it would make MoEArc a llama.cpp wrapper, which is not the
> project.

**Recommendation: (a), hybrid form.** It preserves the thing that is actually ours — control of
memory and of the queue — and takes the thing that is not.

### Migration order — and I argue **against** attention first

The brief's case for attention-first is *"~69% of our decode step at depth 8192, and the
oneDNN/MKL paths are things we would never write."* I could not verify either half.

**🔴 On the 69%.** The committed depth-8192 profile
(`bench/results/2026-09-06-moearc-host-policy-8192.txt:18`, `frac:0.75`) reads:

```
| depth | decode.total | attn.attend | attn.qkv | attn.proj | moe.stage | moe.expert_matvec | moe.host_sync | moe.readback |
| 8192  | 269.73       | 0.08        | 0.45     | 0.23      | 3.82      | 0.23              | 42.72         | 207.85       |
```

`attn.attend` is **0.08 ms of 269.73** — 0.03%, not 69%. `bench/results/2026-09-06-moearc-depth-curve.txt:17`
agrees (0.11 of 466.91). ⚠️ **Neither number should be believed either**: `moe.readback` at
207.85 ms is a synchronisation point on an async queue, and it is absorbing the whole queue's
device time — the exact instrument failure this repo already retracted once
(`README.md:219`, *"a profile that mis-attributed device time under an async queue"*). The
uncommitted working tree shows another agent adding `moearc_track(c, "attn_decode", …)` to
`moearc_attn_decode` **right now**, which is the instrument that would settle it.

Two bounding arguments, so the decision does not have to wait:

- **Bandwidth floor.** At depth 8192 the profile reports **296 MiB of KV** (sliding-window
  attention). Attention cannot read more than that. At MoEArc's *worst* measured efficiency
  (25% of 456 GB/s = 114 GB/s) that is **~2.6 ms**. 2.6 ms cannot be 69% of a 269 ms step, and it
  cannot be 69% of the 466 ms step either. On bandwidth grounds the claim is impossible.
- **The only way it could be true** is that `moearc_attn_decode` is *latency*-bound, not
  bandwidth-bound: it launches `nd_range<1>{n_heads * head_dim, head_dim}` — **one work-group per
  query head**, 32 work-groups on a 20-Xe-core card, with no tiling and no vectorised load. That
  under-parallelisation is real and could put it far above its bandwidth floor. If the new
  instrument shows that, **attention-first becomes right** — and ggml's `fattn-tile`/`fattn-vec`
  do parallelise over the KV axis, so it would be recoverable.

**🔴 On oneDNN and MKL.** Both are gated on `Q->ne[1] >= 32` — thirty-two query tokens.
`fattn.cpp:135`:

```cpp
// ONEDNN requires min 32 query tokens — short-circuit decode to avoid
// calling _supported() on every decode FA call.
if (Q->ne[1] >= 32 && ggml_sycl_flash_attn_ext_onednn_supported(dst)) …
```

and the MKL gate additionally requires `K->ne[1] >= 1024`. **Both XMX paths are prefill-only.**
MoEArc has no batched prefill (`README.md:206`: *"Prefill. There is none."*). At decode, ggml
falls through to `BEST_FATTN_KERNEL_VEC` or `_TILE` — plain SYCL, no oneDNN, no oneMKL.

So of the 1,707 attention lines, the part MoEArc could use *today* is `fattn-vec.hpp` (684) +
`fattn-tile.cpp` + its 10 instances (~250) + `fattn-common.hpp` (1,185 of shared scaffolding).
Good code, worth having — but **the two libraries that justified "things we would never write"
would be linked, packaged, and never executed**, at a cost of ~300 MB of runtime fetch (§4).
`docs/roadmap.md:17` already says the same thing from the other direction: *"XMX and the matrix
engines are irrelevant to decode."*

### The order

| # | step | why here |
| --- | --- | --- |
| **0** | **Scaffolding.** Vendor + `vendor.sh` + `UPSTREAM` + `NOTICE` + trimmed build in `build.rs` + the content-addressed cache. A `moearc_ggml_smoke` binary that calls `ggml_sycl_info()` and prints the B580. **No kernel swapped.** | Everything else is blocked on this compiling; and the licence work must land before any vendored byte is pushed to a public repo |
| **1** | 🔴 **Contiguous `ExpertPool` + `ids_dev` + `ggml_sycl_mul_mat_vec_q_id`** for the expert GEMV. | The largest *measured* share of the step (expert matvecs are 8.50 ms of the 17.38 ms tracked matvec busy at 2 K), and the cheapest integration — **no `ggml_tensor` and no context needed at all** |
| **2** | **Single matvecs** — `attn_q/k/v/output` and `lm_head` — through `…_q_id` with `n_experts_used=1`. | Fixes the worst efficiency case in the repo: `lm_head` Q6_K at **63 GB/s, 14% of peak, 4.03 ms of a 37.66 ms step** |
| **3** | **Dequant + embedding** through `ggml_get_to_fp32_sycl`. Retire `elem_at` / `unit_acc` / the quant block structs. | Small perf, but it is what actually deletes the ~730-line helper region and ends our maintenance of a second copy of `ggml-quants.c` |
| **4** | **Re-measure.** With `moearc_track` on `attn_decode` and a second queue. Decide attention on that number, not on this document. | |
| **5** | **Attention**, only if (4) justifies it — and only after the paged-KV question in §8 has an answer | |
| **6** | **Prefill** (`mmq.cpp` + oneDNN/MKL attention) as a separate project with its own packaging decision | This is where XMX and the 300 MB actually pay |

Stages 1–3 need **no `ggml_tensor` at all**. That is a strong argument for this order
independent of performance: the tensor shim, the `fattn` over-allocation contract, and the
`ggml_backend_sycl_context` lifetime question are all deferred to a stage that may never happen.

---

## 7. 🔴 How the correctness gate survives

The rule, stated first because everything else is subordinate to it:

> **The end-to-end token-id gate against llama.cpp is never relaxed. If a step turns it red, the
> step is reverted. A migration that adjusts a gate to match a kernel is not a migration.**

The gates split into two classes and they must be treated differently.

### Class A — end-to-end token-id gates. These stay, unchanged, and should get *stronger*.

`crates/moearc-engine/tests/qwen3moe_forward.rs`, `bench/references/*.ids`, and the sweep in
`bench/baselines/qwen3-30b-a3b.md` (*"every one of the 42 rows produced identical token ids, and
every one matched llama.cpp for all 64 ids"*).

The reason this class survives is not luck. **llama.cpp itself computes these matvecs with
`mmvq` against Q8_1-quantised activations.** Today MoEArc dequantises to f32 and does fp MACs —
a *different* algorithm that happens to agree on token ids. Adopting `ggml_sycl_mul_mat_vec_q_id`
means adopting the oracle's own arithmetic. Agreement should **improve**, not degrade.

Two additions, both cheap, both to be made **before** stage 1:

1. **Lengthen the reference id streams.** They are generated by llama.cpp and cost nothing.
2. 🔴 **Add a logit-margin check.** Record, from the reference run, the top-1/top-2 logit gap at
   every step. A token-id gate is a step function: it passes until it catastrophically doesn't,
   and it cannot tell "numerically identical" from "one ULP short of flipping." If the margin
   distribution collapses while ids still match, numerics degraded and the gate did not see it.
   **This is the instrument that makes "the gate stayed green" mean something.**

### Class B — kernel-level gates against the CPU reference. These must legitimately change.

`tests/kernels_gpu.rs` compares each kernel to `src/reference.rs` with a ULP-derived tolerance
(`2.0*(n_cols/32.0+5.0)*U*Σ|w|`, `U = 2^-24`). Swapping in `mmvq` breaks this **by construction**:
Q8_1 activation quantisation is ~1/256 relative error, orders of magnitude outside a model built
for f32 rounding. Widening the tolerance until green would be exactly the prohibited move.

**The legitimate change:**

- `reference.rs` gains a **second arm**, `matvec_q8_1`, that reproduces llama.cpp's
  quantise-then-integer-dot arithmetic in the same term-for-term style the existing dequant
  formulas already use.
- The GPU kernel is compared against **that** reference, with a tolerance **re-derived from the
  new error model** (Q8_1 block quantisation error + the integer dot's exact accumulation + the
  final f32 scale). Re-derived, printed, and justified in the test — not tuned until green.
- 🔴 **The new reference is itself oracled.** Golden-dump llama.cpp's own CPU
  `quantize_row_q8_1` into `tools/ggml_q8_1_ref.c`, exactly as `tools/ggml_dequant_dump.c` and
  `tools/ggml_attn_ref.c` already do, gated on a `MOEARC_Q8_1_GOLDEN` env var. Without this the
  new reference is a self-assertion and the gate is worthless.
- The **bit-exact** assertions in `kernels_gpu.rs` (batched-vs-loop, `moe_combine`-vs-axpy) are
  self-consistency checks, not oracle checks. They survive if both arms move together; if the
  batched arm becomes ggml's and the loop arm stays ours, the assertion is meaningless and must
  be deleted rather than loosened.
- `gguf_crosscheck.rs` and `f16_crosscheck.rs` are **unaffected** — they test dequantisation
  against llama.cpp's `to_float`, which is what stage 3 adopts wholesale. They should go from
  ulp-level to exact.

### The A/B harness — the mechanism that makes every step revertible

Keep **both** implementations behind a per-op runtime switch for the entire migration:

```
MOEARC_KERNEL_MATVEC = ours | ggml      (default: ours, until the stage lands)
MOEARC_KERNEL_DEQUANT = ours | ggml
MOEARC_KERNEL_ATTN   = ours | ggml
```

Every gate runs **both arms** in CI. A stage is "done" when the ggml arm has been green on
**OLMoE, Qwen3-30B and gpt-oss** for a release. Only then does our kernel get deleted. This costs
a few hundred lines of dispatch and it is the difference between a migration and a rewrite.

### The one gate that cannot be preserved as-is

`attention_crosscheck.rs` validates that MoEArc's **paged** key layout is transparent — ggml
computed over a flat contiguous run, we computed over scattered pages, same answer. If attention
moves to `ggml_sycl_flash_attn_ext`, that property is **lost, not re-tested**, because ggml's
attention has no block table (§8). The gate does not need to change; the *feature it protects*
would be gone, and that is a design decision, not a test decision.

---

## 8. Honest risks — and how much of the gap I actually expect back

### 🔴 The most useful sentence in this document

**The 29% and the 63% were not measured the same way on the same model.**

- **63.4%** comes from `docs/roadmap.md:10`: llama.cpp SYCL at **283.31 tok/s on OLMoE**,
  converted to 289 GB/s by multiplying by active bytes per token. It is an **end-to-end inferred
  whole-step** figure on a **3.9 GiB fully-resident** model.
- **25–29%** comes from `bench/baselines/qwen3-30b-a3b.md:211`: **per-kernel device time** from
  SYCL event timestamps, on **Qwen3-30B-A3B**, a **17.3 GiB streamed** model.

Two of those differences cut in opposite directions and one is unknown:

- Whole-step *understates* llama.cpp's matvecs (its step also contains attention, norms, sampling,
  none of which read weights), so its matvec-only figure is **above** 63.4% — the gap is worse
  than stated.
- Different models, different quant mixes, different residency — unknown sign.

**Before spending weeks on this, spend an hour deriving llama.cpp's *per-kernel* bandwidth on
Qwen3-30B the same way MoEArc's was derived.** That is a llama.cpp measurement, not a MoEArc
benchmark, and it is the cheapest possible de-risking of this entire plan. If the honest gap is
2.16× the plan is strong; if it is 1.4× most of this is not worth doing.

### Decomposing the gap — where I think it lives

**1. Dequant ALU cost — large, and recoverable. I believe this is the dominant term.**

llama.cpp's `vec_dot_q4_K_q8_1` is an **integer** dot product against Q8_1-quantised activations:
int8 MACs plus two fp scales per 32 values. MoEArc's `unit_acc` dequantises to f32 and does fp
MACs, and the crate's own comment puts the saving from hoisting per-block constants at
*"~25 instructions/MAC."* Twenty-five instructions per MAC is not a memory-bound kernel.

Arithmetically: the B580 is ~13.6 TFLOP/s fp32 (20 Xe-cores × 128 lanes × 2 × ~2.67 GHz) and
456 GB/s. A Q4_K dequant-to-float at ~10–20 ops/byte needs **4.5–9 TOP/s to keep 456 GB/s fed** —
within a small factor of the card's fp32 ceiling. **MoEArc's matvec is plausibly ALU-bound on its
own dequantisation.** The corroborating evidence is already in the repo and was not read this
way at the time: `bench/baselines/qwen3-30b-a3b.md:275` reports Q6_K **flat at ~13 ps/element
from `n_cols` 256 to 4096**. Flat in `n_cols` is the signature of a per-element *compute* cost.

This is exactly what the int8 path fixes, and it is why I expect the swap to work at all.

**2. Launch geometry — already spent.** `docs/roadmap.md:29` identified 384 launches/token and
batching took matvecs 14.8% → 25–29%. ggml's `mul_mat_vec_q_moe` uses the same work-group shape.
**Little left here.**

**3. 🔴 Data layout — the part that may not be recoverable.** `ggml-sycl.cpp:170` sets
`opt_feature.reorder = true` for **every** Intel GPU, and llama.cpp's headline SYCL numbers come
from the **reorder/SoA** path (`ggml_sycl_mul_mat_vec_q_id_reorder`), which repacks quant blocks
into separate quant and scale planes so loads coalesce. It does this **once, at model load, for a
resident model.**

MoEArc stages **raw GGUF blocks** from host into slots, continuously. `ggml_sycl_mul_mat_vec_q_id`
(non-reorder) works on that layout — its header says so explicitly — so we get the int8 dot for
free. Getting the *reorder* win requires a repack: on the host during staging (CPU on the hot
path, and staging is already the dominant cost) or a device kernel after each H2D (a full extra
read+write of every newly-resident expert — i.e. **exactly the `gather_experts` traffic we just
deleted, reintroduced**).

**If a material share of llama.cpp's 63% is the reorder path, that share is structurally out of
reach for a streaming engine.** I do not know the share. It is cheap to find out and it should be
the *second* hour spent: run llama.cpp with reorder disabled and compare. That single number
decides whether the ceiling here is ~63% or ~45%.

**4. 🔴 Residency indirection — I expected a win here and there is none.** The draft of this
document assumed `gather_experts` was on the hot path and that `ids_dev` would delete a
read+write of every active expert byte. **It is dead code** (§5), the engine uses a by-value
pointer table, and this repo has *already measured and retracted* the claim that base-pointer +
stride beats that table. So the indirection change is **overhead, not upside** — a forced
refactor of `ExpertPool` bought at zero measured performance.

That leaves **term 1 carrying the entire case**, which is worth saying out loud.

### The estimate

**I expect roughly half to two-thirds of the 29%→63% bandwidth gap, landing matvecs around
45–55% of peak (200–250 GB/s).**

The reasoning: term 1 is large and directly addressed; term 2 is spent; term 3 is the ceiling and
is partly out of reach by construction; term 4 is nothing. Translated into a step, using the only
per-kernel attribution this repo has (2048 ctx, 2952 slots): tracked matvec busy **17.38 ms of a
37.66 ms step at 28%**. At 50% that is ~9.7 ms — a **7.7 ms saving, ~20% of the step, ~1.25×
throughput**. Not the 2× the headline gap implies.

⚠️ **The confidence interval on that is wide and it is asymmetric downward.** Term 1 is an
argument from instruction counts, not a measurement, and this repo has now had **three**
well-argued kernel hypotheses die when measured in place (the device-`half` conversion, the
`mat_table` layout, and — in this document — the gather). The standing rule from those
retractions applies to this estimate too.

⚠️ **And that is the *decode-at-2048* picture.** At depth 8192 the same repo says staging is 80%
of the growth and matvecs are a small share of a 269 ms step — **so at depth this whole project
moves a rounding error.** The kernel gap is a *shallow-context* problem. The thing that dominates
at depth is expert staging, and no kernel in `ggml-sycl` addresses it.

### Other risks, named

- 🔴 **Paged KV blocks the attention half.** `ggml_sycl_flash_attn_ext` expects K/V as strided
  tensors (`nb[1..3]`). MoEArc's KV is **paged**, reached through a `block_table`, and
  `moe.rs:1420` describes one K and one V pool per block indexed that way. ggml has **no block
  table**. So attention-via-ggml means either abandoning paging (physically contiguous KV per
  sequence) or gathering pages into contiguous scratch every step — which is the same class of
  copy stage 1 exists to delete. **This is a first-class blocker and it is not visible in the
  line counts.** It must be answered before stage 5 is scheduled.
- **The swap is not confined to kernels.** Q8_1 activations mean a `quantize_row_q8_1_sycl` call
  enters every block and the activation buffers change type. Cheap in time (~2048 floats) but a
  real change to `moe.rs`'s decode loop.
- **`ggml_sycl_pool_vmm`** reserves and commits VRAM through the SYCL virtual-memory extension.
  MoEArc's whole thesis is precise VRAM accounting, and `memory::plan` does not know about it.
  Either force `ggml_sycl_pool_leg`, or teach the planner about the pool's reservation. **Do not
  let two allocators size themselves against the same free-memory reading** — `docs/calibration.md`
  already records that `malloc_device` succeeds far past physical VRAM (38 GiB on an 11.33 GiB
  card), so a naive "allocate until it fails" reconciliation will not detect the conflict.
- **Upstream churn.** 35–65k vendored lines from a project that merges dozens of commits a day.
  `UPSTREAM` + `vendor.sh` make re-vendoring a script; the extracted `ggml-sycl-rt.cpp` (§3) makes
  it partly manual, and that is the piece that will rot silently.
- **Build noise and time.** §4. Mitigated by trimming and the content-addressed cache, not
  eliminated.
- **Reviewability.** MoEArc's public case is that it is a small, auditable, clean-room Rust engine.
  Vendoring 35k lines of C++ changes what the repo *is*, and the README should say so plainly
  rather than let a reader discover it in `third_party/`.

### The case for not doing this at all

Stated fairly, because it is not weak:

1. The dominant cost at depth is **staging**, and none of this touches it.
2. The gap being closed may be **1.4× rather than 2.16×** — nobody has measured llama.cpp's
   per-kernel bandwidth on the model MoEArc's 29% was measured on.
3. Some of the remainder is the **reorder layout**, which a streaming engine cannot amortise.
4. The attention argument, as posed, **does not survive the committed profiles**, and its two
   headline libraries are **prefill-only**.
5. The one structural change the migration forces — the contiguous `ExpertPool` — is a change
   this repo has **already measured as worthless** on its own.
6. And the repo's own profile names a **larger, closer fix that has nothing to do with kernels**:
   `bench/baselines/qwen3-30b-a3b.md:242` — *"Staging cannot overlap with compute. `moe.stage`
   is ~11 ms and its copies go into the same in-order queue as the kernels, so a token's
   transfers and its arithmetic are strictly serialised."* ~11 ms of a 37.66 ms step, against the
   ~7.7 ms this whole plan is estimated to save. **A second queue with explicit event
   dependencies is a smaller change with a larger expected return**, and unlike this plan it also
   pays at depth, where staging dominates.

### 🔴 What to do before deciding — two hours, no MoEArc code

Both are llama.cpp measurements. Neither requires touching this repo, and together they turn the
central estimate from an argument into a number.

1. **Derive llama.cpp's *per-kernel* bandwidth on Qwen3-30B-A3B**, the same way MoEArc's 25–29%
   was derived — not tok/s on OLMoE converted to GB/s. This is the denominator of the entire
   plan and nobody has measured it. Give llama.cpp **the whole machine** (`-t 20`, no `-ncmoe`,
   fully resident) — the standing rule from the 2026-09-06 withdrawal.
2. **Measure the reorder share** — `GGML_SYCL_ENABLE_OPT=0`. Verified to exist
   (`ggml-sycl.cpp:331`) and to gate the reorder decision (`ggml-sycl.cpp:4602`). Run the same
   configuration with it off. The delta is the fraction of llama.cpp's advantage that comes from
   a load-time SoA repack a *streaming* engine cannot amortise — i.e. the part of the gap that is
   structurally out of reach. That one number moves the ceiling between ~63% and ~45%.

**If (1) shows the honest gap is nearer 1.4× than 2.16×, or (2) shows most of it is reorder, this
plan is not worth 35,000 vendored lines and the second queue should be built instead.**
