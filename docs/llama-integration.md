# Driving llama.cpp + SYCL from Rust

Written 2026-09-06, against llama.cpp pinned at `e107984bc`. Everything in the
"Proof" section was run on CHARA's Arc B580 and is quoted from real output.
**No benchmark was run and no throughput claim is made** — another agent owns the
GPU for a measurement sweep, and the one timing printed below is labelled as a
smoke test precisely so nobody quotes it.

This document covers the seam only: how Rust talks to llama.cpp. What sits *above*
the seam (tuning, TUI, installer) and what gets retired *below* it are in
`docs/pivot-inventory.md`.

---

## 0. The decision

**A hand-written C shim compiled against llama.cpp's headers, linked to the
pre-built shared objects from the reference SYCL build, with hand-written Rust FFI
to the shim.** New crate: `crates/moearc-llama`.

Three candidates were considered. The rejections are on facts, not taste.

| option | verdict | the deciding fact |
| --- | --- | --- |
| `llama-cpp-2` (`utilityai/llama-cpp-rs`) | ❌ **rejected** | It has no SYCL backend, and it cannot be pointed at our llama.cpp. |
| Direct FFI to `llama.h` (bindgen or hand-written mirrors) | ❌ **rejected** | Requires reproducing `llama_context_params` by hand; drift corrupts silently. |
| **C shim + FFI to the shim** | ✅ **chosen** | llama.h owns the fragile structs; Rust sees a flat surface we control. |
| Drive `llama-server` as a subprocess | 🟡 **viable fallback** | Loses per-run timers, live thread changes, and load progress. Not foreclosed. |

---

## 1. Why not `llama-cpp-2`

This was the crate the owner pointed at, so it got a real evaluation rather than a
glance. It is a good crate. It does not fit, on two independent grounds — either
one alone is disqualifying.

### 1.1 It has no SYCL backend

The complete feature list of `llama-cpp-sys-2` (read from `Cargo.toml` on `main`):

```
common, cuda, cuda-no-vmm, metal, dynamic-link, vulkan, opencl, mkl, openmp,
static-openmp, rocm, static-stdcxx, shared-stdcxx, system-ggml,
system-ggml-static, mtmd, dynamic-backends
```

There is no `sycl`, no `oneapi`, no `intel`. `mkl` is Intel MKL as a **host BLAS**
library — CPU, not GPU offload. `build.rs` contains no branch that sets
`GGML_SYCL=ON`, and never arranges the `icpx` / `setvars.sh` toolchain that
llama.cpp's own SYCL CMake requires; setting `GGML_SYCL=ON` by hand through its
env-var passthrough would configure a SYCL build with a GCC toolchain, which does
not work.

Its issue tracker has **zero** hits for "SYCL" and **zero** for "oneAPI". "Intel
Arc" appears once, in a closed issue about the **Vulkan** backend on an Intel
iGPU. ⚠️ Vulkan is not a substitute here: the Vulkan build on this box is
`build-vulkan/`, and it is **4.8× slower** than the SYCL build.

### 1.2 It cannot link our llama.cpp — and that breaks the benchmark protocol

It builds a vendored git submodule through the `cmake` crate, always from
`$CARGO_MANIFEST_DIR/llama.cpp`. There is no `LLAMA_CPP_LIB`-style escape hatch.
The nearest thing, `system-ggml`, swaps out **ggml only** and still compiles
llama.cpp's `src/` and `common/` from the submodule.

Its submodule currently pins llama.cpp at `e79e4bf66` (2026-08-13). MoEArc's
reference build pins `e107984bc`. 🔴 **MoEArc's benchmark protocol requires the
engine and the baseline to be the identical commit** — the pin file
`llama.cpp-COMMIT` exists because a rebuild once silently moved it and left the
SYCL and Vulkan references on different commits. A project that has already
withdrawn every llama.cpp comparison it published cannot adopt a dependency whose
design guarantees two differently-pinned llama.cpps in one tree.

Adopting it would mean forking it to add a SYCL feature *and* forking it again to
unpin the submodule. What is left after those two changes is a build script —
which is the part `crates/moearc-llama/build.rs` replaces in ~180 lines.

**One thing it got right, and we copied the lesson:** it exposes
`add_cpu_moe_override` / `add_cpu_buft_override` rather than a scalar
`n_cpu_moe` field, because that is what the C API actually offers. See §3.

---

## 2. Why a C shim rather than direct FFI

The tempting shortcut is `#[repr(C)] struct LlamaContextParams { … }` in Rust,
mirroring `llama.h`. It should be resisted.

`llama_context_params` is passed **by value**, has 30+ members, and mixes
`uint32_t`, `int32_t`, five different enums, ten floats, three function pointers
and a trailing block of eight bools. `llama.h` itself carries the warning twice:

> *Keep the booleans together and at the end of the struct to avoid misalignment
> during copy-by-value.*

A hand-transcribed Rust mirror is correct only against the commit it was written
for. When the pin moves and a field is inserted, **it does not fail to compile**.
It passes `n_threads` into the slot the callee reads as `n_ubatch`, and the result
is a program that runs, produces plausible text, and is quietly misconfigured.
That is the worst available failure mode for this project specifically: three
results have already been invalidated by measuring something other than what was
reported.

So `shim/moearc_llama_shim.cpp` is the only thing that touches those structs.
`llama.h` defines them; the shim recompiles against whatever the pin says; Rust
sees `MlaModelParams` and `MlaCtxParams`, which are small, flat, all-scalar, and
**ours**. When upstream changes shape, the shim either keeps working or fails at
build time. Neither outcome is silent.

This also matches the house convention: `crates/moearc-kernels` already pairs a
C++ TU with a hand-written `ffi.rs` and documents why bindgen was rejected.

---

## 3. The parameter surface for the tuning layer

`crates/moearc-llama/src/params.rs` is pure Rust with no llama.cpp dependency, so
a tuning plan can be built, tested, serialised and diffed on a machine with no GPU.

### 3.1 Load-time — `ModelParams` → `llama_model_params`

| MoEArc field | llama.cpp | CLI | notes |
| --- | --- | --- | --- |
| `n_gpu_layers` | `n_gpu_layers` | `-ngl` | negative = all |
| `main_gpu` | `main_gpu` | `-mg` | ⚠️ indexes what llama.cpp enumerated; see §5.2 |
| **`n_cpu_moe`** | **`tensor_buft_overrides`** | `-ncmoe` | 🔴 **not a scalar — see §3.2** |
| `split_mode` | `split_mode` | `-sm` | MoEArc defaults `None`, upstream `Layer` |
| `load_mode` | `load_mode` | `--mmap` etc. | `Auto/None/Mmap/Mlock/MmapMlock/DirectIo` |
| `use_extra_bufts` | `use_extra_bufts` | `--no-extra-bufts` | weight repacking |
| `no_host` | `no_host` | `--no-host` | |
| `check_tensors` | `check_tensors` | `--check-tensors` | |
| `vocab_only` | `vocab_only` | — | metadata without weights; very cheap |

### 3.2 🔴 `-ncmoe` is a list of regexes, not an integer

**There is no `n_cpu_moe` field anywhere in `llama.h`.** This is the single most
important thing to get right in the mapping, because a tuning layer that assumes a
scalar will set nothing and report success.

`-ncmoe N` expands to N entries of `llama_model_params.tensor_buft_overrides`, a
NULL-terminated array of:

```c
struct llama_model_tensor_buft_override { const char * pattern; ggml_backend_buffer_type_t buft; };
```

where entry `i` is the regex `blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps` bound
to `ggml_backend_cpu_buffer_type()`. Reproduced in the shim from
`common/common.h` (`LLM_FFN_EXPS_REGEX`, `llm_ffn_block_regex`) and
`common/arg.cpp` (`llm_add_n_cpu_ffn_overrides`), so the behaviour is identical to
the CLI flag without depending on `libllama-common`.

⚠️ *Corrected while writing this:* an earlier draft justified that by calling
`libllama-common` a static archive. **It is not** — `build/bin/libllama-common.so.0.3.0`
is a 5.6 MB shared library and linking it would work fine. The actual reason is
that `common/` is llama.cpp's **CLI and example support code, not its public API**:
`common.h` carries no `LLAMA_API` and no `extern "C"`, its signatures are C++
(`std::string`, `std::vector`), `llm_add_n_cpu_ffn_overrides` is an `inline`
function over a function-local `static std::list`, and upstream makes no stability
promise about any of it. Depending on it would commit MoEArc to C++ ABI
compatibility with llama.cpp's build in exchange for ten lines of string
formatting.

Three properties the tuning layer must know:

- **`-ncmoe` is not `-ngl`.** The pattern matches only `*_exps` tensors. Attention
  and the dense parts of those same blocks stay on the GPU. Verified: at
  `-ncmoe 8` the load still reports `offloaded 17/17 layers to GPU`.
- **The per-block regexes do not alias.** `blk\.1\.ffn_…` cannot match
  `blk.10.ffn_up_exps`, because the character after `blk.1` there is `0`, not the
  `.` the pattern requires. Asserted in `params.rs`.
- 🔴 **Lifetime:** `llama_model_params` holds *borrowed* `const char *`. The
  pattern strings must outlive the load call, and the vector holding them must not
  reallocate after the pointers are taken. The shim reserves up front.

### 3.3 Per-context — `ContextParams` → `llama_context_params`

A context is cheap to rebuild against an already-loaded model, so these are the
knobs to sweep.

| MoEArc field | llama.cpp | CLI | notes |
| --- | --- | --- | --- |
| `n_ctx` | `n_ctx` | `-c` | 0 = model's trained context |
| `n_batch` | `n_batch` | `-b` | logical max per `llama_decode` |
| `n_ubatch` | `n_ubatch` | `-ub` | physical max; bounds prefill working set |
| `n_seq_max` | `n_seq_max` | `--parallel` | |
| 🔴 **`n_threads`** | `n_threads` | **`-t`** | **see below** |
| `n_threads_batch` | `n_threads_batch` | `-tb` | prompt/batch threads |
| `flash_attn` | `flash_attn_type` | `-fa` | `Auto(-1)/Disabled(0)/Enabled(1)` |
| `type_k` | `type_k` | `-ctk` | `ggml_type` |
| `type_v` | `type_v` | `-ctv` | `ggml_type` |
| `offload_kqv` | `offload_kqv` | `--no-kv-offload` (inverted) | |
| `op_offload` | `op_offload` | | host op offload |
| `swa_full` | `swa_full` | `--swa-full` | |
| `kv_unified` | `kv_unified` | `--kv-unified` | |

🔴 **`n_threads` is the field that invalidated every published llama.cpp
comparison this project made.** `llama-bench` defaults to 4 threads regardless of
core count; on the 20-core box that is a fifth of the machine. With `-ncmoe`
putting expert weights in host RAM, decode is CPU-bound on the expert matmuls, so
this is a first-order determinant of throughput, not a detail. `ContextParams`
therefore derives it from `available_parallelism()` rather than hardcoding, the
example prints it as `(explicit)`, and it should never be left implicit anywhere.

It is also settable on a **live context** via `Context::set_n_threads`, which is
the cheap way to run a thread sweep without reloading 59 GiB.

### 3.4 ⚠️ Do not budget host RAM from the mmap'd buffer size

Found while checking that `-ncmoe` actually worked, and worth stating because a
tuning layer will want exactly this number.

With mmap on (`load_mode: Auto`), OLMoE at `-ncmoe 8` reports:

```
CPU_Mapped model buffer size =  2254.16 MiB
     SYCL0 model buffer size =  2101.35 MiB      total 4355.51
```

With mmap off (`load_mode: None`), the same split reports:

```
     SYCL0 model buffer size =  2101.35 MiB
 SYCL_Host model buffer size =  1915.27 MiB      total 4016.62
```

4016.62 MiB is exactly the `-ncmoe 0` total (55.27 + 3961.35). The mmap figure
**overstates host residency by 338.89 MiB — 17.7% of the true host share** —
because the reported size is the *span of the mapping*, which includes the
GPU-resident attention and norm tensors that lie between the expert tensors in the
GGUF file.

📌 Budget from the non-mmap figure, or from tensor bytes, never from
`CPU_Mapped`. ⚠️ Note also that the **buffer name changes** between the two modes
(`CPU_Mapped` vs `SYCL_Host`), so a log parser keyed on one string silently sees
nothing in the other mode.

---

## 4. Proof

Real output, Arc B580, `ONEAPI_DEVICE_SELECTOR=level_zero:0`:

```
$ cargo run -p moearc-llama --features runtime --example generate -- \
      /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf "The capital of France is" 24

loading /zfs/swift/models/olmoe-1b-7b-0924-instruct-q4_k_m.gguf
  desc          olmoe A1.7B Q4_K - Medium
  params        6.92 B
  size          3.92 GiB
  blocks        16
  n_embd        2048
  n_ctx_train   4096
  vocab         50304
  experts       64 per block, 8 routed per token
  n_cpu_moe     0 -> 0 override pattern(s)
  n_threads     20 (explicit)
  n_ctx         2048

prompt: "The capital of France is"
output: " Paris.\n\nThe currency of France is Euro.\n\nThe population of France is 67.2 million people."
stopped on: token budget
```

The backend was verified rather than assumed — a load that silently falls back to
CPU still generates correct text, so the text is not the evidence:

```
llama_prepare_model_devices: using device SYCL0 (Intel(R) Arc(TM) B580 Graphics) - 11959 MiB free
Found 1 SYCL devices:
| 0| [level_zero:gpu:0]|  Intel Arc B580 Graphics|   20.1|    160|    1024|   32| 12168M|   1.14.37020|
```

🔴 **Why the timing is labelled and not quoted.** The same binary, same model,
same prompt, run twice minutes apart on an otherwise idle box, reported
**108.42 tok/s** and **62.05 tok/s** — a 1.75× spread. The *token ids were
identical* both times (greedy decode is deterministic, and the shorter run's
sequence is a prefix of the longer one's), so this is not a correctness question;
it is ordinary contention on a machine running 21 containers and an Incus VM.
📌 A single unpinned run on a shared box is a smoke test. Anyone quoting either
number as a throughput result would be repeating the exact mistake that cost this
project every published llama.cpp comparison.

Build cost, which is a design outcome and not an incidental: **0.6 s from
`cargo clean`**, including the `g++` shim compile. Nothing here rebuilds
llama.cpp, so the `icpx` SYCL compile — minutes, and audible — happens once, out
of tree, and never again during development.

Tests: `cargo test -p moearc-llama --features runtime -- --include-ignored` →
**6 passed**, covering the defaults check, a vocab-only metadata + tokenizer
round-trip, GPU generation, greedy determinism across two runs, and the `-ncmoe`
mapping. Default-feature `cargo test --workspace` is green (~500 tests) and
`cargo clippy --workspace --all-targets -- -D warnings` is clean.

---

## 5. Risks, honestly

### 5.1 🔴 We do not build llama.cpp, so the pin is a human protocol

This crate links whatever is at `/zfs/swift/projects/llama.cpp/build/bin`. That is
a **strength** for benchmark integrity (engine and baseline are the same binary by
construction) and a **weakness** for reproducibility: nothing in `cargo build`
guarantees which commit that is. `build.rs` asserts the build is a *SYCL* build —
`libggml-sycl.so` must be present, and a Vulkan or CPU-only directory is a hard
error rather than a silent fallback — but it does not assert *which commit*.

**Mitigation not yet done:** have `build.rs` read `llama.cpp-COMMIT` and record it
in the binary, so a result can name the llama.cpp it ran against. This should
happen before any number is published.

### 5.2 ⚠️ Device selection is by index, and index 0 is not stably the B580

`sycl-ls` on this box lists the B580 at `level_zero:0` and the iGPU at
`level_zero:1`, but that ordering is not contractual, and the iGPU is deliberately
left enabled in BIOS so Jellyfin's transcoding does not contend with inference.
The proof above pinned it with `ONEAPI_DEVICE_SELECTOR=level_zero:0`.

📌 This is the **"succeeds and lies"** failure the release notes already flag for
the stale-driver problem: offering the iGPU and appearing to work. `main_gpu` is
an index into a list we do not control. The installer should select by **device
name / PCI id** via `moearc-device`, not by trusting index 0.

### 5.3 Packaging is the biggest open question

`packaging/` currently ships **one** `.so` (`libmoearc_kernels.so`) via
`include_bytes!` + extract + `dlopen`, plus an Intel runtime that
`fetch-runtime.py` downloads SHA-pinned from Intel's own channel. Default tarball
**4.8 MB**; `bundle.sh --with-runtime` **29 MB** and explicitly not Apache-2.0.

⚠️ *Corrected while writing this:* an earlier draft said Intel's EULA **forbids**
redistribution. It does not, and the real position (`docs/packaging.md` §"The
licence position") is both more precise and more load-bearing:

- Intel's EULA **grants** redistribution of "Redistributables", defined as the
  files listed in a `redist.txt` — **and no such file exists anywhere in the
  oneAPI 2026.1 installation we build against.** The grant is real; we cannot show
  it covers any particular file.
- Two of its conditions would **propagate to our users**: a no-reverse-engineering
  clause, and a **prohibition on SaaS use** — which is one of the things an
  inference server is for.

So the reason for fetching rather than shipping is not a prohibition, it is that
we cannot demonstrate the grant applies *and* would not want to pass those
conditions on. That distinction matters for what follows: because MoEArc never
redistributes, adding more Intel libraries does **not** change the licence
position at all. It changes only size and audit surface.

Under the new architecture the payload becomes **llama.cpp's whole shared-object
family** — `libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-cpu.so.0`,
`libggml-sycl.so.0` — plus our shim. Concrete consequences:

- **The Intel runtime requirement does not shrink; it grows.** `libggml-sycl.so`
  links **oneDNN** (`libdnnl.so.3`) and **four oneMKL libraries**
  (`libmkl_sycl_blas`, `libmkl_intel_ilp64`, `libmkl_tbb_thread`, `libmkl_core`)
  in addition to `libsycl`/`libsvml`/`libimf`/`libintlc`. `docs/kernel-strategy.md`
  measured these at ~108 MB (oneDNN) and ~193 MB (the four MKL libraries). Per the
  correction above, the licence *mechanism* is genuinely unchanged — Intel
  publishes these on the same channel `fetch-runtime.py` already uses, and the user
  still accepts Intel's terms from Intel. What grows is the **fetch size and the
  audit surface**, plausibly ~5×. `runtime.lock.json`'s allowlist exists
  specifically to keep the runtime small and auditable, so this is a direct
  regression against a stated goal. ⚠️ The `~370 MB` figure quoted in
  `kernel-strategy.md` is an estimate from library sizes on disk, **not a measured
  fetch**, and should be measured before it is repeated. **This needs a decision
  before release.**
- **Licence:** llama.cpp is MIT and the tree becomes mixed-licence. The
  `NOTICE`/`third_party` mechanics in `docs/kernel-strategy.md` §1 apply unchanged
  and are already correct — but they must actually be executed, and the README
  must stop implying the whole tree is Apache-2.0.
- **The soname-as-absolute-path trick does not survive packaging.** It is a
  development-build device; it makes the artifact non-relocatable by construction.
  Packaging must use `$ORIGIN` rpaths (`packaging/elf-relocatable.py` already
  exists for this) or `dlopen`. Not a new problem, but now it applies to six
  objects instead of one.
- 🔴 **Blocker interaction:** the announcement is blocked on there being no release
  tag. Rebuilding the packaging story around a different, larger payload is not a
  small task, and it is now on the critical path to that tag.

### 5.4 What we give up

llama.cpp owns the forward pass, which means it owns residency. MoEArc's
`residency.rs` decides *which experts live on the card*; llama.cpp's `-ncmoe`
decides *which blocks' experts live on the CPU*, statically, at load time. Those
are not the same lever, and the measured advantage of dynamic residency
(**45.08 vs 13.44 with 24× less staged**, MoEArc against itself) is **not**
expressible through this API today.

That is a real loss and it should be stated plainly rather than glossed: the
pivot buys a fast, correct, maintained engine and **defers** the project's
original thesis. The `-ncmoe` split is the only residency knob currently reachable,
and it is static. Whether the dynamic version can be reintroduced above llama.cpp
— and `docs/kernel-strategy.md` §2 suggests the MoE GEMV's raw-pointer API is the
place to look — is the open architectural question, not a settled one.

📌 The allocation finding still applies and should shape the tuning defaults:
**KV residency is all-or-nothing, expert residency degrades gracefully.** So
`offload_kqv` should be the last thing given up, and `-ncmoe` the first.

---

## 6. The subprocess path — ⚠️ no longer a fallback, and now what `moearc serve` does

⚠️ **Corrected 2026-09-08.** This section used to read *"the subprocess fallback, kept
open"*, and described driving `llama-server` as a retreat that had not been taken. It has
been taken, deliberately, for one command: **`moearc serve` supervises `llama-server` as a
child.** The full argument is in [`serve.md`](serve.md) §0; the short form is that
`moearc info` already prints a `llama-server` command line, and serving by running *that
exact argv* means there is one description of a configuration rather than two that can
drift.

🔴 **This does not replace the seam.** The choice is per-command, and the table in §0
still stands for everything else:

| | drives llama.cpp by |
| --- | --- |
| `moearc serve` | **`llama-server` as a supervised child** — see [`serve.md`](serve.md) |
| `bench`, tuning sweeps, anything timed | **the C shim + FFI in this document** |

The reasons below are why the shim exists and why `bench` keeps it. They are reasons
about *measurement*, and none of them is a reason about *serving* — which is exactly why
the two commands answer differently. `llama_perf_context` has no per-request equivalent a
supervisor needs (llama-server returns a `timings` object per response); thread counts are
not changed on a live context while serving; and load progress reaches the user through
the child's own log, which `serve` reads and asserts against.

⚠️ One thing §6 got right and should be kept in view: the subprocess path
*"adds a process to supervise, a port to allocate and a startup race to poll"*. All three
turned out to be real. The startup race is polled on `/health`; the port is the user's;
and the process is supervised with `PR_SET_PDEATHSIG` after a measured failure —
`kill -TERM` on `moearc serve` left the engine reparented to init, holding the model in
VRAM and the port bound. See [`serve.md`](serve.md) §4.4.

---

### 6.1 The original argument, for the record

Driving `llama-server` as a child process remains viable and is not foreclosed by
anything here. It was not chosen because three things are awkward or impossible
across a process boundary: reading `llama_perf_context` for per-run timings
without scraping stderr; changing thread counts on a live context during a sweep;
and giving the TUI real load progress and an error string instead of an exit code.
It also adds a process to supervise, a port to allocate and a startup race to poll
— for a product whose pitch is "paste a URL and it works".

If the FFI proves brittle across llama.cpp pin moves, this is the retreat, and the
`params` module is deliberately independent of the FFI so it would survive the
switch unchanged: the same `ModelParams`/`ContextParams` can render a command line
instead of filling a struct.
