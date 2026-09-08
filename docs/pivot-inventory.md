# Pivot inventory: keep / retire / re-purpose

Written 2026-09-06, alongside [`llama-integration.md`](llama-integration.md).

🔴 **Nothing here has been deleted, and nothing should be until the replacement is
proven.** This is a proposal and a map. The seam (`crates/moearc-llama`) exists and
generates tokens on the B580; the things marked RETIRE below are still the only
working implementations of several behaviours, and several carry measured findings
in their comments that must be harvested into `docs/` before the file goes.

Verdicts:

- **KEEP** — correct and needed under the new architecture, unchanged
- **RETIRE** — superseded by llama.cpp; delete once the replacement is proven
- **RE-PURPOSE** — the idea survives, the wiring changes

## Scale

| crate | Rust | C/C++ |
|---|---:|---:|
| `moearc-cli` | 11,129 | 0 |
| `moearc-device` | 2,943 | 0 |
| `moearc-engine` | 13,853 | 0 |
| `moearc-kernels` | 5,299 | 2,411 |
| `moearc-llama` (new) | 1,662 | 403 |
| `moearc-model` | 4,545 | 0 |
| `moearc-server` | 4,247 | 0 |

Roughly **19,150 Rust + 2,411 C/C++ lines are in the retire/re-purpose blast
radius** (engine + kernels), against ~22,900 lines largely untouched.

---

## 🔴 The finding that should be read before any of the tables

`-ncmoe` is part of **model** params and is applied **at load time**
(`llama_model_params.tensor_buft_overrides`; see `llama-integration.md` §3.2).

**Expert placement is therefore fixed when the model is loaded and cannot change
per token.** llama.cpp offers exactly the `Policy::StaticSplit` that
`residency.rs` was written to beat.

Consequences, stated rather than smoothed over:

- The residency **simulator**, **trace corpus** and **measurement** survive intact
  and become the tuning layer's evidence base — they now choose the best
  `n_cpu_moe` for a given card/model/context.
- The residency **mechanism** — `cache.rs`'s online LRU admission, `moe.rs::stage`,
  `runtime.rs`'s stage-before-compute ordering — has **nowhere to live** on stock
  llama.cpp. Re-purposing it means patching llama.cpp, not driving it.
- So the published headline — *dynamic residency beats a static split, 45.08 vs
  13.44 with 24× less staged* — becomes a claim about a capability MoEArc **would
  no longer ship**, unless that gap is deliberately closed.

⚠️ That is a product decision, not a file decision, and it is flagged rather than
made here. `docs/kernel-strategy.md` §2 notes that ggml-sycl's MoE GEMV is already
a raw-pointer API (`ggml_sycl_mul_mat_vec_q_id`), which is the place to look if the
dynamic version is to be reintroduced above llama.cpp.

---

## `crates/moearc-cli/` — 11,129 lines — survives, with two rewires

Believed to survive; **confirmed, with one correction**: `src/bench/` does not
wholly survive — `bench/timed.rs` is engine-coupled.

📌 Also: **`moearc serve` was never wired to the engine.** `plain.rs:400` printed
*"not wired yet: the inference server arrives with the engine. Nothing is
listening."* and returned `EXIT_NOT_WIRED` (2); the TUI's `Action::Serve` built a view
model from a stub. There was no coupling to unwind — there was a hole to fill.
✅ **Filled 2026-09-08** in a new `src/serve.rs`; `plain.rs::serve` is now four lines
handing over to it. ⬜ The TUI's `Action::Serve` and the `ServeStats` fixture are
**unchanged** — the wired path is the subcommand. (`plain.rs:343` is the same pattern for the *download*
path, which `moearc-model::pull` can now fill.)

| file | lines | verdict | reason |
|---|---:|---|---|
| `src/main.rs` | 64 | **KEEP** | Module wiring + `ExitCode`; names no engine type. |
| `src/cli.rs` | 523 | **KEEP** | clap `Cli`/`Command`/`BenchArgs`/`ServeArgs` + the TUI↔flag parity table. Backend-agnostic. |
| `src/detect.rs` | 143 | **KEEP** | `LevelZeroDevices` impl of `DeviceSource`; joins `moearc_device::detect()` and `sysman::telemetry()` **by UUID** (the two enumerate in different orders). Pure adapter. |
| `src/catalog.rs` | 405 | **KEEP** | `LocalCatalog`, `read_card`, `dominant_expert_quant`. Reads GGUF *headers* only — 6.8 ms for a 687-tensor 59 GiB file. Independent of who runs the forward pass. |
| `src/fit.rs` | 793 | **RE-PURPOSE** | `Fit`, `plan()`, `CONTEXT_LADDER`. Calls `moearc_engine::memory::plan`. The translation layer survives; its output vocabulary must become llama.cpp's params instead of "resident expert slots". |
| `src/host.rs` | 198 | **KEEP** (light rewire) | `RealHost`, `parse_size`, `free_space_for`. Only the import path moves if `host_budget` moves crates. |
| `src/source.rs` | 932 | **KEEP** | The whole seam: `DeviceRow`, `ModelCard`, `TransferPlan`, `ServeSample`, the four source traits, `Stub*` fixtures and the `stubbed`/`stub_note` provenance fields. This is what makes the TUI snapshot-testable. `ServeStats` and `TransferSource` are the two implementations `moearc-llama` now owes. |
| `src/plain.rs` | 691 | **RE-PURPOSE** | `--json`/plain renderers at parity with the TUI. Keeps entirely *except* `serve()`. ✅ Done: `serve()` delegates to `src/serve.rs`; `print_plan` became `pub(crate)` so the two share one renderer. |
| `src/serve.rs` | 🆕 | **KEEP** | The supervised `llama-server`: profile → argv, the `-dev` pin against the iGPU trap, `/health`, `PR_SET_PDEATHSIG`, and the crash diagnosis. See [`serve.md`](serve.md). |
| `src/format.rs` | 92 | **KEEP** | `bytes`/`rate`/`count`/`duration`/`percent`. Zero coupling. |
| `src/theme.rs` | 73 | **KEEP** | Palette + `panel()`/`field()`. |
| `src/tui/mod.rs` | 184 | **KEEP** | `TerminalGuard`, `run()`, `perform()` — calls only trait objects. |
| `src/tui/model.rs` | 939 | **KEEP** | Elm reducer; pure, no I/O, no clock. Imports `host_budget` for the RAM dial only. |
| `src/tui/view.rs` | 1,586 | **KEEP** | Pure render + `TestBackend` snapshot tests. |
| `build.rs` | 63 | **KEEP** | Stamps `MOEARC_BUILD_COMMIT` for benchmark provenance. |

### `src/bench/` — 3,443 lines — mostly keep, one rewire

| file | lines | verdict | reason |
|---|---:|---|---|
| `bench/guard.rs` | 968 | **KEEP** | The refusal engine — `Reading`, `Verdict`, `evaluate()`, `threads()`, `backend()`. **Pure functions of an injected `Reading`.** `threads()` is literally the fix for the 4-threads-on-20-cores retraction: **more** valuable now, not less. |
| `bench/probe.rs` | 339 | **KEEP** | The impure half: `load1()`, `meminfo()`, `zfs_arc()`, `build_facts()`. `GPU_COMPILED_IN` re-points at the llama feature. |
| `bench/incumbent.rs` | 432 | **KEEP** | Runs `llama-bench`, parses `-o csv` **by header name**, reads `n_threads` and `backends` back out. llama.cpp is now both incumbent *and* engine, so this becomes the honest A/B — our tuning vs upstream defaults. Arguably the most important file in the repo. |
| `bench/shape.rs` | 700 | **KEEP** | `measure()`, `replay()`, `elbow()`, `discover()`. Replays committed traces through `residency::simulate` — no clock, no device, bit-identical anywhere. Moves with `residency.rs`. |
| `bench/stats.rs` | 217 | **KEEP** | Mean ± stddev, keeps every raw value so a discard is auditable. |
| `bench/report.rs` | 598 | **KEEP** | `Artefact` + `render()` — human tables and embedded JSON from the same data. |
| `bench/mod.rs` | 675 | **RE-PURPOSE** | Orchestration. Imports `residency::Policy` for `parse_policy`; the absolutes leg re-points at llama.cpp. |
| `bench/timed.rs` | 514 | **RE-PURPOSE** | The **only** engine-coupled bench file (imports `host_experts`, `moe`, `profile`, `session`). The *design* — N independent child processes, decode fenced by the sampling callback so prefill isn't averaged in, cold and warm as separate questions — is exactly right and must be preserved. Only the inner call becomes `moearc_llama::Context::decode` + `Perf`. |

---

## `crates/moearc-engine/` — 13,853 lines — the split runs through this crate

### Survives as the tuning/telemetry layer (llama.cpp has no equivalent)

| file | lines | verdict | main API | reason |
|---|---:|---|---|---|
| `src/residency.rs` | 1,648 | **RE-PURPOSE** | `Trace`, `Policy` (incl. `Optimal` = Belady, `StaticSplit`), `Residency::hit_rate`, `simulate()` | Offline cache simulation over a routing trace. llama.cpp has **nothing** like this — no Belady bound, no policy comparison, no hit-rate curve. This should now *choose* `n_cpu_moe`; its `StaticSplit` is literally a model of what llama.cpp does. **Highest-value file in the crate.** |
| `src/host_budget.rs` | 894 | **KEEP** | `HostBudget`, `Tier`, `place()`, `host_residency()` | Answers "when an expert isn't on the card, is it in RAM or on disk?" Pure arithmetic, no device. llama.cpp does not model this at all. Drives the TUI's host-RAM dial. |
| `src/memory.rs` | 797 | **RE-PURPOSE** | `plan()` (line 395), `max_context_tokens()`, `Reason`, `Headroom::PROVISIONAL` | Subtracts headroom then dense weights, reserves the active-expert floor, splits the remainder per `Bias` — recording a `Vec<Reason>` rationale so the CLI shows its work. **The arithmetic and the rationale are the product**; llama.cpp makes you guess `-ngl`/`-ncmoe`/`-c` by hand. Output unit must be re-expressed as llama.cpp params. |
| `src/profile.rs` | 101 | **RE-PURPOSE** or retire | `Phase`, `scope()`, `report()` | Near-zero-cost phase timer. Its phase names were our forward pass's. ⚠️ `llama_perf_context` (exposed as `Perf`) may make it simply redundant; 101 lines either way. |

### Retires with the forward pass

| file | lines | verdict | reason |
|---|---:|---|---|
| `src/moe.rs` | **2,911** | **RETIRE** | Our whole forward pass — RMSNorm, QK-normed attention, NeoX RoPE, softmax router, SwiGLU experts — transcribed from llama.cpp's own `olmoe.cpp`/`qwen3moe.cpp`. Superseded by the original. ⚠️ **Harvest its doc comments first**: the ~85 %-of-reported-free allocation cliff and `malloc_device` succeeding past physical VRAM are real measured findings. |
| `src/host_experts.rs` | **1,466** | **RETIRE** (harvest the idea) | Multithreaded CPU expert FFN **overlapped** with GPU work via a fork-join epoch handshake. The *claim* (overlap, not substitution) is genuinely absent from llama.cpp — `-ncmoe` is substitution — but it can only exist inside a forward pass we control. The `frac:0.75` finding becomes a docs artifact. |
| `src/session.rs` | 599 | **RETIRE** | Device-on-its-own-thread mailbox in front of `moe::Model`. 📌 **Copy its contract verbatim** — blocking, `&self`, `on_token` once per accepted token in order, `false` stops — because `moearc-server`'s `Generator` was written against exactly that shape. |
| `src/cache.rs` | 381 | **RETIRE** | The **online** LRU slot manager with per-step pinning. Unreachable under load-time `tensor_buft_overrides`. *(Retiring it does not weaken the simulator — `residency::simulate` is a separate implementation.)* |
| `src/kv.rs` | 353 | **RETIRE** | vLLM-style paged KV allocator. llama.cpp owns its KV cache entirely. ⚠️ **But** `KvUsage::utilisation()` feeds `ServeSample::kv_utilisation` in the TUI — that dial needs a llama.cpp-side source or it goes dark. |
| `src/runtime.rs` | 411 | **RETIRE** | The token loop's ordering contract (route → admit → stage → compute). Orchestrates the pass llama.cpp now owns. |
| `src/lib.rs` | 27 | **RE-PURPOSE** | Shrinks; the `gpu` feature and its optional deps disappear. |

### `examples/` and `tests/`

| file | lines | verdict | reason |
|---|---:|---|---|
| `examples/plan_b580.rs` | 156 | **KEEP** | The only example with no `gpu` gate — pure `memory::plan` over measured B580 geometry. Documentation-as-code. |
| `examples/trace_report.rs` | 184 | **KEEP** | No GPU. Working set + implied residency from a captured trace. |
| `examples/ctx_attrib.rs` | 424 | **RE-PURPOSE** | **Produced the 80 % staging vs 6.5 % attention finding.** The question is still live under llama.cpp ("is it attention or `-ncmoe` traffic?"). |
| `examples/ctx_curve.rs` | 485 | **RE-PURPOSE** | Separates `n_ctx` (an allocation) from actual prompt depth — the distinction `llama-bench -d` makes. |
| `examples/residency_sweep.rs` | 247 | **RE-PURPOSE** | Asserts **every row emits identical token ids** — a paging-bug detector. The analogue is sweeping `n_cpu_moe` with the same invariance check. Worth keeping as a shape. |
| `examples/hybrid_sweep.rs` | 254 | **RETIRE** | Both axes are `host_experts`. |
| `examples/profile_decode.rs` | 231 | **RETIRE** | Phases won't exist. |
| `examples/host_expert_bench.rs` | 221 | **RETIRE** | Dies with `host_experts.rs`. |
| `examples/olmoe_generate.rs` | 80 | **RETIRE** | Superseded exactly by `crates/moearc-llama/examples/generate.rs`. |
| `tests/adversarial_corpus.rs` | 170 | **KEEP** | No GPU. Replays zero budgets / single-byte pages / reserves > budget through `memory::plan` asserting *invariants*. Follows `memory.rs`. |
| `tests/olmoe_forward.rs` | 357 | **RETIRE** | Asserts our forward pass against llama.cpp. llama.cpp cannot disagree with itself. |
| `tests/qwen3moe_forward.rs` | 382 | **RETIRE** | Same. |
| `tests/gptoss_forward.rs` | 509 | **RETIRE** | Same. |
| `tests/host_experts_gpu.rs` | 300 | **RETIRE** | Both sides going. |

---

## `crates/moearc-kernels/` — retires almost entirely; confirmed

| file | lines | verdict | reason |
|---|---:|---|---|
| `kernels.cpp` | 1,922 | **RETIRE** | Every SYCL kernel. Superseded by `libggml-sycl.so`. |
| `src/lib.rs` | 1,253 | **RETIRE** | `Context`, `DeviceBuffer`, ~40 kernel methods. The seam llama.cpp replaces. |
| `src/ffi.rs` | 266 | **RETIRE** | Hand-written decls for the above. |
| `build.rs` | 144 | **RETIRE — but read it first** | Documents the `_intel_fast_memcpy`/`libintlc` static-link wall and the `DT_SONAME`-carries-the-path trick. `moearc-llama/build.rs` already cites and inherits both. Its knowledge survives; its code does not. |
| `tests/*` (6 files) | 2,169 | **RETIRE** | Every kernel and forward-pass crosscheck against ggml. |
| `examples/*` (2) | 269 | **RETIRE** | Matvec scaling; launch overhead. |
| `tools/*.c` | 489 | **RETIRE** | Golden-file generators for tests that are going. |

### Two exceptions

🔴 **`src/reference.rs` — 931 lines — RE-PURPOSE, do not delete.**
Not a kernel. A pure-Rust CPU oracle with **no `unsafe`, no FFI, no SYCL, no
device**, depending on nothing else in the crate. Exports `QuantType`, `dequant`,
`matvec_q`, `rmsnorm`, `swiglu*`, `softmax_ext`, `rope` (+ YaRN), `topk_router`,
`f16`↔`f32`, `attn_decode`. Reductions accumulate in `f64` where ggml uses
`ggml_float`, so it is **the higher-precision side of every comparison, not a
mirror**. Usable as an oracle for llama.cpp's dequantisation and elementwise/matvec
paths — it was already cross-checked against ggml's own `to_float` over real
tensors. It becomes the second opinion when a quant path is suspect.
⚠️ Its `QuantType` block-geometry table duplicates `moearc-model/src/quant.rs`;
resolve that duplication deliberately if it is kept.

**`src/bin/moearc-kernels-smoke.rs` — 24 lines — RE-PURPOSE.**
"A binary whose only job is to be started." It exists because **309 tests passed
while `moearc-server` died in the dynamic loader before `main`** — tests inherit
the build script's rpath and nothing downstream does. 🔴 **The same trap exists
verbatim for `libllama.so`/`libggml-sycl.so`.** Keep the idea as a llama-linked
`moearc-selftest`; `packaging/bundle.sh:73` already ships it under that name.

---

## `crates/moearc-model/` — mostly keeps; one file becomes redundant

llama.cpp parses GGUF itself — but only *after* you load a model. Everything MoEArc
does before that (list what's on disk, size a download, decide what fits) still
needs a header reader.

| file | lines | verdict | reason |
|---|---:|---|---|
| `src/pull.rs` | 1,164 | **KEEP** | The clearest survivor. Resumable (append-only `.part`, `SIGKILL`-safe), **silent** (progress via callback so it lives under the TUI), and **verified** (declared size + its own GGUF parse before reporting success). llama.cpp has no downloader worth this. |
| `src/gguf.rs` | 483 | **KEEP** | Header-only reader; reads ~11 MB of a 20.6 GiB file and stops. Every length checked against real file length before allocating. Makes `moearc ls` cost 12.6 ms across 103 GiB. |
| `src/quant.rs` | 160 | **KEEP** | The ggml type table, cross-checked against `ggml.c`. Removed ids deliberately absent so a bogus file gets `None`. |
| `src/lib.rs` | 854 | **KEEP** | `ModelInfo`, `inspect()` — the four facts the planner needs, from the header alone. |
| `src/tensors.rs` | 929 | **RETIRE** | `MappedModel`, `TensorView`, `slice_last_dim`, `ExpertBank` — zero-copy addressing into the tensor blob so *our* forward pass could reach one expert inside a stacked tensor. llama.cpp owns weight loading now. ⚠️ Verified: `catalog.rs::card_from` uses only `gguf::GgufHeader` + `ModelInfo`, so it does not depend on this. |
| `examples/pull.rs` | 157 | **KEEP** | Exercises `pull` standalone. |
| `examples/inspect.rs` | 73 | **KEEP** | Eyeball our header read against `llama-gguf`. |
| `examples/map.rs` | 177 | **RETIRE** | Evidence for `tensors.rs`. |
| `examples/expert_probe.rs` | 94 | **RETIRE** | Dies with `slice_last_dim`. |
| `tests/pull_hub.rs` | 203 | **KEEP** | Live-Hub integration, gated. |
| `tests/real_model.rs` | 93 | **KEEP** | Header facts against a real GGUF. |
| `tests/mapped_model.rs` | 158 | **RETIRE** | Tests `MappedModel` offsets. |

---

## `crates/moearc-device/` — KEEP entirely. More load-bearing now, not less.

**llama.cpp's SYCL backend does not make this redundant**, for four independent
reasons:

1. **Ordering.** llama.cpp's enumeration needs `libsycl.so.9` + the UR adapters
   loadable. `packaging/launcher.sh:37–41` gives the `moearc` binary
   `needs_runtime=0` — it **must run and explain the machine before the SYCL
   runtime has been fetched**. A pre-flight detector that needs the thing it is
   checking for is not a pre-flight detector.
2. **Failure legibility.** `src/ze.rs` `dlopen`s `libze_loader.so.1` rather than
   linking it, so a machine without it gets *our sentence* instead of
   `error while loading shared libraries` before `main`.
3. 🔴 **`fitness.rs` has no llama.cpp counterpart and encodes a measured finding.**
   On the reference box the **iGPU reports 91,890,372,608 bytes (85.58 GiB)** of
   "device memory" — 93.5 % of system RAM — confirmed across three independent APIs
   and **not** a stale-driver artefact. `vram_budget()`, `inference_target()` and
   `VERIFIED_RUNTIME_BUILD = 37_020` are the guard against the release-blocking
   "stale driver offers the iGPU and *succeeds and lies*" problem.
   **llama.cpp will happily pick that device.**
4. **`pci.rs` answers what Level Zero cannot.** Scans `/sys/bus/pci/devices` with
   **no PCI-ID table** to distinguish "no GPU" from "there is an Arc in the slot
   but no driver is bound".

Confirmed independently: `moearc-llama`'s shim exposes **no device-enumeration
function at all**. There is currently no path from llama.cpp back to "which device,
how much free VRAM".

| file | lines | verdict |
|---|---:|---|
| `src/lib.rs` | 892 | **KEEP** |
| `src/fitness.rs` | 616 | **KEEP** — the iGPU trap + driver caution |
| `src/sysman.rs` | 602 | **KEEP** — live free-VRAM telemetry, dlopen'd, deliberately unable to fail detection |
| `src/ze.rs` | 302 | **KEEP** — Level Zero FFI, field offsets pinned by tests |
| `src/pci.rs` | 232 | **KEEP** — sysfs second opinion |
| `src/bin/report.rs` | 178 | **KEEP** |
| `tests/hardware.rs` + `tests/legible_failure.rs` | 121 | **KEEP** — the negative control |

**One small rewire:** `bench/guard.rs:597` reads `fitness::VERIFIED_RUNTIME_BUILD`
to gate results on driver version; that constant must now also cover llama.cpp's
SYCL requirements.

---

## `crates/moearc-server/` — KEEP 4,132 of 4,247; one 115-line file rewires

Every file was grepped for `moearc_engine`. **Exactly one imports it.**

⚠️ `grep -rl moearc_engine crates/moearc-server/src/` returns **three** files, which
looks like three couplings and is not: `lib.rs:46` and `generate.rs:18` are both
`//!` doc-comment examples showing the intended integration. Only `engine.rs` has
`use` statements (lines 15, 72–74, 100). Verified line by line.

| file | lines | verdict | reason |
|---|---:|---|---|
| `src/engine.rs` | **115** | **RE-PURPOSE** | The *entire* engine coupling. Becomes `impl Generator for moearc_llama::Context`, roughly the same size. |
| `src/generate.rs` | 316 | **KEEP** | Defines `Generator`, `StopReason`, `EchoGenerator`. **Zero engine dependency by design.** |
| `src/state.rs` | 169 | **KEEP** | `generator: Arc<dyn Generator>` is "the *only* place the inference engine enters this crate". |
| `src/routes.rs` | 721 | **KEEP** | OpenAI routes + SSE framing. Names only `Generator`. |
| `src/openai.rs` | 412 | **KEEP** | Wire types; unsupported options (`n=4`) **refused by name**, not ignored. |
| `src/chat.rs` | 400 | **KEEP** | minijinja rendering of the model's own `chat_template`. |
| `src/tokenize.rs` | 548 | **KEEP** ⚠️ | HF `tokenizers`; SentencePiece **refused rather than approximated**. See the open decision below. |
| `src/sampling.rs` | 386 | **KEEP** ⚠️ | Pure `&[f32]`; order documented and deliberate. See below. |
| `src/gguf.rs` | 298 | **KEEP** | Metadata-only reader, explicitly scoped away from tensor data. |
| `src/error.rs` | 160 | **KEEP** | OpenAI error envelope. |
| `src/testing.rs` | 79 | **KEEP** | `tiny_tokenizer` shipped in the library so the socket test can use it. |
| `src/bin/moearc-server.rs` | 204 | **RE-PURPOSE** | One expression changes. |
| `tests/http.rs` | 362 | **KEEP** | Real socket, real SSE, engine-independent. |

---

## `packaging/` — survives structurally; three specific things break

The launcher / `LD_LIBRARY_PATH` / fetch-runtime architecture is **right** for
llama.cpp — more so, since llama.cpp's SYCL build already produces shared objects.
But it assumes **one** object with **one** pathological property.

1. 🔴 **`elf-relocatable.py` may become a no-op that silently stops protecting
   you.** Its trick works because `moearc-kernels/build.rs` set `DT_SONAME` to an
   absolute path, of which the bare name is a suffix — so it only adjusts a string
   offset. **llama.cpp's CMake sets normal short sonames**, so there is nothing to
   shorten. The script is safe to run but does nothing; **the guard at
   `bundle.sh:117–125`** (assert no `DT_NEEDED` contains a slash) is the part to
   keep, now run over `libllama.so`, `libggml*.so` and their transitive deps.
2. 🔴 **`bundle.sh` is hard-wired to a single object** — line 64 globs exactly
   `libmoearc_kernels.so`, line 66 fails without it, line 84 asserts the server
   links it. All three must change: one file becomes five-plus, and
   `libggml-sycl.so`'s own `DT_NEEDED` names `libsvml`/`libirng`/`libimf`/
   `libintlc.so.5`, so `launcher.sh`'s `LD_LIBRARY_PATH` ordering becomes
   load-bearing in a way it was not before.
3. 🔴 **The `moearc` binary's "needs no SYCL runtime" exemption must be defended.**
   It holds only while the device report goes through `moearc-device`'s dlopen'd
   Level Zero. Wire that report through llama.cpp and **the tarball stops working
   on first unpack** — the exact user-facing property the packaging exists to
   deliver. A second independent argument for keeping `moearc-device`.
4. ⚠️ **Size.** `dist/` is currently a 5.0 MB tarball with one kernel object.
   Adding llama.cpp's shared objects multiplies it; `RELEASE.md` asset names and
   `install.sh` expectations need re-checking.
5. ⚠️ **`verify-clean.sh` still works and matters more** — but relabel what it
   proves: the reference-id comparison becomes llama.cpp-vs-llama.cpp, i.e. a
   regression check on **our parameter plumbing** rather than on our kernels.

| file | lines | verdict |
|---|---:|---|
| `README.md` | 107 | **RE-PURPOSE** — layout diagram names the old object |
| `bundle.sh` | 191 | **RE-PURPOSE** — the three hard-wired references |
| `launcher.sh` | 70 | **KEEP** — the dlopen-gap fix now covers more libraries |
| `fetch-runtime.py` | 196 | **KEEP** — stdlib-only, digest-verified |
| `runtime.lock.json` | 67 | **KEEP** — re-verify pins against llama.cpp's SYCL build |
| `elf-relocatable.py` | 123 | **RE-PURPOSE** — likely shrinks to a verifier |
| `verify-clean.sh` | 165 | **KEEP** — relabel what it proves |
| `Containerfile.clean` | 52 | **KEEP** — `DRIVER=distro` reproduces the too-old-driver case |
| `install.sh` | 212 | **KEEP** — still blocked on the missing release tag |
| `RELEASE.md` | 213 | **RE-PURPOSE** — asset list changes |
| `THIRD-PARTY.md` | 95 | 🔴 **RE-PURPOSE — required.** Shipping llama.cpp binaries is a **new licence obligation** (MIT + attribution) this file does not cover. Apache-2.0 + MIT are compatible, but the NOTICE must be written **before** the release tag. |

---

## Top-level

| path | verdict | reason |
|---|---|---|
| `bench/PROTOCOL.md` | **KEEP — the most valuable non-code asset in the repo** | The spec `guard.rs` implements; every rule cites a withdrawn result. §1 is the 4-threads failure. |
| `bench/traces/` (26 `.ndjson` + `capture.sh` + eval-callback patch) | **KEEP** | **Irreplaceable data, not code.** What `residency::simulate` and `bench/shape.rs` replay to get bit-identical results anywhere. Regenerable via `capture.sh`. |
| `bench/references/` (5 `.ids`) | **KEEP** | Golden token ids; now the regression oracle for parameter plumbing. |
| `bench/baselines/` | **KEEP as the record** ⚠️ | Retractions stay in the tree with their evidence. **Do not quote from them** — this is where the withdrawn numbers live. |
| `bench/policy-sweep.md` | **KEEP** | Nine policies against Belady; the "+44 % slots beats any policy by ~3×" finding. Survives as tuning evidence. |
| `bench/results/` | **KEEP** | Includes the `-t` measurements that caused the withdrawal and a file named `...-DISCARDED-round1.txt`. Audit trail. |
| `bench/tuning/` | **RE-PURPOSE** | Shell sweep harness; loop shape survives, invocations change. |
| `bench/reproduce.sh` | **RE-PURPOSE** | Shipped in the tarball. Pins engine concepts, but its **provenance-first** structure — box, commit, model, runtime, the number, then an explicit list of what it did **not** measure — is the pattern to preserve. |
| `tools/gen_adversarial_corpus.py` | **KEEP** | Feeds the surviving planner test. |
| `tools/scan-secrets.sh` | **KEEP** | Architecture-independent. |
| `tools/vram_probe.cpp` | **KEEP** | Produced a finding that **outlives our engine**: `malloc_device` returns valid pointers ~38 GiB past an 11.33 GiB card because pages commit on touch, so it writes every block before counting. **That trap applies to llama.cpp too.** |
| `tools/stream_bench.cpp` | **RE-PURPOSE / archive** | The "MoE serialisation costs ~2 %, not 2×" result that killed speculative prefetch. |
| `registry/`, `site/`, `src/{cli,engine,kernels,server}/`, `third_party/ipex-llm/` | **RETIRE** | All contain **only `.gitkeep`**, untouched since 2026-09-02 — the pre-`crates/` layout, never populated. |
| `dist/` | **RETIRE the artifact, keep the dir** | Holds a built 5.0 MB tarball packaging the **old** architecture; invalid the moment the engine swaps. Must not be published. |

---

## What actually has to be built

Five gaps, all small relative to what is being deleted:

1. `moearc-server/src/engine.rs` — 115 lines, `Session` → `moearc_llama::Context`.
2. ~~`moearc-cli`'s `serve` path — the never-wired hole, now wireable for the first
   time.~~ ✅ **Done 2026-09-08** — `crates/moearc-cli/src/serve.rs`. It **supervises
   `llama-server`** rather than embedding a server, so gap 1 below is *not* on its
   critical path and the tokenizer/sampler question at the foot of this document stays
   open on purpose. Verified end to end on the B580, including a 59.0 GiB gpt-oss-120b.
   Reasoning and proof: [`serve.md`](serve.md).
   🔴 It found two things this document should carry:
   **(a)** `tuning::resolve::derive` applies the trained-context cap only on its MXFP4
   path, so `moearc info olmoe-1b-7b-0924-instruct` prints `-c 47360` for a 4,096-token
   model — `serve` re-plans around it, `info` is still wrong (`serve.md` §3.4).
   **(b)** a supervised child needs `PR_SET_PDEATHSIG`; `Drop` and terminal `Ctrl-C`
   together are not sufficient (`serve.md` §4.4).
3. `bench/timed.rs` — swap the inner call, keep the process-spawning design.
4. `moearc-cli/src/fit.rs` — re-express `memory::plan`'s output as llama.cpp params.
5. `source.rs`'s two fixture traits: `TransferSource` (pure wiring — `pull` exists)
   and `ServeStats`. ⚠️ `Perf` covers tok/s, but **`kv_utilisation` and
   `expert_hit_rate` have no llama.cpp source** — the serving panel loses two dials
   unless something is built.

## Two decisions deliberately not made here

- 🔴 **Which tokenizer and which sampler is authoritative.** `moearc-server` has
  both (`tokenize.rs`, `sampling.rs`); `moearc-llama` now also has both
  (`mla_tokenize`, `Sampler`). Running two tokenizers in one system is a
  silent-divergence risk; two samplers means two answers for seed 42.
- 🔴 **Whether the dynamic-residency claim is retired along with its mechanism.**
  See the finding at the top.
