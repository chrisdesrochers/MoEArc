# Product ergonomics

> **This is a constraint on every module, not a phase at the end.** If a component needs the
> user to know a number, that component is unfinished.

MoEArc exists because running a large MoE model on an Intel Arc card is currently a science
experiment. The engine work is only half the product; the other half is that a person with a
B580 can go from nothing to a running server without learning what an expert slot is.

## The whole user journey

```
$ curl -fsSL https://moearc.dev/install.sh | sh     # one static binary, no runtime
$ moearc                                            # detects the card, says what will fit
$ moearc pull qwen3-30b-a3b                         # or any HF repo id
$ moearc serve qwen3-30b-a3b                        # OpenAI-compatible on :8080
```

Four commands, and the middle two are optional once a model is present. Anything that forces a
fifth needs justification.

### 1. Install is one file

A single statically linked binary. **No Python, no conda, no toolkit version to match, no
`pip install` that resolves a different wheel than the docs assumed.** This is the first and
biggest reason the engine is Rust: the hard part of GPU inference tooling today is rarely the
math, it is the environment. We refuse to have one.

The SYCL kernels are the one component that cannot be Rust — they ship as a bundled shared
library behind an FFI seam, not as something the user builds.

**The binary brings its own dependencies.** MoEArc must never hand the user a list of things to
go install. Today, getting an Arc card to run inference means chasing oneAPI runtimes, Level
Zero loaders and compute-runtime packages across distro versions, and the failure mode is a wall
of messages about missing libraries. We absorb that:

- The oneAPI/SYCL runtime and Level Zero loader are **bundled or fetched by the installer**, not
  prerequisites. The user installs one thing.
- The only genuine external requirement is the **kernel-side GPU driver** (`xe` / `i915`), which
  ships with the kernel and cannot be vendored. If it is missing, that is the *one* thing we ask
  for — named exactly, with the reason, and nothing else alongside it.
- No message the user sees may be a dependency complaint they are expected to resolve. Either we
  install it, or we state precisely what is missing and why we cannot.

### 2. It finds the hardware itself

`moearc` with no arguments enumerates the devices, names them, reports usable VRAM, and states
plainly what it can and cannot run. A user who has just installed it should learn whether their
card works from the tool, not from a forum thread.

Failure here must be *legible*: "no Level Zero device found — is the `xe` driver loaded?" beats
a stack trace. Known-bad configurations get named. (We already have one: `sycl-ls` reporting
CPU-only because oneAPI was not sourced looks exactly like a dead GPU stack, and cost us real
time. The tool should recognise that state and say so.)

### 3. Models are discoverable, not prerequisites

A curated list of known-good MoE models with their measured footprint on *this* card, plus
"paste any Hugging Face repo id". Download with a progress bar, resumable, checksummed.

The curated list carries measured numbers, not estimates. A model we have not run does not get
a green checkmark.

### 4. The split is computed, not configured

**This is where the engine work meets the product.** `plan_cache_budget` exists precisely so the
user never types a number: given the card, the model and the measured free VRAM, MoEArc decides
how many expert slots stay resident and how many KV pages to allocate.

Configuration is an *override for people who want it*, never a prerequisite:

```
moearc serve <model> --ctx 32768          # what a user actually thinks about
moearc serve <model> --moe-cache 48       # escape hatch, rarely needed
```

The unit a user reasons in is context length, not pages. Translating that into pages is our job.

🔴 **Defaults must be measured on the user's card, never inherited.** Two constants already
found in ported code make this concrete: `memory_ratio = 0.9` was tuned against the CUDA caching
allocator, and the `992` slot cap is a property of NVIDIA's Marlin kernel. Shipping either as an
Arc default would be a guess wearing the costume of a default. See `calibration.md`.

### 5. Serving is the boring part

OpenAI-compatible `/v1/chat/completions` on a predictable port, so every existing client works
unchanged. Startup prints what it decided — devices, split, context — so a user can see the
reasoning without enabling debug logging.

## What this rules out

- Config files required before first run.
- Numbers in the quickstart that the user cannot derive.
- Build steps on the user's machine.
- Errors that report a symptom without naming the cause.
- Benchmarks published from a run we know was contaminated.

## Open

- ⬜ Name the binary's zero-arg output format (device report).
- ⬜ Decide the curated model list's source of truth and how measured footprints get in.
- ⬜ Installer hosting — `moearc.dev` is aspirational, not registered.
