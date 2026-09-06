# MoEArc

**Run 30B–120B mixture-of-experts models on an ordinary Intel Arc gaming PC.**
One command to install, one command to run, OpenAI- and Anthropic-compatible
out of the box.

> **Status: the thesis is demonstrated in the engine.**
>
> ✅ **Qwen3-30B-A3B — 17.3 GiB of model — runs on an 11.3 GiB Arc B580**, with greedy
> output matching llama.cpp **token id for token id** (64/64), independently re-verified.
>
> ✅ **Dynamic residency beats a static split at matched capacity: 45.08 vs 13.44 tok/s,
> and 24× less data across the bus.** Identical output. That is the claim this project
> exists to make, measured in the engine rather than simulated.
>
> ✅ **45.08 tok/s against llama.cpp's 50.13 on the same model and card — 90% of it.**
> And llama.cpp needs 24 of 48 MoE layers pinned to the CPU permanently to run this model
> at all; MoEArc decides per block, per token.
>
> 🔴 **Not yet:** no batched prefill at all, so there is no counterpart to llama.cpp's
> 3218 tok/s prefill. Every matvec runs at 25–29% of the card's peak bandwidth against
> llama.cpp's 63%. And no adaptive residency policy — the numbers above come from constant
> policies chosen by sweeping, not by the engine deciding for itself.

## Install

**One command, and you do not need to know what oneAPI is.**

```sh
curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
moearc                       # device report: what card you have and what will fit on it
```

⚠️ **Pre-release, and this is the honest state of it.** There is no tagged release yet, so the
installer has nothing to download until there is one. Build the tarball yourself and point the
installer at it:

```sh
packaging/bundle.sh --build              # needs Intel DPC++ on the BUILD machine, never on yours
MOEARC_TARBALL=dist/moearc-*.tar.gz sh packaging/install.sh
```

### What you need

| | |
|---|---|
| an Intel Arc GPU | B-series (Battlemage) or A-series. Also runs, slowly, on Arrow Lake / Lunar Lake iGPUs. |
| the kernel GPU driver | `xe` for B-series, `i915` for A-series. Ships with your kernel. |
| the Intel GPU user-space driver | `libze1` + `libze-intel-gpu1` + `libigdgmm12` (Debian/Ubuntu), `intel-level-zero-gpu` (Fedora), `intel-compute-runtime` (Arch) |
| glibc 2.39 or newer | Ubuntu 24.04+, Fedora 40+, Debian 13+. **Debian 12 and RHEL 9 are too old.** |
| python3 | used once, to fetch the SYCL runtime |

🔴 **The GPU driver is the one dependency we are allowed to have, and its version matters more
than you would expect.** Three things were measured on a B580, and each fails later than the
last — which is the dangerous part, because the earlier steps keep passing:

| driver | `moearc` finds the card | SYCL starts | a model actually runs |
|---|---|---|---|
| Ubuntu 24.04 stock (build 27642) | ❌ reports your **iGPU** instead | — | — |
| Intel's client repo for 24.04 (25.18.33578) | ✅ | ✅ | ❌ `host-to-device copy failed` |
| Ubuntu 26.04 (26.05.37020) | ✅ | ✅ | ✅ |

**On Ubuntu 24.04, add [Intel's GPU repository](https://dgpu-docs.intel.com/driver/client/overview.html)**
— and be aware that inference has been verified only against the newer stack. A recent distro,
or Intel's current repo, is the supported answer.

📌 **`libigdgmm12` is a `Recommends`, not a `Depends`.** If it is absent, detection succeeds, a
SYCL queue is created, and only the model load dies — with an abort inside the driver's
`gmm_helper`. Install it explicitly.

**That list is the whole list.** No oneAPI toolkit, no Python environment, no conda, no
matching a wheel to a driver version.

The tarball is **4.8 MB**. The SYCL runtime MoEArc needs is downloaded once at install time
from Intel's own published packages, pinned by SHA-256 — about 230 MB fetched, 78 MiB kept, for
a 92 MB install. It is deliberately **not** redistributed inside our tarball;
[`packaging/THIRD-PARTY.md`](packaging/THIRD-PARTY.md) says exactly why not, and the short
version is that Intel's redistribution grant points at a file list that does not exist in the
toolkit we build against, and two of its conditions would follow you home.

For an air-gapped machine, `packaging/bundle.sh --with-runtime` produces a 29 MB tarball that
needs no network at all — proven under `podman --network none`. Read `THIRD-PARTY.md` before
you publish one: that archive is not wholly Apache-2.0.

### Verifying it on your machine

```sh
moearc --no-tui        # names your card, or says precisely what is missing
moearc-selftest        # loads the SYCL kernels and reports the device they found
```

This has been run end to end on a machine that has never had oneAPI installed — Ubuntu 24.04,
glibc 2.39, an empty environment — and finds the B580. The procedure is
`packaging/verify-clean.sh`, and it is a container rather than a unit test on purpose: a
previous packaging bug survived 309 green tests because every one of them ran in a shell where
`setvars.sh` had been sourced.

### Reproducing the numbers below

```sh
bench/reproduce.sh /path/to/Qwen3-30B-A3B-Q4_K_M.gguf
```

Prints the box, the driver, the commit, the model and the runtime *before* it prints a number,
runs the headline configuration three times because the run-to-run spread on this hardware is
about ±10%, and ends with an explicit list of the claims on this page that it did **not**
measure. It also prints the box's load average and refuses to call a contended run citable.

⏱️ Budget time: each repeat reloads the model, and on the reference box a single 128-token run
of the 30B model takes several minutes. `--quick` does one short run and checks the harness
rather than the number. See [`bench/README.md`](bench/README.md) for the protocol.

### Building from source

You need Intel's DPC++ compiler (`icpx`) and a Rust toolchain. That requirement lands on the
build machine and never on a user's — which is the entire point of the packaging above.

```sh
source /opt/intel/oneapi/setvars.sh
cargo build --release -p moearc-cli -p moearc-server \
  --features moearc-server/engine,moearc-engine/gpu
```

📌 A `cargo build` produces a **development** build: the kernel object's soname carries this
build tree's absolute path, so the binaries only run on the machine that built them. That is
deliberate, and `packaging/bundle.sh` is what makes a copy that travels.
[`docs/packaging.md`](docs/packaging.md) explains both.

## Why

MoE models are the reason a consumer box can punch above its VRAM: only a
fraction of the parameters are active per token, so attention and the KV cache
can live in VRAM while routed experts stream from system RAM.

That approach is proven — on NVIDIA. [FreeToken](https://github.com/FlashML-org/FreeToken)
does it well: bandwidth-adaptive CPU–GPU co-execution, an LRU expert cache, elastic VRAM.
It supports **RTX 30/40/50 and nothing else.**

**Arc owners have no equivalent, and Arc now ships the cheapest VRAM per dollar on the
market.** Intel's own `ipex-llm` got there first and was archived on 2026-01-28 with no
community fork. Ollama on Arc falls back to Vulkan — which we measured at **4.8× slower
than SYCL on the same card**. llama.cpp SYCL is genuinely good, but its CPU/GPU split is
*static*: you pick `--n-cpu-moe` by hand, and picking it wrong costs you 20% or fails to
load at all.

**MoEArc is FreeToken for Intel GPUs** — the same co-execution ideas, SYCL-first, with an
install experience that does not require you to know what oneAPI is.

**Inspired by, not derived from.** FreeToken proved the shape of the answer on NVIDIA and we
are grateful for it. MoEArc is an independent implementation in Rust: no FreeToken code is
vendored, linked, or ported here, and where their design carries NVIDIA-specific constants we
deliberately do not inherit them. We intend to benchmark honestly against them — see
`bench/README.md` for what that comparison can and cannot show.

This is an open-source project because the gap is a community problem, not a private one:
anyone who bought an Arc card to run local models is currently choosing between a slow
Vulkan path and hand-tuning flags. Contributions and hardware reports are welcome.

## Design in one paragraph

VRAM holds attention weights, the KV cache, shared experts and an LRU cache of
routed experts. System RAM holds the full expert store in pinned memory. Per
expert, per step, the scheduler picks the cheapest of three options: run it from
the VRAM cache, fetch it over PCIe and run it on the GPU, or run it on the CPU
and ship only the activation. The crossover is calibrated on first launch and
refined online, so the same binary behaves sensibly on a B580 over PCIe 4.0 x8
and a B70 over 5.0 x16 — with no flags.

See [`docs/architecture.md`](docs/architecture.md).

## Hardware

| Tier | Hardware |
|---|---|
| Reference | Core Ultra 7 265K, 96 GB DDR5, Arc B580 12 GB |
| Target at launch | Arc A-series, B570/B580, Arc Pro B50/B60/B65/B70 |
| Works but slow | Core Ultra iGPUs (Lunar Lake / Arrow Lake) |

PCIe bandwidth is the bottleneck for expert fetch, so cache hit rate is the whole
game. That is the thesis this project is testing.

## Benchmarks

Numbers here are reported against llama.cpp on the identical box and commit hash.
Protocol: [`bench/README.md`](bench/README.md). Results: `bench/results/`.

⚠️ `moearc bench` does not exist yet; today's figures come from the named tools and examples in
this repo, each cited in the table below.

### 🔴 The baseline in an earlier version of this file was contaminated

That version quoted **46.9 tok/s** as llama.cpp's tuned SYCL decode. The run it came from is
marked `CONTAMINATED-DISCARDED` — GPU contention — and its files are deliberately not
committed. **It should not be quoted, including by us, and it is retracted here.** A clean
re-run at the swept `-ncmoe 22` is outstanding.

The one comparable figure we currently hold is **46.48 tok/s** from `llama-bench` on the
Vulkan backend, single run, `-r 1`, device not pinned. That is indicative, not a baseline.

### What is measured

All figures below are **Qwen3-30B-A3B Q4_K_M (17.3 GiB) on an Arc B580 (11.33 GiB)** unless
stated. Every row was re-verified independently of the run that produced it, and every
configuration generates **identical token ids**.

| | result | how |
|---|---|---|
| **Correctness** | greedy output identical to llama.cpp, **64/64** token ids | patched `llama-eval-callback`, both its CPU and SYCL backends |
| **MoEArc decode** | **45.08 tok/s** at 2952 resident slots with `frac:0.75` host routing | `examples/hybrid_sweep` |
| **vs the static split** | **45.08 vs 13.44 tok/s** at matched capacity, **24× less staged** | same sweep, same slots, same pool |
| **vs llama.cpp** | 45.08 against its **50.13** — which needs `-ncmoe 24` to run at all | `bench/baselines/qwen3-30b-a3b.md` |
| **The floor** | stream-only falls to **7.76 tok/s** at 4.3% residency; with host routing, **29.74** | `examples/hybrid_sweep` |
| Overlap | **94%** of host arithmetic runs while the device is busy | `MOEARC_PROFILE_EVENTS=1` |
| Numerical agreement | 1−cos **2.37e-3** vs llama.cpp CPU | its *own* Vulkan backend differs from its CPU by more |

### The result that was not predicted

Routing misses to the CPU raises the cache hit rate from **92.2% to 99.6%** and cuts expert
traffic **24×** — and *not* because of the CPU's arithmetic. **A miss routed host-side is never
admitted, so it never evicts a resident expert.** The host path works as a pressure-release
valve on the cache. Above ~1000 slots that is most of the gain, and it is a bigger effect than
the overlap it was built for.

📌 Bit-exactness against llama.cpp is **unavailable by construction**: `ggml-cpu` quantises the
f32 activation to 8 bits before every K-quant matmul, while MoEArc keeps f32. So the spread was
measured rather than a tolerance chosen — and llama.cpp's two backends disagree with each other
more than MoEArc disagrees with either.

### What is not measured

- **Prefill.** There is none. llama.cpp's 3218 tok/s has no counterpart here, and every number
  above is decode.
- **Kernel efficiency.** Every matvec sits at 25–29% of the card's 456 GB/s peak where
  llama.cpp reaches 63%. Two well-argued explanations for that gap were tested this session and
  **both died when measured in the engine** — see `docs/roadmap.md`.
- **An adaptive policy.** `frac:0.75` and `frac:1.0` were found by sweeping. The engine does not
  yet choose for itself, and the mechanism costs 1–4% when it routes almost nothing.
- **Anything above 17.3 GiB.** A 63.4 GB model is staged for the next test; nothing is claimed
  about it yet.

📌 **Three findings were retracted during this work** — a fabricated driver-version string, a
profile that mis-attributed device time under an async queue, and two microbenchmark results
that did not survive measurement in place. They are documented rather than deleted, because a
retraction is a claim like any other and this repo has been wrong confidently before.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
