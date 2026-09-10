# MoEArc v0.1.0

First release. **MoEArc runs large mixture-of-experts models on an Intel Arc card at settings
somebody actually measured.** It detects your card, tells you what will fit, and either prints
or runs the exact `llama-server` command line for the model you want.

A 59.02 GiB model on a 12 GB card, answering OpenAI-format requests, with no flags typed.

---

## 🔴 Two things to know before you download this

**1. MoEArc is not an inference engine. llama.cpp on SYCL is the engine, and MoEArc uses it.**
Nothing here is faster than llama.cpp, because everything here *is* llama.cpp. Every multiplier
below is **tuned-vs-default on the same engine** — same binary, same commit, different flags.
There is no MoEArc-vs-llama.cpp number in this project, on purpose (`bench/PROTOCOL.md` §1).

**2. This tarball does not contain llama.cpp.** The payload is exactly one binary. `moearc
serve` supervises **your own** `llama-server` as a child process, resolved beside the `moearc`
binary first, then `$MOEARC_LLAMA_SERVER`, then `$PATH`. A target machine needs a llama.cpp
build with the SYCL backend already on it. Bundling those binaries is a licence question that
has not been written yet (`packaging/THIRD-PARTY.md`), and shipping them is tracked, not done.

What works with no llama.cpp at all: the device report (`moearc`), the catalog (`moearc ls`,
`moearc pull`), the tuned command line (`moearc info`), and the shape half of `moearc bench`.
Everything that computes needs the engine.

---

## Requirements

- **x86-64 Linux only.** No aarch64, no Windows, no macOS.
- **glibc floor, measured per binary and recorded in `share/moearc/BUILD-INFO.txt`:**
  `moearc` requires `<FILL FROM BUILD-INFO.txt: GLIBC_2.xx>`. At the `GLIBC_2.39` floor this
  build inherits from its host, that is **Ubuntu 24.04, Fedora 40, Debian 13 or newer — and it
  silently excludes Debian 12 and RHEL 9.** Read `BUILD-INFO.txt` in the tarball; it states the
  floor this artefact actually has, not the one this sentence assumes.
- **One dependency, and it ships with your kernel:** the Intel GPU driver — `xe` for Arc
  B-series (Battlemage) or `i915` for A-series — together with its Level Zero user space
  (`libze_loader.so.1`, `libze_intel_gpu.so.1`), which every distribution ships in the same
  package as the kernel-side half. MoEArc deliberately does not pin its own copy: that is the
  layer that has to agree with *your* kernel.
  🔴 **It has a floor and it bites.** Ubuntu 24.04's own `libze-intel-gpu1` is Level Zero build
  27642, which predates Battlemage: on a box with a B580 it enumerates the Arrow Lake iGPU and
  nothing else, and detection then reports a machine that looks broken. Build 33578 detects the
  card and fails at model load; 37020 loads and decodes. `moearc` prints the Level Zero build
  with every result for exactly this reason.
- **No oneAPI, no conda, no Python — and no runtime download.** `install.sh` fetches nothing
  beyond the tarball. Nothing in the payload loads Intel's SYCL runtime: the device report
  reaches the card through a `dlopen`'d Level Zero and links no SYCL at all, so a default
  install creates no `runtime/` directory. `MOEARC_FETCH_RUNTIME=1` stages it anyway —
  **199.5 MiB downloaded, 78.6 MiB kept**, against the SHA-256 pins in
  `packaging/runtime.lock.json` — and is worth setting if your own llama.cpp SYCL build has no
  oneAPI beside it, or to prepare a machine that will be offline.
- An **Arc card** is what the profiles were measured on. The tool runs and reports honestly on
  anything Level Zero can see, and badges every unmeasured value as derived.

---

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
```

Or pin this release: `MOEARC_VERSION=v0.1.0 sh install.sh`. `MOEARC_PREFIX` (default
`~/.local/share/moearc`) and `MOEARC_BINDIR` (default `~/.local/bin`) move where it lands;
`MOEARC_TARBALL` installs a local file and skips the download; `MOEARC_FETCH_RUNTIME=1` stages
Intel's SYCL runtime, which is otherwise not downloaded at all. One command is linked, `moearc`.

A default install prints a two-line note saying no SYCL runtime was fetched. **That is not a
warning and nothing is missing** — running a model needs your own `llama-server`, which links
its own runtime.

```sh
moearc                         # what card you have, and what will fit on it
moearc pull gpt-oss-120b       # or any Hugging Face repo id
moearc info gpt-oss-120b       # the tuned command, and where each flag came from
moearc serve gpt-oss-120b      # run it, tuned, on the pinned device
moearc bench                   # the shape result, from the committed traces
```

### Verify the download

```
moearc-linux-x86_64.tar.gz      sha256  <FILL: sha256 of the uploaded asset>
```

The `.sha256` file is attached to this release and `install.sh` checks it before `tar` sees a
byte. It refuses a short or non-gzip download with the byte count, and prints both digests on a
mismatch.

---

## What is in the tarball

```
moearc-0.1.0-linux-x86_64/
  moearc                <- launcher; sets LD_LIBRARY_PATH and execs the real binary
  libexec/moearc        <- the one ELF binary  (+ fetch-runtime.py)
  share/moearc/         <- runtime.lock.json, BUILD-INFO.txt
  share/doc/moearc/     <- LICENSE, NOTICE, THIRD-PARTY.md
  bench/                <- reproduce.sh, the routing traces, the reference token ids
```

About **3.8 MB**, one binary, **no third-party code at all** — read the size `bundle.sh` prints
beside the artefact path, which is measured, rather than this line, which is copied. The tarball
is wholly Apache-2.0.

Two things a reader will notice and should not mistake for bugs:

- **`moearc` needs no SYCL runtime.** The device report talks to Level Zero directly, so the
  first thing a new user runs works immediately after unpacking and can explain a machine whose
  GPU stack is broken.
- **There is therefore no `runtime/` directory after a default install.** Nothing in a
  one-binary payload opens one, and the installer stopped downloading 199.5 MiB on every install
  to fill it. `MOEARC_FETCH_RUNTIME=1` stages it for the one real use — a user's own llama.cpp
  SYCL build resolving against it through the launcher's `LD_LIBRARY_PATH`. That is a side
  effect rather than a design, nobody has measured it across an ABI skew, and it is a flag
  someone chooses rather than a download everybody pays for. The machinery is kept, not deleted:
  the question settles the day llama.cpp's shared objects are bundled.

**Not in the tarball, deliberately:** no inference engine (above); no `moearc-server`, which
without a linked llama.cpp is an **echo stub** that speaks fluent OpenAI and is indistinguishable
from a model to anyone not reading `/health` — `moearc serve` is the server and does not use it;
and none of the retired SYCL engine (`libmoearc_kernels.so`, `moearc-bench`, `moearc-selftest`).
`bundle.sh` fails the build if any of the three comes back.

---

## What is measured

Seven models on one Arc B580 (12 GB, 11,959 MiB free at load), Core Ultra 7 265K (8 P + 12 E,
20 cores, no SMT), 91 GiB RAM, llama.cpp `e107984bc` build 10788, SYCL backend asserted in every
CSV row. Decode (`tg64`), warm, pooled over independent invocations, flash attention on, KV f16.

| model | size | `-ncmoe` | `-t` | tok/s @ d0 | @ d8192 | VRAM free @ 8K |
|---|---:|---:|---:|---:|---:|---:|
| `olmoe-1b-7b-0924-instruct` q4_K | 3.92 GiB | 0 | any | **282.9** | — (ctx 4096) | 7,964 MiB |
| `gpt-oss-20b` MXFP4 | 11.28 GiB | 24 | 16 | **44.25 ± 0.36** | 43.40 | 10,097 MiB |
| `Qwen3-30B-A3B` q4_K | 17.28 GiB | 21 | 16 | 69.23 | **52.73 ± 0.65** | 690 MiB |
| `Qwen3-Coder-30B-A3B` q4_K | 17.28 GiB | 21 | 16 | 69.38 | **53.47** | 690 MiB |
| `Qwen3.6-35B-A3B-UD` q4_K | 20.61 GiB | 22 | 16 | **54.81 ± 1.09** | 52.79 | 859 MiB |
| `Llama-4-Scout-17B-16E-UD` q3_K | 45.65 GiB | 46 | 16 | **18.69 ± 0.47** | 11.68 | 1,967 MiB |
| **`gpt-oss-120b` MXFP4** | **59.02 GiB** | 36 | 16 | **29.56 ± 0.16** | **28.52 ± 0.13** | **9,416 MiB** |

🔴 **Absolute throughput is an artefact of this one box.** What should carry to another Arc
machine is the *shape* of the findings below, not the numbers (`bench/PROTOCOL.md` §0).

### Why a tuning layer is a product

**1. llama.cpp runs 4 threads on a 20-core CPU, and it costs 2.09×.**
`common_cpu_get_num_math()` mis-reads Arrow Lake's hybrid topology and `llama-server` prints
`n_threads = 4` on a 20-core part. Nothing warns you. On gpt-oss-120b at `-ncmoe 36`, arms
interleaved so page-cache state is common-mode: `-t 4` → **14.11 ± 0.33**, `-t 16` →
**29.54 ± 0.04**. And it is not a rule you could generalise — on OLMoE with every expert on the
GPU, `-t` moves throughput by **0.27% across `t4`→`t20`, nothing at all**. `-t $(nproc)` is not
the answer either: `-t 20` lost on five of six models (−5.6% to −10.8%) and was 3–4× noisier.

**2. `-ncmoe`'s correct direction flips with the quantisation.** Everyone assumes "as few
experts on the CPU as VRAM allows". MXFP4 wants the opposite — every expert on the CPU, even
when the whole model fits in VRAM. On gpt-oss-20b (11.28 GiB, fits entirely on the card),
A/B/A/B inside one process: `-ncmoe 0` → **36.36 ± 0.01**, `-ncmoe 24` → **43.42 ± 0.08**.
**1.19× by taking the model off the GPU it fits on.** Q4_K goes the other way — Qwen3-30B is
monotone in the opposite direction, 74.08 tok/s at `-ncmoe 18` against 58.58 at 30.
⚠️ We measured *that* this happens, not *why*; the SYCL MXFP4 matmul path was never profiled.

**3. The OOM floor moves with context.** Below it llama.cpp does not degrade, it dies with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` *after* loading tens of gigabytes. On Qwen3-30B the floor
is **18** blocks at depth 0, **21** at 8K, **28** at 32K. MoEArc's planner computes `-ncmoe` and
`-c` from one call, so the two flags it prints cannot contradict each other.

**4. Sitting on that floor is a bad trade.** The obvious `-ncmoe` for gpt-oss-120b is 31, the
lowest that loads. Against 36 the two are **1.9% apart, inside the noise floor** — and 36 hands
back **8.1 GiB** of VRAM (9,416 MiB free at 8K against 1,327), which is what makes 32K context
reachable at **26.83 ± 0.11 tok/s**. Two other things that look free: `-fa off` costs **1.72×**
at depth, and quantised KV (`q8_0`) costs **19%** on Arc while buying back exactly one block —
and one block below that it hard-aborts rather than returning a catchable OOM, so MoEArc does
not offer it.

### `moearc bench` — the result that reproduces on your machine

The shape half is a deterministic replay of the routing traces shipped in the tarball. It reads
no clock, touches no GPU and needs no model, so it should come out identical on your box, to the
last digit:

> **dynamic residency beat the widest matched-capacity static split on 11/11 traces, by 19.4 to
> 58.0 points.**

```sh
moearc bench            # or bench/reproduce.sh, from the unpacked tarball
```

If your run disagrees in any digit, that is a real disagreement and worth an issue.

---

## What this release does not do

- **`moearc bench --absolutes` is developer-only in v0.1.0.** The timed half still reaches the
  card through the retired SYCL engine's `gpu` feature, which `bundle.sh` refuses to put in a
  release payload. On the shipped binary it **exits 3** — a refusal, not a crash — with
  `this binary has no GPU backend compiled in`. The published absolute throughput numbers
  therefore **cannot be reproduced from this release**; that is `PROTOCOL.md` §0's position
  stated plainly, not a workaround. The shape half is the headline and it does reproduce.
- **The clean-room gate proves less than it used to, and says so.**
  `packaging/verify-clean.sh` runs the tarball in a container that has never had oneAPI and
  asserts by name that the Arc card is found. It passes. It also prints **`NOT PROVEN`** for a
  model loading, a forward pass, and correct token ids — those gates ran binaries that left the
  payload with the retired engine. A PASS means *the card is found*, not *it computes*. A driver
  stack can pass detection and still fail inference; they are different questions, measured.
- **`-c` is derived, not measured.** `moearc serve` plans context from free VRAM after the
  experts are placed; the largest context actually benchmarked on this card is 32K. A derived
  `-c` is a claim about capacity, not about throughput at that depth.
- **One card, one CPU, one box.** On unmeasured hardware MoEArc falls back to arithmetic over
  your card's real free VRAM, renders every value as **derived**, and says so on screen —
  including that its fallback thread count (50% of physical cores) is conservative, and that
  16-of-20 measured better on the one box there is.
- **The MXFP4 rule comes from two models on one backend.** A third MXFP4 model inherits a
  direction nobody measured for it, and stays badged `derived` for exactly that reason.
- **Every number here is decode.** Prefill (`-b` / `-ub`) is untuned and unmeasured.
- **`moearc serve` has been exercised on one card by one person.**
- **`-t` between 14 and 18 on the flagship is unknown.** Two attempts were page-cache-bound and
  disagreed, so both were withdrawn rather than averaged. They remain in the tree with their
  disk counters so the discard is auditable. The rule this project runs on: **when two attempts
  disagree, withdraw — do not replace** (`PROTOCOL.md` §9).

### Two figures that are not in these notes

A warm decode measurement of the research engine on gpt-oss-120b exists at depths 128 and 512.
It is withheld: the model is far larger than this box's page cache, the **cold** first
invocation of each point is storage-bound, and at depth 512 it carries a spread of 34% of its
mean — which `PROTOCOL.md` §5 says is not a measurement. Publishing the warm half alone, with
its cold twin refused, would be reporting the half that happened to look good.

<!-- DECISION PENDING: warm figures 12.14 ± 0.12 (depth 128) and 12.67 ± 0.05 (depth 512) withheld because their cold twins failed PROTOCOL §5 (11.25 ± 1.42 and 10.54 ± 3.53). To headline them, add here. -->

---

## Reproducibility and provenance

`share/moearc/BUILD-INFO.txt` inside the tarball records the commit, the toolchain, the build
host's glibc, and the measured glibc floor per shipped binary.

`bundle.sh` pins the `built:` stamp and every tar member's mtime, uid, gid and order to
`SOURCE_DATE_EPOCH` (defaulting to the commit's own timestamp), and `gzip -n` keeps the name and
time out of the gzip header. **The same commit, on the same machine, with the same toolchain,
produces the same sha256** — measured twice before that sentence was written.

⚠️ **That is repeatability, not reproducibility across machines.** Nobody has built this on a
second host and compared, so a different `rustc` or a different absolute build path may well
produce different bytes. The claim is the honest one: *these bytes came from this commit with
the toolchain recorded in `BUILD-INFO.txt`*, and the sha256 lets anyone check they got what was
published. Auditable, not bit-for-bit reproducible from source by a third party.

```
tag      v0.1.0
commit   <FILL: commit the tag lands on — must match BUILD-INFO.txt>
rustc    <FILL FROM BUILD-INFO.txt>
```

---

## Licence and credits

Apache-2.0 (`LICENSE`, `NOTICE`). **The tarball contains no third-party code**, which is what
makes that licence true of the bytes you download, and it does not depend on when — or whether —
Intel's SYCL runtime arrives. MoEArc redistributes none of it: four of those libraries are
Intel's proprietary compiler runtime, and Intel's EULA grants redistribution only for files
listed in a `redist.txt` that does not exist in the oneAPI installation this is built against.
So the runtime is never shipped here, and as of this release it is not downloaded by default
either; `MOEARC_FETCH_RUNTIME=1` pulls it from Intel's own channel, on your machine, into a
directory carrying Intel's EULA beside it — the same arrangement PyTorch's XPU builds use, and
the user accepts Intel's terms from Intel. `packaging/THIRD-PARTY.md` is the full position,
including the two downstream terms that would have followed a vendored copy, and what changes
the day llama.cpp's binaries ship alongside.

⚠️ `bundle.sh --with-runtime` vendors the runtime into the archive for air-gapped installs. **The
result is not wholly Apache-2.0 and must not be published as if it were.**

[llama.cpp](https://github.com/ggml-org/llama.cpp) and its SYCL backend are the engine here, and
this project exists to help people get more out of them.

[FreeToken](https://github.com/FlashML-org/FreeToken) proved the shape of the answer on NVIDIA —
which knobs matter and how they interact. That knowledge transfers; its constants do not, since
every one is calibrated against hardware we do not have. **No FreeToken code is copied, ported or
vendored.**
