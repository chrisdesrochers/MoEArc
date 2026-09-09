# MoEArc

**Run large mixture-of-experts models on an Intel Arc card, at settings somebody actually
measured.**

A 59 GiB model on a 12 GB card, generating at ~30 tok/s, is not a trick — it is what
llama.cpp's SYCL backend can already do on an Arc B580 when the flags are right. The problem is
that nobody can guess the flags, the defaults are wrong in ways you would not predict, and one
of them costs **2×**.

MoEArc is the measured answer. It detects your card, tells you what will fit, and hands you the
exact command line for the model you want to run.

```
gpt-oss-120b   59.02 GiB of weights   →   Arc B580, 11.7 GiB free VRAM
               29.56 ± 0.16 tok/s at depth 0 · 28.52 at 8K · 26.83 at 32K
               with 9,416 MiB of VRAM still unused
```

*Measured under [`bench/PROTOCOL.md`](bench/PROTOCOL.md); the run is
[`bench/tuning-profiles.md`](bench/tuning-profiles.md).*

---

## 🔴 MoEArc is not an inference engine

**llama.cpp on SYCL is the engine, and MoEArc uses it.** Nothing in this repository is faster
than llama.cpp, because everything here *is* llama.cpp. Every speedup quoted below is
**tuned-vs-default on the same engine** — the same binary, the same commit, different flags.

There is no MoEArc-vs-llama.cpp number in this repository, and that is deliberate. See
[`bench/PROTOCOL.md`](bench/PROTOCOL.md) §1.

---

## Who this is for

People who bought an Intel Arc card and discovered that the local-LLM world is built for
NVIDIA.

- Intel's own **`ipex-llm` was archived 2026-01-28**, with no community fork.
- **Ollama on Arc falls back to Vulkan.** On this box llama.cpp's Vulkan build measures
  **4.8× slower** than its SYCL build on the same card.
- **llama.cpp's SYCL backend is genuinely good** — it is the reason this project is a wrapper
  and not an engine. But it expects you to know what `-ncmoe 31` means, its memory floor is
  model-specific and undocumented, and below that floor it does not degrade gracefully: it dies
  with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, *after* loading tens of gigabytes.

Sources: [`docs/strategy.md`](docs/strategy.md), [`docs/dependencies.md`](docs/dependencies.md).

---

## Install and run

```sh
curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
```

One static binary, no Python, no conda, no oneAPI to install. The SYCL runtime is fetched at
install time against SHA-256 pins in [`packaging/runtime.lock.json`](packaging/runtime.lock.json).

🔴 **You need two things this installer does not give you.** The kernel-side GPU driver (`xe` or
`i915`), which ships with your kernel — and **a llama.cpp build with the SYCL backend**, which
does not. MoEArc is a wrapper: llama.cpp is the engine, and the tarball does not carry it yet
(bundling somebody else's binaries is a licence question, tracked in
[`packaging/THIRD-PARTY.md`](packaging/THIRD-PARTY.md)). `moearc` itself — the device report,
the catalog, `moearc info` — runs with neither. Everything that computes needs `llama-server`
on `$PATH`, beside the `moearc` binary, or named by `$MOEARC_LLAMA_SERVER`.

```sh
moearc                         # what card you have, and what will fit on it
moearc pull gpt-oss-120b       # or any Hugging Face repo id
moearc info gpt-oss-120b       # the tuned command, and where each flag came from
```

`moearc info` prints a `llama-server` command line you can paste, a table giving each flag's
**provenance** — measured on this exact card and model, extrapolated from a neighbour, derived
from your card's free VRAM, or untuned — and the caveats that apply to it. For a B580 running
the flagship that comes out as:

```
llama-server -m gpt-oss-120b-mxfp4.gguf -ngl 99 -ncmoe 36 -t 16 -c <planned> \
             --cache-type-k f16 --cache-type-v f16 -fa on
```

**Requirements:** x86-64 Linux, glibc ≥ 2.39 (Ubuntu 24.04, Fedora 40, Debian 13 or newer). An
Arc card is what the profiles are measured on; the tool runs and reports honestly on anything
Level Zero can see.

**`moearc serve` runs it for you**, which is the point of the tuning being a product rather
than a table you read:

```sh
moearc serve gpt-oss-120b
#   device confirmed — SYCL0 = Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11753 MiB free)
#   model loaded in 37s   (59.0 GiB, -ngl 99 -ncmoe 36 -t 16 -c 86016 -fa on -dev SYCL0)
#   listening on http://127.0.0.1:8080/v1
```

A 59 GiB model on a 12 GB card, answering OpenAI-format requests, with no flags typed. It
supervises `llama-server` with the resolved argv, so **the command it prints is the command it
runs** — verified by checking 64 token ids against the same command launched by hand. It pins
`-dev SYCL0` and confirms that against the child's own device block before loading, which is
what closes the trap where a run silently lands on the iGPU and succeeds anyway.

⚠️ Two caveats on that output. It needs a `llama-server` on the machine — see the note under
*Install*. And `-c 86016` is **derived**, not measured: the planner computes it from free VRAM
after the experts are placed, and the largest context actually benchmarked on this card is 32K.
A derived `-c` is a claim about capacity, not about throughput at that depth.

---

## What is actually measured

Seven models on one Arc B580 (12 GB, 11,959 MiB free at load), Core Ultra 7 265K (8 P + 12 E,
20 cores, no SMT), 91 GiB RAM, llama.cpp `e107984bc` build 10788, SYCL backend asserted in every
CSV row. Full method and raw data: [`bench/tuning-profiles.md`](bench/tuning-profiles.md).

| model | size | `-ncmoe` | `-t` | tok/s @ d0 | @ d8192 | VRAM free @ 8K |
|---|---:|---:|---:|---:|---:|---:|
| `olmoe-1b-7b-0924-instruct` q4_K | 3.92 GiB | 0 | any | **282.9** | — (ctx 4096) | 7,964 MiB |
| `gpt-oss-20b` MXFP4 | 11.28 GiB | 24 | 16 | **44.25 ± 0.36** | 43.40 | 10,097 MiB |
| `Qwen3-30B-A3B` q4_K | 17.28 GiB | 21 | 16 | 69.23 | **52.73 ± 0.65** | 690 MiB |
| `Qwen3-Coder-30B-A3B` q4_K | 17.28 GiB | 21 | 16 | 69.38 | **53.47** | 690 MiB |
| `Qwen3.6-35B-A3B-UD` q4_K | 20.61 GiB | 22 | 16 | **54.81 ± 1.09** | 52.79 | 859 MiB |
| `Llama-4-Scout-17B-16E-UD` q3_K | 45.65 GiB | 46 | 16 | **18.69 ± 0.47** | 11.68 | 1,967 MiB |
| **`gpt-oss-120b` MXFP4** | **59.02 GiB** | 36 | 16 | **29.56 ± 0.16** | **28.52 ± 0.13** | **9,416 MiB** |

Decode (`tg64`), warm, pooled over independent invocations. Flash attention on everywhere; KV
stays f16 everywhere. Absolute throughput is an artefact of this box — what should carry to
another Arc machine is the **shape** of these findings, not the numbers
([`bench/PROTOCOL.md`](bench/PROTOCOL.md) §0).

---

## The four things worth knowing

### 1. llama.cpp runs 4 threads on a 20-core CPU, and it costs 2.09×

`common_cpu_get_num_math()` mis-reads Arrow Lake's hybrid P-core/E-core topology, and
`llama-server` prints `n_threads = 4` on a 20-core part. Nothing warns you. Measured on
gpt-oss-120b with the two arms **interleaved** (`-t 16,4,16,4,16`), so page-cache state is a
common-mode error:

| gpt-oss-120b, `-ncmoe 36` | readings | mean |
|---|---|---|
| `-t 4` — the default | 13.88, 14.34 | **14.11 ± 0.33** |
| `-t 16` | 29.51, 29.56 | **29.54 ± 0.04** |

**2.09× from one flag.** And it is not universally true, which is the point: on OLMoE with
every expert on the GPU, `-t` moves throughput by **0.27% across `t4`→`t20` — nothing at all**.
`-t` and `-ncmoe` are not independent knobs, so neither can be advice. They have to be a
per-model profile.

Also measured, and counter-intuitive: **`-t $(nproc)` is never the answer here.** `-t 20` lost
on five of six models (−5.6% to −10.8%) and its cells were 3–4× noisier. Eight P-cores and
twelve E-cores are not twenty equal cores.

### 2. `-ncmoe`'s correct direction flips with the quantisation

This inverted our own prior and is the most useful thing in the file. `--n-cpu-moe` decides how
many blocks' experts live in host RAM instead of VRAM. Everyone assumes "as few as VRAM allows."

**MXFP4 wants the opposite — every expert on the CPU, even when the whole model fits in VRAM.**
A/B/A/B inside a single process (`-ncmoe 24,0,24,0`, so drift is common-mode), on gpt-oss-20b,
whose 11.28 GiB fits entirely on the card:

| gpt-oss-20b, `-t 16`, d0 | readings | mean |
|---|---|---|
| `-ncmoe 0` — every expert on the **GPU** | 36.35, 36.36 | **36.36 ± 0.01** |
| `-ncmoe 24` — every expert on the **CPU** | 43.36, 43.47 | **43.42 ± 0.08** |

**1.19× by taking the model off the GPU.** The thirteen-point sweep between the ends is monotone
(+21%) and it survives at depth.

Q4_K and Q3_K go the other way: Qwen3-30B is monotone in the opposite direction, **74.08 tok/s
at `-ncmoe 18` against 58.58 at 30** (1.26×). *No single rule a user could reason out is right
for half this catalogue.*

⚠️ We measured **that** this happens, not **why** — the SYCL MXFP4 matmul path was never
profiled. The plausible reading is that it is weak relative to this CPU's, but that is a
hypothesis and it is the obvious thing to take upstream.

### 3. The OOM floor moves with context

Below its floor, llama.cpp fails at load or first graph with `OUT_OF_DEVICE_MEMORY`. The floor
is not a property of the model. On Qwen3-30B it is **18** blocks at depth 0, **21** at 8K and
**28** at 32K — ten blocks eaten by KV growth. A profile that quotes a floor without quoting a
context will OOM a long-context user *after* they have loaded tens of gigabytes.

MoEArc's planner computes `-ncmoe` and `-c` from a single call, so the two flags it prints
cannot contradict each other.

### 4. Sitting on the floor is a bad trade

The obvious `-ncmoe` for gpt-oss-120b is **31** — the lowest value that loads, so the most
experts on the card. Measured against **36**, the two are **1.9% apart, inside the noise
floor**:

| gpt-oss-120b @ 8K ctx | total VRAM | **free** |
|---|---:|---:|
| `-ncmoe 31` — on the OOM floor | 10,632 MiB | 1,327 MiB |
| `-ncmoe 36` — what ships | **2,543 MiB** | **9,416 MiB** |

Same speed, and **8.1 GiB handed back**. That VRAM is what makes 32K context reachable on this
card at 26.83 ± 0.11 tok/s. Two other things that look free and are not: `-fa off` costs
**1.72×** at depth, and quantised KV (`q8_0`) costs **19%** on Arc while buying back exactly one
block — and one block below that it *hard-aborts* rather than returning a catchable OOM, so
MoEArc does not offer it.

---

## Picking a card

The deciding number is not VRAM, it is what fraction of expert *touches* the card intercepts —
and MoE routing is skewed enough that those are very different. Measured on real gpt-oss-120b
routing traces:

| card | expert pool | bank resident | coverage (prose → code) | worst-case miss |
|---|---:|---:|---:|---:|
| Arc B580 (11.3 GB) | 7.4 GiB | 13.1% | 79.0% → 53.6% | **46.4%** |
| Arc Pro B60 (24 GB) | 20.1 GiB | 35.5% | 98.0% → 84.6% | **15.4%** |
| Arc Pro B70 (32 GB) | 28.1 GiB | 49.6% | 99.8% → 93.7% | **6.3%** |

📌 **Buy the 12 → 24 GB step first.** It removes 31 points of miss traffic; 24 → 32 GB removes 9
more. And **what you run matters as much as what you buy** — at 13% residency, prose covers
78.8% of touches and code covers 53.3%. Code and reasoning revisit experts less and punish a
small pool much harder.

⚠️ Coverage predicts **staged bytes**, not tok/s. This project has no validated model between
the two and publishes none. Full analysis, including its own correction:
[`docs/hardware-sizing.md`](docs/hardware-sizing.md).

---

## What this costs you, honestly

- **No head-to-head against llama.cpp, on purpose.** MoEArc *is* llama.cpp plus flags. Every
  multiplier above is tuned-vs-default on one engine.
- **Throughput falls with prompt depth, by wildly different amounts.** From d0 to 8K:
  Llama-4-Scout **1.65×**, Qwen3-30B **1.31×**, Qwen3.6-35B **1.01×**, gpt-oss-120b **1.04×**.
  Scout is the warning case — 192 KiB of KV per token against 1,967 MiB of headroom — and
  longer context forces `-ncmoe` higher, which costs again.
- **Every number here is decode.** Prefill (`-b` / `-ub`) is untuned and unmeasured; `-ngl` was
  left at "all layers" throughout, because with `-ncmoe` doing the expert placement it is the
  wrong lever for MoE.
- **One card, one CPU, one box.** Seven models on a single Arc B580 and a single Core Ultra 7
  265K. On unmeasured hardware MoEArc falls back to arithmetic over your card's real free VRAM,
  renders every value as **derived**, and says so on screen — including that its fallback thread
  count (50% of physical cores) is deliberately conservative and that 16-of-20 measured better
  on the one box we have.
- **The MXFP4 rule comes from two models on one backend.** A third MXFP4 model inherits a
  direction nobody measured for it. It stays badged `derived` for exactly that reason.
- **`moearc serve` has been exercised on one card by one person**, and quantised KV is not
  offered on Arc (q8_0 measured a 19% *loss* here).
- **`-t` between 14 and 18 on the flagship is unknown.** Two attempts were page-cache-bound and
  disagreed, so both were withdrawn rather than averaged. They remain in the tree with their
  disk counters so the discard is auditable.

The rule the project runs on: **when two attempts disagree, withdraw — do not replace.**
Publishing a second wrong number to correct the first is the worse error
([`bench/PROTOCOL.md`](bench/PROTOCOL.md) §9).

---

## Reproducing this

The raw CSV, the load-average guard readings, `/proc/diskstats` deltas and ZFS ARC counters for
every timed run are committed under `bench/results/tuning/`. The harness that enforces the
protocol is [`bench/tuning/harness.sh`](bench/tuning/harness.sh); the ten sweep drivers beside
it are `d1.sh` … `d10.sh`.

The protocol itself is [`bench/PROTOCOL.md`](bench/PROTOCOL.md), and every rule in it exists
because this project got it wrong first — each one cites the failure that produced it. The
short version:

- Give both sides the whole machine, and read the thread count back out of the tool's own
  output rather than assuming it.
- Assert the backend and build commit. A benchmark that runs cleanly is not evidence it
  benchmarked the thing you meant. (`ls build*/bin/llama-bench | head -1` once silently
  selected a Vulkan build: real CSV, plausible numbers, exit 0, and 4.8× wrong.)
- Refuse to measure on a loaded box, and print the load average beside every number.
- Compare model size against page cache and record disk reads. A run that faulted gigabytes off
  disk measured the storage.
- Report mean ± stddev over **independent invocations**, and discard the first cell of every
  process as warm-up.

`moearc bench` is the executable form of that document: it detects and reports the hardware,
refuses a loaded box, pins and prints thread counts, and **declines to print a number it cannot
stand behind**.

---

## Status

Working today: device detection and the "what will fit" report; the model catalog and
downloader; the tuning resolver with provenance badging on every flag; `moearc bench`;
`moearc serve`, which supervises `llama-server` with the resolved argv; and the seven measured
profiles compiled into the binary, so a shipped build tunes without the repository beside it.

Not yet: **llama.cpp is not bundled**, so every computing path needs one already on the
machine; prefill tuning; profiles for any card but the B580; and a published release tag, until
which `install.sh` will tell you plainly that there is nothing to download and point you at
building locally.

The repository also contains a research engine — a Rust/SYCL MoE runtime with dynamic expert
residency, which generates text on the B580 and matches llama.cpp token for token. It is what
produced the routing traces and the cache analysis this tuning work is built on, and
[`docs/pivot-inventory.md`](docs/pivot-inventory.md) is the honest accounting of which parts of
it survive now that llama.cpp is the engine. **Its results are not what you install.**

---

## Credits

[llama.cpp](https://github.com/ggml-org/llama.cpp) and its SYCL backend are the engine here,
and this project exists to help people get more out of them.

[FreeToken](https://github.com/FlashML-org/FreeToken) proved the shape of the answer on NVIDIA —
which knobs matter and how they interact. That knowledge transfers; its constants do not, since
every one is calibrated against hardware we do not have. **No FreeToken code is copied, ported
or vendored.** Inspired by, not derived from —
[`docs/freetoken-priors.md`](docs/freetoken-priors.md) tags each finding *transfers* /
*NVIDIA-specific* / *must re-measure*.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The published tarball is a single
binary and contains no third-party code at all; Intel's SYCL runtime is fetched from Intel on
your machine at install time, and llama.cpp is yours to install.
[`packaging/THIRD-PARTY.md`](packaging/THIRD-PARTY.md) is the full position, including what
changes the day llama.cpp's binaries are shipped alongside.
