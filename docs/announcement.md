# Announcement draft — MoEArc

**Target:** Hacker News, Friday 2026-09-11. Register: plain, specific, technical readers who
will check the claims. No marketing voice.

**Candidate titles** (pick one; the first is the strongest hook and the second is the safest):

1. *llama.cpp's server uses 4 of my 20 cores by default, and it costs 2×*
2. *Show HN: MoEArc – measured llama.cpp tuning profiles for Intel Arc*
3. *A 59 GiB model on a 12 GB Intel Arc card, and the flags that get it there*

**Why the 4-thread lead.** The 59 GiB result is impressive but not surprising to this audience —
`--n-cpu-moe` is llama.cpp's feature and a reader can correctly say so. The thread default is
surprising, is a five-minute check on their own machine, and is the thing that makes a tuning
layer necessary rather than nice. Beat 2 then has to be the `-ncmoe` inversion, because that is
what survives the obvious "so pass `-t`, why a project?" reply.

---

## Post body

I have an Intel Arc B580 — a 12 GB card — and I have been running MoE models on it with
llama.cpp's SYCL backend. In the course of benchmarking that setup properly I found something I
should have found months earlier.

**`llama-server` starts with 4 threads on my 20-core CPU.** It prints it:
`n_threads = 4`. The CPU is a Core Ultra 7 265K — 8 P-cores, 12 E-cores, no SMT.
`common_cpu_get_num_math()` mis-reads Arrow Lake's hybrid topology and lands on 4. Nothing warns
you, nothing looks broken, and the model generates text at a rate that seems fine until you have
something to compare it against.

On `gpt-oss-120b` with experts offloaded to host RAM, that default costs **2.09×**:

```
gpt-oss-120b MXFP4, -ncmoe 36, warm, arms interleaved (-t 16,4,16,4,16)

  -t 4  (the default)   13.88, 14.34   →  14.11 ± 0.33 tok/s
  -t 16                 29.51, 29.56   →  29.54 ± 0.04 tok/s
```

The arms are interleaved inside one sweep on purpose, so page-cache state is a common-mode error
rather than the finding. If you have a hybrid Intel part, this is a five-minute check: start
`llama-server`, read `n_threads` out of its own startup log.

I want to be exact about the scope of that claim. It is one CPU, one box, one llama.cpp build
(`e107984bc`). The mechanism — a hybrid P/E topology that the core-count heuristic reads
wrongly — should generalise to other Arrow Lake and Meteor Lake parts. Whether it does on yours
is a question I would like answered by people who are not me.

### The reason this is a project and not a bug report

Here is the same knob on a different model:

```
olmoe-1b-7b, -ncmoe 0 (whole model on the GPU)
  t4 282.83 · t8 282.81 · t12 282.13 · t16 282.88 · t20 282.77

  -t is worth 0.27% — nothing at all.
```

`-t` is worth between **2× and nothing**, with very little in between, and which one you get
depends entirely on `-ncmoe`. They are not independent knobs, so no amount of advice fixes this.
It has to be a measured profile per (card, model, quantisation).

And it gets worse, in an interesting way. `-ncmoe` / `--n-cpu-moe` sets how many blocks' experts
live in host RAM instead of VRAM. The universal assumption — mine included — is *offload as
little as your VRAM allows*. That is right for Q4_K and Q3_K. It is **exactly backwards for
MXFP4**:

```
gpt-oss-20b MXFP4 (11.28 GiB — the whole model fits in 12 GB of VRAM)
A/B/A/B inside one process: -ncmoe 24,0,24,0

  -ncmoe 0   every expert on the GPU    36.35, 36.36  →  36.36 ± 0.01 tok/s
  -ncmoe 24  every expert on the CPU    43.36, 43.47  →  43.42 ± 0.08 tok/s
```

**1.19× faster by taking the model off the GPU it fits on.** The full thirteen-point sweep
between those ends is monotone, +21%, and it holds at depth. Qwen3-30B at Q4_K is monotone in
the *opposite* direction over its range: 74.08 tok/s at `-ncmoe 18` against 58.58 at 30.

I measured *that* this happens, not *why*. The obvious hypothesis is that the SYCL MXFP4 matmul
path is weak relative to this CPU's, but I did not profile the kernel, so I am reporting it as a
measurement and not as an explanation. It seems like the useful thing to take upstream.

Two more that cost people real throughput:

- **The `-ncmoe` OOM floor is a function of context, not of the model.** Qwen3-30B needs 18
  blocks offloaded at depth 0, 21 at 8K and **28 at 32K**. Publish a single number and you OOM
  every long-context user *after* they have waited for tens of gigabytes to load.
- **Sitting on that floor is a bad trade, and it is what everyone does.** The obvious value for
  the flagship is `-ncmoe 31`, the lowest that loads. `-ncmoe 36` measures 1.9% different —
  inside the noise floor — and leaves **9,416 MiB of VRAM free instead of 1,327**. Same speed,
  8.1 GiB back, and that VRAM is what makes 32K context reachable on the card at all.

### What the tuned settings actually get you

`gpt-oss-120b` — **59.02 GiB of weights** — on an Arc B580 with **11.7 GiB of free VRAM**:
**29.56 ± 0.16 tok/s** at depth 0, 28.52 at 8K, 26.83 at 32K, with 9.4 GiB of card still unused.
Seven models are profiled this way, from a 3.9 GiB OLMoE through Qwen3-30B, Qwen3-Coder-30B,
Qwen3.6-35B and Llama-4-Scout up to that 59 GiB flagship.

### What MoEArc is

A tuning and distribution layer over llama.cpp + SYCL. One static binary, no Python, no oneAPI
to install. It detects your card, tells you what will fit, and prints the `llama-server` command
for the model you want — with a provenance column saying whether each flag was **measured** on
this exact card and model, extrapolated from a neighbour, or derived from arithmetic over your
card's free VRAM. On hardware nobody has benchmarked it says so, on screen, rather than handing
you a confident number.

```sh
curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
moearc                      # your card, and what will fit
moearc info gpt-oss-120b    # the command, and where every flag came from
```

### 🔴 What I am not claiming

**MoEArc is not faster than llama.cpp. MoEArc is llama.cpp.** Every multiplier above is
tuned-vs-default *on the same engine, the same binary, the same commit* — not one engine against
another. There is no MoEArc-vs-llama.cpp benchmark in the repository, deliberately, and
`bench/PROTOCOL.md` §1 is the standing rule about why: give the baseline the whole machine, pin
the thread count on both sides, read it back from the tool's own output, and quote its best
configuration rather than its first.

The project exists because Intel Arc owners have nowhere good to go. Intel's own `ipex-llm` was
archived in January with no community fork, and Ollama on Arc falls back to Vulkan — on this box
llama.cpp's Vulkan build measures 4.8× slower than its SYCL build on the same card. llama.cpp's
SYCL backend is the good option. It just ships one bad default and expects you to know what
`-ncmoe 31` means.

### Limits, stated up front

- One card, one CPU, one box. Absolute throughput does not travel; what should travel is the
  *shape* — the MXFP4 inversion, `-t` collapsing to nothing when nothing runs host-side,
  `nproc` losing to 16, the floor moving with context.
- Every figure is decode. Prefill is untuned and unmeasured.
- Throughput falls with prompt depth by very different amounts per model — 1.01× for Qwen3.6-35B
  from d0 to 8K, 1.65× for Llama-4-Scout.
- `moearc serve` is not wired yet; today it prints the command and you run `llama-server`.
- `-t` between 14 and 18 on the flagship is unknown. Two attempts were page-cache-bound and
  disagreed with each other, so both were withdrawn rather than averaged. They are still in the
  tree with their disk counters.

Raw CSV, load-average guards, `/proc/diskstats` deltas and ARC counters for every timed run are
committed. Apache-2.0.

**What I would most like:** numbers from Arc hardware that is not a B580, and from hybrid Intel
CPUs that are not Arrow Lake. `moearc bench` writes a single self-describing artefact meant to
be pasted into an issue.

github.com/chrisdesrochers/MoEArc

---

## Prepared reply — *"Is this faster than llama.cpp?"*

*(This will be the first comment. Paste as-is.)*

No, and it can't be — MoEArc runs llama.cpp. The SYCL backend is the engine; MoEArc is a layer
that picks its flags for you.

The 2.09× in the post is **tuned-vs-default on the same engine**: same binary, same commit,
`-t 4` against `-t 16`, arms interleaved in one sweep. It is not one engine against another.

There is also no MoEArc-vs-llama.cpp head-to-head in the repo, deliberately. A fair one is
harder than it looks on this hardware, and both traps are in the post: any comparison that
leaves the baseline on its 4-thread default is measuring the default, not the engine; and once
that is pinned, a 59 GiB model against 16 GiB of page cache means you can end up measuring the
disk instead — two of my attempts disagreed by 2×, and about which side won. My standing rule is
that when two attempts disagree you withdraw rather than replace, because publishing a second
wrong number to correct the first is the worse error. So there is no number there on purpose.
`bench/PROTOCOL.md` has the full protocol.

What I'm actually shipping is the tuning: measured profiles for seven models on an Arc B580, the
per-model `-ncmoe` floor (which moves with context), and the `-t` value that makes llama.cpp use
your whole CPU. If you already know your flags, you don't need this. Most people don't, and the
defaults are worse than they look.
