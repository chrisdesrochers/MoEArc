# Strategy

*Set 2026-09-07. This is the document everything else builds toward. If a decision
contradicts this file, this file is wrong and should be changed deliberately — not worked
around.*

## What MoEArc is

**A one-command way for Intel Arc owners to run MoE models far larger than their VRAM, tuned
for the exact card they own.**

The headline experience, and the thing to protect: *"I fit a 59 GB model on a 12 GB GPU and it
works."*

## Who it is for

People who bought an Arc card and discovered the local-LLM world is built for NVIDIA.

- Intel's own **`ipex-llm` was archived 2026-01-28** with no community fork.
- **Ollama on Arc falls back to Vulkan** — measured at **4.8× slower than SYCL** on the same card.
- **llama.cpp SYCL is genuinely good**, and it is our engine. But it makes you know what
  `-ncmoe 31` means, its floor is model-specific and undocumented, and below that floor it does
  not degrade — it crashes with `OUT_OF_DEVICE_MEMORY`. 🔴 Its `llama-bench` thread default is
  **4 on a 20-core box**, which costs **2.1×** (13.6 vs 28.5 tok/s on gpt-oss-120B, measured).

**Nobody is guessing these flags well. That is the opportunity.**

## The user journey

Paste a URL into a terminal. It installs the runtime, detects the card, shows which models fit
and how well, downloads one, and starts a local server.

**No flags. No oneAPI. No science experiment.**

## The three layers

| layer | what it is |
| --- | --- |
| **Rust** | installer · hardware detection · model catalog · TUI · server · `moearc bench` |
| **Tuning profiles** | measured best settings per (GPU, model) — **this is the product's value** |
| **llama.cpp + SYCL** | the engine. Used to its fullness. Not rewritten. |

🔴 **We do not write kernels.** MoEArc hand-wrote its own SYCL kernels and paid for it: our
matvecs measured **25–29% of the card's peak bandwidth**. That work is being retired. See
`docs/kernel-strategy.md` for the assessment, including its correction of two figures this
project had been quoting as findings.

## What "optimization" means here

> *"The optimization per model, or even per GPU model, is simply what we've benchmarked is the
> best way to push the hardware + model + llama.cpp to get the best results. That's all."*

**It is measured knowledge, not a novel engine.** For a B580 running gpt-oss-120B: this
`-ncmoe`, these threads, this context, this KV type. The user gets the answer without running
the experiment.

## FreeToken is a source of priors, not code

[FreeToken](https://github.com/FlashML-org/FreeToken) solved the expensive problem of working
out **which knobs matter and how they interact** on NVIDIA. That knowledge transfers. **Its
constants do not** — every one is calibrated against hardware we do not have.

🔴 **No FreeToken code is copied, ported or vendored.** Inspired by, not derived from. See
`docs/freetoken-priors.md` for findings tagged *transfers* / *NVIDIA-specific* /
*must re-measure*.

## The optimization loop

> *"If we hit 114 tok/s and things are solid, we apply some tweaks — if that gets it to 128,
> that's a win. If we hit 40, we back up."*

**Hill-climb from a known-good baseline. Not a grid search.**

1. **Baseline** — the recorded profile for this (GPU, model). The "114".
2. **Predict** — say what the change should do and why, before running it. A prior that predicts
   nothing is not a prior.
3. **Change one variable**, so the delta has a cause.
4. **Measure** under `bench/PROTOCOL.md`.
5. **Accept only if it beats the noise floor.** 🔴 Not the mean — the error bar. Some of our runs
   have carried **±32%**; at that noise a 12% win is invisible and you will keep changes that
   hurt and revert ones that help. That is a random walk, not a climb.
6. **Record it** in the profile, with its provenance.

📌 **A change that loses badly is information, not just a revert.** It means the model of what
that knob does was wrong — and correcting the model is worth more than the tweak was.

## What MoEArc is not

- **Not a competitor to llama.cpp.** It is a distribution and tuning layer on top of it.
- **Not a new inference engine.**
- **Not a benchmark-score chase.** We measure to find out where the wins are, not to win an
  argument. The benchmark is an instrument.

## Phases

**1 — Ship the wrapper.** Rust drives llama.cpp+SYCL. Install, detect, catalog, tune, serve.
Publish measured profiles for every model we support. **This is shippable and it is the product.**

**2 — Deepen the tuning.** More hardware, more models, family-level profiles, and the
extrapolation rules that let an unmeasured card get a sensible starting point.

**3 — Only if measurement justifies it**, add scheduling of our own, in FreeToken's shape.
`docs/kernel-strategy.md` names the two-hour experiment that decides whether that is worth
35,000 vendored lines.

## Honest current state (2026-09-07)

- ✅ A **59.0 GiB model runs on an 11.33 GiB Arc B580**, output matching llama.cpp token for token.
- ✅ **Dynamic residency beats a static split 45.08 vs 13.44 tok/s** at matched capacity with
  **24× less data staged** — MoEArc measured against itself.
- 🔴 **No claim is made about speed relative to llama.cpp.** An earlier comparison was withdrawn:
  `llama-bench` was run at its 4-thread default while MoEArc used 19. **Two corrected attempts
  disagree by 2× and about which engine wins**, so no corrected figure is published. See
  `bench/PROTOCOL.md` §1.
- 🔴 **Throughput falls with prompt depth** (3.6× from shallow to 8192, corrected from a published
  5.9×). Cause is contested between staging and attention; `docs/kernel-strategy.md` shows the
  69%-attention figure does not survive a bounding argument.
- 🔴 **Cache capacity, not cache policy.** Nine policies scored against Belady: the best
  non-regressing one recovers 7–25% of the gap to optimal, while **+44% more slots beats any of
  them by ~3×**.
- ⬜ **No release tag** — `install.sh` cannot download anything until one exists
  (`packaging/RELEASE.md` is the checklist).
