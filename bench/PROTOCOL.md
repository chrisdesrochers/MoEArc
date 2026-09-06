# Measurement protocol

Every rule here exists because this project got it wrong first. The failures are cited so the
rules are auditable rather than folklore, and so nobody relaxes one without knowing what it cost.

**This is also the specification for `moearc bench`.** A user reproducing our numbers should not
have to know any of this — the tool should enforce it and **refuse to print a number it does not
trust.** A benchmark that reports a figure it cannot stand behind is worse than one that reports
nothing.

---

## 0. The bar

**Absolute throughput does not reproduce and we should stop implying it does.** It depends on CPU,
memory bandwidth, PCIe generation, filesystem, and whether the model fits in page cache. What must
reproduce on any Arc box is the **shape**:

1. Dynamic residency beats a static split at matched capacity.
2. Staging, not attention, dominates the cost of prompt depth.
3. The hit-rate-vs-slots curve has the same knee.

**Report shape as the result and absolutes as an artefact of one machine.**

---

## 1. Give the baseline the whole machine

🔴 **The failure:** we published *"beats llama.cpp"* for weeks. `llama-bench` defaults to **4
threads** on this 20-core box, and **no invocation anywhere in this repo passed `-t`**. With
`-ncmoe 31` putting 31 of 36 blocks' experts on the CPU, the incumbent ran on a quarter of the
machine while MoEArc used 19 threads. Every comparison had to be withdrawn.

- **Pin thread count on both sides and read it back from the tool's own output**
  (`llama-bench -o csv`, field `n_threads`). Never infer it from a timing, and never accept a
  default.
- Sweep the baseline's tuning knobs and quote its **best** configuration, not its first.
- State both engines' thread counts in every published table.

## 2. Verify you benchmarked the thing you meant

🔴 **The failure:** `ls build*/bin/llama-bench | head -1` silently selected a **Vulkan** build —
**4.8× slower than SYCL** on this card. It produced real CSV, plausible numbers, exit 0. Only the
`backends` field revealed it. Reporting it would have been wrong in the *opposite* direction.

- **Assert the backend, device, and build commit** in the output of every run.
- Never select a binary by glob order.
- 📌 **A benchmark that runs cleanly is not evidence it benchmarked the thing you meant.**

## 3. The box must be quiet

🔴 **The failure:** a sweep reported host offload *losing* 60–75% when the box was at load 9.50.
It reported the opposite of the truth, reproducibly.

- **Read the 1-minute load average immediately before every timed run and print it with the
  result.** Refuse to measure above a stated threshold.
- Never run two engines, or two agents' benchmarks, concurrently.
- Note the box's idle baseline so a reader can judge the margin.

## 4. The model must fit in page cache — or you must say it does not

🔴 **The failure:** a `-r 2` sweep gave **17.59 ± 5.56** where an `-r 5` triplicate gave
**28.5 ± 0.2**. Not a busy box and not unfair threads: the model is **59.03 GiB** against ZFS
`arc c_max` of **16 GiB**, so it streams off disk and throughput depends on what happens to be
resident. This is the third distinct route to the same class of contamination.

- **Compare model size against available page cache and report the ratio.** Warn loudly when the
  model cannot be cached.
- Record **disk read bytes** (`/proc/diskstats`) and, on ZFS, ARC hits/misses **around every timed
  run**. A run that faulted gigabytes from disk measured the storage, not the engine.
- 📌 This applies to **both** engines equally — it is not a handicap, it is a confound.

## 5. Repeats must justify the conclusion

- Report **mean ± stddev**, never a single run.
- 🔴 **A run whose stddev is 20–30% of its mean is not a measurement.** Say so and discard it —
  and keep it in the record with its error bars visible, so the discard is auditable rather than
  convenient.
- Prefer **several independent invocations** over more iterations inside one: process-level
  variance is the variance that bit us, and `-r` inside one process cannot see it.

## 6. Warm and cold are different questions

- **Report both, always.** They differed by up to **2.2×** here, and they *converge* as depth grows
  — which is itself a finding about the pool, not noise to be averaged away.
- 🔴 A tool's **first test in a process pays warm-up.** `llama-bench`'s `tg64 @ d0` cells carried
  error bars 5–20× every other row's for exactly this reason. Discard or amortise deliberately.

## 7. Measure the phase you claim to measure

🔴 **The failure:** host wall-clock around an **asynchronous** queue bills device work to whichever
call later drains it. A profile read this way once made us retract a *correct* conclusion.

- For decode-at-depth, **the timer must start after prefill** on both sides. `llama-bench -d N`
  does this; MoEArc's harness fences decode with the sampling callback.
- To attribute cost between phases, force synchrony (`MOEARC_SYNC_EACH=1`). ⚠️ This **destroys
  overlap**, so absolute values are inflated — **the growth ratio is the finding, not the
  milliseconds.**
- Know what your counters cannot see. MoEArc's device event counters instrument **only matvec
  paths**, so attention is invisible to them; neither instrument alone could attribute the depth
  penalty, and using either alone would have given a confident wrong answer.

## 8. Prompts are part of the protocol

- Use **real text**, not a repeated phrase — a tiled prompt revisits the same experts and flatters
  the hit rate.
- Commit the exact token ids so the run is reproducible from the repo.
- 🔴 For **correctness** comparisons, choose a prompt whose greedy margins are wide, **by
  measurement**. Three candidates here failed at margins of 0.16–0.19 against a ~0.4
  implementation difference — and at one such position **llama.cpp disagreed with itself**
  (one-shot prefill chose 279, incremental decode chose 13). A near-tie measures the tie-break,
  not the engine. The shipped prompt has a minimum margin of **5.81**.
- **Check the capture's own output.** A trace capture here produced degenerate all-ellipsis text
  and would have shipped silently if nobody had looked.

## 9. Publishing

- 🔴 **When two attempts disagree, withdraw — do not replace.** Publishing a second wrong number to
  correct the first is the worse error. Say what is certain (*the old comparison was unfair*) and
  what is not (*by how much*).
- **Never convert a proxy into a headline unit you have not validated.** Hit rate predicts **staged
  bytes**; it does not predict tok/s, and this project has no validated model between them —
  so it publishes none.
- Retractions stay in the tree with their evidence. A retraction is a claim like any other.
- **A measurement transferred from one model is not a measurement of another.** A coverage curve
  taken from Qwen3-30B (8 of 128 active) was applied to gpt-oss (4 of 128) and called
  conservative; it was optimistic. Regenerate from the subject's own trace.

---

## What `moearc bench` must do

1. Detect and **report** hardware, backend, driver, and build commit.
2. **Refuse to run** on a loaded box; refuse silently-degraded configurations.
3. **Warn** when the model exceeds available page cache, and report disk reads per run.
4. **Pin and print** thread counts for every engine it invokes.
5. Emit **warm and cold** separately, with mean ± stddev over independent invocations.
6. Emit the **shape** results (§0) as the headline and absolutes as machine-specific context.
7. Write a single self-describing artefact a user can paste into an issue — one that contains
   enough context for a stranger to tell whether the number is trustworthy.
