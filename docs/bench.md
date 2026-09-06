# `moearc bench`

A benchmark that refuses to print a number it does not trust.

`bench/PROTOCOL.md` is the specification. Every rule in it was learned from a result this
project published and then had to withdraw, and this command exists so that a user reproducing
our numbers does not have to know any of them. The short version:

> **Absolute throughput does not reproduce across machines, and a tool that implies it does is
> lying by omission.** What reproduces is the *shape*. `moearc bench` leads with the shape and
> labels the absolutes as an artefact of one box.

---

## The two halves, and why they are different kinds of thing

| | **shape** | **absolutes** |
| --- | --- | --- |
| what it is | replay of committed routing traces | timed decode on your card |
| reads a clock | no | yes |
| touches the GPU | no | yes |
| needs a quiet box | **no** | **yes, and it refuses without one** |
| reproduces on your machine | **exactly, to the last digit** | not at all |
| is the headline | **yes** | no |

```sh
moearc bench                       # the shape results. Deterministic, no GPU, no model needed.
moearc bench --check --absolutes   # what would stop a timed run on this box right now
moearc bench --all --model <M> --prompt-ids bench/references/<M>.ids
```

Run it from the repository root, or pass `--traces <DIR>`: the shape half replays the captures
in `bench/traces`, and they are what make it reproducible.

⚠️ **Build release.** The replay is a tight loop over hundreds of thousands of cache
operations, and `moearc-engine` is a workspace member, so a `cargo run` debug build does not
get the `opt-level` the dependency profile grants everything else. Three gpt-oss captures over
nine capacities each, with Belady's bound as well, take **8.3 s** in release on the reference
box and the better part of an hour without it.

**Exit codes.** `0` ran and produced a headline · `1` failed · `2` a subsystem is not built ·
`3` **refused** — the guards stopped it, and the artefact says which.

---

## The six things it checks, and the numbers it checks them against

Every threshold is a flag, and the value that was actually in force is printed in the artefact,
so raising one is visible rather than silent. `moearc bench --check` prints them without
measuring anything.

### 1. Is the box quiet? (§3)

Reads `/proc/loadavg` immediately before **every** timed run — not once for the sweep, because
a sweep that drives a host pool across nineteen threads carries its own previous row into the
average.

- **Refuses above `max(2.0, cpus / 8)`** — 2.50 on a 20-thread box. Warns above 60% of that.
- **Why an eighth.** The failure is `bench/PROTOCOL.md` §3: a sweep at **load 9.50** on this
  20-thread box reported host offload *losing* 60–75% when it in fact gains, reproducibly. One
  eighth refuses that by 3.8×. The reference box's documented idle baseline is ~1.2, so a quiet
  machine passes with about 2× of margin rather than tripping on itself. And the effects being
  compared here are tens of percent — a confound the same size as the finding is not a small
  one.
- The floor exists because the fraction is meaningless on a small machine: an eighth of four
  threads is 0.5, which almost any desktop exceeds while idle.
- **An unreadable load average is a refusal, not a pass.** An unmeasured box is not a quiet box.
- Override with `--max-load <LOAD>`.

### 2. Does the model fit in page cache? (§4)

- Compares the model file against the cache ceiling and prints the ratio. **On ZFS the ceiling
  is `zfs_arc_max`, not free memory** — a 96 GB box with `c_max` at 16 GiB cannot cache a
  59 GiB model however much RAM is idle.
- Records **disk read bytes** (`/proc/diskstats`) and ZFS ARC hits/misses around every timed
  child. A run that faulted gigabytes off the drive measured the storage, not the engine.
- 🔴 **This warns; it never refuses.** The model this project exists for is 3.7× its ARC.
  Refusing it would make the tool useless for its own headline case, and §4 asks for the
  confound to be *stated loudly and travel with every number*, which is what the warning does.
- The disk device is resolved from the model's own filesystem (`/proc/self/mountinfo`, and
  `zpool status -P` for ZFS). If it cannot be attributed to a specific device the counter is
  reported as unknown rather than summed across the machine — a number that cannot be
  attributed is worse than no number. Override with `--disk-dev`.

### 3. Are the thread counts pinned — and did the pin take? (§1)

- MoEArc's host pool is pinned with `MOEARC_HOST_THREADS` and **read back** from the engine's
  own `ResidencyReport::host_threads`.
- `llama-bench` is given `-t` and **read back** from the `n_threads` column of its own
  `-o csv`. Never inferred from a timing.
- **A mismatch is a refusal, and so is a value that could not be read back.** The failure this
  guards is §1's: `llama-bench`'s default is 4 threads on a 20-core box, no invocation in this
  repository passed `-t`, and every "beats llama.cpp" claim had to be withdrawn.
- The incumbent is swept over `--llama-bench-threads` and quoted at its **best** configuration,
  not its first.

### 4. Cold and warm, separately (§6)

Each child runs a cold pass (pool cleared) and a warm pass, and reports them as two numbers.
The parent aggregates them as two samples. **Nothing anywhere averages them together**: they
differed by up to **2.2×** here and they *converge* as depth grows, which is a finding about
the pool rather than noise. The ratio is reported as its own column.

For `llama-bench`, the first test in a process pays warm-up — its `tg64 @ d0` cells carried
error bars 5–20× every other row's for exactly this reason — so the depth is requested twice
and the first row discarded deliberately. If the tool collapses the duplicate, the row is kept
and labelled rather than dropped.

### 5. Are there enough repeats, and is the spread small enough? (§5)

- **Independent invocations, not iterations.** The parent process measures nothing; it
  re-executes this same binary as a hidden `bench-run` child, once per repeat. A loop would
  share a page cache, an allocator, a warmed pool and a driver context, and would report a
  spread narrower than the one a user meets.
- Mean ± **sample** stddev (`n-1`). One run prints "no error bar" rather than "± 0.00".
- **Refuses to headline** below `--repeats 3`, or at a stddev **≥ 20%** of the mean; **warns**
  at ≥ 10%. Twenty is §5's own stated floor — *"a run whose stddev is 20–30% of its mean is not
  a measurement"* — and the good triplicate it is contrasted against sits at 0.7%, so a warning
  fires an order of magnitude before a result is worthless.
- 🔴 **A refused figure stays in the artefact with its error bars**, and every individual
  invocation is listed. §5 asks that the discard be auditable rather than convenient.

### 6. Is this the build you meant? (§2)

- Asserts and prints the **backend, device, driver, Level Zero build and this binary's commit**
  (plus whether the tree was dirty). A mismatch against `--expect-backend` is a refusal.
- The `llama-bench` path is **given, never searched for.** §2's failure was
  `ls build*/bin/llama-bench | head -1` selecting a **Vulkan** build, 4.8× slower than SYCL —
  real CSV, plausible numbers, exit 0, and only the `backends` field revealed it.
- The CSV is indexed by header name, never by position: `cpu_info` on this box contains a
  comma, and a naive split reads the wrong column as `n_threads`.
- The **Level Zero runtime build is printed with every result, passing or not**, because two
  users on the same card get different answers for reasons that have nothing to do with their
  hardware. Measured here, same card, same free VRAM: build **27642** does not enumerate the
  card at all, **33578** detects it and then fails at model load, **37020** loads and decodes.
  🔴 An older build is a **caution with provenance and never a refusal** — the Level Zero
  specification assigns `driverVersion` no encoding and no minimum is published, so there is
  nothing to gate on and nothing is invented.

---

## What it reports

### The shape (the result)

Per trace, over a capacity ladder that runs geometrically from the trace's **peak single-step
demand** — the least capacity at which it is servable — to its **working set**, where every
policy holds everything and ties. Both ends are properties of the trace, so the ladder means
the same thing on a model that activates 144 experts a step and one that activates 320.

1. **Dynamic residency against a static split at matched capacity.** The baseline is
   `Trace::widest_static_split(capacity)` — the most generous static split that fits the *same*
   budget, so any remaining advantage belongs to the policy rather than to an unequal
   allowance. A row where the whole working set is resident is marked and **excluded from the
   claim**: there the dynamic policy never evicts, so every miss it takes is compulsory
   warm-up, while the static split is modelled as resident from step zero and is charged none.
   The gap goes slightly negative, and that is a statement about the two models rather than
   about residency.
2. **The hit-rate-versus-slots curve**, with gain per doubling and a stated knee: the
   **elbow**, the rung farthest from the straight line joining the first and last rungs in
   normalised log2(slots) × hit-rate space. The definition is printed, because "the knee" is
   otherwise a matter of eyesight. 🔴 It replaced a threshold rule — *"the first rung where the
   previous doubling bought fewer than five percentage points"* — which on real captures only
   ever fired at the last rung, where the whole working set is resident and the curve has
   necessarily gone flat. It reported the working set as the knee: true, useless, and
   indistinguishable from a tool that has not looked. A curve with no elbow reports none rather
   than naming an end point.
3. **Whether the claim held**, reported either way. A benchmark that only prints the claim when
   the claim holds is not a benchmark.

Optionally `--optimal` adds Belady's bound: what any online policy could have reached there.

### The absolutes (this machine only)

Decode-only throughput at each `--depths` value, cold and warm, mean ± stddev over independent
invocations, with the disk-read counters beside them.

🔴 The timer starts **after prefill**. `Session::generate` decodes prompt tokens through the
same path as generated ones, so a stopwatch around the whole call divides by `depth + n` steps
and reports mostly prefill. The sampling callback fires once per *generated* token and its
first call lands the instant prefill finished, so its marks fence the decode phase exactly.

### What it will never report

🔴 **A predicted tok/s.** §9: *never convert a proxy into a headline unit you have not
validated.* Hit rate predicts **staged bytes** — one miss is one expert's bytes across the bus,
and that conversion is exact — and it does **not** predict throughput, because nothing in the
replay models overlap, host offload, or the drain that follows a transfer. There is no
validated model between them in this project, so none is published. `shape.rs` carries a test
that fails if a throughput field ever appears in that half of the output.

🔴 **A slot size carried from another model.** Byte columns are attached only to captures whose
own header names the model `--model` resolved to. gpt-oss-120B's slot is 12.607 MiB against
Qwen3-30B-A3B's 2.92 MiB, and §9's last rule was learned by transferring a coverage curve from
one to the other and calling it conservative when it was optimistic.

**Prefill captures are skipped**, with the reason printed. Residency is a decode-time question,
and the prefill traces are 102–143 steps and therefore dominated by compulsory misses; folding
them in would move every number in the flattering direction.

**Captures from other models are skipped too** when `--model` is given, for the §9 reason
above; `--all-traces` replays every capture in the directory. Either way each skipped file is
named with its reason, so a shorter table is never a silently shorter table.

---

## The artefact

One file, `--out <FILE>`, meant to be pasted whole into an issue. It carries the verdict, every
check with the threshold it was judged against, the machine, the tables, a **Not measured**
section naming what the run did not do so a gap reads as a gap rather than as a zero, the exact
command line that produced it, and — in a collapsed block at the end — the same data as JSON,
so pasting it somewhere does not silently drop half of it. `--json` prints the JSON alone.

A refused run still writes a full artefact. What it does not write is a headline:

```text
**VERDICT: REFUSED — no number below is a measurement**

## Result

**There is none.** At least one check below refused, so anything this run produced describes
the state of the machine rather than the engine.
```

`--force` proceeds past a refusal. It stamps the artefact untrusted and says so where the
result would have been. There is no flag that turns a refusal into a headline.

---

## Reference

```text
what to run   --shape (default) · --absolutes · --all · --check · --force
shape         --traces DIR · --trace FILE... · --all-traces · --policy SPEC · --slots N,...
              --optimal · --slot-bytes SIZE
timed         --model M · --prompt-ids FILE · --depths N,... · --tokens N · --repeats N
              --threads N · --residency SPEC · --host-policy SPEC · --ctx N · --attribution
incumbent     --llama-bench PATH · --llama-bench-threads N,... · --llama-bench-arg ARG
              --llama-bench-inner-repeats N
checks        --expect-backend NAME · --max-load LOAD · --disk-dev DEV
output        --out FILE · --json
```

`--policy` accepts `lru` (default), `lfu`, `lru-k:<k>`, `slru:<pct>`, `2q:<kin>:<kout>`,
`w-tinylfu:<window>:<protected>`, `phase-lru` and `optimal`. A static split is deliberately not
spellable: it is derived from the capacity so the baseline always gets every slot it can
legitimately use.

`--prompt-ids` has no default and no generator. §8: a tiled prompt revisits the same experts
and flatters the hit rate, and the exact ids have to be in the repository for the run to be
reproducible from it. `bench/references/*.ids` holds the committed ones.

`moearc bench` never opens the interface, whichever way it is invoked. A measurement taken
underneath a renderer redrawing on the same box measures the renderer too, and what this
command produces is a file rather than a screen.
