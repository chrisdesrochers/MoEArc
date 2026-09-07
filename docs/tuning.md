# Tuning profiles

**The feature that makes MoEArc worth installing.** The engine underneath is llama.cpp on SYCL.
What this project adds is the answer to *"what should I actually pass it?"* — and two of those
flags are worth real money to a user:

- **`-t`.** `llama-bench` defaults to **4 threads**. On this project's 20-core box that default
  cost **2.1×** — 13.6 tok/s against 28.5 on a 59 GiB model. It is not documented as a
  performance cliff anywhere, and nobody guesses their way to the right value.
- **`-ncmoe` / `--n-cpu-moe`.** It has a **model-specific floor**. One block below it, llama.cpp
  aborts with `OUT_OF_DEVICE_MEMORY` — after loading tens of gigabytes. Nothing computes that
  floor, nothing documents it, so users find it by crashing and then back off further than they
  need to, losing throughput to superstition.

🔴 **And one rule governs the whole thing: a setting with a measurement behind it must never
look like one without.** This project has already reported a guessed constant to its owner as a
result. The fix is not diligence — it is a type. `tuning::schema::Setting<T>` carries an
[`Origin`](#origins) and has no `Deref`, no `From<T>` and no constructor that omits it, so a
bare number cannot reach a renderer.

---

## Origins

Every value that reaches a screen is one of four things, and every renderer prints which.

| origin | glyph | means |
| --- | :---: | --- |
| **measured** | ◆ | benchmarked on **this** GPU with **this** model at **this** quantisation, under `bench/PROTOCOL.md` |
| **extrapolated** | ◇ | carried from a profile measured on different hardware or a different model, with the differences reasoned about. **A starting point, not a measurement** |
| **derived** | · | computed by `moearc_engine::memory` from the model's geometry and this card's free VRAM, plus the two **directional** rules in `bench/tuning-profiles.md` that hold across the whole catalogue. Real arithmetic and a measured direction; nothing was ever run **on this pairing** — see [the fallback](#the-fallback-what-happens-with-no-profile) |
| **untuned** | *(blank)* | not tuned at all. llama.cpp's own default stands and we make no claim about it. Never emitted as a flag |

A profile's badge is its **basis**; each flag also carries its own origin, and they can differ —
see [the two recomputations](#two-things-are-recomputed-even-on-an-exact-match).
`Resolved::weakest_field_origin()` reports the weakest field, and the CLI prints a red line
whenever it is not `measured`.

---

## How resolution degrades

```
exact (gpu, model, quant)  ──▶ MEASURED
      │ no
      ├─ same model, other quant, this card  ──┐
      ├─ same model, other Arc card           ──┼─▶ EXTRAPOLATED
      ├─ sibling model, this card             ──┘
      │ no
      └─ moearc_engine::memory::plan_llama    ──▶ DERIVED
             │ planner refuses (VRAM held by another process, …)
             └─▶ no settings, and the planner's own sentence saying why
```

**Sibling** is decided **structurally, not by name**: same `moe_blocks`, same
`experts_per_block`, same `active_experts_per_block`, same quantisation, and parameter counts
within 5%. `qwen3-30b-a3b` and `qwen3-coder-30b-a3b-instruct` match on all five; a name prefix
alone would happily match `qwen3-4b` to `qwen3-235b-a22b`, and a routing width of 4-of-128
against 8-of-128 is a materially different cache whatever the name says. Names are used only to
rank among several structural matches.

⚠️ **One hop only.** A sibling model must have been measured on *this* card, and a different
card must carry *this* model. A profile two hops away — another model on another card — falls
through to derived. The second hop would buy a KV width and `-fa` while badging the answer as
though someone had reasoned about the pair, and what it replaces is exact arithmetic on the
user's own hardware and file.

### The invariant

🔴 **Only an exact match can produce a `measured` field.** Not "usually" — never. The moment a
value is carried onto hardware or a model it was not measured on, it is extrapolated, whatever
we believe about how well it transfers. `only_an_exact_match_can_produce_a_measured_field`
asserts it over every field of every non-exact basis. This is what stops the badge decaying
into decoration.

### Two things are recomputed even on an exact match

- **`-ncmoe`, when this card has less free VRAM right now than the plan needs.** A compositor
  holding a gigabyte is enough. The measured value would then be the first setting past the
  cliff. ⚠️ The headroom behind our floor is `Headroom::PROVISIONAL` — a *stated guess* — so
  the recomputed value may be one block more conservative than the card needs. The asymmetry
  justifies it: too conservative costs throughput, too aggressive costs a crash after loading
  59 GiB.
- **`-t`, when the host CPU is not the one it was measured on.** A thread count measured on a
  20-core box is not a measurement about an 8-core one. Physical cores are read from
  `/proc/cpuinfo` as distinct `(physical id, core id)` pairs; a machine that will not say gets
  no suggestion and is told why, rather than an invented number.

There is a third, quieter one: **a profile's measured `ctx_size` is re-planned before it is
adopted.** `-c` and `-ncmoe` come out of the same pool, so emitting a measured 4,096-token
context on top of a split planned for "largest that fits" would emit two flags that contradict
each other — and llama.cpp would discover the contradiction by running out of device memory.

### Extrapolating across cards

The quantitative basis is `docs/hardware-sizing.md`: on a real gpt-oss-120B routing trace, 13%
of the expert bank resident intercepts 53–79% of expert touches and 35% intercepts 84–98%. The
resolver **re-derives** `-ncmoe` from the new card's actual VRAM rather than scaling the old
number, because the planner already does that arithmetic correctly; the curve is used to say
*what the new residency is worth*.

🔴 **A coverage curve is read only for the model it was measured on.** `bench/PROTOCOL.md` §9
records the cost of the alternative: a curve taken from Qwen3-30B (8 of 128 active) applied to
gpt-oss (4 of 128) and called conservative was **optimistic** — 40.6% / 7.3% / 1.2% predicted
miss against a measured 46.4% / 15.4% / 6.3%. So there is no built-in curve. One arrives
attached to its own model's profile, or the tool makes no coverage claim. Reading one outside
its measured span clamps to the nearest measured point and says so.

⚠️ Coverage predicts **staged bytes**, not tok/s. This project has no validated model between
them and publishes none.

---

## The fallback: what happens with no profile

Most machines will never have a profile. The fallback is what they get, and 🔴 **every value it
produces renders as `derived`, never `measured`** — `no_fallback_value_can_ever_render_as_measured`
asserts it over four models and three contexts, alongside the
`only_an_exact_match_can_produce_a_measured_field` invariant above.

The rule that shapes all of it: **a fallback exists because *not choosing* is measurably the
expensive option.** llama.cpp's own defaults are not neutral — one of them costs 2.09× — so
declining to answer is not the humble choice, it is the wrong one. What we owe the user is a
defensible answer with its provenance attached, not silence.

### `-t` — half the cores, and we say what that costs

`FALLBACK_THREAD_FRACTION = 0.50`, in `tuning::resolve`. One named constant, so changing the
fallback for every unmeasured machine is one edit. Physical cores are the base; hyperthread
siblings share load/store units and a memory-bound expert gather gets nothing from a second thread
on one core.

| | |
|---|---|
| **What llama.cpp does** | **4 threads**, whatever the machine is. `common_cpu_get_num_math()` mis-reads Arrow Lake's hybrid topology; `llama-server` prints `n_threads = 4` on a 20-core part |
| **What that costs** | **2.09×** on gpt-oss-120B — 14.11 ± 0.33 tok/s against 29.54 ± 0.04 at `-t 16`, arms interleaved so page-cache state is common-mode |
| **What we ship** | 50% of physical cores. On the measured box that is **10** |
| ⚠️ **What we know is better** | **16 of 20 — about 80%.** `-t 16` wins on every model in the catalogue where threads matter |
| ⚠️ **What 50% costs** | roughly **a fifth** of what `-t 16` reaches, on the models that care. The in-tree `-t 8` readings bracket it: Llama-4-Scout 14.27 → 18.52 (−23%), Qwen3-30B 68.20 → 73.04 (−6.6%), Qwen3.6-35B 53.18 → 54.04 (−1.6%) |
| 🔴 **Never `nproc`** | `-t 20` lost on **five of six** models — gpt-oss-120B −8.3%, gpt-oss-20B −10.8%, olmoe at full offload −10.8%, Qwen3.6 −7.3%, Qwen3-30B −5.6%, Scout a tie — and its cells were 3–4× noisier. 8 P-cores and 12 E-cores are not 20 equal cores |

**50% is deliberately conservative for silicon nobody has benchmarked, and is not claimed to be
optimal.** 16-of-20 is one measurement on one hybrid Intel part. Half the cores leaves the other
half of an unknown machine alone, the loss is bounded, the screen says the value is derived, and a
user who measures moves it up. `threads_note` prints all of the above beside the flag.

### `-ncmoe` — the direction flips with the quantisation

🔴 **A single derivation rule would be wrong for half the catalogue.** `experts_belong_on_the_host`
branches on the quantisation, and both halves are measured on Arc B580 + SYCL:

- **MXFP4 → `-ncmoe` = every block.** A/B/A/B inside one process on gpt-oss-20B
  (`-ncmoe 24,0,24,0`, so drift is common-mode): every expert on the **CPU** measured
  **43.42 ± 0.08** against **36.36 ± 0.01** with every expert on the **GPU**. **1.19×, on a model
  whose 11.28 GiB fits entirely in the card's VRAM.** The sweep between the ends is monotone
  (+21%) and it survives at depth.
- **Q4_K / Q3_K → the planner's floor.** Qwen3-30B is monotone the other way: **74.08 tok/s at
  `-ncmoe 18` against 58.58 at 30**, 1.26×.

⚠️ What is measured is *that* it happens, not *why* — the SYCL MXFP4 matmul path was never
profiled — so the rule is carried as a property of **this backend at this quantisation**, and
everything it produces stays `derived`.

**The MXFP4 branch is applied after the plan succeeds, never instead of it.** The plan that
succeeded put *more* on the card, so taking every expert off spends strictly less VRAM: a feasible
plan cannot become an `OUT_OF_DEVICE_MEMORY`. The freed bytes are then re-planned into context by
`host_expert_context` — without that step the recommendation would free gigabytes and spend them
on nothing, and gpt-oss-20B's measured 65K-token configuration is worth what it is *because*
`-ncmoe 24` leaves 10,097 MiB free. That re-plan holds the bank at `active_experts` (the floor
`Policy::max_resident_experts` cannot go below) and caps at the trained context, so it is
conservative in both directions.

⚠️ **Consequence worth knowing:** for an MXFP4 model the `Tuning` block's `-c` and the *What will
fit* block's context now differ, because they are answers to two different questions — MoEArc's own
expert cache against llama.cpp's flags. The residency figures already differed for the same reason.

### The context tier — `-ncmoe` is a floor *for a context*

🔴 **The `-ncmoe` floor is a function of context, not of the model alone.** Measured on Qwen3-30B:
**18** blocks offloaded at depth 0, **21** at 8K, **28** at 32K — ten blocks consumed by KV growth.
A single-number floor OOMs a long-context user *after* they have loaded tens of gigabytes.

The planner already honours this, because `-ncmoe` and `-c` come out of **one** call to
`plan_llama` and therefore cannot contradict each other. What the fallback adds is
`context_coupling_note`, which states on screen that the `-ncmoe` printed is the floor **for the
`-c` printed beside it**, names the measured 18 → 21 → 28 movement, and points at `--ctx`.

⚠️ **No default context constant, on purpose.** `fit.rs` already refuses to invent one — *"a tool
that silently assumes 8k and reports success has answered a question the user did not ask"* — and a
second module quietly assuming a different one would put two contradictory context numbers on one
screen.

### `-fa` and the KV width

| flag | value | why it is stated rather than inherited |
|---|---|---|
| `-fa` | **on** | `-fa off` costs **1.72×** at 8K (Qwen3-30B, 52.4 against 30.52). llama.cpp's `auto` already resolves to on, so this changes nothing about what runs — it changes what a user can be talked out of, because `-fa off` is advice they copy from other backends |
| `--cache-type-k/-v` | **f16** | 🔴 `q8_0` measured **−19%** on this card (42.87 against 52.97), bought back exactly **one** `-ncmoe` block, and one block below that **hard-aborts** inside `ggml_backend_sycl_synchronize` instead of returning an OOM a tool can catch. It is not offered. f16 is also the width every plan here is computed at |

🔴 **`-fa` carries its value.** In this engine the option is
`common_arg({"-fa", "--flash-attn"}, "[on|off|auto]", …)` — a bare `-fa` swallows the next token as
its argument, so a command line ending in one is not a command line. `flags()` always prints the
value.

### Host memory — a reserve, not a cap

🔴 **A fraction-of-RAM cap on the model was proposed (50%) and refused. Do not re-litigate it; the
reasoning is recorded on `Reserve::DEFAULT` and guarded by
`a_fifty_nine_gib_model_is_never_refused_on_a_ninety_one_gib_box`.**

1. **It would refuse this project's own headline result.** Half of the 91 GiB box every number in
   `bench/` was taken on is 45.5 GiB. **gpt-oss-120B is 59.0 GiB** and runs there at a measured
   29.56 ± 0.16 tok/s with 32K of context.
2. **It would not protect anything, because the weights are `mmap`ped.** They live in the page
   cache, which the kernel reclaims the instant something else wants it. The cap would have no
   effect on memory pressure at all — its only effect would be to decline models that
   demonstrably work.

What stands is the existing ceiling — **`MemAvailable` less a 20% reserve**, which no budget may
exceed — and the existing three-state classification, in which a model past the budget is
*classified* rather than *refused*:

| tier | means |
|---|---|
| `RunsFromRam` | every cache miss is a copy from RAM |
| `RunsPagesFromDisk` | the excess is read from the drive on demand. **Slower, and not a failure** — `mmap` degrades, it does not fail |
| `WillNotFit` | not on this machine, and no room to fetch it. The only refusal, and it is about the drive |

---

## Is a tweak a win?

> *"If we hit 114 tok/s and things are solid, we apply some tweaks — if that gets it to 128,
> that's a win. If we hit 40, we back up."*

That loop is only sound if *a win* means **beat the noise**, not **beat the mean**.

```
moearc info gpt-oss-120b --candidate 31.9±0.3/5
```

```
  baseline          28.50 ± 0.20 tok/s over 5 runs
  candidate         31.90 ± 0.30 tok/s over 5 runs
  noise floor       ±0.32 tok/s (1.1% of the baseline) at k=2.0
  verdict           win

  win: +3.40 tok/s (+11.9%), clear of a ±1.1% noise floor at 21.1σ
  Keep it, and record it as the new baseline.
```

The headline output is **not** the verdict. It is the **noise floor as a percentage of the
baseline** — the smallest improvement this pair of measurements could possibly have resolved.

🔴 At the ±32% error bars some of this project's runs carried, that figure is ~36%. A real 12%
gain is **invisible**, and a loop run on means alone would accept it — and then accept the next
regression with exactly the same confidence.

```
  --candidate 29.5±4.0/3   →  indistinguishable: +1.00 tok/s (+3.5%) is inside a ±16.2% noise
                              floor. This experiment could not have seen an improvement smaller
                              than 16.2% — take more runs before deciding
  --candidate 31.9±9.0/3   →  no verdict — the candidate is not a measurement: its stddev is
                              28% of its mean, past the 20% at which PROTOCOL §5 stops calling
                              a run a measurement
  --candidate 128.4        →  no verdict — the candidate is not a measurement: 1 independent
                              run cannot carry an error bar; PROTOCOL §5 asks for at least 3
```

**Refusals, in the order they are checked.** Trust is decided *before* the difference is looked
at, because a difference computed from an untrustworthy sample is not a small finding — it is
not a finding.

1. **Different metric** → no verdict.
2. **Different prompt depth** → no verdict. Throughput on this engine falls 5.9× with depth;
   two depths are two questions, and averaging them into a verdict would be the tidiest
   possible lie.
3. **Either side untrustworthy** (`runs < 2`, or stddev ≥ 20% of mean — PROTOCOL §5) → no
   verdict, naming which side and why.
4. Only then: `win` / `regression` / `indistinguishable`, at
   `|Δ| > k · sqrt(sd_b²/n_b + sd_c²/n_c)` with `k = 2.0`.

`k = 2.0` is a **chosen constant**, the conventional ~95% two-sided interval under a normal
approximation. It is exported, printed with every verdict, and **optimistic at these run
counts** — with n = 3 a t-distribution would demand roughly twice the margin. It is documented
rather than corrected because erring toward *indistinguishable* is the safe direction: the
failure this guards against is calling a non-result a win.

A profile whose own score fails PROTOCOL §5 is **kept in the file** — a retraction stays in the
tree with its evidence — but is never handed out as a baseline.

🔴 **A baseline never crosses machines.** PROTOCOL §0: absolute throughput does not reproduce
across CPU, memory bandwidth, PCIe generation and page cache. An extrapolated profile carries
settings and **no score**; the tool says so rather than offering a number to beat that means
nothing on this box.

---

## The file contract

**`bench/tuning-report.json` is the benchmark harness's own report and is NOT this contract —
different shape, and it was renamed on 2026-09-07 precisely so it can no longer shadow the search
path below. The contract file is what this half consumes
it.**

### The one rule the producer must know

**The file contains only measurements.** There is no `origin` field and deliberately no way to
write one: anything in the file was measured, by definition, and anything *absent* is untuned.
Every downgrade is decided at resolution time from what actually matched. A producer that could
label its own output "measured" would be a producer that could lie in one field.

So: **omit a field you did not measure.** Do not write a default into it.

### Schema

```jsonc
{
  "schema": 1,                       // required; a different value loads nothing
  "generated_at": "2026-09-06T21:00:00Z",   // optional
  "profiles": [
    {
      "id": "arc-b580/gpt-oss-120b/mxfp4",  // required, unique, citable in a report
      "hardware": {
        "gpu": "Intel(R) Arc(TM) B580 Graphics",  // required, as the driver reports it
        "gpu_key": "arc-b580",                    // required — see below
        "vram_bytes": 12133400576,                // required, non-zero
        "driver": "xe / L0 build 37020",          // optional
        "cpu": "Intel Core Ultra 9 285K",         // optional
        "physical_cores": 20,                     // optional, but see below
        "ram_bytes": 98257694720                  // optional
      },
      "model": {
        "id": "gpt-oss-120b",          // required — matched case-insensitively against `moearc ls`
        "quant": "mxfp4",              // required — ggml's spelling, from the tensor index
        "file_bytes": 63387346208,     // optional
        "moe_blocks": 36,              // required
        "experts_per_block": 128,      // required
        "active_experts_per_block": 4, // required
        "expert_slots_total": 4608,    // required, and must equal blocks × experts_per_block
        "parameters": 116829156672     // optional; used only for the sibling test
      },
      "engine": { "name": "llama.cpp", "version": "b7013", "backend": "SYCL" },  // optional
      "settings": {                    // every field optional; absent = untuned
        "threads": 16,                 // -t
        "n_cpu_moe": 31,               // -ncmoe   (≤ moe_blocks, or the profile is rejected)
        "n_gpu_layers": 99,            // -ngl
        "ctx_size": 4096,              // -c
        "batch_size": null,            // -b
        "ubatch_size": null,           // -ub
        "kv_cache_type": "f16",        // --cache-type-k / --cache-type-v
        "flash_attn": true,            // -fa
        "extra_args": []               // reproduced verbatim, never parsed
      },
      "score": {                       // optional — see below
        "metric": "decode_tokens_per_second",
        "depth_tokens": 0,
        "mean": 28.5,
        "stddev": 0.2,
        "runs": 5                      // INDEPENDENT INVOCATIONS, not -r inside one process
      },
      "coverage": {                    // optional, and per-model
        "model": "gpt-oss-120b",       // must match `model.id` or the curve is ignored
        "trace": "bench/traces/",
        "points": [
          { "bank_resident": 0.13, "prose": 0.788, "code": 0.533, "reasoning": 0.569 }
        ]
      },
      "measured_at": "2026-09-06",     // required, YYYY-MM-DD; newest wins among candidates
      "protocol": "bench/PROTOCOL.md", // optional but wanted — it is what makes the number auditable
      "notes": "…"                     // optional
    }
  ]
}
```

### Field notes the producer should not skip

- **`gpu_key`** is the matching key and must be the *normalised* device name. The consumer
  computes the same normalisation from the detected device: lowercase, split on non-alphanumerics,
  drop `intel` / `r` / `tm` / `graphics` / `gpu` / `series`, join with `-`. So
  `"Intel(R) Arc(TM) Pro B60 Graphics"` → `"arc-pro-b60"`. It is separate from `gpu` because
  driver strings are not stable across driver versions and a profile must not stop matching
  because somebody updated a package.
- **`physical_cores`** is what makes `threads` transferable. Without it the thread count is
  carried as *extrapolated* with a caveat saying it could not be confirmed; with it, a matching
  host gets `measured` and a differing host gets its own derived count and is told why.
- **`score` is optional and a profile without one still tunes.** An `-ncmoe` floor found by
  bisecting until llama.cpp stopped crashing is a real measurement with no tok/s attached. Such
  a profile just cannot be a baseline to climb from.
- **`score.runs` is independent invocations.** PROTOCOL §5: process-level variance is the
  variance that bit this project, and `-r` inside one process cannot see it.
- **`score.depth_tokens` is part of the number's identity.** A candidate at a different depth
  gets no verdict.
- **`coverage.points`** may arrive in any order; they are sorted on read and interpolated
  linearly between measured points. Reading outside the measured span clamps and is flagged.

### Validation — a profile is *rejected and named*, never repaired

| condition | why |
| --- | --- |
| `schema` ≠ 1 | the whole file loads nothing; a newer producer may mean something different by the same field names |
| empty `id` or `hardware.gpu_key` | nothing to match on or to cite |
| `hardware.vram_bytes == 0` | not a card |
| `moe_blocks × experts_per_block ≠ expert_slots_total` | **the 36× bug** — the per-block count where the slot count belongs makes every residency figure wrong by the block count while looking entirely plausible |
| `active_experts_per_block > experts_per_block`, or any of the three zero | not an MoE geometry |
| `settings.n_cpu_moe > moe_blocks` | not a value llama.cpp accepts |
| `coverage` present with no points | an empty claim |

Rejected profiles are listed by id under `moearc info` as `note: profile rejected — <id>: <why>`.
A silently dropped row looks exactly like a row nobody measured.

### Where the file is looked for

First hit wins, so a user's own file beats ours:

1. `$MOEARC_PROFILES` — a file, or a directory containing `tuning-profiles.json`
2. `./bench/tuning-profiles.json` (a repository checkout, run from its root — absent by default;
   the harness's own output lives at `bench/tuning-report.json` in a different shape)
3. `<exe dir>/tuning-profiles.json`
4. `<prefix>/share/moearc/tuning-profiles.json`
5. `$XDG_DATA_HOME/moearc/tuning-profiles.json`, else `~/.local/share/moearc/…`
6. the built-in set

**The built-in set is live** — `store.rs`'s `BUILTIN` carries the seven models
`bench/tuning-profiles.md` measured, so a shipped binary tunes without the repository beside it.

🔴 **It is not `include_str!("../../../../bench/tuning-profiles.json")`, and that is a producer
bug rather than a design choice.** The file the harness committed **is not written to the schema
above**: it is a report — one top-level `hardware` block, a `models` array keyed by GGUF filename,
`recommended` settings, `ncmoe_floor_by_ctx`, `knob_value` — where this half expects `schema`,
`profiles[]`, and a `model.id` matching what `moearc ls` prints. `serde` refuses it at the first
field.

Two consequences, and neither is papered over here:

- `crates/moearc-cli/src/tuning/builtin-profiles.json` is a **transcription** of the same
  measurements into the documented shape. Settings, scores and notes come from
  `bench/tuning-profiles.md`; ids, quantisations, geometry, parameter and byte counts come from
  `moearc ls --json` run against the same GGUF files. Nothing was measured or inferred to write
  it — but **it will drift**, and the fix is for the producer to emit the contract shape, after
  which `BUILTIN` becomes the one-liner it was meant to be.
- ✅ **Resolved 2026-09-07.** The harness's report was renamed to `bench/tuning-report.json`,
  so it no longer sits on the search path and no longer suppresses the built-in set in a
  repository checkout. A checkout now behaves exactly like a shipped binary: one source of
  truth, compiled in. The producer emitting the contract shape directly remains the tidier
  end state, and is still open.

**Absence is silent and normal.** A machine with no file resolves everything to *derived* and
says so. A **malformed** file is never silent: nothing is loaded from it and the parse error is
printed, because reading half a file and tuning from what parsed is how a user ends up running
a configuration nobody wrote.

---

## Where it surfaces

- **`moearc info <model>`** — the `Tuning` block: confidence badge, the pasteable
  `llama-server` command, a flag table with a provenance column and a plain-English purpose,
  the residency fraction and (when a curve exists) what it is worth, the baseline with its
  noise floor, and the caveats.
- **`moearc info <model> --candidate <mean[±sd][/runs]>`** — the comparison above.
- **`moearc` / `moearc --no-tui`** — a per-row badge in *What will fit*, with a legend.
- **The TUI** — the same per-row badge, plus a one-line `tuning` field beside `footprint` near
  the top of the detail pane and the full flag table at the bottom. 🔴 The badge is placed high
  deliberately: the pane has a fold, the planner's rationale alone runs to six wrapped lines,
  and whether any of this was measured must never be the thing that scrolls.
- **`--json`** — `tuning`, `tuning_source`, `tuning_profile`, `tuning_command`, `cpu` and
  `candidate` on `moearc info`; `tuning` and `tuning_source` on the device report.

---

## The planner keeps its job

`moearc_engine::memory` was written to plan MoEArc's *own* VRAM. It now also speaks llama.cpp:

```rust
pub struct BlockGeometry { pub moe_blocks: u32, pub experts_per_block: u32 }
pub fn llama_split(alloc: &Allocation, model: &ModelFootprint, geom: BlockGeometry)
    -> Result<LlamaSplit, PlanError>;
pub fn plan_llama(device, model, geom, policy, want)
    -> Result<(Allocation, LlamaSplit), PlanError>;
```

`Allocation` carries the reasoning a user can read and `LlamaSplit` carries the flags a user can
paste, and they cannot disagree because there is one calculation behind them. Partial blocks
round **down** — llama.cpp offloads whole blocks' expert tensors, so a plan with room for 612
slots on a 128-expert model gets 4 blocks and 100 slots of free margin against the cliff, which
is reported rather than quietly absorbed. `llama_split` refuses a geometry that does not
multiply out to the model's slot count, which makes the 36× bug unrepresentable rather than
merely unlikely.

## Known limits

- **`-ngl 99`** means "every layer". That every layer *fits* is derived — `plan_llama` refuses a
  card the dense weights do not fit on — but 99 is llama.cpp's convention, not a tuned number,
  and the caveat says so.
- **KV width is f16** everywhere the planner is concerned, because that is what `moearc-model`
  reads out of the header. A measured profile may carry `q8_0`; the plan behind it is still
  computed at f16, which is the conservative direction.
- **No throughput is predicted from coverage.** There is no validated model between staged bytes
  and tok/s in this project.
- **The comparison's `k` is a normal approximation**, optimistic at n = 3.
- **A derived `-c` can exceed any context that was benchmarked.** It is the planner's arithmetic
  against free VRAM, and llama.cpp's compute buffers grow with context in a way nothing here
  models — that growth is what `Headroom::PROVISIONAL` is holding back, and 12% is a stated guess.
  gpt-oss-20B derives `-c 131072` where `bench/` measured 65,536. A context that does not fit
  fails as a clean `OUT_OF_DEVICE_MEMORY` at load, which is detectable; it is still a claim made
  by arithmetic rather than by a run, and the badge says so.
- **The MXFP4 rule is read off two models on one backend.** `experts_belong_on_the_host` matches
  the quantisation string, so a *third* MXFP4 model — or the same one on a different backend —
  inherits a direction nobody measured for it. It stays `derived` for exactly that reason.
- **The transcribed built-in set will drift** from `bench/tuning-profiles.md` until the harness
  emits the schema this document specifies. See *Where the file is looked for*.
