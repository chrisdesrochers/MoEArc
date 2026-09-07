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
| **derived** | · | computed by `moearc_engine::memory` from the model's geometry and this card's free VRAM. Real arithmetic; nothing was ever run |
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

**`bench/tuning-profiles.json` is produced by the benchmark harness. This half only consumes
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
2. `./bench/tuning-profiles.json` (a repository checkout, run from its root)
3. `<exe dir>/tuning-profiles.json`
4. `<prefix>/share/moearc/tuning-profiles.json`
5. `$XDG_DATA_HOME/moearc/tuning-profiles.json`, else `~/.local/share/moearc/…`
6. the built-in set

**The built-in set is `None` today** — a build that `include_str!`s a missing file does not
compile. When the harness commits `bench/tuning-profiles.json`, `store.rs`'s `BUILTIN` becomes
one line and a shipped binary carries its measurements without needing the repository beside it.

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
