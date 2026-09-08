# `moearc serve` — the resolved profile, actually running

Written 2026-09-08. Everything quoted below is real output from CHARA's Arc B580, and
the two runs are labelled so nobody has to guess which model produced which line.
**No throughput claim is made anywhere in this document.** The box was not quiet by
`bench/PROTOCOL.md`'s standard and no timed run was taken; the numbers that do appear
are token ids and load times, neither of which is a benchmark.

This closes the last gap in the product story. Every other link existed:
`moearc` explains the machine, `moearc ls` finds the models, `moearc info` prints the
exact command line with per-flag provenance — and `moearc serve` parsed its arguments
and printed *"not wired yet: the inference server arrives with the engine. Nothing is
listening."* It now listens.

---

## 0. The decision

**`moearc serve` supervises llama.cpp's own `llama-server` as a child process**,
launched with the argv `crates/moearc-cli/src/tuning/resolve.rs` produced for this
card, this CPU and this file.

The alternative was a server inside the `moearc` binary, over
`moearc_llama::runtime` — which `docs/llama-integration.md` §6 deliberately kept open,
and which `docs/pivot-inventory.md` sketched as *"`moearc-server/src/engine.rs` — 115
lines, `Session` → `moearc_llama::Context`"*. Both are defensible. The four reasons
below are why one won, and the fourth is the one that would be hardest to undo.

### 0.1 The command MoEArc prints is the command MoEArc runs

`moearc info` renders `Resolved::command()` for a user to paste. `moearc serve` builds
its argv from the **same `Resolved`**, through the same `flags()` call. There is exactly
one description of what a configuration means.

An in-process server would be a second one — a translation from `Resolved` into
`ContextParams` and `ModelParams` — free to drift from the printed one. This project has
withdrawn three results for reporting a number that described something other than what
ran. A `serve` path whose behaviour can silently diverge from its own `info` output is
that failure with a socket attached.

### 0.2 The product is the tuning, not the HTTP

What MoEArc knows that nobody else does is that this box wants `-t 16` (llama.cpp
defaults to 4, which cost **2.09×** on this machine) and that gpt-oss-120b's `-ncmoe`
floor is 36 (below it, llama.cpp aborts with `OUT_OF_DEVICE_MEMORY` *after* loading
59 GiB). Chat templates, SSE framing, `/v1/models`, stop strings and OpenAI's error
envelope are solved work. Re-solving them buys a user nothing and costs a release.

### 0.3 It defers a decision that should not be forced by a ship date

`docs/pivot-inventory.md` closes on two questions it deliberately does not answer. The
first: **which tokenizer and which sampler is authoritative.** `moearc-server` has both
(`tokenize.rs` on HF `tokenizers`, `sampling.rs`); `moearc-llama` now also has both
(`mla_tokenize`, `Sampler`). Two tokenizers in one system is a silent-divergence risk;
two samplers means two answers for seed 42.

Supervising `llama-server` means the tokenizer, the chat template and the sampler that
produce a served token are **the same ones that produced the verified 40-of-40 token-id
match against llama.cpp**. The question stays open, to be answered on its merits.

### 0.4 🔴 The serving contract is preserved by not reimplementing it

`crates/moearc-engine/src/session.rs` records the contract `moearc-server`'s `Generator`
was written against, and one clause is subtler than the rest:

> **A token that is never emitted must not reach the KV cache.**

Sample a token, decide not to emit it — a stop token, a stop-string match, a cancelled
request — and if it has already been appended to the cache, the sequence is corrupted for
every subsequent token. Silently. Visible only as drift, later.

That clause is not a detail of an implementation; it is a property a serving path either
has or does not. `llama-server`'s slot machinery already has it, tested by upstream
against every sampler and stop condition it ships. Re-deriving it inside a two-day window,
in a project whose own retired forward pass carried a ring-victim bug that **diverged from
llama.cpp at token 18 — nowhere earlier, fluent throughout** — is the wrong trade. See
§4.3 for the check this argument obliges us to actually run.

### 0.5 A crash stays a crash in the child

`-ncmoe` one block too low is answered by ggml with an abort. As a child that is an exit
status this module can explain (`serve::diagnose`, §3.4); in-process it is a `GGML_ABORT`
inside our own address space and the user gets a core dump instead of a sentence.

### 0.6 ⚠️ What it costs, stated rather than discovered

It needs a second executable. Three things make that smaller than it sounds, and one
thing genuinely has to change.

- **It is not a `PATH` dependency.** `find_binary()` looks beside the `moearc`
  executable **first**, which is the layout `packaging/bundle.sh` already produces —
  four executables under `libexec/`, each behind `launcher.sh`. `PATH` is the last
  resort, not the mechanism.
- **The payload does not grow by a library.** Under the llama.cpp architecture the
  tarball must ship `libllama.so.0` and the `libggml-*.so.0` family *regardless* of which
  design is chosen, because `moearc-llama` links them either way. `llama-server` is one
  more ELF in a directory that already has to exist.
- 🔴 **`bundle.sh` must still be rewritten, and would have had to be anyway.** Lines 64,
  66 and 84 are hard-wired to a single `libmoearc_kernels.so`.
  `docs/pivot-inventory.md` §`packaging/` already flags all three as required work,
  independent of this module. What this decision adds to that list is one file name.

---

## 1. 🔴 The integrated GPU, and where it is closed

`moearc-device`'s `fitness.rs` encodes a measured finding: on this machine the iGPU
reports **91,890,372,608 bytes** of "device memory", which is 93.5% of system RAM, and
`docs/pivot-inventory.md` says the quiet part out loud — **"llama.cpp will happily pick
that device."**

It is not hypothetical. Unpinned, on the reference box:

```
$ llama-server --list-devices
Available devices:
  SYCL0: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)
  SYCL1: Intel(R) Graphics (76029 MiB, 23270 MiB free)
```

With `-ngl 99` and llama.cpp's default layer split, both get used. The run succeeds and
lies — the release-blocking failure mode the announcement notes already carry for the
stale-driver problem.

`moearc serve` closes it with two independent enumerations that have to agree:

1. **Before anything loads**, `llama-server --list-devices` is parsed and matched
   **by name** against the device `moearc-device` chose — never by index.
   `docs/llama-integration.md` §5.2: index 0 is not contractually the discrete card. The
   match becomes `-dev SYCL0`.
2. **From the child's own mouth**, the `device_info:` block it prints at startup is read
   back and the pin is confirmed against the name. This happens *before* the model is
   read, so a disagreement costs seconds rather than a 59 GiB load.

⚠️ The second step is why `-lv 4` is on every command line. At llama.cpp's default
verbosity of 3 the `device_info:` block **is not emitted at all**, and the pin would be
asserted against nothing. The child's resulting chatter is captured, not printed; it
reaches the terminal only under `-v`.

A machine with no matching device, or a `llama-server` that reports no SYCL device at
all, is a **refusal**, not a fallback.

---

## 2. What the user sees

```
$ moearc serve gpt-oss-120b

Serve

  model             gpt-oss-120b
                    /zfs/swift/models/gpt-oss-120b-MXFP4.gguf · 59.0 GiB · mxfp4
  device            Intel(R) Arc(TM) B580 Graphics · 11.3 GiB free
  endpoint          http://127.0.0.1:8080/v1
  engine            /zfs/swift/projects/llama.cpp/build/bin/llama-server
                    llama.cpp `llama-server`, the MoEArc reference build

  confidence        ◆ MEASURED
                    measured on this card with this model — profile
                    `arc-b580/gpt-oss-120b/mxfp4`, 2026-09-07

  flag            value     provenance    what it does
  -ngl            99        derived       layers on the GPU
  -ncmoe          36        measured      MoE blocks whose experts stay in host RAM
  -t              16        measured      host threads
  -c              86016     derived       context length, in tokens
  --cache-type-k  f16       measured      KV cache width
  --cache-type-v  f16       measured      KV cache width
  -fa             on        measured      flash attention
  -dev            SYCL0     selected      the card to run on, pinned by name

  🔴 not every setting above was measured — the weakest is derived. This is a starting
  point; measure from it with `moearc bench`.
  badges:  ◆ measured on this card   ◇ extrapolated from a nearby measurement
           · derived from the model's geometry, never run
```

followed by the pasteable command, any overrides, the resolver's caveats, and then:

```
  starting the engine…
  ✓ device confirmed — SYCL0 = Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11680 MiB free)
    (2 other device(s) were visible and are not being used: SYCL1 = Intel(R) Graphics …)
  ✓ model loaded in 4s
  ✓ listening on http://127.0.0.1:8080/v1
```

Three properties of that screen are deliberate.

- **The confidence line comes before the settings.** Same order `moearc info` uses, for
  the same reason: a reader who takes in one line must take in the one that says whether
  any of this was measured.
- 🔴 **The weakest field is restated as a claim, not left as a column.** A reader who
  skims the table and acts on it must not be able to miss that some of it was never run.
- **The devices *not* being used are named.** The one thing a user cannot check for
  themselves is which of several plausible devices got picked.

### 2.1 The provenance vocabulary

`tuning::schema::Setting` guarantees inside the tuning layer that a value cannot reach a
renderer without its `Origin`. `serve` extends that across the two things the tuning
layer has no word for, rather than borrowing one of its four and quietly stretching it:

| shown as | means |
| --- | --- |
| `measured` / `extrapolated` / `derived` / `untuned` | straight from the resolver, its own verdict |
| **`replanned`** | the resolver's value was recomputed because an override moved something upstream of it |
| **`selected`** | chosen by MoEArc's own measurement of this machine, not from any profile — today, the `-dev` pin |

🔴 `Provenance::is_measured()` is true for exactly one of the six, and
`nothing_but_a_measured_profile_may_read_measured` asserts it over the whole enum. An
override can never inherit a measured badge.

---

## 3. Overrides re-plan; they never contradict

`-c` and `-ncmoe` come out of the same pool of VRAM. Raising one without the other is how
a run dies with `OUT_OF_DEVICE_MEMORY` after loading 59 GiB. So no override edits a flag
in place — each one goes back through a planner.

### 3.1 `--ctx N`

Passed **into** `tuning::resolve::resolve`, which recomputes the split for that depth and
writes its own prose about the new pair. `crate::fit::plan` refuses first if `N` is past
the model's trained context, in the planner's own words.

```
$ moearc serve gpt-oss-120b --dry-run --ctx 8192
  -ncmoe          36        measured      MoE blocks whose experts stay in host RAM
  -c              8192      derived       context length, in tokens
```

### 3.2 `--moe-cache S`

Re-planned through `crate::fit::plan_with_slot_override`. **Both** `-ncmoe` and `-c` are
taken from that one `Fit`, so the pair cannot disagree, and both are badged `replanned`:

```
$ moearc serve gpt-oss-120b --dry-run --moe-cache 1024
  -ncmoe          32        replanned     MoE blocks whose experts stay in host RAM
  -c              2048      replanned     context length, in tokens

  Overrides and adjustments
  · 🔴 `--moe-cache 1024` pinned residency, so the split was re-planned from it: 612 of
    4,608 slots stay on the card, which is `-ncmoe 32`, and `-c 2,048` is what the same
    plan has room for beside them. Both flags come out of that one calculation — raising
    context without moving the split is how a run dies with OUT_OF_DEVICE_MEMORY after
    loading the whole model.
```

Slots become **whole blocks**, rounded so a partly-covered block counts as *not*
resident: `-ncmoe` has no half-block setting, and the other rounding ends in an abort.

⚠️ An override also **supersedes the resolver's prose about the value it replaced**.
The resolver writes a paragraph explaining the `-ncmoe` it chose; leaving that on screen
beside a different `-ncmoe` would put two contradictory numbers in front of the user,
one of them in a sentence arguing it is right. `supersede_split_caveats` drops them and
the replacement note above carries the same warning for the new pair. The residency
percentage is recomputed from the override for the same reason.

### 3.3 `--host-budget`

Changes **no** llama.cpp flag, and says so on screen rather than being silently dropped:
llama.cpp memory-maps the weights and the kernel decides what stays in the page cache, so
there is nothing to carry the budget into. It still sets the host tier that is reported.

### 3.4 The trained-context clamp

Found while wiring this up, and worth recording because it is a live inconsistency in a
module `serve` only reads.

`crate::fit` caps the context it prints at the model's trained length and states why:
*olmoe-1b-7b is a 4,096-token model and a B580 with 11.3 GiB free has room for 47,360
tokens of its KV cache.* 🔴 **`tuning::resolve::derive` applies that cap only on its
all-experts-on-host (MXFP4) path**, so a Q4_K model on a roomy card resolves to
`-c 47360` — and `moearc info olmoe-1b-7b-0924-instruct` prints exactly that today,
beside a "Planned split" that says 4,096.

`serve` will not launch at that length. It **re-plans through the resolver** at the
trained context rather than overwriting `-c` afterwards, so the split, the flag and the
paragraph all agree:

```
  -c              4096      derived       context length, in tokens

  Overrides and adjustments
  · this card has room for 47,360 tokens of KV cache and this model was trained for
    4,096, so the split was re-planned at the smaller figure. The pages would allocate
    either way; the answers past 4,096 would not mean anything.
```

⬜ **The cap belongs in the resolver, not here.** `crates/moearc-cli/src/tuning/` was
out of scope for this change; `serve` compensates, and `moearc info` is still wrong.

---

## 4. Proof

### 4.1 It serves

Real output, `moearc serve olmoe-1b-7b-0924-instruct --port 18100`, Arc B580:

```
  ✓ device confirmed — SYCL0 = Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11680 MiB free)
    (2 other device(s) were visible and are not being used:
     SYCL1 = Intel(R) Graphics (75473 MiB, 21953 MiB free);
     CPU   = Intel(R) Core(TM) Ultra 7 265K (93705 MiB, 93705 MiB free))
  ✓ model loaded in 4s
  ✓ listening on http://127.0.0.1:18100/v1

$ curl -s http://127.0.0.1:18100/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"olmoe-1b-7b-0924-instruct","messages":[{"role":"user","content":
        "List three reasons a mixture-of-experts model is cheaper to run than a dense one."}],
        "max_tokens":40,"temperature":0,"seed":0}'

"content": "A mixture-of-experts (MoE) model is a type of neural network architecture
            that combines multiple sub-networks, or \"experts,\" to solve a given task.
            Each expert"
"model":   "olmoe-1b-7b-0924-instruct"
"usage":   {"completion_tokens": 40, "prompt_tokens": 33, "total_tokens": 73}

$ curl -s http://127.0.0.1:18100/v1/models
['olmoe-1b-7b-0924-instruct']
```

`/v1/models` reports the **handle**, not the file path, because `serve` passes
`--alias`. A client configured against `moearc ls` output works unchanged.

### 4.2 The 59 GiB demo model, on an 11.33 GiB card

`moearc serve gpt-oss-120b`, one command, no flags. The plan above (§2) is verbatim from
this run; what followed it:

```
  ✓ device confirmed — SYCL0 = Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11680 MiB free)
    (2 other device(s) were visible and are not being used:
     SYCL1 = Intel(R) Graphics (77945 MiB, 17340 MiB free);
     CPU   = Intel(R) Core(TM) Ultra 7 265K (93705 MiB, 93705 MiB free))
  ✓ model loaded in 12s
  ✓ listening on http://127.0.0.1:18104/v1

$ curl -s …/v1/chat/completions -d '{"model":"gpt-oss-120b","messages":[…],
      "max_tokens":64,"temperature":0,"seed":0}'

"content":           "**Three reasons a Mixture‑of‑Experts (MoE)"
"reasoning_content": "We need to answer: list three reasons a mixture-of-experts (MoE)
                      model is cheaper to run than a dense one. Provide concise bullet
                      points, maybe with brief explanation. Should be clear."
"model":             "gpt-oss-120b"
"usage":             {"completion_tokens": 64, "prompt_tokens": 85, "total_tokens": 149}

$ curl -s …/v1/models   →   ['gpt-oss-120b']
```

📌 **`reasoning_content` is the line worth looking at twice.** gpt-oss speaks Harmony,
whose chat template carries a separate reasoning channel, and llama.cpp parsed it into
the right field without being asked. That is §0.2 and §0.3 made concrete: an in-process
server would have had to reproduce Harmony's channel semantics in minijinja, correctly,
this week — and a client that renders `content` would otherwise have shown the model's
scratchpad to the user.

⚠️ **No throughput figure from this run is quoted, here or anywhere.** The box was not
quiet — a background container fleet, and the one-minute load average moved from 1.11
before to 6.52 during — and `bench/PROTOCOL.md` §5 does not call a single unpinned run a
measurement. The load times above are reported as evidence that a load happened, not as
results.

### 4.3 🔴 It changes nothing about what the engine produces

The check §0.4 obliges. Same model, same flags, same greedy prompt, **64 tokens** —
deliberately past the token-18 class of bug, where a divergence is fluent and nobody
diffs that far into a chat reply. Side A goes through `moearc serve`; side B is the argv
`moearc` printed, hand-launched, with no `moearc` in the picture:

```
A (moearc serve): [7785, 15, 187, 187, 510, 14731, 273, 6181, 310, 14029, 15, 187, 187,
                   510, 3072, 273, 6181, 310, 9963, 15, 19, 3041, 952, 15, 187, 187, 510,
                   3565, 3448, 273, 6181, 310, 5112, 15, 187, 187, 510, 3872, 802, 12404,
                   273, 6181, 310, 253, 11414, 280, 687, 7337, 15, 187, 187, 510, 3872,
                   7908, 273, 6181, 310, 253, 492, 49122, 7908, 13, 11253, 273]
B (hand-run)    : … identical …
IDENTICAL over 64 tokens
```

That is a regression check on **parameter plumbing**, which is exactly what
`docs/pivot-inventory.md` says `verify-clean.sh` becomes under this architecture. It
cannot prove llama.cpp correct; it proves MoEArc adds nothing to it.

### 4.4 🔴 The engine cannot outlive its supervisor

Found by testing rather than by reasoning, and it was a real defect. `Drop` on the
supervisor covers the paths the program takes on purpose, and an interactive `Ctrl-C`
is covered by the terminal signalling the whole foreground process group. Both were true.
Both were insufficient:

```
$ kill -TERM <moearc pid>
$ ps -eo pid,ppid,args | grep llama-server
   9339       1 …/llama-server -m …/gpt-oss-120b-MXFP4.gguf -ngl 99 -ncmoe 36 …
```

Reparented to init, still holding the model in VRAM, still bound to the port. A
default-disposition `SIGTERM` terminates without unwinding, so no destructor runs — and
`SIGKILL` and a segfault have the same shape, with no handler able to catch either.

The fix is `PR_SET_PDEATHSIG`, set in the child between `fork` and `exec`, so the kernel
makes the guarantee rather than code of ours that has to still be running. Verified
against the hardest case:

```
supervisor=15282 engine=15312
--- SIGKILL the supervisor (no destructor can run) ---
PASS: engine 15312 died with its supervisor
```

⚠️ Two limits, stated: the signal fires when the *thread* that forked exits — fine here,
the spawn is on the main thread — and it is Linux-only, which matches everything else
about a SYCL-on-Arc target.

### 4.5 When it dies, it says why

The failure a user actually meets is `-ncmoe` one block too low, answered after tens of
gigabytes have loaded. `serve::diagnose` reads the captured log and answers it in terms
of the two flags that cause it and the one that fixes it, above the engine's last 30
lines rather than instead of them.

---

## 5. What this does not do

Stated plainly, because a gap nobody wrote down is a gap somebody will trip over.

- **No live serving stats.** `source.rs`'s `ServeStats` is still a fixture, and the TUI's
  serving screen still draws from it. `docs/pivot-inventory.md` already records that
  `kv_utilisation` and `expert_hit_rate` have **no llama.cpp source**; `Perf`-style
  numbers are available per request in llama-server's `timings` object, and wiring them
  is separate work.
- **The TUI's `Action::Serve` is unchanged.** `moearc serve <model>` is the wired path;
  the interface's serving screen is not.
- **`moearc-server` is not retired by this**, and nothing here says it should be. §0.3
  is a deferral, not a verdict.
- **The dynamic-residency thesis is still deferred**, exactly as
  `docs/llama-integration.md` §5.4 states. `-ncmoe` is a static, load-time split. Nothing
  in this module changes that, and `serve` should not be read as closing it.
- **No throughput number is produced or quoted by `serve`.** It reports a load time,
  which is not a benchmark, and points at `moearc bench` for anything that is.
