# Captured expert-routing traces

Real `ffn_moe_topk` values pulled out of a running llama.cpp decode, one file per prompt.
These replace the synthetic generator in `residency.rs` as the evidence base for the project's
central claim. Nothing in this folder is generated or estimated.

## 🔴 Decode and prefill are different regimes, and only decode is the question

Prefill routes a whole prompt of tokens through one graph; decode routes one token at a time.
Expert **residency** is a decode-time problem — that is when the engine has to decide what to keep
in VRAM between tokens. Both are captured, in separate files, and the headline numbers below are
**decode only**. The prefill files are included for contrast but are short (102–143 steps) and
therefore dominated by compulsory misses; they support no conclusion on their own.

## Files

| File | Steps | What |
|---|---|---|
| `qwen35moe-prose.decode.ndjson` | 1024 | Long-form English essay generation |
| `qwen35moe-code.decode.ndjson` | 1024 | Rust source generation |
| `qwen35moe-reasoning.decode.ndjson` | 1024 | Step-by-step arithmetic/scheduling reasoning |
| `qwen35moe-*.prefill.ndjson` | 102–143 | The prompt pass for each, kept separate |
| `prompts/*.txt` | | The exact prompt text, ChatML markup included |
| `llama.cpp-eval-callback-moearc.patch` | | The llama.cpp patch that produced all of it |
| `capture.sh` | | The capture driver |

## Format — `moearc-trace-v1`

Newline-delimited JSON. **Line 1 is a header object**; every line after it is one step.

```json
{"format":"moearc-trace-v1","phase":"decode","n_layer":40,"n_layers_routed":40,
 "n_expert_used":8,"n_prompt_tokens":114,"n_prefill_steps":114,"n_decode_steps":1024,
 "hit_eog":false,"model_file":"…gguf","quantisation":"Q4_K_M","n_expert":256,
 "llama_cpp_commit":"…","llama_cpp_patched":true,"captured_utc":"…",
 "prompt_name":"prose","prompt":"…","backend":"CPU (-ngl 0)","sampling":"…"}
{"step":0,"phase":"decode","pos":114,"e":[0,181,0,14,1,126,…]}
```

| Field | Meaning |
|---|---|
| `step` | Index within this phase, from 0 |
| `phase` | `prefill` or `decode` |
| `pos` | Absolute token position in the sequence |
| `e` | **Flat** array of `layer, expert, layer, expert, …` — always an even number of entries |

`e` is flat rather than nested so the reader stays dependency-free: `Trace::from_ndjson_file` in
`crates/moearc-engine/src/residency.rs` scans for `"e":[`, reads integers to the `]`, and pairs
them up. It reads nothing else from a step line, so fields may be added to the capture tool
without breaking it. A malformed step is an error, never a skip — a loader that silently dropped
steps would understate every miss count taken from the trace.

## How these were captured

`llama-eval-callback` **was not sufficient as shipped**, for two independent reasons:

1. **It never decodes.** It runs exactly one `llama_decode` over the whole prompt and exits, so it
   can only ever observe the prefill regime.
2. **Its printer truncates.** `common_debug_print_tensor` is called with `n = 3` and elides the
   middle of any dimension where `ne[i] > 2n`. `ffn_moe_topk` is `[8, n_tokens]`, and `8 > 6`, so
   elements 3 and 4 of every row are replaced by `...` — a *complete* tensor of 8 values still
   comes out incomplete. It also prints through the float formatter (`%12.4f`) with no step
   boundaries.

`llama.cpp-eval-callback-moearc.patch` therefore adds, to that example only:

- a callback that answers `ask` with `true` only for `ffn_moe_topk-<il>` and records the raw
  `int32` contents (the layer index is read from the tensor name, not inferred from order);
- a real decode loop driven by `common_sampler`, one `llama_decode` per token;
- NDJSON output, prefill and decode written to separate files;
- ubatch-boundary handling: a non-increasing layer index means a new ubatch has begun.

The patch is inert unless `MOEARC_TRACE_OUT` is set, so the upstream behaviour of the example is
unchanged.

```sh
cd "$LLAMACPP" && git apply /path/to/llama.cpp-eval-callback-moearc.patch
cmake --build build --target llama-eval-callback -j
MODEL=… LLAMACPP=… OUT=… ./capture.sh prose 1024
```

## Provenance of the committed traces

| | |
|---|---|
| Model | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, Q4_K_M, 34.66 B params |
| Architecture | `qwen35moe` — 40 blocks, **all 40 MoE**, 256 experts each, 8 active ⇒ **320 activations per token** |
| Hybrid | `full_attention_interval = 4`: blocks 3, 7, … 39 carry attention; the other 30 are recurrent |
| llama.cpp | `e107984bcffcfd701e82738092a2b000b6fda7a2`, plus the patch above |
| Backend | **CPU, `-ngl 0`** — deliberately. A routing trace is a functional capture, not a benchmark, and the GPU was in use by another measurement |
| Sampling | `--temp 0.7 --top-k 20 --top-p 0.8 --seed 20260904` (reproducible) |
| Captured | 2026-09-05 UTC |
| Thinking | Disabled for `prose` and `code` (empty `<think></think>` prefilled). Left on for `reasoning` |

⚠️ **Thinking mode matters and was nearly a trap.** The first capture attempt left it on, and both
the "prose" and the "code" prompt spent all 512 tokens inside a `<think>` block writing English
planning notes. The two traces would have been compared as prose-vs-code while both were in fact
English prose. The committed prose/code traces have thinking suppressed and generate actual essay
text and actual Rust; `reasoning` keeps it on deliberately, as a third content type.

## Measured results — decode, 3976 slots

3976 slots is what the real card supports (`bench/README.md`: 11.33 GiB allocatable, 2.38 GiB
dense weights, 1.95 MiB per expert). Reproduce with:

```sh
cargo run --release -p moearc-engine --example trace_report -- bench/traces/<file> [capacity]
```

| Prompt | Working set | Top 10% of experts hold | Reused from previous step | **static** | **LRU** | LFU | **optimal** |
|---|---|---|---|---|---|---|---|
| prose | 6725 / 10240 (65.7%) | 50.8% | 42.3% | **55.0%** | **95.2%** | 95.8% | **97.3%** |
| code | 8713 / 10240 (85.1%) | 50.7% | 38.7% | **42.5%** | **88.9%** | 88.0% | **94.5%** |
| reasoning | 8721 / 10240 (85.2%) | 44.5% | 35.5% | **42.5%** | **89.1%** | 86.6% | **94.6%** |

For comparison, the synthetic trace previously quoted gave static 40.0% / LRU 65.9% /
optimal 80.1%. **The real traces show a much larger static→LRU gap than the synthetic one did,
not a smaller one**: +40.2 points on prose, +46.4 on code, +46.6 on reasoning, against the
synthetic +25.9.

### Content changes locality, and by a lot

Prose routes through **6725** distinct experts where code and reasoning use **~8720** — 30% more
of the model touched for the same 1024 tokens. LRU is correspondingly 6 points better on prose.
Whatever residency policy ships must be measured on more than one content type; a prose-only
benchmark would overstate it.

Routing skew is nearly identical across content types (top 10% ≈ 45–51% of activations), so the
difference is *breadth* of the working set, not concentration within it.

### Routing differs by block, but not by block type

Distinct experts used per block ranges from **119 to 256** (prose) and **186 to 256** (code) — a
two-to-one spread, so a uniform per-block VRAM budget is the wrong shape. The gradient is by
*depth*: on prose, blocks 0–6 use 206–252 distinct experts, while everything from block 12 onward
sits between 119 and 178. Block 2 is the widest and block 20 the narrowest on all three prompts.

But the hybrid's attention blocks and recurrent blocks route **indistinguishably**: mean distinct
experts 170.0 vs 167.5 (prose), 221.2 vs 216.7 (code), 219.4 vs 217.6 (reasoning). Despite only
10 of the 40 blocks carrying attention, **the attention/recurrent split has no measurable effect
on expert routing** — depth does, block type does not.

## Caveats on these numbers — read before quoting them

- **The static baseline is modelled generously, in the incumbent's favour.** `Policy::StaticSplit`
  charges *no* compulsory misses: every expert in a resident block is treated as already loaded.
  LRU pays a fetch for each of the 6725–8721 experts on first touch, which alone caps it at 97.9%
  / 97.3%. The dynamic policies are handicapped here, not helped.
- **The static baseline is also sized generously.** `widest_static_split` counts only the experts
  the trace actually *touched* in the resident blocks, giving it 17–22 blocks at 3976 slots. Real
  `--n-cpu-moe` must hold all 256 experts of a resident block, which at 3976 slots is 15 blocks —
  a 37.5% hit rate by the same accounting (derived from the slot count, not separately measured).
- **At capacity ≥ 7680 the static split appears to beat LRU.** That is entirely the compulsory-miss
  asymmetry above: once the whole working set fits, static is scored at 100% while LRU still pays
  each expert's first fetch. It is a modelling artifact, not a result.
- **One model, one quantisation, one seed, three prompts, 1024 tokens each.** These are not
  claims about MoE routing in general.

---

# Qwen3-30B-A3B traces

The `qwen3-30b-*` files are a different model from everything above — `qwen3moe`, **48 blocks,
all 48 MoE, 128 experts each, 8 active ⇒ 384 activations per token**, a pure transformer with no
shared experts and no recurrent blocks. The tables and conclusions earlier in this file are about
`qwen35moe` and do **not** carry over: different expert count, different working-set size,
different slot budget. `qwen3-30b-prose` was captured first; its provenance is in the commit that
added it (`traces: Qwen3-30B-A3B, the first model where residency actually has to work`).

## `qwen3-30b-fibonacci` — the trace the engine can be checked against

| | |
|---|---|
| Prompt | `prompts/qwen3-30b-fibonacci.txt` — `def fibonacci(n):` + newline + four spaces, 5 tokens |
| Sampling | **greedy** (`--temp 0 --top-k 1`), not the 0.7/20/0.8 used for the `qwen35moe` set |
| Steps | 192 decode, 5 prefill |
| Backend | CPU (`-ngl 0`) |
| llama.cpp | `e107984bcffcfd701e82738092a2b000b6fda7a2` + `llama.cpp-eval-callback-moearc.patch` |

🔴 **Greedy is the whole point of this one.** MoEArc reproduces llama.cpp exactly on this prompt
(`crates/moearc-engine/tests/qwen3moe_forward.rs`), so the engine walks the same token sequence
and therefore routes through nearly the same experts — which makes the offline simulator and the
live engine comparable on *this* file in a way they are not on a sampled trace of some other
prompt. Measured at matched capacities, over 192 decode steps:

| slots | simulator LRU | engine LRU, cold | engine LRU, warm |
| ---: | ---: | ---: | ---: |
| 2952 | 91.0% | 91.0% | 93.0% |
| 2056 | 84.1% | 84.3% | 85.3% |
| 1032 | 68.4% | 67.3% | 67.6% |
| 520 | 53.7% | 47.8% | 47.9% |

⚠️ **The gap widens as capacity tightens, and the reason is routing, not caching.** MoEArc keeps
activations in f32 where `ggml-cpu` quantises them to Q8_K before every K-quant matmul, so the
router logits differ slightly and the 8th-ranked expert is not always the same one. Measured on
one token of `The capital of France is`, comparing `ffn_moe_topk` block by block against a
`MOEARC_DUMP_DIR` dump: of 48 blocks, **13 chose the identical list, 21 the same set in a
different order, and 14 a different set**. A different set means a slightly larger working set
over 192 steps, which costs nothing while almost everything fits and costs 6 points at 520 slots.

Note also that an engine run is 197 steps — 5 prompt tokens decoded one at a time, then 192 —
where the trace file holds the 192 decode steps only.

---

# gpt-oss traces — the engine's own geometry

Added 2026-09-06 for `bench/policy-sweep.md`. Everything above is Qwen; **these are the model
MoEArc's published baselines are measured on**, so a residency result taken from them needs no
transfer argument.

| | gpt-oss-120B | gpt-oss-20B |
|---|---|---|
| File | `gpt-oss-120b-MXFP4.gguf` | `gpt-oss-20b-MXFP4.gguf` |
| Architecture | `gpt-oss`, **36 blocks, all MoE, 128 experts, 4 active ⇒ 144 activations/token** | **24 blocks, 32 experts, 4 active ⇒ 96 activations/token** |
| Quantisation | MXFP4 | MXFP4 |
| Expert footprint | **12.607 MiB** (`bench/README.md`) | — |
| Decode steps | **512** | **1024** |
| Prompts | `prompts/gptoss120b-*.txt` | `prompts/gptoss20b-*.txt` |
| llama.cpp | `e107984bcffcfd701e82738092a2b000b6fda7a2` + `llama.cpp-eval-callback-moearc.patch` | same |
| Backend | CPU (`-ngl 0`), `-t 6`, `nice -n 19`, `ionice -c3` | same |
| Sampling | `--temp 0.7 --top-k 20 --top-p 0.8`, seed **20260906** (prose) / **20260904** (code, reasoning) | seed 20260904 |
| Captured | 2026-09-06 UTC | 2026-09-06 UTC |

| trace | working set | as % of model | top 10% of experts hold | step-to-step reuse |
|---|---:|---:|---:|---:|
| `gptoss120b-prose` | 2442 / 4608 | 53.0% | 52.4% | 34.3% |
| `gptoss120b-code` | 3685 / 4608 | 80.0% | 40.9% | 26.0% |
| `gptoss120b-reasoning` | 3486 / 4608 | 75.7% | 41.7% | 22.7% |
| `gptoss20b-prose` | 630 / 768 | 82.0% | 34.0% | 46.8% |
| `gptoss20b-code` | 746 / 768 | 97.1% | 37.8% | 45.8% |
| `gptoss20b-reasoning` | 710 / 768 | 92.4% | 36.8% | 43.4% |

🔴 **gpt-oss-120B is a much tighter residency problem than anything above.** The engine's 600-slot
pool is **13.0%** of its 4608 experts, where Qwen3.5's 3976 slots are **38.8%** of 10240; and
step-to-step reuse is **22.7–34.3%** against Qwen's 35.5–47.1%, because four active experts per
block overlap the previous token less than eight do. Conclusions carried from the Qwen tables into
this regime will be optimistic.

## Capture — not via `capture.sh`

`capture.sh` hardcodes `-t 20`, `"quantisation":"Q4_K_M"` and `"n_expert":256` into the header,
all three wrong for these models, so a separate driver was used with the same binary, the same
patch and the same conventions. It differs only in threads (6, to leave a concurrent GPU
measurement on the same host alone), the header metadata, and a `SEED` override.

⚠️ **Chat template.** gpt-oss uses **harmony**, not ChatML, so these prompts are not the `prose.txt`
/ `code.txt` / `reasoning.txt` above:

```
<|start|>system<|message|>You are a helpful assistant. Reasoning: low<|end|>
<|start|>user<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>
```

`reasoning` uses `Reasoning: high` and opens the **`analysis`** channel instead, which is how the
third content type stays genuinely distinct — the same trap the Qwen set documents with `<think>`.

🔴 **The gpt-oss-120B prose prompt is prefilled and that is load-bearing.** With the plain prompt
the 120B model **refuses the length**: three attempts produced 512 steps of degenerate
`... ... ...`, then `... (the rest of the essay... )` followed by EOG at **10** and at **21**
decode steps. The degenerate 512-step capture was **discarded, not used** — routing through a
collapsed token loop is not a decode trace of anything. The committed file prefills the assistant's
`final` channel with an opening clause (`"The story of lighthouse building around the North
Atlantic begins"`) and adds an explicit no-placeholder instruction, which produces real continuous
prose for the full 512 steps. The exact prompt is in the trace header, as always. The 20B model
needed none of this — **the same prompt is not equally answerable by two models of the same
family**, and a capture script that does not check its own output would have silently produced
a worthless trace.

```sh
MOEARC_TRACE_OUT=bench/traces/gptoss120b-prose \
MOEARC_TRACE_META='…' \
MOEARC_TEXT_OUT=bench/traces/gptoss120b-prose.generated.txt \
nice -n 19 ionice -c3 llama-eval-callback \
  -m /zfs/swift/models/gpt-oss-120b-MXFP4.gguf -ngl 0 \
  -f bench/traces/prompts/gptoss120b-prose.txt -n 512 \
  -t 6 -c 4096 -b 2048 -ub 512 --temp 0.7 --top-k 20 --top-p 0.8 --seed 20260906
```

`llama-eval-callback` needs `source /opt/intel/oneapi/setvars.sh` on this host or it fails to find
`libsvml.so`.

## Measured results

See **`bench/policy-sweep.md`** — nine policies plus Belady, nine capacities, on these traces and
the Qwen ones. Headline: at the engine's 600 slots, LRU reaches 54.6–72.8% against Belady's
75.0–85.6%, and no online policy tested closes more than a quarter of that gap.
