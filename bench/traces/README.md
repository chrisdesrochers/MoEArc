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
