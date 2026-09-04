# Calibration — measure the machine, don't hard-code a card

**Decision (2026-09-04, owner):** MoEArc measures its tuning constants on whatever GPU it runs
on. It does **not** ship a table of Arc-tuned numbers.

## Why

FreeToken's constants are fitted to NVIDIA hardware — several explicitly to an H100. Porting
them as literals would produce a project that works on exactly one Arc card and silently
misbehaves on the rest. MoEArc targets Arc A-series, B570/B580, and Arc Pro B50/B60/B65/B70,
which differ in VRAM, bus width, Xe-core count and host-link bandwidth. A single constant
cannot be right across that range.

The failure mode matters more than the inaccuracy: a mis-tuned constant does not crash. It
produces a plausible number that is quietly 20% low, which is indistinguishable from "this
hardware is just slower" — and that is precisely the class of bug this project exists to avoid
making claims about.

## We are not starting from zero

FreeToken already ships the right shape: `python/freetoken/moe/benchbw.py` (`ft bench bw`)
measures, per expert format:

- `measure_cpu_moe_bw` — CPU MoE GEMV bandwidth
- `measure_pcie_gather_bw` — PCIe expert-gather bandwidth
- `measure_overlap_bw` — **both concurrently, under contention** (2.0 s from a shared barrier)

and caches results to `$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json`, read at startup by
`load_backend_recommendation` and `load_hybrid_fetch_fraction`.

📌 **This reorders the port.** `benchbw` is written against `torch.cuda.synchronize` and pinned
banks, so it needs porting — and until it produces an Arc profile, `load_backend_recommendation`
returns `None` and the hybrid path never activates. The calibrator is therefore both the
prerequisite for mechanism 1 *and* the deliverable of this decision. It moves from last to
foundational.

## What must be measured, not assumed

Derived from the audit of FreeToken's hard-coded constants.

| Constant | Current value | Provenance | Action |
|---|---|---|---|
| `benchbw.recommend` threshold | `2.0` (CPU BW > 2× PCIe) | NVIDIA-measured crossover | **Measure** — derive the crossover from the two measured bandwidths |
| `hybrid_fetch_fraction` (q*) | `min(1, pcie/(pcie+cpu))` | already computed from measurements | ✅ keep — it is already calibration-driven |
| fallback `hybrid_max_fetch` | `1` expert/layer/step | arbitrary safe default | **Measure**, or keep as the no-profile fallback only |
| `memory_ratio` | `0.9` | NVIDIA allocator/activation headroom | **Measure** free VRAM headroom after capture |
| `kv_reserve_tokens` | `8192` | "small by design (MoE-priority)" | Policy, not hardware — keep, expose as a flag |
| flashlib `STREAMING_THRESHOLD` | `40_000` slots | **H100 register-spill boundary** | **Measure** — Xe register file differs |
| flashlib `_SEQ_US`/`_INSERT_US` tables | µs latencies | tagged `device="H100"` | **Measure** |
| `fast_index_copy` `blocks_per_bank` | 8 (PCIe) / 64 (D2D) | "~31 GB/s knee", "~4096 threads/bank" on H100 | **Sweep** on the target card |
| `num_warps` in hybrid kernel | `8 if block_c >= 2048 else 4` | SM occupancy tuning | **Sweep** — Xe sub-group sizing differs |
| `_FLAG_SLOTS_PER_LAYER` | `16` | CPU-executor handshake capacity | Review with the executor rewrite |
| `_SMALL_BANK_FEAT_BYTES` | `256 * 1024` | 🔴 workaround for a **CUDA 13.0 driver bug on H100** | **DELETE** — meaningless on Arc. Do not port. |
| `MARLIN_MAX_CACHE_SIZE` | `992` | vLLM `moe_align_block_size` artefact | **DELETE** — no marlin path on Arc |

## Design constraints

1. **Cache per device, keyed on something stable.** FreeToken keys on GPU UUID via NVML. Arc has
   no NVML; use Level Zero sysman or `xpu-smi`. ⬜ Open: what identifier is stable across driver
   updates on Arc.
2. **A missing profile must degrade safely, not silently.** FreeToken's current behaviour — no
   profile means hybrid never activates — is *correct* but invisible. MoEArc should say so out
   loud at startup rather than quietly running the slow path.
3. **Calibration output is a benchmark artifact.** It states the box, the driver, the card and
   the date. Same rule as `bench/README.md`: a number without its provenance is not a number.
4. **Re-measure on driver change.** A profile taken under a different compute-runtime version is
   suspect; record the version and warn on mismatch (FreeToken already warns on GPU-name change).

## Gate

A calibration run on the reference B580 that produces a profile, plus at least one constant whose
measured value **differs materially from the NVIDIA default** — that is the evidence the decision
was worth making. If every measured value matches the hard-coded one, say so and simplify.

---

## Worked example — why we measure: the XMX flag is wrong

The first thing calibration caught, before a line of engine code was written.

`torch.xpu` reports the Arc B580 as having **no** matrix engine:

```python
>>> torch.xpu.get_device_properties(0).has_subgroup_matrix_multiply_accumulate
False
```

torch's own docstring says this flag means *"whether DPAS (Dot Product Accumulate Systolic) is
supported"*. Taken at face value, it says: do not write an XMX path, this card cannot use one.

**It is wrong.** Measured on the reference B580, 4096×4096 matmul:

| dtype | TFLOP/s |
|---|---|
| float32 | 14.3 |
| **float16** | **109.2** |
| bfloat16 | 100.3 |

The B580's rated bf16 throughput is ~115 TFLOPS **with** XMX and roughly 25–30 without.
**109 TFLOP/s at 7.6× the fp32 rate is only reachable through the systolic array.** XMX is
present and fully engaged.

Two independent sources corroborate, and one of them provides a control:

- `sycl-ls --verbose` lists **`ext_intel_matrix`** among the B580's aspects — while the Arrow
  Lake iGPU in the same machine, which genuinely has no XMX, **does not**. The aspect query
  discriminates correctly; torch's flag does not.
- Vulkan independently reports `matrix cores: KHR_coopmat` for the same card.

### Why this is the founding example

Nothing crashed. Nothing logged a warning. A single boolean, read in good faith, would have
removed the card's most important compute unit from consideration and every subsequent
performance number would have been quietly capped — and attributed to "Arc is just slower."

That is exactly the failure mode this document exists to prevent, and it is why the rule is
**measure the machine, don't trust its self-description.** The same scepticism applies to every
constant in the table above.

⬜ Reported upstream — see the project issue tracker for the filing.
⬜ Unresolved: `has_bfloat16_conversions` and `has_subgroup_2d_block_io` also report `False`.
Whether those are the same defect or genuinely absent on Battlemage is **not yet determined**;
do not assume either way.
