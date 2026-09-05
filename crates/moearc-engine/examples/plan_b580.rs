//! What MoEArc would decide for Qwen3.6-35B-A3B on an Arc B580.
//!
//! Run with: `cargo run -p moearc-engine --example plan_b580`
//!
//! # Provenance of every number below
//!
//! The model geometry is **measured**, read out of the real 20.6 GiB GGUF by `moearc-model`
//! and cross-checked against three independent sources (llama.cpp's `llama-gguf`, its
//! `print_info` output, and a standalone Python reader). It is not estimated.
//!
//! Device memory is **measured too**, by `moearc-device` on the real B580 via Level Zero:
//! 12,168,933,376 bytes allocatable (11.33 GiB) — not the nominal 12 GiB, and not a guess.
//!
//! 🔴 Two caveats on that figure. Core Level Zero reports 12,168,933,376 B allocatable while
//! sysman reports 12,567,810,048 B free; they are different questions and must not be
//! conflated, so the smaller allocatable figure is used here. And it assumes the card is
//! otherwise idle — anything else holding VRAM reduces it.
//!
//! 🔴 **The routing trace is synthetic.** The geometry it runs against is real, but which
//! experts get chosen is generated, so hit rates describe a plausible shape rather than this
//! model's actual behaviour. Capturing a real trace is the next task.

use moearc_engine::memory::{plan, Context, DeviceMemory, Headroom, ModelFootprint, Policy};
use moearc_engine::residency::{simulate, synthetic_trace, Policy as CachePolicy};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn main() {
    // ---- Measured from the GGUF -------------------------------------------------------
    // Qwen3.6-35B-A3B-UD-Q4_K_M: 40 blocks, 256 experts per block, 8 active per block.
    const BLOCKS: u64 = 40;
    const EXPERTS_PER_BLOCK: u64 = 256;
    const ACTIVE_PER_BLOCK: u64 = 8;

    // A residency slot holds ONE expert of ONE block, so the counts the planner wants are
    // per-slot, not per-block. Conflating the two is a factor-of-40 error and an easy one to
    // make: the model is described as having "256 experts", but it has 10,240 slots.
    let total_slots = BLOCKS * EXPERTS_PER_BLOCK;
    let active_slots = BLOCKS * ACTIVE_PER_BLOCK;

    // Max across blocks, not mean. This is an Unsloth dynamic quant: ffn_down_exps is Q5_K in
    // 37 blocks and Q6_K in 3, a 7.3% spread. A slot must hold any expert, so sizing to the
    // mean would fit more slots and then overrun VRAM the first time a heavy expert routed in.
    const PER_EXPERT_BYTES: u64 = 2_039_808;

    // 🔴 Provisional: awaiting expert_weights_bytes from moearc-model. Taking the total weight
    // figure minus an upper-bound estimate of expert bytes understates dense weights, which
    // makes the plan optimistic in the same direction as the free-memory assumption above.
    const TOTAL_WEIGHTS: u64 = 22_123_538_944;
    let expert_bytes_upper = total_slots * PER_EXPERT_BYTES;
    let dense_weights = TOTAL_WEIGHTS.saturating_sub(expert_bytes_upper);

    // Only 10 of 40 blocks carry a KV cache: qwen35moe is a hybrid, full_attention_interval=4,
    // and the other 30 blocks are recurrent. Using block_count here would overstate KV by 4x.
    const KV_BYTES_PER_TOKEN: u64 = 20_480;

    let model = ModelFootprint {
        dense_weights_bytes: dense_weights,
        per_expert_bytes: PER_EXPERT_BYTES,
        total_experts: total_slots as u32,
        active_experts: active_slots as u32,
        kv_bytes_per_token: KV_BYTES_PER_TOKEN,
    };

    println!("Qwen3.6-35B-A3B-UD-Q4_K_M  (geometry measured from the GGUF)");
    println!("  {BLOCKS} blocks x {EXPERTS_PER_BLOCK} experts = {total_slots} residency slots");
    println!("  {ACTIVE_PER_BLOCK} active per block = {active_slots} slots touched per token");
    println!("  {:.2} MiB per slot (max across blocks; mixed quant)", PER_EXPERT_BYTES as f64 / MIB as f64);
    println!("  all experts resident would need {:.1} GiB", expert_bytes_upper as f64 / GIB as f64);
    println!("  dense weights ~{:.2} GiB, KV {KV_BYTES_PER_TOKEN} B/token (10 of 40 blocks)",
             dense_weights as f64 / GIB as f64);
    println!();

    // ---- Plan against several cards ---------------------------------------------------
    // The B580 figure is MEASURED on the real card by moearc-device (Level Zero
    // zeDeviceGetMemoryProperties). The others are nominal capacities for comparison and are
    // marked as such, because we have not run on them.
    const B580_MEASURED_FREE: u64 = 12_168_933_376;
    let cards: [(&str, u64); 3] = [
        ("Arc B580 - MEASURED 11.33 GiB allocatable", B580_MEASURED_FREE),
        ("Arc B770 24 GiB (nominal, unmeasured)", 24 * GIB),
        ("2x B580 (nominal, unmeasured)", 2 * B580_MEASURED_FREE),
    ];

    for (name, total) in cards {
        println!("== {name} ==");
        let device = DeviceMemory { total_bytes: total, free_bytes: total };
        let policy = Policy { headroom: Headroom::PROVISIONAL, ..Policy::default() };

        match plan(device, &model, &policy, Context::Largest) {
            Ok(a) => {
                println!(
                    "  {} of {} slots resident ({:.1}%), {} tokens of context",
                    a.resident_experts,
                    total_slots,
                    100.0 * a.resident_experts as f64 / total_slots as f64,
                    a.context_tokens
                );
                for r in &a.rationale {
                    println!("    - {r}");
                }
                // What that residency is worth, on a synthetic trace of the real geometry.
                report_residency(a.resident_experts, active_slots as u16);
            }
            Err(e) => println!("  cannot serve this model: {e}"),
        }
        println!();
    }
}

fn report_residency(capacity: u32, _active_slots: u16) {
    // Real geometry (40 blocks, 256 experts, 8 active), synthetic routing.
    let trace = synthetic_trace(200, 40, 256, 8, 0.7, 20260904);
    let baseline = trace.widest_static_split(capacity);
    let stat = simulate(&trace, capacity, baseline, 2_039_808);
    let lru = simulate(&trace, capacity, CachePolicy::Lru, 2_039_808);
    let opt = simulate(&trace, capacity, CachePolicy::Optimal, 2_039_808);

    match (stat, lru, opt) {
        (Ok(s), Ok(l), Ok(o)) => {
            println!(
                "    residency (SYNTHETIC routing, real geometry): static {:.1}%  lru {:.1}%  optimal {:.1}%",
                100.0 * s.hit_rate(),
                100.0 * l.hit_rate(),
                100.0 * o.hit_rate()
            );
            // A hit rate only means something once it is time.
            let secs = l.transfer_seconds(13.4e9);
            println!(
                "    lru moves {:.1} GiB over 200 tokens = {:.2} s at the measured 13.4 GB/s link",
                l.bytes_fetched as f64 / (1u64 << 30) as f64,
                secs
            );
        }
        _ => println!("    residency: capacity below one step's demand"),
    }
}
