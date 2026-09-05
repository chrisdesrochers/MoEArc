//! Report what a captured expert-routing trace looks like, and what residency it implies.
//!
//! ```text
//! cargo run --release -p moearc-engine --example trace_report -- <trace.ndjson> [capacity]
//! ```
//!
//! Everything printed here is computed from the file named on the command line. Nothing is
//! generated, assumed, or carried over from the synthetic study in `residency.rs`.

use moearc_engine::residency::{ExpertRef, Policy, Trace, simulate};
use std::collections::HashMap;

/// Slots a 12 GiB Arc B580 can hold, from the measurement recorded in `bench/README.md`.
const DEFAULT_CAPACITY: u32 = 3976;

/// One expert at Q4_K_M in this model, 1.95 MiB. Affects `bytes_fetched` only, never hit rate.
const PER_EXPERT_BYTES: u64 = 2_044_723;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: trace_report <trace.ndjson> [capacity]");
        std::process::exit(2);
    };
    let capacity: u32 = args.next().map_or(DEFAULT_CAPACITY, |a| a.parse().expect("capacity"));

    let loaded = match Trace::from_ndjson_file(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };
    let t = &loaded.trace;

    println!("# {path}");
    println!("\n## provenance\n\n```json\n{}\n```", loaded.header);

    // ---- shape ------------------------------------------------------------
    let mut counts: HashMap<ExpertRef, u64> = HashMap::new();
    for e in t.steps.iter().flatten() {
        *counts.entry(*e).or_insert(0) += 1;
    }
    let mut per_layer: HashMap<u16, HashMap<u16, u64>> = HashMap::new();
    for e in t.steps.iter().flatten() {
        *per_layer.entry(e.layer).or_default().entry(e.expert).or_insert(0) += 1;
    }
    let n_layers = per_layer.len();
    let demands = t.demands();

    println!("\n## shape\n");
    println!("| | |\n|---|---|");
    println!("| steps | {} |", t.steps.len());
    println!("| activations (demands) | {demands} |");
    println!("| activations per step | {} |", t.peak_step_demand());
    println!("| layers routed | {n_layers} |");
    println!("| working set (distinct layer,expert) | {} |", t.working_set());
    println!("| theoretical max working set | {} |", n_layers * 256);
    println!(
        "| working set as % of all experts | {:.1}% |",
        100.0 * t.working_set() as f64 / (n_layers * 256) as f64
    );

    // ---- skew -------------------------------------------------------------
    let mut sorted: Vec<u64> = counts.values().copied().collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let share = |frac: f64| -> f64 {
        let k = ((sorted.len() as f64) * frac).ceil() as usize;
        let taken: u64 = sorted.iter().take(k).sum();
        100.0 * taken as f64 / demands as f64
    };
    println!("\n## skew (share of all activations taken by the busiest experts)\n");
    println!("| top _n_ of experts **touched** | share of activations |\n|---|---|");
    for f in [0.01, 0.05, 0.10, 0.25, 0.50] {
        println!("| top {:.0}% | {:.1}% |", f * 100.0, share(f));
    }
    println!(
        "\nUniform routing over the experts actually touched would give the top 10% exactly \
         10.0%. Perfect concentration would give 100%."
    );

    // ---- temporal locality ------------------------------------------------
    // The property LRU lives on: does step N re-use what step N-1 used?
    let mut overlap = 0u64;
    let mut overlap_denom = 0u64;
    for w in t.steps.windows(2) {
        let mut prev: Vec<ExpertRef> = w[0].clone();
        prev.sort_unstable();
        prev.dedup();
        for e in &w[1] {
            overlap_denom += 1;
            if prev.binary_search(e).is_ok() {
                overlap += 1;
            }
        }
    }
    println!("\n## temporal locality\n");
    println!(
        "| activations also used by the immediately preceding step | {:.1}% |\n|---|---|",
        100.0 * overlap as f64 / overlap_denom.max(1) as f64
    );

    // ---- per-layer --------------------------------------------------------
    // The hybrid has attention on every 4th block (2,6,10,... zero-based: il where
    // (il+1) % 4 == 0). Routing is reported for both groups separately because a difference
    // there would be a real architectural finding, not noise.
    let mut layers: Vec<u16> = per_layer.keys().copied().collect();
    layers.sort_unstable();
    let mut rows: Vec<(u16, usize, f64)> = Vec::new();
    for l in &layers {
        let m = &per_layer[l];
        let mut c: Vec<u64> = m.values().copied().collect();
        c.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = c.iter().sum();
        let k = ((c.len() as f64) * 0.10).ceil() as usize;
        let top10: u64 = c.iter().take(k).sum();
        rows.push((*l, m.len(), 100.0 * top10 as f64 / total as f64));
    }
    let widest = rows.iter().max_by_key(|r| r.1).unwrap();
    let narrowest = rows.iter().min_by_key(|r| r.1).unwrap();
    println!("\n## routing by block\n");
    println!("| | distinct experts used | top-10% share |\n|---|---|---|");
    println!("| widest block (blk {}) | {} | {:.1}% |", widest.0, widest.1, widest.2);
    println!("| narrowest block (blk {}) | {} | {:.1}% |", narrowest.0, narrowest.1, narrowest.2);
    let mean_distinct = rows.iter().map(|r| r.1 as f64).sum::<f64>() / rows.len() as f64;
    println!("| mean across blocks | {mean_distinct:.1} | |");

    let attn: Vec<&(u16, usize, f64)> = rows.iter().filter(|r| (r.0 + 1) % 4 == 0).collect();
    let rec: Vec<&(u16, usize, f64)> = rows.iter().filter(|r| (r.0 + 1) % 4 != 0).collect();
    if !attn.is_empty() && !rec.is_empty() {
        let m =
            |v: &[&(u16, usize, f64)]| v.iter().map(|r| r.1 as f64).sum::<f64>() / v.len() as f64;
        println!("| mean, attention blocks ({}) | {:.1} | |", attn.len(), m(&attn));
        println!("| mean, recurrent blocks ({}) | {:.1} | |", rec.len(), m(&rec));
    }
    println!("\nPer-block detail:\n\n| blk | distinct | top-10% share |\n|---|---|---|");
    for (l, d, s) in &rows {
        println!("| {l} | {d} | {s:.1}% |");
    }

    // ---- residency --------------------------------------------------------
    println!("\n## residency at {capacity} slots\n");
    let baseline = t.widest_static_split(capacity);
    if let Policy::StaticSplit { resident_layers } = baseline {
        println!(
            "Widest static split that fits the same budget: the first **{resident_layers}** \
             blocks resident ({} expert slots used of {capacity}).\n",
            t.experts_in_layers_below(resident_layers)
        );
    }
    println!("| policy | hit rate | hits | misses | compulsory | GiB fetched |");
    println!("|---|---|---|---|---|---|");
    for p in [baseline, Policy::Lru, Policy::Lfu, Policy::Optimal] {
        match simulate(t, capacity, p, PER_EXPERT_BYTES) {
            Ok(r) => println!(
                "| {} | **{:.1}%** | {} | {} | {} | {:.2} |",
                r.policy,
                100.0 * r.hit_rate(),
                r.hits,
                r.misses,
                r.compulsory_misses,
                r.bytes_fetched as f64 / (1u64 << 30) as f64
            ),
            Err(e) => println!("| {} | — | | | | {e} |", p.name()),
        }
    }

    // ---- capacity sweep ---------------------------------------------------
    println!("\n## capacity sweep\n");
    println!("| slots | static | lru | optimal |\n|---|---|---|---|");
    for cap in [512u32, 1024, 2048, 3976, 5120, 7680, 10240] {
        if (cap as usize) < t.peak_step_demand() {
            continue;
        }
        let s = simulate(t, cap, t.widest_static_split(cap), PER_EXPERT_BYTES);
        let l = simulate(t, cap, Policy::Lru, PER_EXPERT_BYTES);
        let o = simulate(t, cap, Policy::Optimal, PER_EXPERT_BYTES);
        let pct = |r: &Result<moearc_engine::residency::Residency, _>| match r {
            Ok(r) => format!("{:.1}%", 100.0 * r.hit_rate()),
            Err(_) => "—".to_string(),
        };
        println!("| {cap} | {} | {} | {} |", pct(&s), pct(&l), pct(&o));
    }
}
