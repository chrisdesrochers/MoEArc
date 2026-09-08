//! Attributing the cost of depth: attention, or a churning expert cache?
//!
//! ```text
//! MOEARC_PROFILE=1 MOEARC_PROFILE_EVENTS=1 \
//! cargo run --release -p moearc-engine --features gpu --example ctx_attrib -- \
//!     <model.gguf> <depth,...> <n-predict> <residency> <host-policy> <ids-file>
//! ```
//!
//! # The question this exists to answer, and why `ctx_curve` cannot
//!
//! Throughput falling as context grows has two candidate causes that a `tok/s` column cannot
//! tell apart. Attention over more keys costs more. And a deeper prompt names more distinct
//! experts, which leaves the resident pool holding a more diluted working set, so the decode
//! steps that follow miss more and stage more bytes. **The first is a cost of context; the
//! second is a cost of the memory strategy** — and they call for opposite fixes, so guessing
//! between them is worthless.
//!
//! `ctx_curve` reports `profile` phases, which are **host** wall time around a submit. On an
//! asynchronous queue that is close to the submit's own cost, and the device work it stands for
//! lands wherever the next synchronisation happens. 🔴 Read naively that says attention is free
//! and `moe.stage` is everything, which is an artefact of where the queue was drained, not a
//! finding. This example reads the **SYCL events themselves**, so every kernel is charged the
//! device time it actually took.
//!
//! # Differencing, and why it is exact rather than approximate
//!
//! Neither the event counters nor the cache counters can be zeroed part-way through a
//! generation: both are reached through the session, and `generate_with` holds its lock for the
//! whole call. So each depth is run **twice from the same warm pool**, differing in one thing
//! only:
//!
//! - `prefill` — the prompt, then a single token. `depth` decode-path steps, none of them a
//!   generated-token step.
//! - `full` — the same prompt, then `n` tokens. The same `depth` steps, plus `n - 1` more.
//!
//! The counters are cumulative, so `full - prefill` is exactly those `n - 1` steps. This is not
//! a model of the decode phase; it is the decode phase, measured by subtraction. ⚠️ The one
//! assumption is that the shared prefix costs the same in both runs, which is why both are
//! preceded by a full warm-up pass and neither is the first thing the pool sees.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::host_experts::HostPolicy;
use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions};

/// Kernel-wide I/O counters, read around a generation so the window can be differenced.
///
/// Both are cumulative and machine-wide, so only a bracketed difference means anything and any
/// other tenant's I/O lands in them too — which is why the load average is read as well.
/// PROTOCOL §4: a run that faulted gigabytes off NVMe measured the storage, not the engine.
#[derive(Clone, Copy, Default)]
struct Io {
    /// 512-byte sectors read from the device backing the model's pool.
    disk_sectors: u64,
    arc_hits: u64,
    arc_misses: u64,
}

/// The partition backing the pool the model lives on, e.g. `nvme0n1p4`, from
/// `MOEARC_BENCH_DISK`. Unset omits the disk column rather than guessing: `/proc/diskstats` is
/// machine-wide, so summing every device would silently count an unrelated tenant's disk.
fn disk_dev() -> Option<String> {
    std::env::var("MOEARC_BENCH_DISK").ok().filter(|s| !s.is_empty())
}

impl Io {
    fn now(dev: Option<&str>) -> Self {
        let mut io = Self::default();
        if let Some(dev) = dev {
            if let Ok(text) = std::fs::read_to_string("/proc/diskstats") {
                for line in text.lines() {
                    let f: Vec<&str> = line.split_whitespace().collect();
                    // major minor name reads_completed reads_merged sectors_read ...
                    if f.len() > 5 && f[2] == dev {
                        io.disk_sectors = f[5].parse().unwrap_or(0);
                        break;
                    }
                }
            }
        }
        if let Ok(text) = std::fs::read_to_string("/proc/spl/kstat/zfs/arcstats") {
            for line in text.lines() {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() >= 3 {
                    match f[0] {
                        "hits" => io.arc_hits = f[2].parse().unwrap_or(0),
                        "misses" => io.arc_misses = f[2].parse().unwrap_or(0),
                        _ => {}
                    }
                }
            }
        }
        io
    }

    fn since(self, earlier: Self) -> Self {
        Self {
            disk_sectors: self.disk_sectors.saturating_sub(earlier.disk_sectors),
            arc_hits: self.arc_hits.saturating_sub(earlier.arc_hits),
            arc_misses: self.arc_misses.saturating_sub(earlier.arc_misses),
        }
    }

    fn disk_mib(self) -> f64 {
        (self.disk_sectors as f64) * 512.0 / (1024.0 * 1024.0)
    }
}

/// One named field out of `arcstats`, or 0 where ZFS is not loaded.
fn read_kstat(name: &str) -> u64 {
    std::fs::read_to_string("/proc/spl/kstat/zfs/arcstats")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.split_whitespace().next() == Some(name))
                .and_then(|l| l.split_whitespace().nth(2))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn load_avg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(f64::NAN)
}

fn read_ids(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut ids = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        for field in line.split([',', ' ', '\t']).filter(|f| !f.is_empty()) {
            ids.push(field.parse::<u32>().map_err(|_| format!("`{field}` is not a token id"))?);
        }
    }
    if ids.is_empty() {
        return Err(format!("{} holds no token ids", path.display()));
    }
    Ok(ids)
}

/// Cumulative counters after one generation.
struct Snapshot {
    /// Device nanoseconds per kernel key, from the SYCL events.
    device: BTreeMap<String, u64>,
    demands: u64,
    hits: u64,
    bytes_staged: u64,
    host_experts: u64,
    /// Wall time of the whole generation. Differenced like every other counter here, so
    /// `full - prefill` is the wall clock of exactly the `n - 1` extra decode steps — the
    /// denominator a share has to be taken against, measured in the same invocation as its
    /// numerator rather than carried in from another run.
    wall_ms: f64,
    io: Io,
}

fn snapshot(
    session: &Session,
    prompt: &[u32],
    n: usize,
    dev: Option<&str>,
) -> Result<Snapshot, String> {
    session.reset_event_profile().map_err(|e| e.to_string())?;
    session.reset_cache_stats().map_err(|e| e.to_string())?;
    let stop = StopConditions { max_tokens: n, stop_tokens: Vec::new() };
    let io0 = Io::now(dev);
    let t0 = Instant::now();
    session.generate(prompt, &stop, &mut |_| true).map_err(|e| e.to_string())?;
    // 🔴 Read the clock before the counters. `event_profile` drains outstanding SYCL events,
    // which blocks until they complete; taking the wall time after it would charge the drain to
    // the generation and inflate the denominator the shares below are taken against.
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let io = Io::now(dev).since(io0);
    let ev = session.event_profile().map_err(|e| e.to_string())?;
    let r = session.residency().map_err(|e| e.to_string())?;
    Ok(Snapshot {
        device: ev.into_iter().map(|(k, ns, _)| (k, ns)).collect(),
        demands: r.stats.demands,
        hits: r.stats.hits,
        bytes_staged: r.bytes_staged,
        host_experts: r.host.experts,
        wall_ms,
        io,
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "usage: ctx_attrib <model.gguf> <depth,...> <n-predict> <residency> <host-policy> \
             <ids-file>"
        );
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(&args[1]);
    let mut depths: Vec<usize> = Vec::new();
    for d in args[2].split(',').filter(|s| !s.is_empty()) {
        match d.parse() {
            Ok(v) => depths.push(v),
            Err(_) => {
                eprintln!("`{d}` is not a depth");
                return ExitCode::FAILURE;
            }
        }
    }
    let Ok(n_predict) = args[3].parse::<usize>() else {
        eprintln!("n-predict must be a number");
        return ExitCode::FAILURE;
    };
    if n_predict < 3 {
        eprintln!("n-predict must leave at least two decode steps to difference");
        return ExitCode::FAILURE;
    }
    let residency: Residency = match args[4].parse() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let host: HostPolicy = match args[5].parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let pool = match read_ids(Path::new(&args[6])) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let dev = disk_dev();
    println!("residency `{}`, host `{}`, {n_predict} generated tokens per depth", args[4], args[5]);
    println!(
        "disk device `{}` (set MOEARC_BENCH_DISK to attribute reads)",
        dev.as_deref().unwrap_or("unset -- disk column omitted")
    );
    let model_bytes = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
    let arc_max = read_kstat("c_max");
    if model_bytes > 0 && arc_max > 0 {
        println!(
            "model {:.1} GiB against an ARC ceiling of {:.1} GiB ({:.2}x): it CANNOT be fully \
             cached here, which is permanent and is not the same condition as a cold hot set",
            model_bytes as f64 / (1u64 << 30) as f64,
            arc_max as f64 / (1u64 << 30) as f64,
            model_bytes as f64 / arc_max as f64,
        );
    }
    println!("device time and cache counters, DECODE STEPS ONLY, by differencing\n");

    for &depth in &depths {
        if depth > pool.len() {
            println!("depth {depth}: only {} ids available\n", pool.len());
            continue;
        }
        let prompt = &pool[..depth];
        let n_ctx = depth + n_predict + 1;
        let opts = SessionOptions { n_ctx: Some(n_ctx), residency, host };
        let session = match Session::load_with(&model, opts) {
            Ok(s) => s,
            Err(e) => {
                println!("depth {depth}: load failed: {e}\n");
                continue;
            }
        };
        let stop = StopConditions { max_tokens: n_predict, stop_tokens: Vec::new() };
        // Warm the pool first, so neither differenced run is the one that pays for filling it.
        if let Err(e) = session.generate(prompt, &stop, &mut |_| true) {
            println!("depth {depth}: warm-up failed: {e}\n");
            continue;
        }
        let load = load_avg();
        let prefill = match snapshot(&session, prompt, 1, dev.as_deref()) {
            Ok(s) => s,
            Err(e) => {
                println!("depth {depth}: prefill pass failed: {e}\n");
                continue;
            }
        };
        let full = match snapshot(&session, prompt, n_predict, dev.as_deref()) {
            Ok(s) => s,
            Err(e) => {
                println!("depth {depth}: full pass failed: {e}\n");
                continue;
            }
        };
        let steps = (n_predict - 1) as f64;

        let d_demands = full.demands.saturating_sub(prefill.demands);
        let d_hits = full.hits.saturating_sub(prefill.hits);
        let d_staged = full.bytes_staged.saturating_sub(prefill.bytes_staged);
        let d_host = full.host_experts.saturating_sub(prefill.host_experts);
        let hit_rate = if d_demands == 0 { 0.0 } else { d_hits as f64 / d_demands as f64 };

        println!("## depth {depth} (load {load:.2} before the pair, {n_ctx} n_ctx)");
        println!(
            "\ndecode-only cache: **{:.1}% hit** over {d_demands} demands, {:.1} MiB staged per \
             step, {:.1} experts/step to the CPU",
            100.0 * hit_rate,
            d_staged as f64 / (1024.0 * 1024.0) / steps,
            d_host as f64 / steps,
        );
        println!("\n| kernel | ms/step (decode) | ms/step (prefill) |");
        println!("|---|---|---|");
        let mut keys: Vec<&String> = full.device.keys().collect();
        keys.sort_by_key(|k| {
            let f = full.device.get(*k).copied().unwrap_or(0);
            let p = prefill.device.get(*k).copied().unwrap_or(0);
            std::cmp::Reverse(f.saturating_sub(p))
        });
        let mut decode_total = 0.0;
        for k in keys {
            let f = full.device.get(k).copied().unwrap_or(0);
            let p = prefill.device.get(k).copied().unwrap_or(0);
            let dec = f.saturating_sub(p) as f64 / 1e6 / steps;
            let pre = p as f64 / 1e6 / depth.max(1) as f64;
            decode_total += dec;
            println!("| `{k}` | {dec:.3} | {pre:.3} |");
        }
        println!("| **tracked device busy** | **{decode_total:.3}** | |");

        // The share section 7.6 had to extrapolate. Numerator: `attn_decode`'s own device time,
        // now that the kernel is tracked. Denominator: the wall clock of the same differenced
        // steps. Both come from this one pair of runs, on an asynchronous queue, so nothing
        // here is inflated by `MOEARC_SYNC_EACH`.
        let attn: f64 = full
            .device
            .keys()
            .filter(|k| k.starts_with("attn_decode"))
            .map(|k| {
                let f = full.device.get(k).copied().unwrap_or(0);
                let p = prefill.device.get(k).copied().unwrap_or(0);
                f.saturating_sub(p) as f64 / 1e6 / steps
            })
            .sum();
        let wall = (full.wall_ms - prefill.wall_ms) / steps;
        let pct = |x: f64| if wall > 0.0 { 100.0 * x / wall } else { f64::NAN };
        println!(
            "\ndecode-only wall: **{wall:.2} ms/step** ({:.2} tok/s). **`attn_decode` is \
             {:.1}% of the step**; every tracked device kernel together is {:.1}%; untracked \
             and non-overlapped host time is the remaining {:.1}%.",
            if wall > 0.0 { 1000.0 / wall } else { f64::NAN },
            pct(attn),
            pct(decode_total),
            pct(wall - decode_total),
        );
        let dio = full.io.since(prefill.io);
        let lookups = dio.arc_hits + dio.arc_misses;
        let miss = if lookups > 0 {
            format!(
                "{:.1}% miss over {lookups} lookups",
                100.0 * dio.arc_misses as f64 / lookups as f64
            )
        } else {
            "no lookups".to_string()
        };
        match dev.as_deref() {
            Some(d) => println!(
                "decode-only I/O: **{:.0} MiB** read from `{d}` ({:.1} MiB/step); ARC {miss}",
                dio.disk_mib(),
                dio.disk_mib() / steps,
            ),
            None => println!("decode-only I/O: disk unattributed; ARC {miss}"),
        }

        // 🔴 The gate the load average cannot see.
        //
        // PROTOCOL §4 says to *report* disk reads. Reporting is not enough: a **cold cache
        // passes a quiet-box check and still produces a ten-fold wrong answer.** This cell was
        // measured on a box at load 1.46 — genuinely idle — and returned 1.36 tok/s at depth
        // 512 against an established 13.38, because it faulted 528 MiB **per decode step** off
        // NVMe. Nothing in the load gate could have known.
        //
        // ⚠️ Two conditions look identical in a disk column and are not the same:
        //
        // - **"this model cannot be fully cached"** — permanent here. 59 GiB of weights against
        //   a 16 GiB ARC ceiling. Refusing on it would make the tool useless for its own
        //   flagship model, so it is a *warning*.
        // - **"this model's hot set is cold right now"** — fixable, by running the cell once and
        //   discarding it. A run in this state did not measure the engine and must not be
        //   quoted.
        //
        // The threshold is one expert slot (12.6 MiB on gpt-oss) per step: below one fault per
        // step the disk cannot be the story, above ten it is.
        if dev.is_some() {
            let per_step = dio.disk_mib() / steps;
            if per_step > 10.0 {
                println!(
                    "\n🔴 **DISCARD THIS CELL — STORAGE-BOUND.** {per_step:.1} MiB/step faulted \
                     off disk during decode. It measured the storage, not the engine (PROTOCOL \
                     §4). Warm the cell by running it once and discarding it, then re-measure."
                );
            } else if per_step > 1.0 {
                println!(
                    "\n⚠️ Qualified: {per_step:.1} MiB/step of decode-time disk reads. Not zero, \
                     so some of this step is the filesystem."
                );
            } else {
                println!(
                    "\n✅ Decode-time disk reads are {per_step:.2} MiB/step: this cell is the engine."
                );
            }
        }
        println!(
            "\n⚠️ The prefill column divides by `depth` steps and is there to show a kernel's \
             cost *scaling*, not as a comparison of like with like: a prefill step at depth \
             {depth} attends over a cache that is filling, so it averages shallower positions \
             than the decode steps do.\n"
        );
    }
    ExitCode::SUCCESS
}
