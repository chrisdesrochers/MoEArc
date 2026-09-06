//! Does the headline survive a context anyone would actually use?
//!
//! ```text
//! MOEARC_PROFILE=1 cargo run --release -p moearc-engine --features gpu --example ctx_curve -- \
//!     <model.gguf> <depth,...> <n-predict> <residency> <host-policy> <ids-file>
//! ```
//!
//! # Why this is not `hybrid_sweep` with a bigger `n_ctx`
//!
//! 🔴 `n_ctx` is an **allocation**, not a depth. Raising it grows the KV cache and takes VRAM
//! away from the expert pool, but a six-token prompt still attends over six positions, so a
//! sweep over `n_ctx` alone measures the memory cost of context and none of its compute cost.
//! What a user actually meets is a *full* cache: this drives the depth by putting `depth` real
//! tokens in front of the generation, which is exactly what `llama-bench -d <depth>` does, so
//! the two engines are asked the same question.
//!
//! # What is timed, and what is deliberately not
//!
//! `generate` decodes the prompt through the same path as a generated token, so a naive
//! stopwatch over the whole call divides one number by `depth + n` steps and reports mostly
//! prefill. **Only the decode steps are timed here.** The sampling closure is called once per
//! generated token and the first call lands the instant prefill finished, so the marks it
//! records fence the decode phase exactly: `n` marks, `n - 1` decode steps between them.
//! Prefill is reported too, in its own column, because it is a real cost — it is just not the
//! one the headline is about.
//!
//! # Separating two explanations that look identical in a throughput number
//!
//! A longer run is slower for two reasons that a `tok/s` column cannot tell apart: attention
//! over more keys costs more (`attn.attend`), and a longer run touches more distinct experts,
//! which churns the cache and stages more bytes (`moe.stage`). **`n-predict` is held fixed
//! across depths on purpose**, so the number of generated tokens — and therefore the amount of
//! router churn generation itself causes — is constant while depth varies. The per-phase
//! columns then attribute what is left: `profile::reset()` is called at the same first mark, so
//! the phase totals cover the decode steps and nothing else.
//!
//! ⚠️ `profile::reset` is a free function over a process-global. That is load-bearing here:
//! `generate_with` holds the session lock while the closure runs, so calling anything on the
//! `Session` from inside it deadlocks. The device thread is idle at that moment, waiting for
//! the next command, so the counters are quiescent when they are cleared.
//!
//! # Cold and warm
//!
//! Both are reported. The first pass fills the expert pool; the second is the steady state a
//! served request meets. On a model five times the card they can converge, and where they do,
//! that is the finding rather than a warm-up artefact.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use moearc_engine::host_experts::HostPolicy;
use moearc_engine::moe::Residency;
use moearc_engine::profile;
use moearc_engine::session::{Session, SessionOptions, StopConditions, argmax};

/// The kernel's own 1-minute load average.
///
/// 🔴 Printed on every row because a throughput number is a measurement of the whole machine,
/// not of the engine. `bench/baselines/gpt-oss-120b.md` section 3.4 records a sweep of this
/// model that reported host offload **losing** 60-75%, which reproduced as a 48-140% *gain*
/// on a quiet box. The only difference was load average. A row that cannot say what else was
/// running is not evidence.
///
/// ⚠️ Read it *before* the run: `frac:` policies drive the host pool across all 20 threads, so
/// the load this process causes is part of the experiment and must not be confused with the
/// load it was contaminated by.
fn load_avg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(f64::NAN)
}

/// Disk reads and ZFS ARC hits, so "slow" can be told apart from "waiting on the disk".
///
/// 🔴 The model is 59.03 GiB against a `zfs_arc_max` of 16 GiB, so it **cannot** be cached.
/// That makes cache residency an uncontrolled variable that moves throughput by more than the
/// setting under test, and it makes two very different failures look identical in a `tok/s`
/// column: staging that is bound by the PCIe bus, and staging that is bound by a page fault to
/// NVMe. Pinned host memory fixes the first and does nothing at all for the second, so the two
/// must be separated before anyone optimises for either.
///
/// Both counters are cumulative and kernel-wide, so only differences over a bracketed window
/// mean anything, and any other tenant's I/O lands in them too — which is the same reason the
/// load average is read before every run.
#[derive(Clone, Copy, Default)]
struct Io {
    /// 512-byte sectors read from the device backing the model's pool.
    disk_sectors: u64,
    arc_hits: u64,
    arc_misses: u64,
}

/// The partition backing the pool the model lives on, e.g. `nvme0n1p4`.
///
/// Unset means the I/O table is omitted rather than guessed: summing every block device would
/// silently count an unrelated tenant's disk, and a number that cannot be attributed is worse
/// than no number.
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

    /// Share of ARC lookups in this window that missed. `None` when nothing was looked up.
    fn arc_miss_pct(self) -> Option<f64> {
        let total = self.arc_hits + self.arc_misses;
        (total > 0).then(|| 100.0 * self.arc_misses as f64 / total as f64)
    }
}

/// Token ids, one per line or comma-separated, `#` to end of line ignored.
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

struct Measured {
    /// Wall time for the whole prompt, through the same decode path.
    prefill_seconds: f64,
    /// Wall time for the decode steps alone.
    decode_seconds: f64,
    decode_steps: usize,
    /// Milliseconds per decode step, in order.
    step_ms: Vec<f64>,
    ids: Vec<u32>,
    hit_rate: f64,
    staged_mib: f64,
    /// Decode-only phase totals, in seconds over `decode_steps` steps.
    phases: Vec<profile::Phase>,
    /// Disk and ARC counters over the prompt, and over the decode steps alone.
    io_prefill: Io,
    io_decode: Io,
}

impl Measured {
    fn decode_toks(&self) -> f64 {
        self.decode_steps as f64 / self.decode_seconds
    }

    fn prefill_toks(&self, depth: usize) -> f64 {
        depth as f64 / self.prefill_seconds
    }

    /// One phase's milliseconds per decode step. Zero when the phase never ran, which is the
    /// honest reading: a phase absent from the report did no work.
    fn phase_ms(&self, name: &str) -> f64 {
        self.phases
            .iter()
            .find(|p| p.name == name)
            .map_or(0.0, |p| p.seconds * 1000.0 / self.decode_steps as f64)
    }

    /// Median step, and the first and last eighth of the run.
    ///
    /// 🔴 The drift between the two ends is the cache-churn signal. Attention cost at a fixed
    /// depth barely moves over `n` more keys; a working set outgrowing the pool shows up as a
    /// run that gets slower as it goes.
    fn shape(&self) -> (f64, f64, f64) {
        let mut sorted = self.step_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        let eighth = (self.step_ms.len() / 8).max(1);
        let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
        let head = mean(&self.step_ms[..eighth]);
        let tail = mean(&self.step_ms[self.step_ms.len() - eighth..]);
        (median, head, tail)
    }
}

fn measure(session: &Session, prompt: &[u32], n: usize) -> Result<Measured, String> {
    session.reset_cache_stats().map_err(|e| e.to_string())?;
    let stop = StopConditions { max_tokens: n, stop_tokens: Vec::new() };
    let mut ids = Vec::new();
    let mut marks: Vec<Instant> = Vec::with_capacity(n);
    let dev = disk_dev();
    let dev = dev.as_deref();
    let io_start = Io::now(dev);
    let mut io_first_mark = io_start;
    let started = Instant::now();
    {
        let mut sample = |logits: &[f32], _: &[u32]| {
            if marks.is_empty() {
                // Prefill has just finished and no decode step has started. See the module note
                // on why this is the free function and not a call on the session.
                profile::reset();
                // Same fence, for the same reason: everything before this is the prompt.
                io_first_mark = Io::now(dev);
            }
            marks.push(Instant::now());
            argmax(logits)
        };
        let mut on_token = |t: u32| {
            ids.push(t);
            true
        };
        session
            .generate_with(prompt, &stop, &mut sample, &mut on_token)
            .map_err(|e| e.to_string())?;
    }
    if marks.len() < 2 {
        return Err(format!("{} generated tokens is too few to time a decode step", marks.len()));
    }
    let prefill_seconds = marks[0].duration_since(started).as_secs_f64();
    let step_ms: Vec<f64> =
        marks.windows(2).map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0).collect();
    let decode_seconds = marks[marks.len() - 1].duration_since(marks[0]).as_secs_f64();
    let io_end = Io::now(dev);
    let r = session.residency().map_err(|e| e.to_string())?;
    Ok(Measured {
        io_prefill: io_first_mark.since(io_start),
        io_decode: io_end.since(io_first_mark),
        prefill_seconds,
        decode_seconds,
        decode_steps: step_ms.len(),
        step_ms,
        ids,
        hit_rate: r.stats.hit_rate(),
        staged_mib: r.bytes_staged as f64 / (1024.0 * 1024.0),
        phases: profile::report(),
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "usage: ctx_curve <model.gguf> <depth,...> <n-predict> <residency> <host-policy> \
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
    if !profile::enabled() {
        eprintln!("note: MOEARC_PROFILE=1 is not set, so the per-phase columns will be empty");
    }

    println!("residency `{}`, host `{}`, {n_predict} generated tokens per depth", args[4], args[5]);
    println!("{} token ids available\n", pool.len());
    println!(
        "| depth | n_ctx | KV MiB | slots | load | prefill tok/s | **cold tok/s** | \
         **warm tok/s** | hit (all) | staged MiB (all) | p50 ms | first 1/8 | last 1/8 | drift |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");

    let mut phase_rows: Vec<(usize, Measured, Measured)> = Vec::new();
    let mut failures = 0;
    for &depth in &depths {
        if depth > pool.len() {
            println!("| {depth} | | | | | | | | | | | | only {} ids available |", pool.len());
            failures += 1;
            continue;
        }
        let prompt = &pool[..depth];
        // Room for the prompt, the generation, and nothing speculative: the cache and the
        // expert pool compete for the same bytes, so an over-allocated context is throughput
        // taken away from residency.
        let n_ctx = depth + n_predict + 1;
        let opts = SessionOptions { n_ctx: Some(n_ctx), residency, host };
        let session = match Session::load_with(&model, opts) {
            Ok(s) => s,
            Err(e) => {
                println!("| {depth} | {n_ctx} | | | | | | | | | | | | load: {e} |");
                failures += 1;
                continue;
            }
        };
        let info = session.info();
        let kv_mib = info.kv.bytes as f64 / (1024.0 * 1024.0);
        let slots = info.residency.resident_slots;
        let load = load_avg();
        if let Err(e) = session.clear_residency() {
            eprintln!("clear failed: {e}");
            return ExitCode::FAILURE;
        }
        let cold = match measure(&session, prompt, n_predict) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "| {depth} | {n_ctx} | {kv_mib:.0} | {slots} | | | | | | | | | | cold: {e} |"
                );
                failures += 1;
                continue;
            }
        };
        let warm = match measure(&session, prompt, n_predict) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "| {depth} | {n_ctx} | {kv_mib:.0} | {slots} | | | | | | | | | | warm: {e} |"
                );
                failures += 1;
                continue;
            }
        };
        if cold.ids != warm.ids {
            eprintln!("depth {depth}: the cold and warm passes produced different ids");
            failures += 1;
        }
        let (p50, head, tail) = warm.shape();
        println!(
            "| {depth} | {n_ctx} | {kv_mib:.0} | {slots} | {load:.2} | {:.2} | **{:.2}** | \
             **{:.2}** | {:.1}% | {:.0} | {p50:.1} | {head:.1} | {tail:.1} | {:+.1}% |",
            cold.prefill_toks(depth),
            cold.decode_toks(),
            warm.decode_toks(),
            100.0 * warm.hit_rate,
            warm.staged_mib,
            100.0 * (tail / head - 1.0),
        );
        phase_rows.push((depth, cold, warm));
    }

    // The attribution. Every column is decode-only and per decode step.
    println!(
        "\n| depth | decode.total | attn.attend | attn.qkv | attn.proj | moe.stage | \
         moe.expert_matvec | moe.host_sync | moe.readback |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for (depth, _cold, m) in &phase_rows {
        println!(
            "| {depth} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
            m.phase_ms("decode.total"),
            m.phase_ms("attn.attend"),
            m.phase_ms("attn.qkv"),
            m.phase_ms("attn.proj"),
            m.phase_ms("moe.stage"),
            m.phase_ms("moe.expert_matvec"),
            m.phase_ms("moe.host_sync"),
            m.phase_ms("moe.readback"),
        );
    }
    println!(
        "\nms per decode step, warm pass, **decode steps only** -- `profile::reset()` runs at \
         the first mark, so prefill is excluded from every column above."
    );
    println!(
        "\n\u{26a0} `hit (all)` and `staged MiB (all)` cover the WHOLE pass, prompt included: \
         the cache counters can only be zeroed through the session, and `generate_with` holds \
         its lock. At depth 8192 the 8,192 prefill steps outnumber the 63 decode steps 130 to \
         1, so those two columns describe prefill and must not be read as a decode hit rate. \
         The decode-only staging cost is `moe.stage` in the phase table, which is fenced \
         correctly."
    );

    // Disk and ARC, the two counters that separate "the bus is busy" from "the page was not
    // resident". Prefill and decode are bracketed separately because at depth 8192 the prompt
    // is 8,192 steps against 63 decode steps, so one figure for the pass would describe the
    // prompt and be read as the decode.
    if disk_dev().is_some() {
        println!(
            "\n| depth | pass | disk read MiB | disk MiB/step | ARC miss % | staged MiB | \
             disk/staged |"
        );
        println!("|---|---|---|---|---|---|---|");
        for (depth, cold, warm) in &phase_rows {
            for (label, m) in [("cold", cold), ("warm", warm)] {
                for (phase, io, steps) in
                    [("prefill", m.io_prefill, *depth), ("decode", m.io_decode, m.decode_steps)]
                {
                    let per_step = if steps > 0 { io.disk_mib() / steps as f64 } else { f64::NAN };
                    let miss =
                        io.arc_miss_pct().map_or_else(|| "-".to_string(), |v| format!("{v:.1}%"));
                    // Staged bytes are tracked only for the whole pass, so the ratio is quoted
                    // on the decode row alone, where both cover a comparable window.
                    let staged =
                        if phase == "decode" { format!("{:.0}", m.staged_mib) } else { "-".into() };
                    let ratio = if phase == "decode" && m.staged_mib > 0.0 {
                        format!("{:.2}", io.disk_mib() / m.staged_mib)
                    } else {
                        "-".to_string()
                    };
                    println!(
                        "| {depth} | {label} {phase} | {:.0} | {per_step:.1} | {miss} | {staged} \
                         | {ratio} |",
                        io.disk_mib()
                    );
                }
            }
        }
        println!(
            "\n\u{1f534} `disk read MiB` is the whole device, not this process: another tenant's \
             reads land in it too, which is why the load average is gated before every run. \
             `staged MiB` covers the WHOLE pass (see the caveat above), so `disk/staged` is an \
             upper bound on the share of staging served from NVMe rather than from ARC."
        );
    } else {
        println!(
            "\nnote: MOEARC_BENCH_DISK is unset (e.g. `nvme0n1p4`), so the disk/ARC table is \
             omitted rather than guessed at."
        );
    }

    if failures > 0 {
        eprintln!("\n{failures} depth(s) failed");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
