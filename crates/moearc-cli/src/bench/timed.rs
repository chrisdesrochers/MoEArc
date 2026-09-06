//! The absolutes: MoEArc's own throughput, measured over **independent invocations**.
//!
//! # Why this spawns processes instead of looping
//!
//! `bench/PROTOCOL.md` §5: *prefer several independent invocations over more iterations inside
//! one — process-level variance is the variance that bit us, and `-r` inside one process
//! cannot see it.* A loop in this process would share a page cache, an allocator, a warmed
//! expert pool and a driver context with every other iteration, and would report a spread
//! narrower than the one a user meets.
//!
//! So the parent measures nothing. It re-executes **this same binary** as
//! `moearc bench-run …`, once per repeat, and aggregates what each child printed. The child is
//! the only thing that touches the device.
//!
//! # What is timed
//!
//! 🔴 §7: *measure the phase you claim to measure.* `Session::generate` decodes prompt tokens
//! through the same path as generated ones, so a stopwatch around the whole call divides by
//! `depth + n` steps and reports mostly prefill. The sampling callback is called once per
//! **generated** token and its first call lands the instant prefill finished, so the marks it
//! records fence the decode phase exactly: `n` marks with `n - 1` decode steps between them.
//! Prefill is reported in its own field, because it is a real cost that is simply not the one
//! the headline is about.
//!
//! # Cold and warm are two questions, not two samples of one
//!
//! §6: they differed by up to 2.2x here and they *converge* as depth grows, which is a finding
//! about the pool rather than noise. Each child runs both and reports them separately; the
//! parent aggregates them separately; nothing anywhere averages them together.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::probe::Io;
use super::stats::Sample;

/// The line prefix a worker's result is printed behind, so a child's ordinary logging (or an
/// oneAPI warning on stderr that finds its way to stdout) cannot be mistaken for the result.
pub const RESULT_PREFIX: &str = "MOEARC-BENCH-RESULT ";

/// One pass — cold or warm — as the worker measured it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pass {
    /// Decode steps between the first and last sampling mark. `generated - 1`.
    pub decode_steps: u64,
    pub decode_seconds: f64,
    /// Decode-only throughput. Prefill is excluded on purpose; see the module docs.
    pub decode_tok_s: f64,
    /// Wall time from the start of `generate` to the first sampling mark.
    pub prefill_seconds: f64,
    /// Expert-cache hit rate over this pass alone, from the engine's own counters.
    pub hit_rate: f64,
    pub demands: u64,
    /// Expert bytes copied across the bus during this pass.
    pub staged_bytes: u64,
    /// The ids produced, so two runs can be checked for having done the same work.
    pub token_ids: Vec<u32>,
}

/// Everything one child invocation reports back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerResult {
    /// Read immediately before the model was loaded, per §3.
    pub load1_before: Option<f64>,
    /// Read after the run, so a reader can see whether the run's own work moved it.
    pub load1_after: Option<f64>,
    pub device: String,
    /// 🔴 The engine's own answer, not the value we set in the environment. §1 forbids
    /// accepting that a flag took.
    pub host_threads: usize,
    pub resident_slots: u32,
    pub total_slots: u32,
    pub slot_bytes: u64,
    pub n_ctx: usize,
    pub depth: u32,
    pub cold: Pass,
    pub warm: Pass,
    /// Disk reads and ARC hits/misses bracketing the whole child, per §4.
    pub io: Io,
    /// Per-phase host time, present only when the child was run with `MOEARC_PROFILE=1`.
    ///
    /// ⚠️ Meaningful for attribution only under `MOEARC_SYNC_EACH=1`. On an asynchronous queue
    /// these are host wall times around a submit, which bill device work to whichever call
    /// later drains the queue — §7's failure, which once made this project retract a *correct*
    /// conclusion.
    pub phases: Vec<(String, f64, u64)>,
}

/// The aggregate over every child, cold and warm kept apart.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimedPoint {
    pub depth: u32,
    pub generated_tokens: u32,
    pub residency: String,
    pub host_policy: String,
    pub threads_requested: usize,
    pub cold: Sample,
    pub warm: Sample,
    pub cold_hit_rate: Sample,
    pub warm_hit_rate: Sample,
    /// Cold-to-warm ratio of the means. §6: they differed by 2.2x here and converge with
    /// depth, which is the finding.
    pub warm_over_cold: Option<f64>,
    pub invocations: Vec<WorkerResult>,
}

impl TimedPoint {
    pub fn from_workers(
        depth: u32,
        generated_tokens: u32,
        residency: &str,
        host_policy: &str,
        threads_requested: usize,
        results: Vec<WorkerResult>,
    ) -> Self {
        let cold = Sample::new(
            format!("moearc decode, cold pool, depth {depth}"),
            "tok/s",
            results.iter().map(|r| r.cold.decode_tok_s).collect(),
        );
        let warm = Sample::new(
            format!("moearc decode, warm pool, depth {depth}"),
            "tok/s",
            results.iter().map(|r| r.warm.decode_tok_s).collect(),
        );
        let warm_over_cold = {
            let (c, w) = (cold.mean(), warm.mean());
            if c > 0.0 && c.is_finite() && w.is_finite() { Some(w / c) } else { None }
        };
        Self {
            depth,
            generated_tokens,
            residency: residency.to_string(),
            host_policy: host_policy.to_string(),
            threads_requested,
            cold,
            warm,
            cold_hit_rate: Sample::new(
                format!("cold hit rate, depth {depth}"),
                "fraction",
                results.iter().map(|r| r.cold.hit_rate).collect(),
            ),
            warm_hit_rate: Sample::new(
                format!("warm hit rate, depth {depth}"),
                "fraction",
                results.iter().map(|r| r.warm.hit_rate).collect(),
            ),
            warm_over_cold,
            invocations: results,
        }
    }

    /// Bytes read off the disk during the timed children, summed.
    ///
    /// 🔴 §4: *a run that faulted gigabytes from disk measured the storage, not the engine.*
    /// This is the number that tells the two apart, and it is reported per point rather than
    /// once for the sweep because the answer changes with what is already resident.
    pub fn disk_read_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for r in &self.invocations {
            if let Some(b) = r.io.disk_read_bytes {
                total += b;
                any = true;
            }
        }
        if any { Some(total) } else { None }
    }

    /// ZFS ARC misses across the timed children, and the miss rate they imply.
    ///
    /// §4 asks for ARC hits and misses around every timed run, not only disk reads, and the two
    /// answer different questions: the disk counter says whether this box faulted, the ARC
    /// ratio says how close it was to doing so. On a machine where the model does fit in cache
    /// both come back clean, and *that is the finding* — this column exists to say so rather
    /// than to imply paging is the normal case.
    pub fn arc(&self) -> Option<(u64, u64)> {
        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut any = false;
        for r in &self.invocations {
            if let (Some(h), Some(m)) = (r.io.arc_hits, r.io.arc_misses) {
                hits += h;
                misses += m;
                any = true;
            }
        }
        if any { Some((hits, misses)) } else { None }
    }

    /// The thread count every child actually ran on, or `None` if they disagreed.
    pub fn threads_reported(&self) -> Option<usize> {
        let mut it = self.invocations.iter().map(|r| r.host_threads);
        let first = it.next()?;
        if it.all(|n| n == first) { Some(first) } else { None }
    }
}

/// Where the worker takes its prompt from.
///
/// 🔴 §8: *use real text, not a repeated phrase — a tiled prompt revisits the same experts and
/// flatters the hit rate*, and *commit the exact token ids so the run is reproducible from the
/// repo*. There is deliberately no generator here and no default: a prompt file is required,
/// and `bench/references/*.ids` holds the committed ones.
pub fn read_ids(path: &PathBuf) -> anyhow::Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading prompt ids {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("");
        for tok in line.split([',', ' ', '\t', '[', ']']).filter(|t| !t.trim().is_empty()) {
            out.push(tok.trim().parse::<u32>().map_err(|_| {
                anyhow::anyhow!("{}:{}: `{tok}` is not a token id", path.display(), n + 1)
            })?);
        }
    }
    if out.is_empty() {
        anyhow::bail!("{} contains no token ids", path.display());
    }
    Ok(out)
}

/// Pull the one result line out of a child's stdout.
pub fn parse_worker_output(stdout: &str) -> anyhow::Result<WorkerResult> {
    let line = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix(RESULT_PREFIX))
        .ok_or_else(|| anyhow::anyhow!("the worker printed no result line"))?;
    serde_json::from_str(line).map_err(|e| anyhow::anyhow!("worker result was not readable: {e}"))
}

// ---------------------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------------------

/// What one child is asked to do.
///
/// Every field is read by the `gpu` half of [`measure`]; on a build without it the stub
/// refuses before looking at any of them, which is the correct behaviour and leaves the fields
/// unread.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub struct WorkerJob {
    pub model: PathBuf,
    pub prompt: Vec<u32>,
    pub depth: u32,
    pub generated_tokens: u32,
    pub residency: String,
    pub host_policy: String,
    pub n_ctx: Option<u32>,
    pub disk_devices: Option<Vec<String>>,
}

#[cfg(feature = "gpu")]
pub fn measure(job: &WorkerJob) -> anyhow::Result<WorkerResult> {
    use moearc_engine::host_experts::HostPolicy;
    use moearc_engine::moe::Residency;
    use moearc_engine::profile;
    use moearc_engine::session::{Session, SessionOptions, StopConditions};

    let load1_before = super::probe::load1();
    let io_before = Io::now(job.disk_devices.as_deref());

    let residency: Residency =
        job.residency.parse().map_err(|e: String| anyhow::anyhow!("--residency: {e}"))?;
    let host: HostPolicy =
        job.host_policy.parse().map_err(|e: String| anyhow::anyhow!("--host: {e}"))?;

    let opts = SessionOptions { n_ctx: job.n_ctx.map(|c| c as usize), residency, host };
    let session = Session::load_with(&job.model, opts)
        .map_err(|e| anyhow::anyhow!("loading {}: {e}", job.model.display()))?;

    let info = session.info().clone();
    let stop = StopConditions {
        max_tokens: job.generated_tokens as usize,
        // No stop tokens: a run that ended early would divide by a different number of steps
        // than the one it was asked for, and the difference would look like throughput.
        stop_tokens: Vec::new(),
    };

    // The cold pass is cold by construction, not by being first.
    session.clear_residency().map_err(|e| anyhow::anyhow!("clearing residency: {e}"))?;
    let cold = one_pass(&session, &job.prompt, &stop)?;
    let warm = one_pass(&session, &job.prompt, &stop)?;

    let after = session.residency().map_err(|e| anyhow::anyhow!("reading residency: {e}"))?;
    let io = Io::now(job.disk_devices.as_deref()).since(&io_before);
    let phases =
        profile::report().into_iter().map(|p| (p.name.to_string(), p.seconds, p.calls)).collect();

    Ok(WorkerResult {
        load1_before,
        load1_after: super::probe::load1(),
        device: info.device.clone(),
        host_threads: after.host_threads,
        resident_slots: after.resident_slots,
        total_slots: after.total_slots,
        slot_bytes: after.slot_bytes,
        n_ctx: info.n_ctx,
        depth: job.depth,
        cold,
        warm,
        io,
        phases,
    })
}

/// One generation, timed decode-only.
#[cfg(feature = "gpu")]
fn one_pass(
    session: &moearc_engine::session::Session,
    prompt: &[u32],
    stop: &moearc_engine::session::StopConditions,
) -> anyhow::Result<Pass> {
    use std::time::Instant;

    use moearc_engine::profile;

    let before = session.residency().map_err(|e| anyhow::anyhow!("reading residency: {e}"))?;

    let mut marks: Vec<Instant> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let started = Instant::now();
    session
        .generate(prompt, stop, &mut |t| {
            // ⚠️ `profile::reset` is a free function over a process global, and that is
            // load-bearing: `generate` holds the session lock while this closure runs, so
            // touching the `Session` here would deadlock. The device thread is idle at this
            // moment, so the counters are quiescent when they are cleared — which is what
            // makes the phase totals cover the decode steps and nothing else.
            if marks.is_empty() {
                profile::reset();
            }
            marks.push(Instant::now());
            ids.push(t);
            true
        })
        .map_err(|e| anyhow::anyhow!("generate: {e}"))?;

    if marks.len() < 2 {
        anyhow::bail!(
            "a decode-only measurement needs at least two generated tokens; got {}",
            marks.len()
        );
    }
    let first = marks[0];
    let last = *marks.last().expect("checked non-empty");
    let decode_seconds = last.duration_since(first).as_secs_f64();
    let decode_steps = (marks.len() - 1) as u64;

    let after = session.residency().map_err(|e| anyhow::anyhow!("reading residency: {e}"))?;
    let demands = after.stats.demands.saturating_sub(before.stats.demands);
    let hits = after.stats.hits.saturating_sub(before.stats.hits);

    Ok(Pass {
        decode_steps,
        decode_seconds,
        decode_tok_s: if decode_seconds > 0.0 {
            decode_steps as f64 / decode_seconds
        } else {
            f64::NAN
        },
        prefill_seconds: first.duration_since(started).as_secs_f64(),
        hit_rate: if demands == 0 { 0.0 } else { hits as f64 / demands as f64 },
        demands,
        staged_bytes: after.bytes_staged.saturating_sub(before.bytes_staged),
        token_ids: ids,
    })
}

#[cfg(not(feature = "gpu"))]
pub fn measure(_job: &WorkerJob) -> anyhow::Result<WorkerResult> {
    anyhow::bail!(
        "this binary has no GPU backend compiled in — rebuild with `--features gpu`. \
         PROTOCOL §2: a benchmark that runs cleanly is not evidence it benchmarked the thing \
         you meant, and a binary that cannot reach the card cannot benchmark it at all."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(tok_s: f64, hit: f64) -> Pass {
        Pass {
            decode_steps: 63,
            decode_seconds: 63.0 / tok_s,
            decode_tok_s: tok_s,
            prefill_seconds: 1.0,
            hit_rate: hit,
            demands: 9072,
            staged_bytes: 1 << 30,
            token_ids: vec![1, 2, 3],
        }
    }

    fn worker(cold: f64, warm: f64, threads: usize) -> WorkerResult {
        WorkerResult {
            load1_before: Some(1.1),
            load1_after: Some(3.4),
            device: "Intel(R) Arc(TM) B580 Graphics".to_string(),
            host_threads: threads,
            resident_slots: 600,
            total_slots: 4608,
            slot_bytes: 13_220_000,
            n_ctx: 4096,
            depth: 512,
            cold: pass(cold, 0.60),
            warm: pass(warm, 0.87),
            io: Io { disk_read_bytes: Some(1_000), arc_hits: Some(10), arc_misses: Some(1) },
            phases: Vec::new(),
        }
    }

    #[test]
    fn cold_and_warm_are_never_averaged_together() {
        // §6: they differed by up to 2.2x here. A single mean over both would hide the finding
        // and understate the spread of each.
        let p = TimedPoint::from_workers(
            512,
            64,
            "600",
            "frac:0.5",
            19,
            vec![worker(5.62, 12.12, 19), worker(5.70, 12.30, 19), worker(5.55, 12.00, 19)],
        );
        assert_eq!(p.cold.n(), 3);
        assert_eq!(p.warm.n(), 3);
        assert!(p.cold.mean() < 6.0);
        assert!(p.warm.mean() > 12.0);
        // And the ratio is reported, because it is the finding rather than a nuisance.
        assert!((p.warm_over_cold.unwrap() - 2.16).abs() < 0.05);
    }

    #[test]
    fn a_disagreement_about_threads_across_children_is_reported_as_unknown() {
        let p = TimedPoint::from_workers(
            512,
            64,
            "600",
            "off",
            19,
            vec![worker(5.0, 10.0, 19), worker(5.0, 10.0, 4)],
        );
        assert_eq!(p.threads_reported(), None);
        let q = TimedPoint::from_workers(
            512,
            64,
            "600",
            "off",
            19,
            vec![worker(5.0, 10.0, 19), worker(5.0, 10.0, 19)],
        );
        assert_eq!(q.threads_reported(), Some(19));
    }

    #[test]
    fn the_arc_counters_are_summed_and_a_clean_run_is_reported_as_clean() {
        // A machine where the model fits in cache comes back with no misses, and that has to
        // be distinguishable from a machine where the counter could not be read.
        let mut clean = worker(5.0, 10.0, 19);
        clean.io.arc_hits = Some(1_000);
        clean.io.arc_misses = Some(0);
        let p = TimedPoint::from_workers(0, 64, "600", "off", 19, vec![clean.clone(), clean]);
        assert_eq!(p.arc(), Some((2_000, 0)));

        let mut unknown = worker(5.0, 10.0, 19);
        unknown.io.arc_hits = None;
        unknown.io.arc_misses = None;
        let q = TimedPoint::from_workers(0, 64, "600", "off", 19, vec![unknown]);
        assert_eq!(q.arc(), None);
    }

    #[test]
    fn disk_reads_are_summed_across_children_and_unknown_stays_unknown() {
        let mut a = worker(5.0, 10.0, 19);
        let b = worker(5.0, 10.0, 19);
        let p = TimedPoint::from_workers(0, 64, "600", "off", 19, vec![a.clone(), b]);
        assert_eq!(p.disk_read_bytes(), Some(2_000));
        a.io.disk_read_bytes = None;
        let q = TimedPoint::from_workers(0, 64, "600", "off", 19, vec![a]);
        assert_eq!(q.disk_read_bytes(), None);
    }

    #[test]
    fn a_worker_result_round_trips_through_the_wire_format() {
        let r = worker(5.62, 12.12, 19);
        let line = format!("{RESULT_PREFIX}{}", serde_json::to_string(&r).unwrap());
        let noise = format!("oneAPI: some warning\n{line}\ntrailing chatter\n");
        assert_eq!(parse_worker_output(&noise).unwrap(), r);
    }

    #[test]
    fn a_child_that_printed_nothing_is_an_error_not_an_empty_result() {
        assert!(parse_worker_output("segfault\n").is_err());
    }

    #[test]
    fn prompt_ids_are_read_from_a_committed_file_in_any_reasonable_layout() {
        let dir = std::env::temp_dir().join("moearc-bench-ids-test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("p.ids");
        std::fs::write(&f, "# gpt-oss-120b, this repo's own docs\n976, 9029\n5030 328\n[10128]\n")
            .unwrap();
        assert_eq!(read_ids(&f).unwrap(), vec![976, 9029, 5030, 328, 10128]);
        std::fs::write(&f, "# only a comment\n").unwrap();
        assert!(read_ids(&f).is_err());
        std::fs::write(&f, "976 not-a-number\n").unwrap();
        assert!(read_ids(&f).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
