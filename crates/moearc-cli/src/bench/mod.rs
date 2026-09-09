//! `moearc bench` — measure this machine, and refuse to print a number it cannot stand behind.
//!
//! `bench/PROTOCOL.md` is this command's specification, not its background reading. Every rule
//! there was learned from a published result this project had to withdraw, and the protocol's
//! own framing is the design brief:
//!
//! > A user reproducing our numbers should not have to know any of this — the tool should
//! > enforce it and **refuse to print a number it does not trust.** A benchmark that reports a
//! > figure it cannot stand behind is worse than one that reports nothing.
//!
//! # The shape of a run
//!
//! ```text
//!   probe        read the machine                       -> guard::Reading
//!   pre-flight   guards that can refuse before any work -> refuse here, cheaply
//!   shape        replay committed traces (no clock)     -> the headline
//!   absolutes    spawn N children, each timing itself   -> this machine's artefact
//!   incumbent    llama-bench, pinned and read back      -> the comparison
//!   post-flight  guards that need the run's own output  -> thread pins, dispersion
//!   artefact     one file, human and machine readable
//! ```
//!
//! Two of those deserve a note.
//!
//! **The guards run twice, on purpose.** Some questions can only be asked before the work
//! (is the box quiet? does the model fit in cache? is this the build I meant?) and some only
//! after it (did the thread pin actually take? is the spread small enough to headline?). A
//! single pass would either start work it should have refused, or fail to check the things
//! that are only knowable at the end.
//!
//! **The parent measures nothing.** Every timed figure comes from a child process — see
//! [`timed`] for why §5 makes that a requirement rather than a style choice.
//!
//! # What this command deliberately does not do
//!
//! 🔴 It never converts a hit rate into a tok/s. §9: *never convert a proxy into a headline
//! unit you have not validated.* Hit rate predicts staged bytes; there is no validated model
//! in this project between staged bytes and throughput, so none is published. [`shape`] has a
//! test that fails if a throughput field ever appears in its output.

pub mod guard;
pub mod incumbent;
pub mod probe;
pub mod report;
pub mod shape;
pub mod stats;
pub mod timed;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{Context, Result, bail};
use moearc_engine::residency::Policy;

use crate::cli::{BenchArgs, BenchRunArgs, Cli};
use crate::source::Sources;

/// Exit code for a run the guards refused. Distinct from a plain failure (1) and from "this
/// part of MoEArc is not built" (2), so a script can tell a refusal from a crash.
pub const EXIT_REFUSED: u8 = 3;

// ---------------------------------------------------------------------------------------
// The worker: one timed invocation, re-executed from this same binary
// ---------------------------------------------------------------------------------------

/// `moearc bench-run` — one child. Prints a single result line and exits.
pub fn run_worker(args: &BenchRunArgs) -> Result<ExitCode> {
    let all = timed::read_ids(&args.prompt_ids)?;
    if (args.depth as usize) > all.len() {
        bail!(
            "{} holds {} token ids, which is fewer than the depth {} asked for. §8 requires \
             the prompt be real committed text; padding it here would tile a prompt and \
             flatter the hit rate.",
            args.prompt_ids.display(),
            all.len(),
            args.depth
        );
    }
    let prompt = all[..(args.depth as usize).max(1)].to_vec();
    let job = timed::WorkerJob {
        model: args.model.clone(),
        prompt,
        depth: args.depth,
        generated_tokens: args.tokens,
        residency: args.residency.clone(),
        host_policy: args.host.clone(),
        n_ctx: args.ctx,
        disk_devices: args.disk_dev.clone().map(|d| vec![d]),
    };
    let result = timed::measure(&job)?;
    println!("{}{}", timed::RESULT_PREFIX, serde_json::to_string(&result)?);
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------------------

pub fn run(cli: &Cli, sources: &Sources, args: &BenchArgs) -> Result<ExitCode> {
    let want_shape = args.shape || args.all || !args.absolutes;
    let want_absolutes = args.absolutes || args.all;

    let thresholds = args.thresholds();
    let model_path = resolve_model(cli, sources, args)?;
    let devices = sources.devices.detect().ok();
    let device = devices.as_ref().and_then(|r| r.primary().map(device_facts));
    let (mem_total, mem_available) = probe::meminfo();

    let mut reading = guard::Reading {
        logical_cpus: probe::logical_cpus(),
        load1: probe::load1(),
        mem_total_bytes: mem_total,
        mem_available_bytes: mem_available,
        zfs_arc: probe::zfs_arc(),
        model: model_path.as_deref().and_then(probe::model_facts),
        engine_threads: None,
        host_policy: None,
        incumbent: None,
        build: probe::build_facts(),
        device,
        gpu_compiled_in: probe::GPU_COMPILED_IN,
        expected_backend: args.expect_backend.clone(),
    };

    let intent = if want_absolutes { guard::Intent::ABSOLUTES } else { guard::Intent::SHAPE };

    // 🔴 The same settle the sweep uses, applied before the sweep, for the same reason and so
    // the rule reads the same way in both places. Without it the second of two back-to-back
    // runs refuses at the pre-flight with `the box is busy` and no hint that the busy process
    // was the first one: measured here, a run that finished at load 15 refused the next
    // invocation four minutes later. `--load-settle 0` refuses at once, as before.
    if want_absolutes && !args.check && !args.force && args.load_settle > 0 {
        let limit = thresholds.load_refuse(reading.logical_cpus);
        if reading.load1.is_some_and(|l| l > limit) {
            eprintln!(
                "moearc bench: load is above {limit:.2}; waiting up to {}s for the box to go \
                 quiet (--load-settle)",
                args.load_settle
            );
            let _ = wait_for_quiet(limit, args.load_settle);
            reading.load1 = probe::load1();
        }
    }

    let preflight = guard::evaluate(&reading, &thresholds, intent);
    let refused_before_work = guard::Verdict::of(&preflight) == guard::Verdict::Refused;

    let mut not_measured: Vec<String> = Vec::new();
    let mut absolutes: Vec<timed::TimedPoint> = Vec::new();
    let mut incumbent_result = None;
    // Findings that only the run itself can produce: what the page-cache warm-up did, and
    // whether the box had to be waited on between invocations.
    let mut run_findings: Vec<guard::Finding> = Vec::new();

    // 🔴 The shape results are a deterministic replay: no clock, no device. They are produced
    // whatever the pre-flight said, because refusing them for load average would be theatre —
    // and a refused absolutes run is exactly when a user most wants the part that still holds.
    let shape_result = if want_shape {
        let s = measure_shape(cli, sources, args, &reading)?;
        // 🔴 The shape half is what this tool calls *the result* — §0, and the table at the top
        // of `docs/bench.md`. A run that replayed nothing of it still reported
        // `VERDICT: trusted`, with a headline bullet reading "**Shape (reproduces anywhere).**
        // no trace was replayed": an absence formatted as a finding, over a verdict that says
        // it can be cited. Every skipped file already names its own reason below; what was
        // missing was the verdict noticing that the total was zero. Measured by running
        // `--all --model olmoe-…` on this repository, whose captures are all from other models.
        if s.traces.is_empty() {
            run_findings.push(guard::Finding {
                level: guard::Level::Warn,
                code: "shape-empty",
                rule: "§0",
                headline: format!(
                    "no routing trace was replayed — {} capture(s) were found and all were \
                     skipped",
                    s.skipped.len()
                ),
                detail: "§0 makes the shape results the headline and the absolutes an artefact \
                         of one machine, so a run that replayed no trace has not produced the \
                         thing it calls its result. Each skipped file is named with its reason \
                         in the Shape section — most often that it was captured from a \
                         different model, which §9 forbids transferring, or that it is a \
                         prefill capture. Pass --all-traces to replay them regardless, --trace \
                         <FILE> to name one, or drop --model to compare hit rates without the \
                         byte columns."
                    .to_string(),
            });
        }
        Some(s)
    } else {
        not_measured.push(
            "the shape results (dynamic-versus-static and the slots curve) — not requested"
                .to_string(),
        );
        None
    };

    if want_absolutes {
        if refused_before_work && !args.force {
            not_measured.push(format!(
                "every timed measurement — the pre-flight checks refused: {}",
                preflight
                    .iter()
                    .filter(|f| f.level == guard::Level::Refuse)
                    .map(|f| f.headline.clone())
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        } else if args.check {
            not_measured.push("every timed measurement — --check only runs the guards".to_string());
        } else {
            let model = model_path
                .clone()
                .context("--absolutes needs --model: there is nothing to time without one")?;
            run_findings.push(warm_page_cache(&reading, &model, args));
            let (points, threads, mut produced) = measure_absolutes(args, &model)?;
            run_findings.append(&mut produced);
            reading.engine_threads = Some(threads);
            reading.host_policy = Some(args.host_policy.clone());
            absolutes = points;

            if let Some(bin) = &args.llama_bench {
                let (result, _raw) = run_incumbent(args, bin, &model)?;
                // `_raw` is the same text, already carried inside `result.raw_output`; it is
                // returned separately only so a caller that wants it need not reach through
                // the parsed struct.
                reading.incumbent = Some(result.facts.clone());
                incumbent_result = Some(result);
            } else {
                not_measured.push(
                    "llama.cpp as a baseline — pass --llama-bench <PATH>. It is never searched \
                     for: PROTOCOL §2 records a glob-ordered pick that silently selected a \
                     Vulkan build 4.8x slower than SYCL."
                        .to_string(),
                );
            }
        }
    } else {
        not_measured.push(
            "absolute throughput on this machine — not requested (pass --absolutes)".to_string(),
        );
    }

    not_measured.push(
        "staging-versus-attention attribution with prompt depth (PROTOCOL §0 claim 2). It \
         needs a synchronous device profile (`MOEARC_SYNC_EACH=1`) at two depths on a model \
         large enough for staging to bind, which is a far longer run than this command takes; \
         `bench/baselines/gpt-oss-120b.md` §6.4 carries the measurement."
            .to_string(),
    );

    // Post-flight: the questions that could only be asked once the run had produced output.
    let mut findings = guard::evaluate(&reading, &thresholds, intent);
    findings.append(&mut run_findings);
    for p in &absolutes {
        findings.push(p.cold.dispersion(&thresholds));
        findings.push(p.warm.dispersion(&thresholds));
    }
    if let Some(inc) = &incumbent_result {
        for p in &inc.points {
            findings.push(p.decode.dispersion(&thresholds));
        }
    }
    if args.force {
        findings.push(guard::Finding {
            level: guard::Level::Warn,
            code: "forced",
            rule: "§0",
            headline: "--force was given: refusals were overridden".to_string(),
            detail: "Every figure in this artefact was produced past a check that said it \
                     should not be. Do not cite it."
                .to_string(),
        });
    }

    let verdict = guard::Verdict::of(&findings);
    let headline = headline_for(verdict, shape_result.as_ref(), &absolutes, &findings);

    let artefact = report::Artefact {
        tool: "moearc bench",
        tool_version: env!("CARGO_PKG_VERSION"),
        generated_utc: report::now_utc(),
        command: std::env::args().collect(),
        verdict,
        headline,
        reading,
        thresholds,
        findings,
        shape: shape_result,
        absolutes,
        incumbent: incumbent_result,
        not_measured,
    };

    emit(cli, args, &artefact)?;
    Ok(artefact.exit_code())
}

fn emit(cli: &Cli, args: &BenchArgs, a: &report::Artefact) -> Result<()> {
    let text = report::render(a);
    // Always on stderr, even when the artefact goes to a file or the JSON goes to a pipe:
    // a refusal that only exists inside a document nobody opened is not a refusal.
    eprintln!(
        "moearc bench: verdict {} ({} check(s), {} refusing)",
        a.verdict.label(),
        a.findings.len(),
        a.findings.iter().filter(|f| f.level == guard::Level::Refuse).count(),
    );
    if let Some(path) = &args.out {
        std::fs::write(path, &text)
            .with_context(|| format!("writing the artefact to {}", path.display()))?;
        eprintln!("moearc bench: artefact written to {}", path.display());
    }
    if cli.global.json {
        println!("{}", serde_json::to_string_pretty(a)?);
    } else if args.out.is_none() {
        print!("{text}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------------------

fn measure_shape(
    cli: &Cli,
    sources: &Sources,
    args: &BenchArgs,
    reading: &guard::Reading,
) -> Result<shape::ShapeResult> {
    let mut files = args.trace.clone();
    if files.is_empty() {
        let dir = &args.traces;
        if !dir.is_dir() {
            bail!(
                "no trace directory at {} — the shape results replay captured routing traces, \
                 which live in the repository at `bench/traces`. Pass --traces <DIR> or \
                 --trace <FILE>.",
                dir.display()
            );
        }
        files = shape::discover(dir)?;
    }
    if files.is_empty() {
        bail!("no `.ndjson` captures found to replay");
    }

    // Slot size and the card's own operating point, when both are knowable — and bound to
    // the model's file name, because §9 forbids carrying either onto a capture from a
    // different model. Neither is required: their absence removes the byte columns rather
    // than filling them with an estimate.
    let card = resolve_card(sources, args);
    let model_context = card.as_ref().map(|c| {
        let card_slots = reading.device.as_ref().and_then(|_| {
            sources.devices.detect().ok().and_then(|r| {
                r.primary().and_then(|d| match crate::fit::plan(d, c, cli.ctx).outcome {
                    crate::fit::FitOutcome::Fits { resident_experts, .. } => Some(resident_experts),
                    crate::fit::FitOutcome::DoesNotFit { .. } => None,
                })
            })
        });
        shape::ModelContext {
            file: c.file.clone().unwrap_or_default(),
            slot_bytes: args.slot_bytes.or(Some(c.per_expert_bytes)),
            card_slots,
        }
    });

    shape::measure(
        &files,
        parse_policy(&args.policy).map_err(|e| anyhow::anyhow!("--policy: {e}"))?,
        args.slots.as_deref(),
        model_context.as_ref(),
        model_context.is_some() && !args.all_traces,
        args.optimal,
    )
}

/// `lru`, `lfu`, `lru-k:2`, `slru:80`, `2q:25:50`, `w-tinylfu:1:80`, `phase-lru`, `optimal`.
///
/// A static split is deliberately *not* spellable here: it is derived from the capacity by
/// [`moearc_engine::residency::Trace::widest_static_split`] so the baseline always gets every
/// slot it can legitimately use. Letting a caller pick it would let them pick a weak one.
pub fn parse_policy(s: &str) -> Result<Policy, String> {
    let (name, rest) = s.split_once(':').unwrap_or((s, ""));
    let part = |i: usize, default: u8| -> Result<u8, String> {
        match rest.split(':').nth(i).filter(|p| !p.is_empty()) {
            Some(p) => p.parse().map_err(|_| format!("`{p}` is not a percentage")),
            None => Ok(default),
        }
    };
    match name {
        "lru" => Ok(Policy::Lru),
        "lfu" => Ok(Policy::Lfu),
        "optimal" | "belady" => Ok(Policy::Optimal),
        "phase-lru" => Ok(Policy::PhaseLru),
        "slru" => Ok(Policy::Slru { protected_pct: part(0, 80)? }),
        "2q" => Ok(Policy::TwoQ { kin_pct: part(0, 25)?, kout_pct: part(1, 50)? }),
        "lru-k" => Ok(Policy::LruK { k: part(0, 2)? }),
        "pinned-hot" => Ok(Policy::PinnedHot {
            pin_pct: part(0, 50)?,
            warmup_steps: rest
                .split(':')
                .nth(1)
                .filter(|p| !p.is_empty())
                .map(|p| p.parse::<u16>().map_err(|_| format!("`{p}` is not a step count")))
                .transpose()?
                .unwrap_or(64),
        }),
        "w-tinylfu" | "tinylfu" => Ok(Policy::WTinyLfu {
            window_pct: part(0, if name == "tinylfu" { 0 } else { 1 })?,
            protected_pct: part(1, 80)?,
        }),
        other => Err(format!(
            "`{other}` is not a policy — try lru, lfu, lru-k:<k>, slru:<pct>, 2q:<kin>:<kout>, \
             w-tinylfu:<window>:<protected>, phase-lru or optimal"
        )),
    }
}

// ---------------------------------------------------------------------------------------
// Absolutes
// ---------------------------------------------------------------------------------------

/// §4 — read the model into page cache before anything is timed.
///
/// 🔴 This exists because of a measurement taken on the reference box, not a preference. The
/// first `--absolutes` invocation against an uncached model returned cold values of **29.63,
/// 103.43, 104.17 tok/s** — a stddev 54% of the mean, refused under §5 — having faulted
/// **3.4 GiB** off NVMe inside the first child. The identical command a minute later returned
/// **103.22 ± 0.89**. Without a warm-up the first run on any machine refuses, and it refuses
/// citing the *spread* rather than the cause; §4 asks that the storage confound be removed
/// where it can be and stated loudly where it cannot.
fn warm_page_cache(reading: &guard::Reading, model: &Path, args: &BenchArgs) -> guard::Finding {
    use std::io::Read;

    let f = |level, headline: String, detail: String| guard::Finding {
        level,
        code: "warm-cache",
        rule: "§4",
        headline,
        detail,
    };
    let Some(m) = &reading.model else {
        return f(guard::Level::Pass, "no model to warm".into(), String::new());
    };
    if args.no_warm_cache {
        return f(
            guard::Level::Warn,
            "the page-cache warm-up was skipped by request".into(),
            "--no-warm-cache was given, so the first invocation's cold pass may include the \
             cost of faulting the model off the drive. The disk-read column is the evidence."
                .to_string(),
        );
    }
    let (ceiling, source) = guard::cache_ceiling(reading);
    if ceiling == 0 || m.bytes > ceiling {
        return f(
            guard::Level::Warn,
            format!(
                "the model cannot be warmed into cache — {} vs a {} ceiling",
                crate::format::bytes(m.bytes),
                crate::format::bytes(ceiling)
            ),
            format!(
                "Reading it in would evict as much as it caches, so it was not attempted \
                 ({source}). Every invocation will fault some of the model off the drive and \
                 the disk-read column beside each row is what says how much. PROTOCOL §4: \
                 this is stated rather than fixed, because the model this project exists for \
                 is 3.7x its ARC and refusing it would make the tool useless for its own \
                 headline case."
            ),
        );
    }
    let started = std::time::Instant::now();
    let mut buf = vec![0u8; 8 << 20];
    let mut read = 0u64;
    match std::fs::File::open(model) {
        Err(e) => f(
            guard::Level::Warn,
            "the model could not be opened to warm the cache".into(),
            format!("{e}. The run continues; the first cold pass may be storage-bound."),
        ),
        Ok(mut fh) => {
            while let Ok(n) = fh.read(&mut buf) {
                if n == 0 {
                    break;
                }
                read += n as u64;
            }
            let secs = started.elapsed().as_secs_f64();
            f(
                guard::Level::Pass,
                format!("model warmed into cache — {} in {secs:.1}s", crate::format::bytes(read)),
                format!(
                    "Read sequentially before the first timed child so that no invocation \
                     pays for faulting it off the drive, and so the cold/warm split describes \
                     the *expert pool* rather than the page cache. Ceiling {} ({source}). \
                     Suppress with --no-warm-cache.",
                    crate::format::bytes(ceiling)
                ),
            )
        }
    }
}

/// Wait for the 1-minute load average to come back under the refusal threshold.
///
/// 🔴 Without this the tool aborts its own sweep. §3 asks for the load average immediately
/// before **every** timed run, which is right — but our own previous invocation is not another
/// tenant, and a 1-minute average does not forget it. Measured from a quiet box at load
/// **0.99**: one `--host-policy frac:0.5` invocation put 19 host threads on the machine and
/// the next pre-run reading was **2.51**, over the 2.50 refusal. The sweep stopped at its
/// second row, deterministically, on nothing but its own decay.
///
/// Waiting preserves the rule and removes the self-collision. `Ok(seconds)` once the box is
/// under the threshold, `Err(last_reading)` if it never got there — which means something
/// *else* is on the machine, and that is the case §3 is actually about.
fn wait_for_quiet(limit: f64, budget_secs: u64) -> Result<f64, Option<f64>> {
    let started = std::time::Instant::now();
    let mut slept = false;
    loop {
        match probe::load1() {
            None => return Err(None),
            // Zero, not the microseconds it took to read `/proc/loadavg`: a box that was
            // already quiet was not waited on, and reporting "waited 0s before 3 of the
            // invocations" reads as a wait that happened.
            Some(l) if l <= limit => {
                return Ok(if slept { started.elapsed().as_secs_f64() } else { 0.0 });
            }
            Some(l) => {
                if started.elapsed().as_secs_f64() >= budget_secs as f64 {
                    return Err(Some(l));
                }
                slept = true;
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
        }
    }
}

fn measure_absolutes(
    args: &BenchArgs,
    model: &Path,
) -> Result<(Vec<timed::TimedPoint>, guard::ThreadPin, Vec<guard::Finding>)> {
    let exe = probe::self_exe().context("could not find this binary to re-execute")?;
    let ids = args.prompt_ids.clone().context(
        "--absolutes needs --prompt-ids <FILE>. PROTOCOL §8: use real text, not a repeated \
         phrase — a tiled prompt revisits the same experts and flatters the hit rate — and \
         commit the exact token ids so the run is reproducible from the repo. \
         `bench/references/*.ids` holds the committed ones.",
    )?;
    // 🔴 Asked here, in the parent, before a single child is launched. The check exists in the
    // worker too, but reaching it meant spawning a process, failing it, and surfacing the
    // reason wrapped in `bench worker exited exit status: 1` — and it is not an exotic case:
    // the default `--depths 128,512` is longer than four of the five prompt files this
    // repository ships, so the invocation printed in `docs/bench.md` failed immediately.
    let available = timed::read_ids(&ids)?.len();
    if let Some(&too_deep) = args.depths.iter().find(|&&d| d as usize > available) {
        bail!(
            "--depths asks for {too_deep} prompt tokens but {} holds {available}. \
             PROTOCOL §8 requires real committed text, so the prompt is never padded or \
             tiled to reach a depth — a tiled prompt revisits the same experts and flatters \
             the hit rate. Either lower --depths to at most {available}, or point \
             --prompt-ids at a longer committed capture: `bench/references/` holds one \
             (`gpt-oss-120b.longctx.ids`, 16,384 ids) built for depth work, and the \
             `*.capital.ids` files are short correctness prompts of 60-64 ids.",
            ids.display()
        );
    }
    let threads = args.threads.unwrap_or_else(|| probe::logical_cpus().saturating_sub(1).max(1));
    let disk = args
        .disk_dev
        .clone()
        .or_else(|| probe::disk_device_for(model).and_then(|v| v.into_iter().next()));

    let mut points = Vec::new();
    let mut reported: Option<usize> = None;
    let mut disagreed = false;
    let mut produced: Vec<guard::Finding> = Vec::new();
    let mut waited_total = 0.0f64;
    let mut waits = 0usize;

    'sweep: for &depth in &args.depths {
        let mut results = Vec::new();
        for i in 0..args.repeats.max(1) {
            // 🔴 §3: the load average is judged immediately before *every* timed run, not
            // once for the sweep. The child reads it again from inside itself and reports it,
            // which is the reading the artefact prints beside each invocation.
            let thresholds = args.thresholds();
            let limit = thresholds.load_refuse(probe::logical_cpus());
            if !args.force {
                match wait_for_quiet(limit, args.load_settle) {
                    Ok(w) => {
                        if w > 0.0 {
                            waited_total += w;
                            waits += 1;
                        }
                    }
                    Err(last) => {
                        produced.push(guard::Finding {
                            level: guard::Level::Refuse,
                            code: "load-settle",
                            rule: "§3",
                            headline: match last {
                                Some(l) => format!(
                                    "the box did not go quiet — load {l:.2} after waiting {}s, \
                                     refusing above {limit:.2}",
                                    args.load_settle
                                ),
                                None => {
                                    "the load average stopped being readable mid-sweep".to_string()
                                }
                            },
                            detail: format!(
                                "Stopped before invocation {} of depth {depth}, so the rows \
                                 already taken are kept and the rest were not attempted. \
                                 §3: a sweep at load 9.50 on this 20-thread box reported the \
                                 opposite of the truth, reproducibly, so finishing a sweep \
                                 whose later rows are not comparable with its earlier ones is \
                                 worse than stopping. The wait is --load-settle; something \
                                 other than this benchmark is using the machine.",
                                i + 1
                            ),
                        });
                        break 'sweep;
                    }
                }
            }
            let mut cmd = Command::new(&exe);
            cmd.arg("bench-run")
                .arg("--model")
                .arg(model)
                .arg("--depth")
                .arg(depth.to_string())
                .arg("--tokens")
                .arg(args.tokens.to_string())
                .arg("--residency")
                .arg(&args.residency)
                .arg("--host")
                .arg(&args.host_policy)
                .arg("--prompt-ids")
                .arg(&ids)
                // §1: pin the thread count, then read it back from the engine's own report.
                .env("MOEARC_HOST_THREADS", threads.to_string());
            if let Some(c) = args.ctx {
                cmd.arg("--ctx").arg(c.to_string());
            }
            if let Some(d) = &disk {
                cmd.arg("--disk-dev").arg(d);
            }
            if args.attribution {
                // §7: host wall time around an asynchronous queue bills device work to
                // whichever call later drains it. Sync-each makes each phase's host time equal
                // its device time — at the cost of destroying overlap, which is why the
                // artefact says the growth ratio is the finding and the milliseconds are not.
                cmd.env("MOEARC_PROFILE", "1").env("MOEARC_SYNC_EACH", "1");
            }

            let out = cmd
                .output()
                .with_context(|| format!("re-executing {} as a bench worker", exe.display()))?;
            if !out.status.success() {
                bail!(
                    "bench worker exited {}: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            let r = timed::parse_worker_output(&String::from_utf8_lossy(&out.stdout))?;
            match reported {
                None => reported = Some(r.host_threads),
                Some(n) if n != r.host_threads => disagreed = true,
                _ => {}
            }
            results.push(r);
        }
        points.push(timed::TimedPoint::from_workers(
            depth,
            args.tokens,
            &args.residency,
            &args.host_policy,
            threads,
            results,
        ));
    }

    if waits > 0 {
        produced.push(guard::Finding {
            level: guard::Level::Pass,
            code: "load-settle",
            rule: "§3",
            headline: format!(
                "waited {waited_total:.0}s in total for the box to go quiet, before {waits} of \
                 the invocations"
            ),
            detail: "🔴 The wait is almost always this benchmark waiting for itself: a \
                     1-minute load average does not forget the 19-thread invocation that just \
                     finished, and on this project's own box one host-offload run takes a \
                     quiet 0.99 to 2.51 — over the 2.50 refusal. §3's threshold is there to \
                     detect *another* tenant, so the tool waits for its own contribution to \
                     decay rather than refusing itself. Set --load-settle 0 to refuse at once \
                     instead."
                .to_string(),
        });
    }

    Ok((
        points,
        guard::ThreadPin { requested: threads, reported: if disagreed { None } else { reported } },
        produced,
    ))
}

fn run_incumbent(
    args: &BenchArgs,
    binary: &Path,
    model: &Path,
) -> Result<(incumbent::IncumbentResult, String)> {
    let inv = incumbent::Invocation {
        binary: binary.to_path_buf(),
        model: model.to_path_buf(),
        threads: if args.llama_bench_threads.is_empty() {
            vec![args.threads.unwrap_or_else(|| probe::logical_cpus().saturating_sub(1).max(1))]
        } else {
            args.llama_bench_threads.clone()
        },
        depths: args.depths.clone(),
        generated_tokens: args.tokens,
        inner_repeats: args.llama_bench_inner_repeats,
        invocations: args.repeats.max(1),
        extra: args
            .llama_bench_arg
            .iter()
            .flat_map(|a| a.split_whitespace().map(str::to_string))
            .collect(),
        discard_warmup: true,
    };
    incumbent::run(&inv)
}

impl BenchArgs {
    /// The numbers every check compares against, with the caller's overrides folded in.
    ///
    /// `--max-load` replaces the derived threshold outright rather than scaling it: a user who
    /// states a limit has stated a limit, and silently taking the larger of theirs and ours
    /// would make the flag look ignored on a small machine. The value that was actually in
    /// force is printed in the artefact either way, so raising it is visible.
    pub fn thresholds(&self) -> guard::Thresholds {
        let mut t = guard::Thresholds::default();
        if let Some(l) = self.max_load {
            t.load_fraction = 0.0;
            t.load_floor = l;
        }
        t
    }
}

// ---------------------------------------------------------------------------------------
// Odds and ends
// ---------------------------------------------------------------------------------------

fn device_facts(d: &crate::source::DeviceRow) -> guard::DeviceFacts {
    guard::DeviceFacts {
        name: d.name.clone(),
        backend: d.backend.label().to_string(),
        driver: d.driver.clone(),
        driver_build: d.driver_build,
        budget_source: d.budget_source.map(str::to_string),
        total_bytes: d.total_bytes,
        free_bytes: d.free_bytes,
    }
}

/// The catalogue entry for `--model`, whether it was given as a handle or as a path.
///
/// A path is the ordinary case for a benchmark — the model is wherever the user put it — and
/// without this the byte columns and the card marker silently disappear for exactly the
/// invocation a benchmarker is most likely to type. The directory searched is the file's own,
/// so the lookup does not depend on `$MOEARC_MODELS` pointing at it.
fn resolve_card(sources: &Sources, args: &BenchArgs) -> Option<crate::source::ModelCard> {
    use crate::source::ModelCatalog;
    let id = args.model.as_deref()?;
    if let Ok(card) = sources.models.resolve(id) {
        return Some(card);
    }
    let path = Path::new(id);
    let name = path.file_name()?.to_string_lossy().to_string();
    let dir = path.parent()?.to_path_buf();
    crate::catalog::LocalCatalog::new(dir)
        .installed()
        .ok()?
        .into_iter()
        .find(|c| c.file.as_deref() == Some(name.as_str()))
}

/// A path if it is one, otherwise a handle resolved through the catalogue.
fn resolve_model(cli: &Cli, sources: &Sources, args: &BenchArgs) -> Result<Option<PathBuf>> {
    let Some(id) = args.model.as_deref() else { return Ok(None) };
    let direct = PathBuf::from(id);
    if direct.is_file() {
        return Ok(Some(direct));
    }
    let card = sources
        .models
        .resolve(id)
        .with_context(|| format!("`{id}` is neither a file nor a model this machine has"))?;
    let file = card.file.with_context(|| {
        format!("`{id}` is a catalogue entry with no local file — pull it first")
    })?;
    Ok(Some(crate::catalog::models_dir(cli.global.models_dir.as_deref()).join(file)))
}

/// The lines that lead the artefact — or `None`, which is this tool's whole point.
fn headline_for(
    verdict: guard::Verdict,
    shape: Option<&shape::ShapeResult>,
    absolutes: &[timed::TimedPoint],
    findings: &[guard::Finding],
) -> Option<Vec<String>> {
    if verdict == guard::Verdict::Refused {
        return None;
    }
    let mut lines = Vec::new();
    // An empty shape half contributes no bullet: `verdict_line()` renders "no trace was
    // replayed", which under a "**Shape (reproduces anywhere).**" label reads as a result. The
    // `shape-empty` finding says it instead, where a reader looks for what went wrong.
    if let Some(s) = shape.filter(|s| !s.traces.is_empty()) {
        lines.push(format!("**Shape (reproduces anywhere).** {}", s.verdict_line()));
        for t in &s.traces {
            if let Some(k) = t.knee_slots {
                lines.push(format!(
                    "`{}`: the hit-rate curve flattens at **{}** slots.",
                    t.name,
                    crate::format::count(k as i64)
                ));
            }
        }
    }
    // An absolute is headlined only if its own dispersion finding passed. A figure whose error
    // bars disqualify it stays in the table below and out of the summary.
    let blocked =
        findings.iter().any(|f| f.code == "dispersion" && f.level == guard::Level::Refuse);
    if !absolutes.is_empty() {
        if blocked {
            lines.push(
                "**Absolutes.** Withheld: at least one figure's spread across independent \
                 invocations disqualifies it (PROTOCOL §5). The table below keeps them with \
                 their error bars."
                    .to_string(),
            );
        } else {
            for p in absolutes {
                lines.push(format!(
                    "**This machine, depth {}.** {} tok/s warm, {} cold, decode-only — an \
                     artefact of this box, not a portable number.",
                    p.depth,
                    p.warm.render(),
                    p.cold.render()
                ));
            }
        }
    }
    if lines.is_empty() { None } else { Some(lines) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_policy_spelling_parses_and_a_typo_names_the_alternatives() {
        for spec in [
            "lru",
            "lfu",
            "optimal",
            "phase-lru",
            "slru",
            "slru:70",
            "2q",
            "2q:25:50",
            "lru-k:3",
            "pinned-hot:40:128",
            "tinylfu",
            "w-tinylfu:1:80",
        ] {
            assert!(parse_policy(spec).is_ok(), "{spec} failed: {:?}", parse_policy(spec));
        }
        let err = parse_policy("lru2").unwrap_err();
        assert!(err.contains("lru-k"), "{err}");
    }

    #[test]
    fn tinylfu_and_w_tinylfu_differ_only_in_the_window() {
        // Policy::name() renders window_pct 0 as "tinylfu" and anything else as "w-tinylfu",
        // so the default has to match the spelling the user typed.
        assert_eq!(parse_policy("tinylfu").unwrap().name(), "tinylfu");
        assert_eq!(parse_policy("w-tinylfu").unwrap().name(), "w-tinylfu");
    }

    #[test]
    fn an_empty_shape_half_contributes_no_headline_bullet() {
        // 🔴 It used to contribute `**Shape (reproduces anywhere).** no trace was replayed`,
        // under a `trusted` verdict. The `shape-empty` finding is what says it now.
        let empty = shape::ShapeResult {
            policy: "lru".to_string(),
            knee_definition: String::new(),
            ladder_definition: String::new(),
            model_context: None,
            traces: Vec::new(),
            skipped: vec![("a.ndjson".to_string(), "another model".to_string())],
        };
        assert!(headline_for(guard::Verdict::Qualified, Some(&empty), &[], &[]).is_none());
    }

    #[test]
    fn a_refused_verdict_has_no_headline_whatever_was_measured() {
        // The single most important behaviour in this module.
        let h = headline_for(guard::Verdict::Refused, None, &[], &[]);
        assert!(h.is_none());
    }

    #[test]
    fn a_disqualifying_spread_withholds_the_absolutes_but_not_the_shape() {
        let finding = guard::Finding {
            level: guard::Level::Refuse,
            code: "dispersion",
            rule: "§5",
            headline: "x".to_string(),
            detail: "y".to_string(),
        };
        let point = timed::TimedPoint::from_workers(512, 64, "600", "off", 19, Vec::new());
        // Verdict::Qualified rather than Refused, so the artefact still leads with something —
        // but not with the number whose error bars disqualified it.
        let h = headline_for(guard::Verdict::Qualified, None, &[point], &[finding]).unwrap();
        assert!(h.iter().any(|l| l.contains("Withheld")), "{h:?}");
        assert!(!h.iter().any(|l| l.contains("tok/s warm")), "{h:?}");
    }
}
