//! The guards: everything `moearc bench` checks before it is willing to print a number.
//!
//! `bench/PROTOCOL.md` is the specification, and its central demand is that the tool **refuse
//! to print a number it does not trust**. Each rule there cites a failure this project
//! actually suffered; each function here implements one and names the rule it answers.
//!
//! # Why this module is pure
//!
//! Every guard is a function from a [`Reading`] to a [`Finding`]. Nothing here opens a file,
//! reads `/proc`, or spawns a process — [`crate::bench::probe`] does all of that and hands the
//! result over as data.
//!
//! 🔴 That split is not tidiness. The refusal paths are the part of this tool that must be
//! correct, and the only honest way to test them is to *inject the reading*: loading a machine
//! until its load average crosses a threshold, in order to see whether the threshold works,
//! contaminates the box for whoever is measuring on it and still only exercises one point.
//! Every threshold below therefore has a unit test that hands it the number.

use serde::Serialize;

use crate::format;

// ---------------------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------------------

/// How much a finding matters. Ordered, so the worst of a set is `iter().max()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    /// Checked, and the machine is in a state where the measurement means something.
    Pass,
    /// The number can be taken, but it carries a caveat that must travel with it.
    Warn,
    /// The number would not mean what it appears to mean. Do not take it.
    Refuse,
}

/// One check, its verdict, and the evidence behind it.
///
/// `detail` carries the numbers *and the threshold they were compared against*. A refusal that
/// does not say what the limit was cannot be argued with, and a reader of the artefact has to
/// be able to decide for themselves whether they agree with the limit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Finding {
    pub level: Level,
    /// Stable machine-readable name, so a script can key on a specific refusal.
    pub code: &'static str,
    pub headline: String,
    pub detail: String,
    /// The section of `bench/PROTOCOL.md` this check implements.
    pub rule: &'static str,
}

impl Finding {
    fn new(
        level: Level,
        code: &'static str,
        rule: &'static str,
        headline: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self { level, code, rule, headline: headline.into(), detail: detail.into() }
    }
}

/// The verdict over a whole set of findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every check passed. Numbers produced under this verdict are citable.
    Trusted,
    /// Nothing refused, but at least one caveat must travel with the numbers.
    Qualified,
    /// At least one check refused. No number produced under this verdict is a measurement.
    Refused,
}

impl Verdict {
    pub fn of(findings: &[Finding]) -> Self {
        match findings.iter().map(|f| f.level).max() {
            Some(Level::Refuse) => Self::Refused,
            Some(Level::Warn) => Self::Qualified,
            _ => Self::Trusted,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Trusted => "trusted",
            Self::Qualified => "qualified",
            Self::Refused => "refused",
        }
    }
}

// ---------------------------------------------------------------------------------------
// What the guards are given
// ---------------------------------------------------------------------------------------

/// ZFS ARC counters. Cumulative and machine-wide, so only differences mean anything —
/// except `c_max`, which is the cap and is the number the page-cache guard compares against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ZfsArc {
    pub c_max_bytes: u64,
    pub size_bytes: u64,
    pub hits: u64,
    pub misses: u64,
}

/// The model file the absolutes are being taken on.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelUnderTest {
    pub path: String,
    pub bytes: u64,
    /// Filesystem type, from `/proc/self/mountinfo`. `zfs` changes which cache ceiling applies.
    pub filesystem: String,
}

/// A thread count we asked for, beside the one the engine reported back.
///
/// 🔴 Two fields rather than one, because §1's failure was believing a flag took. `reported`
/// is read out of the engine's own output — `ResidencyReport::host_threads` for MoEArc,
/// the `n_threads` column of `llama-bench -o csv` for the incumbent — never inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ThreadPin {
    pub requested: usize,
    pub reported: Option<usize>,
}

/// What the incumbent binary said about itself.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IncumbentFacts {
    pub binary: String,
    pub build_commit: Option<String>,
    /// The `backends` column. §2's failure is invisible in every other field.
    pub backends: Option<String>,
    pub threads: ThreadPin,
    pub model_filename: Option<String>,
    /// `ONEAPI_DEVICE_SELECTOR` as the incumbent was launched with it.
    ///
    /// ⚠️ Recorded because the `backends` column **cannot** answer the question it looks like
    /// it answers. `SYCL` says the build targets Intel's oneAPI stack; which SYCL backend the
    /// runtime then picked — Level Zero or OpenCL — is this variable's decision at run time,
    /// and `llama-bench` does not print it. Printing the value in force is honest; inferring
    /// Level Zero from `SYCL` would not be.
    pub device_selector: Option<String>,
}

/// How this binary was built.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BuildFacts {
    pub commit: Option<String>,
    /// `None` when it could not be determined, which is itself worth saying.
    pub dirty: Option<bool>,
    pub profile: &'static str,
    pub target: &'static str,
    /// Cargo features this binary was compiled with, so the artefact names the build exactly.
    pub features: Vec<&'static str>,
}

/// The device the timed work would run on.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeviceFacts {
    pub name: String,
    pub backend: String,
    pub driver: String,
    /// The Level Zero compute-runtime build.
    ///
    /// 🔴 Carried as its own field, not only inside `driver`, because it belongs in every
    /// machine-readable result: two users on the same card get different answers for reasons
    /// that have nothing to do with their hardware. Measured on this project's own B580 in
    /// clean containers — build 27642 does not enumerate the card, 33578 detects it and then
    /// fails at model load, 37020 loads and decodes. A benchmark that does not record it
    /// cannot tell a hardware difference from a distribution difference.
    pub driver_build: Option<u32>,
    /// Where the free-VRAM figure came from: measured on the device, or installed capacity
    /// assumed idle.
    pub budget_source: Option<String>,
    /// The device's installed memory.
    pub total_bytes: u64,
    /// Free device memory as the driver reported it, when it could be measured.
    ///
    /// 🔴 In the artefact because of a contaminated run on this box: `moearc bench` passed its
    /// `box quiet — load 1.29` check while another agent's `llama-server` held **5.0 GiB of
    /// the 11.33 GiB card** with the same model loaded. Load average cannot see that — a
    /// second inference engine spends its time waiting on a GPU queue, not on CPUs — and the
    /// one figure that could have revealed it was read, used for planning, and then left out
    /// of the record entirely.
    pub free_bytes: u64,
}

/// Everything the guards look at, as data.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Reading {
    pub logical_cpus: usize,
    /// `None` when `/proc/loadavg` could not be read — which is a refusal for a timed run, not
    /// a pass. An unmeasurable box cannot be declared quiet.
    pub load1: Option<f64>,
    pub mem_total_bytes: u64,
    pub mem_available_bytes: u64,
    pub zfs_arc: Option<ZfsArc>,
    pub model: Option<ModelUnderTest>,
    pub engine_threads: Option<ThreadPin>,
    /// The `--host-policy` the timed children ran under, or `None` when nothing was timed.
    ///
    /// 🔴 The §1 guard needs this to tell a pin that failed from a pool that was never asked
    /// for. See [`host_offload_enabled`].
    pub host_policy: Option<String>,
    pub incumbent: Option<IncumbentFacts>,
    pub build: BuildFacts,
    pub device: Option<DeviceFacts>,
    /// Whether the GPU backend is compiled into *this* binary.
    pub gpu_compiled_in: bool,
    /// The backend the caller asserts the work must run on.
    pub expected_backend: String,
}

// ---------------------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------------------

/// Every number a guard compares against, in one struct so the artefact can print them and a
/// reader can disagree with them explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Thresholds {
    /// Fraction of the machine's logical CPUs of background load that makes a timing
    /// untrustworthy. See [`Thresholds::load_refuse`] for the derivation.
    pub load_fraction: f64,
    /// Absolute floor for the refusal, so a four-thread laptop is not refused for running its
    /// own desktop.
    pub load_floor: f64,
    /// Warn band, as a fraction of the refusal threshold.
    pub load_warn_ratio: f64,
    /// Model size as a fraction of the cache ceiling, above which the fit is called tight.
    pub cache_tight_ratio: f64,
    /// Coefficient of variation at which a result is qualified.
    pub cv_warn: f64,
    /// Coefficient of variation at which a result stops being a measurement.
    pub cv_refuse: f64,
    /// Independent invocations below which a stddev is not worth quoting.
    pub min_invocations: usize,
    /// Fraction of the card's memory that may already be allocated before a timed run is
    /// refused for sharing the device. See [`device_idle`].
    pub vram_occupied_refuse: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            load_fraction: 0.125,
            load_floor: 2.0,
            load_warn_ratio: 0.6,
            cache_tight_ratio: 0.8,
            cv_warn: 0.10,
            cv_refuse: 0.20,
            min_invocations: 3,
            // A desktop compositor's framebuffer on a card this size is a small fraction of
            // it; the second inference engine that provoked this guard held 44%. The value is
            // printed in the artefact so it can be argued with rather than discovered.
            vram_occupied_refuse: 0.10,
        }
    }
}

impl Thresholds {
    /// The 1-minute load average above which a timed run is refused.
    ///
    /// **One eighth of the machine, with a floor of 2.0.** The derivation, stated so it can be
    /// argued with:
    ///
    /// * The failure this guards against is `bench/PROTOCOL.md` §3 — a sweep at **load 9.50**
    ///   on a 20-thread box (47% of the machine) that reported host offload *losing* 60–75%
    ///   when it in fact gains. One eighth refuses that by 3.8x.
    /// * The reference box's documented idle baseline is **~1.2** with its ordinary daemons.
    ///   One eighth of 20 threads is 2.5, so an otherwise-quiet reference machine passes with
    ///   about 2x of margin rather than tripping on itself.
    /// * The effects being compared here are tens of percent (dynamic residency against a
    ///   static split, host offload against streaming). Background contention of more than an
    ///   eighth of the machine is the same order as the signal, and a confound the same size
    ///   as the finding is not a small one.
    /// * The floor exists because the fraction is meaningless on a small machine: an eighth of
    ///   four threads is 0.5, which almost any desktop exceeds while idle.
    pub fn load_refuse(&self, logical_cpus: usize) -> f64 {
        (logical_cpus as f64 * self.load_fraction).max(self.load_floor)
    }

    /// Where the load guard starts warning rather than refusing.
    pub fn load_warn(&self, logical_cpus: usize) -> f64 {
        self.load_refuse(logical_cpus) * self.load_warn_ratio
    }
}

// ---------------------------------------------------------------------------------------
// The guards
// ---------------------------------------------------------------------------------------

/// What the run intends to do, because a check that is irrelevant to a run must not fire on it.
///
/// The shape results are a deterministic replay of committed traces: no clock is read, no
/// device is touched, and the same input produces the same output on a loaded box and a quiet
/// one. Refusing them for load average would be theatre.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intent {
    /// True when something will be timed. Turns on the load, cache and build guards.
    pub timed: bool,
    /// True when device work will be submitted. Turns on the backend guards.
    pub device: bool,
}

impl Intent {
    pub const SHAPE: Self = Self { timed: false, device: false };
    pub const ABSOLUTES: Self = Self { timed: true, device: true };
}

/// Run every guard that applies to `intent`.
pub fn evaluate(reading: &Reading, thresholds: &Thresholds, intent: Intent) -> Vec<Finding> {
    let mut out = Vec::new();
    out.push(build(reading));
    if intent.timed {
        out.push(load(reading, thresholds));
        out.push(page_cache(reading, thresholds));
        out.extend(threads(reading));
    } else {
        out.push(Finding::new(
            Level::Pass,
            "untimed",
            "§0",
            "no clock is read",
            "The shape results are a deterministic replay of committed traces. They do not \
             depend on load average, page cache, thread count or device, so those guards are \
             not applied and their thresholds are not reported as if they had been."
                .to_string(),
        ));
    }
    if intent.device {
        out.extend(backend(reading));
        out.push(device_idle(reading, thresholds));
    }
    out
}

/// §3 — the box must be quiet.
pub fn load(reading: &Reading, t: &Thresholds) -> Finding {
    let refuse_at = t.load_refuse(reading.logical_cpus);
    let warn_at = t.load_warn(reading.logical_cpus);
    let Some(load1) = reading.load1 else {
        return Finding::new(
            Level::Refuse,
            "load-unreadable",
            "§3",
            "the 1-minute load average could not be read",
            "A timed run must state what else was on the machine. `/proc/loadavg` was not \
             readable, so this box cannot be declared quiet — and an unmeasured box is not a \
             quiet one."
                .to_string(),
        );
    };
    let detail = format!(
        "1-minute load average {load1:.2} on {} logical CPUs. Refuse above {refuse_at:.2} \
         (one eighth of the machine, floor {floor:.1}); warn above {warn_at:.2}. PROTOCOL §3 \
         records a sweep at load 9.50 on this 20-thread box that reported the opposite of the \
         truth, reproducibly.",
        reading.logical_cpus,
        floor = t.load_floor,
    );
    if load1 > refuse_at {
        Finding::new(
            Level::Refuse,
            "load",
            "§3",
            format!("the box is busy — load {load1:.2}, refusing above {refuse_at:.2}"),
            detail,
        )
    } else if load1 > warn_at {
        Finding::new(
            Level::Warn,
            "load",
            "§3",
            format!("the box is not idle — load {load1:.2}"),
            detail,
        )
    } else {
        Finding::new(Level::Pass, "load", "§3", format!("box quiet — load {load1:.2}"), detail)
    }
}

/// The cache ceiling that actually applies to `reading`, and what set it.
///
/// On ZFS, file data is cached in the ARC and **`zfs_arc_max` is the cap** — a 96 GB box with
/// `c_max` at 16 GiB cannot cache a 59 GiB model no matter how much memory is free. Everywhere
/// else the ordinary page cache applies and `MemAvailable` is the kernel's own answer.
pub fn cache_ceiling(reading: &Reading) -> (u64, &'static str) {
    match (&reading.model, reading.zfs_arc) {
        (Some(m), Some(arc)) if m.filesystem == "zfs" => (arc.c_max_bytes, "zfs_arc_max"),
        _ => (reading.mem_available_bytes, "MemAvailable"),
    }
}

/// §4 — the model must fit in page cache, or the tool must say it does not.
///
/// 🔴 This never refuses, and that is deliberate rather than lenient. The model this project
/// exists for is 59 GiB against a 16 GiB ARC; refusing it would make the tool useless for its
/// own headline case. What the protocol asks for is that the confound be **stated loudly and
/// travel with every number**, which is what a `Warn` does here — the caveat is carried into
/// the artefact and stamps the absolutes.
pub fn page_cache(reading: &Reading, t: &Thresholds) -> Finding {
    let Some(model) = &reading.model else {
        return Finding::new(
            Level::Pass,
            "page-cache",
            "§4",
            "no model file under test",
            "Nothing is being read off the disk, so there is no cache confound to report."
                .to_string(),
        );
    };
    let (ceiling, source) = cache_ceiling(reading);
    if ceiling == 0 {
        return Finding::new(
            Level::Warn,
            "page-cache",
            "§4",
            "the cache ceiling could not be read",
            format!(
                "{} is {} on {}, so whether it can be cached is unknown. Treat the disk-read \
                 counters, not this line, as the evidence.",
                model.path,
                format::bytes(model.bytes),
                model.filesystem
            ),
        );
    }
    let ratio = model.bytes as f64 / ceiling as f64;
    let detail = format!(
        "{} is {} against a cache ceiling of {} ({source} on {}) — a ratio of {ratio:.2}x. \
         PROTOCOL §4: a `-r 2` sweep of a model 3.7x its ARC gave 17.59 ± 5.56 where an `-r 5` \
         triplicate gave 28.5 ± 0.2. The disk-read counters bracketing each run are the check.",
        model.path,
        format::bytes(model.bytes),
        format::bytes(ceiling),
        model.filesystem,
    );
    if ratio > 1.0 {
        Finding::new(
            Level::Warn,
            "page-cache",
            "§4",
            format!("the model CANNOT be cached — {ratio:.2}x the cache ceiling"),
            detail,
        )
    } else if ratio > t.cache_tight_ratio {
        Finding::new(
            Level::Warn,
            "page-cache",
            "§4",
            format!("the model barely fits in cache — {ratio:.2}x the ceiling"),
            detail,
        )
    } else {
        Finding::new(
            Level::Pass,
            "page-cache",
            "§4",
            format!("the model fits in cache — {ratio:.2}x the ceiling"),
            detail,
        )
    }
}

/// §1 — pin the thread count on both sides, and read it back.
pub fn threads(reading: &Reading) -> Vec<Finding> {
    let mut out = Vec::new();
    if let Some(pin) = reading.engine_threads {
        let offload = reading.host_policy.as_deref().unwrap_or("off");
        out.push(if host_offload_enabled(offload) {
            pin_finding("threads-moearc", "moearc host pool", pin)
        } else {
            host_pool_disabled(offload, pin)
        });
    }
    if let Some(inc) = &reading.incumbent {
        out.push(pin_finding("threads-incumbent", "llama-bench", inc.threads));
    }
    if out.is_empty() {
        out.push(Finding::new(
            Level::Pass,
            "threads",
            "§1",
            "no engine invoked",
            "Nothing was run, so there is no thread count to pin.".to_string(),
        ));
    }
    out
}

/// Whether a `--host-policy` spec builds a host expert pool at all.
///
/// 🔴 `off` is the **default**, and under it `ResidencyReport::host_threads` is `0` by
/// construction — `Session` builds no `HostExecutor` to report a thread count from. Comparing
/// that zero against the requested pin made §1 refuse *every default `--absolutes` run*, on a
/// correctly built binary and a quiet box: measured here as
/// `moearc host pool ran on 0 threads, not the 19 asked for`, on a run whose figures were
/// otherwise clean at 0.9% spread. A pin on a pool that does not exist is vacuous, not broken,
/// and refusing it made the tool unable to produce its own headline.
pub fn host_offload_enabled(spec: &str) -> bool {
    !matches!(spec.trim(), "" | "off" | "frac:0" | "frac:0.0" | "frac:0.00")
}

/// §1 where there is deliberately no host pool to pin.
fn host_pool_disabled(spec: &str, pin: ThreadPin) -> Finding {
    match pin.reported {
        Some(0) => Finding::new(
            Level::Pass,
            "threads-moearc",
            "§1",
            "moearc ran GPU-only — no host pool to pin",
            format!(
                "`--host-policy {spec}` builds no host expert pool, so the engine reports 0 \
                 host threads by construction and the {} that were pinned were never drawn \
                 on. §1's failure was *accepting that a flag took*; here there is nothing for \
                 it to take. What this configuration compares is a GPU-only MoEArc against \
                 whatever the incumbent was given, and the host policy is printed beside every \
                 row so that asymmetry travels with the number.",
                pin.requested
            ),
        ),
        Some(n) => Finding::new(
            Level::Refuse,
            "threads-moearc",
            "§1",
            format!("moearc reported {n} host threads with offload disabled"),
            format!(
                "`--host-policy {spec}` should build no host pool, yet the engine reported {n} \
                 threads. The configuration did not take, and a run whose host policy is not \
                 the one that was asked for is not the run that was requested."
            ),
        ),
        None => pin_finding("threads-moearc", "moearc host pool", pin),
    }
}

fn pin_finding(code: &'static str, who: &str, pin: ThreadPin) -> Finding {
    match pin.reported {
        Some(got) if got == pin.requested => Finding::new(
            Level::Pass,
            code,
            "§1",
            format!("{who} pinned to {got} threads"),
            format!(
                "Asked for {}, and {who} reported {got} in its own output. PROTOCOL §1: \
                 `llama-bench`'s default is 4 threads on a 20-core box and every published \
                 comparison that accepted it had to be withdrawn.",
                pin.requested
            ),
        ),
        Some(got) => Finding::new(
            Level::Refuse,
            code,
            "§1",
            format!("{who} ran on {got} threads, not the {} asked for", pin.requested),
            "The pin did not take. A comparison in which one side silently ran on a \
             different number of cores than intended is the exact failure §1 records, so the \
             run is refused rather than annotated."
                .to_string(),
        ),
        None => Finding::new(
            Level::Refuse,
            code,
            "§1",
            format!("{who}'s thread count could not be read back"),
            format!(
                "{} threads were requested but {who} did not report what it used. §1 forbids \
                 inferring a thread count from a timing or accepting that a flag took, so this \
                 is a refusal rather than an assumption.",
                pin.requested
            ),
        ),
    }
}

/// §2 — verify you benchmarked the thing you meant.
pub fn backend(reading: &Reading) -> Vec<Finding> {
    let mut out = Vec::new();
    let want = reading.expected_backend.to_ascii_lowercase();

    if !reading.gpu_compiled_in {
        out.push(Finding::new(
            Level::Refuse,
            "build-no-gpu",
            "§2",
            "this binary has no GPU backend compiled in",
            "`moearc-engine`'s device half is behind the `gpu` feature. Rebuild with \
             `--features gpu` (and an oneAPI toolchain on PATH) to take absolutes. Reporting a \
             throughput figure from a binary that cannot reach the card would be §2's failure \
             in its purest form."
                .to_string(),
        ));
    }

    match &reading.device {
        None => out.push(Finding::new(
            Level::Refuse,
            "device",
            "§2",
            "no inference device was detected",
            "Every timed number has to name the device it was taken on. None was found, so \
             there is nothing to name."
                .to_string(),
        )),
        Some(d) => {
            let got = d.backend.to_ascii_lowercase();
            let level = if got == want { Level::Pass } else { Level::Refuse };
            out.push(Finding::new(
                level,
                "device",
                "§2",
                format!("{} on {} ({})", d.name, d.backend, d.driver),
                format!(
                    "Expected backend `{want}`, found `{got}`. PROTOCOL §2: selecting a binary \
                     by glob order once picked a Vulkan build 4.8x slower than SYCL — it \
                     produced real CSV, plausible numbers and exit 0, and only the backend \
                     field revealed it."
                ),
            ));
        }
    }

    if let Some(build) = reading.device.as_ref().and_then(|d| d.driver_build) {
        out.push(runtime_build(build));
    }

    if let Some(inc) = &reading.incumbent {
        out.push(incumbent_backend(inc, &want));
    }

    out
}

/// The `llama.cpp` backend names that correspond to a MoEArc backend.
///
/// 🔴 These are two different naming systems, and conflating them made §2's check a
/// **tautology**. `--expect-backend` names what `moearc-device` enumerates — a Level Zero
/// driver — while `llama-bench`'s `backends` column carries `ggml_backend_reg_name`, and
/// llama.cpp registers Intel's oneAPI path as **`SYCL`**. A substring comparison therefore
/// refused the correct build with the same sentence as the wrong one: measured on this box,
/// the SYCL build was refused as ``is a `SYCL` build, not `level_zero` `` and the Vulkan build
/// as ``is a `Vulkan` build, not `level_zero` ``. A check that refuses everything detects
/// nothing, and would have told a stranger their working build was broken.
///
/// An unmapped name returns an empty slice, which is reported as *unknown* rather than as a
/// pass or a refusal: this project does not invent a correspondence it has not verified.
pub fn incumbent_backends_for(expected: &str) -> &'static [&'static str] {
    match expected {
        "level_zero" => &["sycl", "level_zero"],
        "opencl" => &["opencl"],
        "cpu" => &["cpu"],
        _ => &[],
    }
}

/// §2 for the incumbent's build, judged on its own registry's names.
fn incumbent_backend(inc: &IncumbentFacts, want: &str) -> Finding {
    let code = "incumbent-backend";
    let Some(raw) = &inc.backends else {
        return Finding::new(
            Level::Refuse,
            code,
            "§2",
            format!("{} did not report a backend", inc.binary),
            "`llama-bench -o csv` carries a `backends` column. Its absence means the output \
             was not understood, and an unverified build is not a baseline."
                .to_string(),
        );
    };
    let selector = match inc.device_selector.as_deref() {
        Some(v) => format!("`ONEAPI_DEVICE_SELECTOR={v}`"),
        None => "`ONEAPI_DEVICE_SELECTOR` was unset".to_string(),
    };
    let accept = incumbent_backends_for(want);
    if accept.is_empty() {
        return Finding::new(
            Level::Warn,
            code,
            "§2",
            format!("no llama.cpp backend is known to correspond to `{want}`"),
            format!(
                "{} reports backends `{raw}`. `--expect-backend {want}` has no mapping onto \
                 `ggml_backend_reg_name`, so there is nothing to compare it against and this \
                 check says so rather than passing or refusing on a guess.",
                inc.binary
            ),
        );
    }
    let found: Vec<String> =
        raw.split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
    match found.iter().find(|f| accept.contains(&f.as_str())) {
        Some(hit) => Finding::new(
            Level::Pass,
            code,
            "§2",
            format!("{} is a `{raw}` build — the `{want}` counterpart", inc.binary),
            format!(
                "Read out of the tool's own `-o csv`, build {}. Never chosen by glob: the \
                 binary path was given explicitly. llama.cpp calls this backend `{hit}` and \
                 MoEArc calls it `{want}`; the names differ because they come from two \
                 different registries, not because the builds differ. ⚠️ This field does not \
                 say which SYCL backend the runtime then selected — that is decided at run \
                 time and llama-bench does not print it, so the value in force is recorded \
                 instead of inferred: {selector}.",
                inc.build_commit.as_deref().unwrap_or("unknown")
            ),
        ),
        None => Finding::new(
            Level::Refuse,
            code,
            "§2",
            format!("{} is a `{raw}` build, not the `{want}` counterpart", inc.binary),
            format!(
                "This is §2's failure verbatim: `ls build*/bin/llama-bench | head -1` once \
                 selected a Vulkan build 4.8x slower than SYCL — real CSV, plausible numbers, \
                 exit 0, and only this column revealed it. Expected one of `{}`, found \
                 `{raw}`. {selector}.",
                accept.join("`, `")
            ),
        ),
    }
}

/// §3 — is the card itself idle?
///
/// 🔴 The failure, measured on this box on 2026-09-09 rather than reasoned about: a
/// `--absolutes` run passed **`box quiet — load 1.29`** and ran for sixteen minutes against
/// gpt-oss-120B while another agent's `llama-server` held the same model on the same card.
/// The 1-minute load average is blind to it — a second engine spends its life waiting on a GPU
/// queue and contributed about 0.4 — and `bench/PROTOCOL.md` §3's own words are *"never run
/// two engines, or two agents' benchmarks, concurrently"*, a rule nothing enforced. The tool's
/// job is to enforce the protocol so a user does not have to know it.
///
/// The reading is the device's own, and both ends of it were measured on the same card within
/// half an hour: with the second engine up, **6.33 GiB free of 11.33 GiB installed** — 44%
/// spoken for; with it gone, **11.3 GiB free of 11.3 GiB** — nothing held at all, on a box
/// running a desktop session. So the threshold is not separating a busy card from a
/// theoretically perfect one, it is separating 44% from 0%, and the 10% default sits an order
/// of magnitude from the measured idle case in one direction and a factor of four from the
/// measured failure in the other. It refuses above a stated fraction rather than at any
/// occupancy because a compositor's framebuffer is a legitimate tenant, and the value in force
/// is printed like every other threshold.
///
/// ⚠️ An *unmeasurable* free figure is a warning, not a refusal: on a device with no memory of
/// its own there is nothing to measure and refusing would be refusing the wrong thing. The
/// guard says what it could not determine instead of assuming the card was free.
pub fn device_idle(reading: &Reading, t: &Thresholds) -> Finding {
    let code = "device-busy";
    let Some(d) = &reading.device else {
        return Finding::new(
            Level::Pass,
            code,
            "§3",
            "no device to check for other tenants",
            "There is no device reading, so the backend guard has already refused.".to_string(),
        );
    };
    if d.budget_source.as_deref() != Some("measured free VRAM") || d.total_bytes == 0 {
        return Finding::new(
            Level::Warn,
            code,
            "§3",
            "whether another process holds the card could not be determined".to_string(),
            format!(
                "Free device memory was not measured on {} ({}), so this run cannot say the \
                 card was idle when it started. §3 asks that two engines never share a box, \
                 and the check for it here is the device's own free-memory figure.",
                d.name,
                d.budget_source.as_deref().unwrap_or("no budget source reported"),
            ),
        );
    }
    let occupied = d.total_bytes.saturating_sub(d.free_bytes);
    let frac = occupied as f64 / d.total_bytes as f64;
    let detail = format!(
        "{} of {} on {} was already allocated when this run started — {:.0}% of the card. \
         Refuse above {:.0}%. 🔴 Measured failure: a run on this box passed `box quiet — load \
         1.29` and measured for sixteen minutes while another agent's `llama-server` held \
         5.0 GiB of the same card with the same model; a second engine waits on a GPU queue \
         rather than on CPUs, so the load average never moved. PROTOCOL §3: never run two \
         engines, or two agents' benchmarks, concurrently.",
        format::bytes(occupied),
        format::bytes(d.total_bytes),
        d.name,
        frac * 100.0,
        t.vram_occupied_refuse * 100.0,
    );
    if frac > t.vram_occupied_refuse {
        Finding::new(
            Level::Refuse,
            code,
            "§3",
            format!(
                "the card is not idle — {} of {} already allocated",
                format::bytes(occupied),
                format::bytes(d.total_bytes)
            ),
            detail,
        )
    } else {
        Finding::new(
            Level::Pass,
            code,
            "§3",
            format!(
                "card idle — {} free of {}",
                format::bytes(d.free_bytes),
                format::bytes(d.total_bytes)
            ),
            detail,
        )
    }
}

/// The Level Zero runtime build, as a caution and never as a gate.
///
/// 🔴 **Not a version check.** The Level Zero specification defines `driverVersion` as "a
/// non-zero, monotonically increasing value" and assigns it no encoding, and Intel publishes no
/// minimum for this workload, so there is nothing here to gate on and nothing is invented. What
/// exists is a measurement, taken on this project's own B580 in clean containers with the same
/// card and the same ~1.4 GiB free — three distributions, three different answers. The lowest
/// of them **reports success and is wrong**, which is exactly why it is worth printing beside
/// every result rather than only when something fails.
pub fn runtime_build(observed: u32) -> Finding {
    let verified = moearc_device::fitness::VERIFIED_RUNTIME_BUILD;
    let detail = format!(
        "Level Zero compute-runtime build {observed}. Measured on this project's B580, same \
         card, same free VRAM: build 27642 (Ubuntu 24.04 stock `libze-intel-gpu1`) does not \
         enumerate the card at all and reports the integrated GPU's host RAM as VRAM; build \
         33578 (Intel's client repo for noble) detects the card and then fails at model load \
         with `host-to-device copy failed on the device`; build {verified} (Ubuntu 26.04, \
         `26.05.37020.3`) loads and decodes, matching llama.cpp's token ids. That makes 33578 \
         a distribution constraint rather than a memory problem — proven by the 26.04 pass on \
         identical hardware. These are measurements on one machine, not a published minimum, \
         so an older build is a caution here and never a refusal."
    );
    if observed < verified {
        Finding::new(
            Level::Warn,
            "runtime-build",
            "§2",
            format!(
                "Level Zero build {observed} is older than the {verified} MoEArc has been \
                 measured on"
            ),
            detail,
        )
    } else {
        Finding::new(
            Level::Pass,
            "runtime-build",
            "§2",
            format!("Level Zero build {observed}"),
            detail,
        )
    }
}

/// The build this binary is, named exactly. §2 asks for the commit in the output of every run.
pub fn build(reading: &Reading) -> Finding {
    let b = &reading.build;
    let features = if b.features.is_empty() { "none".to_string() } else { b.features.join(",") };
    let detail = format!(
        "moearc {} / {} / features {features} / commit {} / working tree {}",
        b.profile,
        b.target,
        b.commit.as_deref().unwrap_or("unknown"),
        match b.dirty {
            Some(true) => "dirty",
            Some(false) => "clean",
            None => "unknown",
        }
    );
    match (b.commit.as_deref(), b.dirty) {
        (None, _) => Finding::new(
            Level::Warn,
            "build-commit",
            "§2",
            "this build does not know its commit",
            format!("{detail}. A result that cannot name its source cannot be reproduced."),
        ),
        (Some(_), Some(true)) => Finding::new(
            Level::Warn,
            "build-commit",
            "§2",
            "built from a dirty working tree",
            format!("{detail}. The commit named here does not describe what was compiled."),
        ),
        (Some(c), _) => {
            Finding::new(Level::Pass, "build-commit", "§2", format!("build {c}"), detail)
        }
    }
}

// ---------------------------------------------------------------------------------------
// Tests: every refusal path, by injected reading
// ---------------------------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn quiet_reading() -> Reading {
    Reading {
        logical_cpus: 20,
        load1: Some(1.20),
        mem_total_bytes: 96 << 30,
        mem_available_bytes: 72 << 30,
        zfs_arc: Some(ZfsArc {
            c_max_bytes: 16 << 30,
            size_bytes: 15 << 30,
            hits: 1_000,
            misses: 10,
        }),
        model: Some(ModelUnderTest {
            path: "/models/olmoe.gguf".to_string(),
            bytes: 4 << 30,
            filesystem: "zfs".to_string(),
        }),
        engine_threads: Some(ThreadPin { requested: 19, reported: Some(19) }),
        host_policy: Some("frac:0.75".to_string()),
        incumbent: None,
        build: BuildFacts {
            commit: Some("4056ebc".to_string()),
            dirty: Some(false),
            profile: "release",
            target: "x86_64-unknown-linux-gnu",
            features: vec!["gpu"],
        },
        device: Some(DeviceFacts {
            total_bytes: 12_168_933_376,
            free_bytes: 12_000_000_000,
            name: "Intel(R) Arc(TM) B580 Graphics".to_string(),
            backend: "level_zero".to_string(),
            driver: "xe / L0 build 37020".to_string(),
            driver_build: Some(37_020),
            budget_source: Some("measured free VRAM".to_string()),
        }),
        gpu_compiled_in: true,
        expected_backend: "level_zero".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(f: &'a [Finding], code: &str) -> &'a Finding {
        f.iter().find(|x| x.code == code).unwrap_or_else(|| panic!("no finding `{code}`"))
    }

    // --- §3, the busy box -------------------------------------------------------------

    #[test]
    fn the_load_threshold_is_an_eighth_of_the_machine_with_a_floor() {
        let t = Thresholds::default();
        // The reference box.
        assert!((t.load_refuse(20) - 2.5).abs() < 1e-9);
        // A four-thread laptop is not refused for running its own desktop.
        assert!((t.load_refuse(4) - 2.0).abs() < 1e-9);
        // A big box scales up rather than being pinned to the floor.
        assert!((t.load_refuse(128) - 16.0).abs() < 1e-9);
    }

    #[test]
    fn the_documented_idle_baseline_passes_and_the_documented_failure_is_refused() {
        let t = Thresholds::default();
        let mut r = quiet_reading();

        // ~1.2 is the reference box's own idle baseline, recorded in bench/baselines.
        r.load1 = Some(1.20);
        assert_eq!(load(&r, &t).level, Level::Pass);

        // PROTOCOL §3's failure: a sweep at 9.50 that reported the opposite of the truth.
        r.load1 = Some(9.50);
        assert_eq!(load(&r, &t).level, Level::Refuse);

        // The band between them warns rather than passing silently.
        r.load1 = Some(1.80);
        assert_eq!(load(&r, &t).level, Level::Warn);
    }

    #[test]
    fn a_refusal_states_the_threshold_it_used() {
        let mut r = quiet_reading();
        r.load1 = Some(21.69);
        let f = load(&r, &Thresholds::default());
        assert_eq!(f.level, Level::Refuse);
        assert!(f.detail.contains("2.50"), "{}", f.detail);
        assert!(f.headline.contains("21.69"), "{}", f.headline);
    }

    #[test]
    fn an_unreadable_load_average_is_a_refusal_not_a_pass() {
        // The dangerous default. A box whose load could not be read is not a quiet box.
        let mut r = quiet_reading();
        r.load1 = None;
        assert_eq!(load(&r, &Thresholds::default()).level, Level::Refuse);
    }

    #[test]
    fn load_is_not_checked_for_a_replay_that_reads_no_clock() {
        let mut r = quiet_reading();
        r.load1 = Some(40.0);
        let f = evaluate(&r, &Thresholds::default(), Intent::SHAPE);
        assert_eq!(Verdict::of(&f), Verdict::Trusted);
        assert!(f.iter().all(|x| x.code != "load"));
    }

    // --- §4, page cache ---------------------------------------------------------------

    #[test]
    fn a_model_larger_than_the_arc_warns_loudly_and_says_by_how_much() {
        let mut r = quiet_reading();
        // gpt-oss-120b: 59.03 GiB against zfs_arc_max of 16 GiB.
        r.model = Some(ModelUnderTest {
            path: "/models/gpt-oss-120b-MXFP4.gguf".to_string(),
            bytes: 63_387_346_208,
            filesystem: "zfs".to_string(),
        });
        let f = page_cache(&r, &Thresholds::default());
        assert_eq!(f.level, Level::Warn);
        assert!(f.headline.contains("CANNOT"), "{}", f.headline);
        assert!(f.detail.contains("3.6") || f.detail.contains("3.7"), "{}", f.detail);
    }

    #[test]
    fn the_arc_cap_beats_free_memory_on_zfs() {
        // 72 GiB free would say a 20 GiB model caches fine. The ARC cap says otherwise, and
        // the ARC cap is what actually holds file data on ZFS.
        let mut r = quiet_reading();
        r.model = Some(ModelUnderTest {
            path: "/m.gguf".to_string(),
            bytes: 20 << 30,
            filesystem: "zfs".to_string(),
        });
        assert_eq!(page_cache(&r, &Thresholds::default()).level, Level::Warn);
        // The same file on ext4 is judged against MemAvailable and fits.
        r.model.as_mut().unwrap().filesystem = "ext4".to_string();
        assert_eq!(page_cache(&r, &Thresholds::default()).level, Level::Pass);
    }

    #[test]
    fn a_model_that_cannot_be_cached_is_never_a_refusal() {
        // The model this project exists for is 3.7x its ARC. Refusing it would make the tool
        // useless for its own headline; §4 asks for a loud warning, not a stop.
        let mut r = quiet_reading();
        r.model = Some(ModelUnderTest {
            path: "/m.gguf".to_string(),
            bytes: 200 << 30,
            filesystem: "zfs".to_string(),
        });
        let f = evaluate(&r, &Thresholds::default(), Intent::ABSOLUTES);
        assert_eq!(Verdict::of(&f), Verdict::Qualified);
    }

    // --- §1, threads ------------------------------------------------------------------

    #[test]
    fn a_thread_pin_that_did_not_take_is_refused() {
        let mut r = quiet_reading();
        r.engine_threads = Some(ThreadPin { requested: 19, reported: Some(4) });
        assert_eq!(find(&threads(&r), "threads-moearc").level, Level::Refuse);
    }

    #[test]
    fn a_thread_count_that_could_not_be_read_back_is_refused_not_assumed() {
        // §1 forbids inferring a thread count or accepting that a flag took.
        let mut r = quiet_reading();
        r.engine_threads = Some(ThreadPin { requested: 19, reported: None });
        assert_eq!(find(&threads(&r), "threads-moearc").level, Level::Refuse);
    }

    #[test]
    fn the_incumbents_default_four_threads_is_caught() {
        // The exact §1 failure: -t 16 asked for, llama-bench's own csv says 4.
        let mut r = quiet_reading();
        r.incumbent = Some(IncumbentFacts {
            binary: "/opt/llama.cpp/build-sycl/bin/llama-bench".to_string(),
            build_commit: Some("e107984".to_string()),
            backends: Some("SYCL".to_string()),
            threads: ThreadPin { requested: 16, reported: Some(4) },
            model_filename: Some("gpt-oss-120b-MXFP4.gguf".to_string()),
            device_selector: Some("level_zero:0".to_string()),
        });
        assert_eq!(find(&threads(&r), "threads-incumbent").level, Level::Refuse);
    }

    // --- §3, another engine on the card -----------------------------------------------

    #[test]
    fn a_card_another_engine_is_already_holding_is_refused() {
        // 🔴 The measured failure, as numbers: 6.33 GiB free of 11.33 GiB installed while a
        // second engine held the rest, on a box whose load average read 1.29.
        let mut r = quiet_reading();
        if let Some(d) = r.device.as_mut() {
            d.free_bytes = 6_794_121_216;
        }
        let f = device_idle(&r, &Thresholds::default());
        assert_eq!(f.level, Level::Refuse, "{}", f.headline);
        assert!(f.detail.contains("44%"), "{}", f.detail);
    }

    #[test]
    fn a_compositors_framebuffer_does_not_refuse_the_run() {
        let mut r = quiet_reading();
        if let Some(d) = r.device.as_mut() {
            d.free_bytes = d.total_bytes - (300 << 20);
        }
        assert_eq!(device_idle(&r, &Thresholds::default()).level, Level::Pass);
    }

    #[test]
    fn an_unmeasurable_free_figure_warns_rather_than_assuming_the_card_is_free() {
        let mut r = quiet_reading();
        if let Some(d) = r.device.as_mut() {
            d.budget_source = Some("installed capacity, assumed idle".to_string());
        }
        assert_eq!(device_idle(&r, &Thresholds::default()).level, Level::Warn);
    }

    // --- §2, the wrong build ----------------------------------------------------------

    #[test]
    fn a_vulkan_incumbent_is_refused() {
        // 4.8x slower than SYCL, exit 0, plausible CSV. Only this field reveals it.
        let mut r = quiet_reading();
        r.incumbent = Some(IncumbentFacts {
            binary: "/opt/llama.cpp/build/bin/llama-bench".to_string(),
            build_commit: Some("e107984".to_string()),
            backends: Some("Vulkan".to_string()),
            threads: ThreadPin { requested: 16, reported: Some(16) },
            model_filename: None,
            device_selector: Some("level_zero:0".to_string()),
        });
        assert_eq!(find(&backend(&r), "incumbent-backend").level, Level::Refuse);
    }

    #[test]
    fn the_correct_sycl_incumbent_is_not_refused_alongside_the_vulkan_one() {
        // 🔴 The regression this exists for. `--expect-backend level_zero` was compared as a
        // substring against llama.cpp's own `backends` column, so `SYCL` — the *correct*
        // build — was refused with the same sentence as `Vulkan`. The check refused every
        // build and therefore detected none of them.
        let mut r = quiet_reading();
        r.incumbent = Some(IncumbentFacts {
            binary: "/zfs/swift/projects/llama.cpp/build/bin/llama-bench".to_string(),
            build_commit: Some("e107984".to_string()),
            backends: Some("SYCL".to_string()),
            threads: ThreadPin { requested: 16, reported: Some(16) },
            model_filename: None,
            device_selector: Some("level_zero:0".to_string()),
        });
        let findings = backend(&r);
        let f = find(&findings, "incumbent-backend");
        assert_eq!(f.level, Level::Pass, "{}", f.headline);
        // And it must not claim more than the field can support.
        assert!(f.detail.contains("ONEAPI_DEVICE_SELECTOR"), "{}", f.detail);
    }

    #[test]
    fn a_multi_backend_incumbent_is_read_as_a_list_not_a_substring() {
        let mut r = quiet_reading();
        r.incumbent = Some(IncumbentFacts {
            binary: "llama-bench".to_string(),
            build_commit: None,
            backends: Some("SYCL,BLAS,RPC".to_string()),
            threads: ThreadPin { requested: 16, reported: Some(16) },
            model_filename: None,
            device_selector: None,
        });
        assert_eq!(find(&backend(&r), "incumbent-backend").level, Level::Pass);
    }

    // --- §1, the default host policy --------------------------------------------------

    #[test]
    fn the_default_host_policy_has_no_pool_to_pin_and_is_not_refused_for_it() {
        // 🔴 The other regression. `off` is the default; under it the engine reports 0 host
        // threads by construction, and comparing that to the pin refused every default run.
        let mut r = quiet_reading();
        r.host_policy = Some("off".to_string());
        r.engine_threads = Some(ThreadPin { requested: 19, reported: Some(0) });
        assert_eq!(find(&threads(&r), "threads-moearc").level, Level::Pass);
    }

    #[test]
    fn a_host_pool_that_appeared_with_offload_off_is_still_refused() {
        let mut r = quiet_reading();
        r.host_policy = Some("off".to_string());
        r.engine_threads = Some(ThreadPin { requested: 19, reported: Some(19) });
        assert_eq!(find(&threads(&r), "threads-moearc").level, Level::Refuse);
    }

    #[test]
    fn offload_on_still_demands_the_pin_take() {
        let mut r = quiet_reading();
        r.host_policy = Some("frac:0.5".to_string());
        r.engine_threads = Some(ThreadPin { requested: 19, reported: Some(0) });
        assert_eq!(find(&threads(&r), "threads-moearc").level, Level::Refuse);
    }

    #[test]
    fn an_incumbent_that_reports_no_backend_is_refused() {
        let mut r = quiet_reading();
        r.incumbent = Some(IncumbentFacts {
            binary: "llama-bench".to_string(),
            build_commit: None,
            backends: None,
            threads: ThreadPin { requested: 16, reported: Some(16) },
            model_filename: None,
            device_selector: None,
        });
        assert_eq!(find(&backend(&r), "incumbent-backend").level, Level::Refuse);
    }

    #[test]
    fn a_binary_without_the_gpu_feature_cannot_take_absolutes() {
        let mut r = quiet_reading();
        r.gpu_compiled_in = false;
        assert_eq!(find(&backend(&r), "build-no-gpu").level, Level::Refuse);
    }

    #[test]
    fn a_device_on_the_wrong_backend_is_refused() {
        let mut r = quiet_reading();
        r.device.as_mut().unwrap().backend = "opencl".to_string();
        assert_eq!(find(&backend(&r), "device").level, Level::Refuse);
    }

    #[test]
    fn no_device_at_all_is_refused_rather_than_defaulted() {
        let mut r = quiet_reading();
        r.device = None;
        assert_eq!(find(&backend(&r), "device").level, Level::Refuse);
    }

    #[test]
    fn an_old_runtime_is_a_caution_with_its_measurement_never_a_refusal() {
        // 🔴 The rule is: refuse on the observable, caution on the version. 33578 detects the
        // card and then fails at model load — but that is one machine's measurement, and this
        // project has already been burned asserting a guessed value decoded from a field the
        // Level Zero spec gives no encoding for.
        let mut r = quiet_reading();
        r.device.as_mut().unwrap().driver_build = Some(33_578);
        let findings = backend(&r);
        let f = find(&findings, "runtime-build");
        assert_eq!(f.level, Level::Warn);
        assert!(f.detail.contains("33578"), "{}", f.detail);
        assert!(f.detail.contains("not a published minimum"), "{}", f.detail);
        // A whole run is still usable on an old runtime; it is qualified, not refused.
        assert_eq!(
            Verdict::of(&evaluate(&r, &Thresholds::default(), Intent::ABSOLUTES)),
            Verdict::Qualified
        );
    }

    #[test]
    fn the_runtime_build_is_reported_even_when_it_is_current() {
        // It goes in every artefact, not only failing ones: a result that does not name the
        // driver build cannot be compared with another machine's.
        let findings = backend(&quiet_reading());
        let f = find(&findings, "runtime-build");
        assert_eq!(f.level, Level::Pass);
        assert!(f.headline.contains("37020"), "{}", f.headline);
    }

    #[test]
    fn a_dirty_tree_qualifies_the_result() {
        let mut r = quiet_reading();
        r.build.dirty = Some(true);
        assert_eq!(build(&r).level, Level::Warn);
        r.build.commit = None;
        assert_eq!(build(&r).level, Level::Warn);
    }

    // --- the verdict ------------------------------------------------------------------

    #[test]
    fn a_quiet_machine_with_a_cacheable_model_is_trusted() {
        let f = evaluate(&quiet_reading(), &Thresholds::default(), Intent::ABSOLUTES);
        assert_eq!(Verdict::of(&f), Verdict::Trusted, "{f:#?}");
    }

    #[test]
    fn one_refusal_refuses_the_whole_run() {
        let mut r = quiet_reading();
        r.load1 = Some(21.69);
        let f = evaluate(&r, &Thresholds::default(), Intent::ABSOLUTES);
        assert_eq!(Verdict::of(&f), Verdict::Refused);
    }

    #[test]
    fn every_finding_names_the_rule_it_implements() {
        // The artefact's whole claim to being auditable is that a reader can go from a line of
        // output to the paragraph that justifies it.
        for f in evaluate(&quiet_reading(), &Thresholds::default(), Intent::ABSOLUTES) {
            assert!(f.rule.starts_with('§'), "{f:?}");
            assert!(!f.detail.is_empty(), "{f:?}");
        }
    }
}
