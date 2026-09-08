//! `moearc serve` — the resolved tuning profile, actually running.
//!
//! # The architecture, and why
//!
//! `moearc serve <model>` **supervises llama.cpp's own `llama-server` as a child process**,
//! launched with the argv that [`crate::tuning::resolve`] produced for this card, this CPU and
//! this file. It is deliberately not an in-process HTTP layer over
//! [`moearc_llama::runtime`], and the reasons are worth stating because the other choice is
//! defensible and `docs/llama-integration.md` §6 kept it open.
//!
//! * 🔴 **The command MoEArc prints is the command MoEArc runs.** `moearc info` renders
//!   `Resolved::command()` for a user to paste; this module builds its argv from the *same*
//!   [`Resolved`] by the same code path, so there is exactly one description of a
//!   configuration. An in-process server would be a second implementation of "what these
//!   settings mean", free to drift from the printed one — and a printed number that does not
//!   describe what ran is the specific failure this project has withdrawn results over three
//!   times.
//! * **The product is the tuning, not the HTTP.** What MoEArc knows that nobody else does is
//!   that this box wants `-t 16` and this model's `-ncmoe` floor is 36. Chat templates, SSE
//!   framing, `/v1/models` and OpenAI's error envelope are solved work; re-solving them buys
//!   the user nothing and costs a release.
//! * **It defers a decision that is not ours to force.** `docs/pivot-inventory.md` closes on
//!   two open questions, the first being *which tokenizer and which sampler is authoritative* —
//!   `moearc-server` has both, `moearc-llama` has both, and running two is a silent-divergence
//!   risk. Supervising `llama-server` means the tokenizer and sampler that generate a served
//!   token are the ones that produced the verified 40-of-40 token-id match against llama.cpp.
//!   Answering that question under release pressure is how a project acquires a bug it cannot
//!   see.
//! * **A crash stays a crash in the child.** `-ncmoe` one block too low is answered by ggml
//!   with an abort *after* loading tens of gigabytes. As a child that is an exit status this
//!   module can explain (see [`diagnose`]); in-process it is a `GGML_ABORT` inside our own
//!   address space, and the user gets a core dump instead of a sentence.
//!
//! ⚠️ **The packaging consequence, stated rather than discovered.** This needs a second
//! executable — but not one on `PATH`. `packaging/bundle.sh` already stages four executables
//! behind `launcher.sh`, and under this architecture the tarball must ship llama.cpp's whole
//! shared-object family (`libllama.so.0`, `libggml*.so.0`) regardless of which of the two
//! designs is chosen, because `moearc-llama` links them either way. `llama-server` is one more
//! ELF in a directory that already has to exist. What genuinely changes is that
//! `bundle.sh`'s single-`.so` wiring (lines 64/66/84) must become a list —
//! `docs/pivot-inventory.md` already flags that as required work independent of this module.
//! [`find_binary`] resolves the child next to our own executable **first**, so the packaged
//! layout never depends on the user's `PATH`.
//!
//! # 🔴 The iGPU is a live trap, and this module is where it is closed
//!
//! On the reference machine `llama-server --list-devices` reports:
//!
//! ```text
//!   SYCL0: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)
//!   SYCL1: Intel(R) Graphics (76029 MiB, 23270 MiB free)
//! ```
//!
//! `SYCL1` is the integrated GPU, and the 76,029 MiB it claims is **system RAM**, which is the
//! measured finding `moearc_device::fitness` exists to encode. Left to itself llama.cpp will
//! use it — with `-ngl 99` and the default layer split it will use *both* — and the run
//! succeeds and lies. So the device is chosen by **name**, from MoEArc's own detection, and
//! pinned with `-dev`; and then the child's own startup log is read back and the pin is
//! confirmed against the name before the model is allowed to load. Two independent enumerations
//! have to agree, or nothing runs.
//!
//! # Overrides re-plan; they never contradict
//!
//! 🔴 `-c` and `-ncmoe` come out of the same pool of VRAM, and raising one without the other
//! is how a run dies with `OUT_OF_DEVICE_MEMORY` after loading 59 GiB. Every override
//! therefore goes back through a planner rather than editing one flag in place:
//!
//! | override | what re-plans |
//! | --- | --- |
//! | `--ctx N` | passed *into* [`crate::tuning::resolve::resolve`], which recomputes the split for that depth. Refused first by [`crate::fit::plan`] if `N` is past the model's trained context. |
//! | `--moe-cache S` | re-planned through [`crate::fit::plan_with_slot_override`]; **both** `-ncmoe` and `-c` are taken from that one `Fit`, so the pair cannot disagree. |
//! | `--host-budget` | changes no llama.cpp flag, and says so on screen rather than being silently dropped. |

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::cli::{Cli, ServeArgs};
use crate::fit::{self, Fit, FitOutcome};
use crate::format;
use crate::source::{DeviceRow, ModelCard, Sources};
use crate::tuning::resolve::Resolved;
use crate::tuning::schema::Origin;

/// Environment variable naming the `llama-server` to run.
pub const LLAMA_SERVER_ENV: &str = "MOEARC_LLAMA_SERVER";

/// The verbosity `llama-server` needs before it will print the device table.
///
/// 🔴 Not a preference. At llama.cpp's default (3, "info") the `device_info:` block is not
/// emitted at all, and that block is the only place the child states which device it resolved
/// `-dev` to. Without it the pin above is asserted against nothing. The child's chatter is
/// captured rather than printed — see [`Supervisor`] — so this costs the user no output.
const CHILD_LOG_VERBOSITY: &str = "4";

/// Lines of the child's log kept for a crash report.
const LOG_RING: usize = 240;

/// How long to wait for `/health` before giving up on a load.
///
/// Generous because it has to cover a cold 59 GiB read from spinning-rust-backed ZFS. The
/// child is watched for death throughout, so a real failure is reported in seconds regardless.
const READY_TIMEOUT: Duration = Duration::from_secs(45 * 60);

// ---------------------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------------------

/// Where one flag's value came from.
///
/// 🔴 This exists so that a value MoEArc re-planned, or one the user pinned, can never be
/// rendered with the word `measured`. [`crate::tuning::schema::Setting`] guarantees that
/// property inside the tuning layer by having no constructor that omits an [`Origin`]; this
/// type extends it across the two things the tuning layer has no vocabulary for — an override
/// and a device pin — rather than borrowing [`Origin`] and quietly picking one of its four
/// words for something that is none of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Straight from the resolver, carrying the resolver's own verdict.
    Tuned(Origin),
    /// The resolver's value was recomputed, because something upstream of it moved. The string
    /// names what moved.
    Replanned(&'static str),
    /// Chosen by MoEArc itself, from something it measured on this machine rather than from a
    /// tuning profile. Today: the `-dev` pin.
    Chosen(&'static str),
}

impl Provenance {
    /// The word in the table's `provenance` column.
    pub fn label(self) -> &'static str {
        match self {
            Self::Tuned(o) => o.label(),
            Self::Replanned(_) => "replanned",
            Self::Chosen(_) => "selected",
        }
    }

    /// 🔴 True **only** for a value a profile measured on this card with this model. A
    /// re-planned or pinned value is not a measurement, whatever it was derived from.
    pub fn is_measured(self) -> bool {
        matches!(self, Self::Tuned(o) if o.is_measured())
    }

    /// The origin this counts as when summarising the weakest field on screen.
    fn as_origin(self) -> Origin {
        match self {
            Self::Tuned(o) => o,
            // Real arithmetic on this machine's own numbers, and nothing was run — which is
            // exactly what `Origin::Derived` means.
            Self::Replanned(_) | Self::Chosen(_) => Origin::Derived,
        }
    }
}

/// One argument of the child's command line, with the standing of the value in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedFlag {
    pub flag: String,
    pub value: String,
    pub provenance: Provenance,
    pub purpose: String,
}

// ---------------------------------------------------------------------------------------
// The engine binary
// ---------------------------------------------------------------------------------------

/// The `llama-server` this run will drive, and how it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binary {
    pub path: PathBuf,
    /// A clause for the screen. Printed always: which llama.cpp ran is half of what a result
    /// means, and `bench/PROTOCOL.md` §2 records a glob-ordered pick that silently selected a
    /// Vulkan build 4.8× slower than SYCL.
    pub how: &'static str,
}

/// Candidate locations, strongest first.
///
/// 🔴 **`PATH` is last and is not a glob.** The `build-vulkan/` directory on the reference
/// machine holds a `llama-server` that runs, answers correctly and is 4.8× slower; nothing
/// about the file name distinguishes it. Being last is not the safety mechanism — the backend
/// assertion in [`Supervisor::await_ready`] is — but a packaged install must never depend on
/// what happens to be earlier in a user's `PATH`, so the bundled copy wins by construction.
fn candidates() -> Vec<(PathBuf, &'static str)> {
    let mut out: Vec<(PathBuf, &'static str)> = Vec::new();

    if let Some(p) = std::env::var_os(LLAMA_SERVER_ENV) {
        out.push((PathBuf::from(p), "from $MOEARC_LLAMA_SERVER"));
    }
    // The packaged layout: bundle.sh stages every executable side by side under libexec/.
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push((dir.join("llama-server"), "shipped beside moearc"));
    }
    if let Some(p) = std::env::var_os("MOEARC_LLAMA_CPP_BUILD") {
        out.push((PathBuf::from(p).join("llama-server"), "from $MOEARC_LLAMA_CPP_BUILD"));
    }
    if let Some(p) = std::env::var_os("MOEARC_LLAMA_CPP_DIR") {
        out.push((PathBuf::from(p).join("build/bin/llama-server"), "from $MOEARC_LLAMA_CPP_DIR"));
    }
    // The development reference build. The same path `crates/moearc-llama/build.rs` looks in,
    // and for the same reason: the engine and the benchmark baseline must be one binary.
    out.push((
        PathBuf::from("/zfs/swift/projects/llama.cpp/build/bin/llama-server"),
        "the MoEArc reference build",
    ));

    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            out.push((dir.join("llama-server"), "found on $PATH"));
        }
    }
    out
}

/// Whether a path is a file we could execute.
fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

/// Resolve the engine binary, or say precisely what was looked for.
pub fn find_binary() -> Result<Binary> {
    let all = candidates();
    for (path, how) in &all {
        if is_executable(path) {
            return Ok(Binary { path: path.clone(), how });
        }
    }
    // An explicit setting that does not resolve is a typo, not an absence, and deserves to be
    // named rather than folded into "nothing found anywhere".
    if let Some(p) = std::env::var_os(LLAMA_SERVER_ENV) {
        bail!(
            "{LLAMA_SERVER_ENV} is set to {} and there is no executable there",
            PathBuf::from(p).display()
        );
    }
    bail!(
        "no `llama-server` to run. MoEArc drives llama.cpp rather than reimplementing it, so \
         the engine has to be on disk somewhere. Looked beside this binary, in \
         $MOEARC_LLAMA_CPP_BUILD, in $MOEARC_LLAMA_CPP_DIR/build/bin, in the reference build \
         directory and on $PATH. Set {LLAMA_SERVER_ENV} to point at one, and make sure it is a \
         SYCL build — a Vulkan or CPU build will be refused when it reports its devices."
    )
}

// ---------------------------------------------------------------------------------------
// Devices, as llama.cpp sees them
// ---------------------------------------------------------------------------------------

/// One row of llama.cpp's own device enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaDevice {
    /// `SYCL0`. The token `-dev` takes.
    pub id: String,
    pub name: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

impl LlamaDevice {
    /// Whether this is a device the SYCL backend enumerated, rather than the CPU row or a
    /// Vulkan/CUDA one from a build that is not ours.
    fn is_sycl(&self) -> bool {
        self.id.to_ascii_uppercase().starts_with("SYCL")
    }

    fn describe(&self) -> String {
        format!("{} = {} ({} MiB, {} MiB free)", self.id, self.name, self.total_mib, self.free_mib)
    }
}

/// `Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)` → name and the two figures.
///
/// Split out because the same tail appears in two different llama.cpp outputs whose *heads*
/// differ, and one parser for the shared half is one place to be wrong.
fn parse_device_body(id: &str, rest: &str) -> Option<LlamaDevice> {
    // `rfind`, not `find`: the device name itself contains `(R)` and `(TM)`.
    let open = rest.rfind(" (")?;
    let name = rest[..open].trim();
    let tail = rest[open + 2..].trim_end_matches(')');
    let mut parts = tail.split(',');
    let total = leading_u64(parts.next()?)?;
    let free = leading_u64(parts.next()?)?;
    if name.is_empty() || id.is_empty() {
        return None;
    }
    Some(LlamaDevice {
        id: id.to_string(),
        name: name.to_string(),
        total_mib: total,
        free_mib: free,
    })
}

/// The number at the start of `" 12216 MiB"`.
fn leading_u64(s: &str) -> Option<u64> {
    let t = s.trim();
    let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Parse `llama-server --list-devices`.
///
/// ```text
/// Available devices:
///   SYCL0: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)
/// ```
pub fn parse_list_devices(out: &str) -> Vec<LlamaDevice> {
    let mut devices = Vec::new();
    let mut inside = false;
    for line in out.lines() {
        let t = line.trim();
        if t.eq_ignore_ascii_case("available devices:") {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        let Some((id, rest)) = t.split_once(": ") else {
            // The list is contiguous; anything else ends it rather than being skipped over,
            // so a second unrelated section cannot contribute a phantom device.
            if !t.is_empty() {
                inside = false;
            }
            continue;
        };
        if let Some(d) = parse_device_body(id.trim(), rest) {
            devices.push(d);
        }
    }
    devices
}

/// Parse one row of the child's own `device_info:` block.
///
/// ```text
/// 0.00.104.673 I cmn  common_param:   - SYCL0   : Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)
/// ```
///
/// The log prefix is skipped by anchoring on `" - "` rather than by matching the prefix, which
/// carries a wall-clock-ish timestamp and a subsystem tag that are not ours to depend on.
pub fn parse_device_info_row(line: &str) -> Option<LlamaDevice> {
    let after_dash = line.split_once(" - ")?.1;
    let (id, rest) = after_dash.split_once(" : ")?;
    parse_device_body(id.trim(), rest)
}

/// Which enumerated device is the one MoEArc planned against.
///
/// 🔴 Matched by **name**, never by index. `docs/llama-integration.md` §5.2: `main_gpu` and
/// `-dev` index a list this program does not control, the iGPU is deliberately left enabled in
/// BIOS on the reference box, and index 0 is not contractually the discrete card.
pub fn select_device(
    devices: &[LlamaDevice],
    want: &DeviceRow,
) -> Result<(LlamaDevice, Option<String>)> {
    if devices.is_empty() {
        bail!(
            "`llama-server --list-devices` enumerated nothing. MoEArc found {} on this machine, \
             so the difference is in llama.cpp's runtime rather than in the hardware — most \
             often a SYCL runtime it cannot load.",
            want.name
        );
    }
    if !devices.iter().any(LlamaDevice::is_sycl) {
        bail!(
            "that `llama-server` reports no SYCL device — it enumerated {}. MoEArc targets \
             Intel Arc through SYCL; a Vulkan or CPU-only build would run and quietly serve \
             from the wrong backend, so it is refused here instead.",
            devices.iter().map(LlamaDevice::describe).collect::<Vec<_>>().join("; ")
        );
    }

    let matches: Vec<&LlamaDevice> = devices
        .iter()
        .filter(|d| d.is_sycl() && d.name.trim().eq_ignore_ascii_case(want.name.trim()))
        .collect();

    match matches.as_slice() {
        [] => bail!(
            "llama.cpp does not offer the device MoEArc planned against. MoEArc chose `{}`; \
             llama.cpp enumerated {}. Serving would fall onto whichever device it picked \
             instead — on this class of machine that is the integrated GPU, which reports \
             system RAM as video memory and therefore fits everything and is wrong.",
            want.name,
            devices.iter().map(LlamaDevice::describe).collect::<Vec<_>>().join("; ")
        ),
        [only] => Ok(((*only).clone(), None)),
        several => {
            // Two identical cards is a legitimate machine. Pick the one whose reported size is
            // closest to the device MoEArc measured, and say that a choice was made.
            let want_mib = want.total_bytes / (1024 * 1024);
            let best = several
                .iter()
                .min_by_key(|d| d.total_mib.abs_diff(want_mib))
                .expect("several is non-empty");
            let note = format!(
                "{} devices report the name `{}`; `{}` was pinned because its {} MiB is the \
                 closest match to the {} MiB MoEArc measured. Pin another with \
                 ONEAPI_DEVICE_SELECTOR.",
                several.len(),
                want.name,
                best.id,
                best.total_mib,
                want_mib
            );
            Ok(((*best).clone(), Some(note)))
        }
    }
}

/// Ask the engine binary what devices it can see.
fn list_devices(bin: &Path) -> Result<Vec<LlamaDevice>> {
    let out = Command::new(bin)
        .arg("--list-devices")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| anyhow!("could not run {} --list-devices: {e}", bin.display()))?;
    let text =
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let devices = parse_list_devices(&text);
    if devices.is_empty() && !out.status.success() {
        // The same explanation the supervised child gets. This is the *first* thing a user
        // meets on a machine whose SYCL runtime is not reachable, and an exit status with a
        // loader message under it reads as "MoEArc is broken" rather than "the runtime is not
        // on the library path".
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let hint = diagnose(&lines).map(|d| format!("\n\n  {d}")).unwrap_or_default();
        bail!(
            "{} --list-devices exited {} and listed nothing:\n{}{hint}",
            bin.display(),
            out.status,
            text.trim()
        );
    }
    Ok(devices)
}

// ---------------------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------------------

/// Everything decided before a process exists.
pub struct Plan {
    pub card: ModelCard,
    pub device: DeviceRow,
    pub model_path: PathBuf,
    pub resolved: Resolved,
    pub flags: Vec<PlannedFlag>,
    pub binary: Binary,
    pub llama_device: LlamaDevice,
    pub bind_host: String,
    pub port: u16,
    /// Overrides, clamps and anything else the user has to know was done to their request.
    pub notes: Vec<String>,
    /// The resolver's caveats, minus any that describe a value an override has replaced.
    pub caveats: Vec<String>,
    /// Fraction of the expert bank left on the card, recomputed when an override moved it.
    pub bank_resident: Option<f64>,
    /// Set when nothing can be run. The sentence is the planner's own.
    pub refusal: Option<String>,
    /// The residency plan, kept for `-v` and for `--moe-cache`'s re-plan.
    pub fit: Fit,
}

impl Plan {
    /// The child's argv, `[0]` being the binary.
    pub fn argv(&self) -> Vec<String> {
        let mut v = vec![self.binary.path.display().to_string()];
        v.push("-m".to_string());
        v.push(self.model_path.display().to_string());
        for f in &self.flags {
            v.push(f.flag.clone());
            v.push(f.value.clone());
        }
        v.extend(self.resolved.extra_args.iter().cloned());
        v.push("-lv".to_string());
        v.push(CHILD_LOG_VERBOSITY.to_string());
        v.push("--host".to_string());
        v.push(self.bind_host.clone());
        v.push("--port".to_string());
        v.push(self.port.to_string());
        v.push("--alias".to_string());
        v.push(self.card.id.clone());
        v
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}:{}/v1", self.bind_host, self.port)
    }

    /// The weakest standing among the flags actually being passed.
    pub fn weakest(&self) -> Origin {
        self.flags.iter().fold(Origin::Measured, |acc, f| acc.weakest(f.provenance.as_origin()))
    }

    fn json(&self, sources: &Sources) -> Value {
        json!({
            "model": self.card,
            "model_path": self.model_path,
            "device": self.device,
            "llama_device": { "id": self.llama_device.id, "name": self.llama_device.name },
            "engine": { "binary": self.binary.path, "found": self.binary.how },
            "endpoint": self.endpoint(),
            "tuning": self.resolved,
            "flags": self.flags.iter().map(|f| json!({
                "flag": f.flag,
                "value": f.value,
                "provenance": f.provenance.label(),
                "measured": f.provenance.is_measured(),
                "purpose": f.purpose,
            })).collect::<Vec<_>>(),
            "weakest_provenance": self.weakest().label(),
            "bank_resident": self.bank_resident,
            "caveats": self.caveats,
            "argv": self.argv(),
            "notes": self.notes,
            "refusal": self.refusal,
            "plan": self.fit,
            "stubbed": sources.stubbed,
        })
    }
}

/// llama.cpp's `-ncmoe N` offloads the experts of the **first N** MoE blocks, so a residency
/// expressed in slots becomes a block count.
///
/// 🔴 Rounded so that a partly-covered block counts as **not** resident. `-ncmoe` has no
/// half-block setting: leaving a block on the card that the slot budget only partly covers is
/// the direction that ends in `OUT_OF_DEVICE_MEMORY`, and the other direction merely gives up
/// a little throughput.
pub fn ncmoe_for_slots(resident_slots: u32, experts_per_block: u32, moe_blocks: u32) -> u32 {
    if experts_per_block == 0 {
        return moe_blocks;
    }
    let whole_blocks = (resident_slots / experts_per_block).min(moe_blocks);
    moe_blocks - whole_blocks
}

/// Fraction of the expert bank `-ncmoe n` leaves on the card.
///
/// The same arithmetic `tuning::resolve` does for its own figure, repeated here for the one
/// case it cannot answer: `--moe-cache` moves `-ncmoe` after the resolver has finished, and a
/// residency percentage describing the superseded value is worse than none.
pub fn bank_resident(card: &ModelCard, n_cpu_moe: u32) -> Option<f64> {
    if card.expert_slots_total == 0 {
        return None;
    }
    let on_gpu = card.moe_blocks.saturating_sub(n_cpu_moe) * card.experts_per_block;
    Some(f64::from(on_gpu) / f64::from(card.expert_slots_total))
}

/// Drop the resolver caveats that describe an `-ncmoe` an override has replaced.
///
/// 🔴 Two contradictory numbers on one screen is the failure this repository keeps naming, and
/// an override creates exactly that: the resolver explains the value it chose, in prose, with
/// the number in it. Every caveat it writes about the split names the flag, so naming the flag
/// is the filter — and [`build_plan`] pushes its own coupling sentence for the new pair in
/// their place, so the warning itself is never lost.
pub fn supersede_split_caveats(caveats: Vec<String>) -> Vec<String> {
    caveats.into_iter().filter(|c| !c.contains("-ncmoe")).collect()
}

/// Hold `-c` down to what the model was trained for.
///
/// 🔴 `crate::fit` already does this for the numbers it prints and states why: *olmoe-1b-7b is
/// a 4,096-token model and a B580 with 11.3 GiB free has room for 47,360 tokens of its KV
/// cache*, so the larger figure is a claim about the model that the model does not make. The
/// tuning resolver's own derivation applies that cap only on the all-experts-on-host path, so
/// a Q4_K model on a roomy card can arrive here with a `-c` past its training. Serving at that
/// length is worse than printing it: the pages allocate, and the answers past the trained
/// length do not mean anything.
///
/// Clamping is always **downward**, so it can only free VRAM — it can never turn a feasible
/// split into one that does not fit.
pub fn clamp_to_trained_context(planned: u32, trained: u32) -> Option<u32> {
    (trained > 0 && planned > trained).then_some(trained)
}

/// Work out everything, and run nothing.
fn build_plan(cli: &Cli, sources: &Sources, args: &ServeArgs) -> Result<Plan> {
    let card = sources.models.resolve(&args.model)?;
    let devices = sources.devices.detect()?;
    let Some(device) = devices.primary() else {
        bail!("{}", devices.verdict.headline());
    };

    let models_dir = crate::catalog::models_dir(cli.global.models_dir.as_deref());
    let file = card.file.clone().ok_or_else(|| {
        anyhow!("`{}` has no file on this machine — `moearc pull {}` first", card.id, card.id)
    })?;
    let model_path = models_dir.join(&file);
    if !model_path.is_file() {
        bail!(
            "{} is not a file — the catalogue listed it but it is not there now",
            model_path.display()
        );
    }

    let binary = find_binary()?;
    let enumerated = list_devices(&binary.path)?;
    let (llama_device, ambiguity) = select_device(&enumerated, device)?;

    let mut notes: Vec<String> = Vec::new();
    notes.extend(ambiguity);

    // The residency plan. With `--moe-cache` it is also the source of both overridden flags.
    let fit = match args.moe_cache {
        Some(slots) => fit::plan_with_slot_override(device, &card, args.ctx, slots),
        None => fit::plan(device, &card, args.ctx),
    };

    let profiles = crate::tuning::store::Store::load();
    let cpu = crate::host::cpu();
    let mut resolved =
        crate::tuning::resolve::resolve(&profiles, Some(device), &card, &cpu, args.ctx);

    // 🔴 The trained-context clamp, done by **re-planning through the resolver** rather than by
    // editing `-c` afterwards.
    //
    // `crate::fit` already caps the context it prints, and says why: *olmoe-1b-7b is a
    // 4,096-token model and a B580 with 11.3 GiB free has room for 47,360 tokens of its KV
    // cache*, so the larger figure is a claim about the model that the model does not make. The
    // resolver applies that cap only on its all-experts-on-host path, so a Q4_K model on a roomy
    // card arrives here with a `-c` past its training.
    //
    // Overwriting the flag here would leave the resolver's own prose quoting the superseded
    // number — two contradictory contexts on one screen, which is the failure this project keeps
    // writing rules about. Asking it again for the length we actually want gets a split, a `-c`
    // and a paragraph that agree, all out of its arithmetic rather than ours.
    let mut clamp_note = None;
    if args.ctx.is_none()
        && let Some(planned) = resolved.ctx_size.as_ref().map(|s| s.value)
        && let Some(capped) = clamp_to_trained_context(planned, card.trained_context_tokens)
    {
        resolved =
            crate::tuning::resolve::resolve(&profiles, Some(device), &card, &cpu, Some(capped));
        clamp_note = Some(format!(
            "this card has room for {} tokens of KV cache and this model was trained for {}, so \
             the split was re-planned at the smaller figure. The pages would allocate either \
             way; the answers past {} would not mean anything.",
            format::count(i64::from(planned)),
            format::count(i64::from(card.trained_context_tokens)),
            format::count(i64::from(card.trained_context_tokens)),
        ));
    }

    notes.extend(clamp_note);

    let mut refusal = match &fit.outcome {
        FitOutcome::DoesNotFit { headline, reason } => Some(format!("{headline} — {reason}")),
        FitOutcome::Fits { .. } => None,
    };

    let base = resolved.flags();
    if base.is_empty() && refusal.is_none() {
        refusal = Some(resolved.basis.sentence());
    }

    // --- overrides, each of which re-plans rather than editing one flag ------------------

    let mut ctx_override: Option<(u32, &'static str)> = None;
    let mut moe_override: Option<(u32, &'static str)> = None;

    if let (Some(slots), FitOutcome::Fits { resident_experts, context_tokens, .. }) =
        (args.moe_cache, &fit.outcome)
    {
        let ncmoe = ncmoe_for_slots(*resident_experts, card.experts_per_block, card.moe_blocks);
        moe_override = Some((ncmoe, "--moe-cache"));
        ctx_override = Some((*context_tokens, "--moe-cache"));
        notes.push(format!(
            "🔴 `--moe-cache {slots}` pinned residency, so the split was re-planned from it: \
             {} of {} slots stay on the card, which is `-ncmoe {ncmoe}`, and `-c {}` is what \
             the same plan has room for beside them. Both flags come out of that one \
             calculation — raising context without moving the split is how a run dies with \
             OUT_OF_DEVICE_MEMORY after loading the whole model.",
            format::count(*resident_experts as i64),
            format::count(card.expert_slots_total as i64),
            format::count(*context_tokens as i64),
        ));
    }

    // `--host-budget` reaches no llama.cpp flag. Said, rather than silently dropped.
    if cli.host_budget().is_some() {
        notes.push(
            "`--host-budget` describes this run rather than steering it: llama.cpp memory-maps \
             the weights and the kernel decides what stays in the page cache, so there is no \
             flag to carry the budget into. It still sets the host tier printed above."
                .to_string(),
        );
    }

    let mut flags: Vec<PlannedFlag> = base
        .into_iter()
        .map(|f| {
            let (value, provenance) = match (f.flag, ctx_override, moe_override) {
                ("-c", Some((v, by)), _) => (v.to_string(), Provenance::Replanned(by)),
                ("-ncmoe", _, Some((v, by))) => (v.to_string(), Provenance::Replanned(by)),
                _ => (f.value, Provenance::Tuned(f.origin)),
            };
            PlannedFlag {
                flag: f.flag.to_string(),
                value,
                provenance,
                purpose: f.purpose.to_string(),
            }
        })
        .collect();

    // 🔴 The device pin. Last in the table because it is last on the command line, and present
    // even though no profile has an opinion about it: it is the flag that stops llama.cpp
    // choosing the integrated GPU.
    flags.push(PlannedFlag {
        flag: "-dev".to_string(),
        value: llama_device.id.clone(),
        provenance: Provenance::Chosen("MoEArc's own device detection"),
        purpose: "the card to run on, pinned by name".to_string(),
    });

    // 🔴 An override that moved the split invalidates the resolver's prose about it, and
    // leaving both on screen puts two `-ncmoe` values in front of the user. The replacement
    // sentence pushed into `notes` above carries the same warning for the new pair.
    let caveats = match moe_override {
        Some(_) => supersede_split_caveats(resolved.caveats.clone()),
        None => resolved.caveats.clone(),
    };
    let bank_resident = match moe_override {
        Some((ncmoe, _)) => bank_resident(&card, ncmoe),
        None => resolved.bank_resident,
    };

    Ok(Plan {
        card,
        device: device.clone(),
        model_path,
        resolved,
        flags,
        binary,
        llama_device,
        bind_host: args.host.clone(),
        port: args.port,
        notes,
        caveats,
        bank_resident,
        refusal,
        fit,
    })
}

// ---------------------------------------------------------------------------------------
// The start screen
// ---------------------------------------------------------------------------------------

fn section(title: &str) {
    println!();
    println!("{title}");
    println!();
}

/// What was chosen, and how much of it was measured — on one screen, before anything starts.
fn render(plan: &Plan, cli: &Cli, sources: &Sources) {
    let r = &plan.resolved;
    section("Serve");
    println!("  {:<18}{}", "model", plan.card.id);
    println!(
        "  {:<18}{} · {} · {}",
        "",
        plan.model_path.display(),
        format::bytes(plan.card.file_bytes),
        plan.card.quant
    );
    println!(
        "  {:<18}{} · {} free",
        "device",
        plan.device.name,
        format::bytes(plan.device.free_bytes)
    );
    println!("  {:<18}{}", "endpoint", plan.endpoint());
    println!("  {:<18}{}", "engine", plan.binary.path.display());
    println!("  {:<18}llama.cpp `llama-server`, {}", "", plan.binary.how);

    // 🔴 The confidence line before the settings, the same order `moearc info` uses: a reader
    // who takes in one line must take in the one that says whether any of this was measured.
    println!();
    println!("  {:<18}{} {}", "confidence", r.origin.glyph(), r.basis.label().to_uppercase());
    println!("  {:<18}{}", "", r.basis.sentence());

    if let Some(reason) = &plan.refusal {
        println!();
        println!("  ✗ {reason}");
        return;
    }

    println!();
    println!("  {:<16}{:<10}{:<14}what it does", "flag", "value", "provenance");
    for f in &plan.flags {
        println!("  {:<16}{:<10}{:<14}{}", f.flag, f.value, f.provenance.label(), f.purpose);
    }

    // 🔴 Restated as a claim, not left as a column. A reader who skims the table and acts on it
    // must not be able to miss that some of it was never run.
    let weakest = plan.weakest();
    println!();
    if weakest.is_measured() {
        println!("  every setting above was measured on this card with this model.");
    } else {
        println!(
            "  🔴 not every setting above was measured — the weakest is {}. This is a starting \
             point; measure from it with `moearc bench`.",
            weakest.label()
        );
    }
    println!("  {}", Origin::legend());

    if let Some(fraction) = plan.bank_resident {
        println!();
        println!(
            "  {:<18}{:.1}% of the expert bank stays on the card",
            "residency",
            fraction * 100.0
        );
        if let Some(c) = r.coverage {
            println!(
                "  {:<18}{:.0}% of expert touches on prose, {:.0}% on code — measured on this \
                 model's own routing trace. It predicts staged bytes, not tok/s.",
                "coverage",
                c.prose * 100.0,
                c.code * 100.0
            );
        }
    }

    println!();
    println!("  Command");
    println!("    {}", shell_line(&plan.argv()));
    println!(
        "    the trailing `-lv/--host/--port/--alias` are MoEArc's, not tuning: the log level \
         is what makes the device confirmation below possible."
    );

    if !plan.notes.is_empty() {
        println!();
        println!("  Overrides and adjustments");
        for n in &plan.notes {
            println!("  · {n}");
        }
    }

    if !plan.caveats.is_empty() {
        println!();
        println!("  Read this before you trust it");
        for c in &plan.caveats {
            println!("  · {c}");
        }
    }

    if cli.global.verbose >= 1 {
        println!();
        crate::plain::print_plan(&plan.fit, &plan.device, cli.global.verbose);
    }

    if sources.stubbed {
        println!();
        println!("  note: {}.", sources.stub_note);
    }
}

/// The argv as a line someone could paste.
fn shell_line(argv: &[String]) -> String {
    argv.iter().map(|a| shell_quote(a)).collect::<Vec<_>>().join(" ")
}

fn shell_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "._-/=:+@,".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

// ---------------------------------------------------------------------------------------
// Supervision
// ---------------------------------------------------------------------------------------

/// A child that is killed if we leave by any path, including `?`.
struct Supervisor {
    child: Child,
    lines: Receiver<String>,
    ring: VecDeque<String>,
    /// Every device row the child printed in its own `device_info:` block.
    seen: Vec<LlamaDevice>,
    verbose: u8,
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        // Ordinary exit has already reaped it and this is a no-op. The case that matters is an
        // error between spawn and wait: without this the model stays loaded, the port stays
        // bound, and the user's next attempt fails for a reason that is not the real one.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 🔴 Ask the kernel to kill the engine when this process dies — **whatever** killed it.
///
/// [`Drop`] above covers the paths this program takes on purpose, and an interactive `Ctrl-C`
/// is covered by the terminal delivering `SIGINT` to the whole foreground process group. Both
/// were true, and both were insufficient. Measured on the reference machine: `kill -TERM` on a
/// running `moearc serve` left `llama-server` reparented to init, still holding the model in
/// VRAM and still bound to the port — because a default-disposition `SIGTERM` terminates
/// without unwinding, so no destructor runs. `SIGKILL` and a segfault have the same shape and
/// no handler could catch them either.
///
/// `PR_SET_PDEATHSIG` is set in the child between `fork` and `exec`, so it is the kernel that
/// makes the guarantee rather than any code of ours that has to still be running.
///
/// ⚠️ Two limits, stated rather than glossed: the signal fires when the *thread* that forked
/// exits (fine here — the spawn happens on the main thread, and this process exits with it),
/// and it is Linux-only, which matches everything else about a SYCL-on-Arc target.
#[cfg(target_os = "linux")]
fn set_death_signal(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;

    let supervisor = std::process::id();
    // SAFETY: the closure runs in the forked child before `exec`, where only
    // async-signal-safe calls are permitted. `prctl`, `getppid` and `_exit` are each a single
    // syscall; nothing here allocates, locks or touches the Rust runtime.
    unsafe {
        cmd.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // The window between `fork` and the `prctl` above is real: if the supervisor died
            // inside it, the death signal was already sent to nobody and would never arrive.
            // Checking the parent afterwards closes it.
            if libc::getppid() != supervisor as libc::pid_t {
                libc::_exit(1);
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn set_death_signal(_cmd: &mut Command) {}

impl Supervisor {
    fn spawn(plan: &Plan, verbose: u8) -> Result<Self> {
        let argv = plan.argv();
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]).stdin(Stdio::null()).stdout(Stdio::inherit()).stderr(Stdio::piped());
        set_death_signal(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("could not start {}: {e}", plan.binary.path.display()))?;

        let stderr = child.stderr.take().expect("stderr was piped");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            child,
            lines: rx,
            ring: VecDeque::with_capacity(LOG_RING),
            seen: Vec::new(),
            verbose,
        })
    }

    /// Take whatever the child has said since the last call.
    fn drain(&mut self, block_for: Duration) {
        let deadline = Instant::now() + block_for;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) => {
                    if let Some(d) = parse_device_info_row(&line)
                        && !self.seen.iter().any(|s| s.id == d.id)
                    {
                        self.seen.push(d);
                    }
                    if self.verbose >= 1 {
                        eprintln!("{line}");
                    }
                    if self.ring.len() == LOG_RING {
                        self.ring.pop_front();
                    }
                    self.ring.push_back(line);
                }
                Err(RecvTimeoutError::Timeout) => return,
                // The reader thread is gone, which means the child closed stderr. Nothing more
                // will arrive; the caller's `try_wait` is what notices.
                Err(RecvTimeoutError::Disconnected) => {
                    // The child closed stderr, so nothing more is coming. Sleep out the slice
                    // the caller asked for rather than returning instantly, or the supervision
                    // loop becomes a spin on `try_wait`.
                    std::thread::sleep(left);
                    return;
                }
            }
        }
    }

    /// 🔴 Confirm, from the child's own mouth, that the pin landed on the card MoEArc planned
    /// against — before a byte of the model is read.
    fn confirm_device(&self, plan: &Plan) -> Result<()> {
        let Some(row) = self.seen.iter().find(|d| d.id == plan.llama_device.id) else {
            return Ok(()); // not printed yet
        };
        if row.name.trim().eq_ignore_ascii_case(plan.device.name.trim()) {
            return Ok(());
        }
        bail!(
            "the engine resolved `-dev {}` to `{}`, and MoEArc planned against `{}`. Two \
             enumerations of the same machine disagreed, so the split above was computed for a \
             different device than the one about to run it. Refusing before the model loads. \
             Pin the card explicitly with ONEAPI_DEVICE_SELECTOR and try again.",
            plan.llama_device.id,
            row.name,
            plan.device.name
        );
    }

    fn tail(&self, n: usize) -> Vec<String> {
        self.ring.iter().skip(self.ring.len().saturating_sub(n)).cloned().collect()
    }
}

/// What to say when the child dies.
///
/// 🔴 The `OUT_OF_DEVICE_MEMORY` case is the one this project has been warning about in prose
/// for weeks. It is the failure a user meets after waiting for tens of gigabytes to load, and
/// meeting it with an exit status and a wall of ggml output is how a person concludes the tool
/// does not work.
pub fn diagnose(log: &[String]) -> Option<String> {
    let hay = log.join("\n").to_ascii_lowercase();
    if hay.contains("out_of_device_memory")
        || hay.contains("out of device memory")
        || hay.contains("failed to allocate")
    {
        return Some(
            "the card ran out of memory while allocating. 🔴 `-c` and `-ncmoe` come out of the \
             same pool: more context means fewer expert slots, and one block too few on the \
             host is exactly this. Ask for a shorter context with `--ctx` — the split is \
             recomputed for it — or pin residency lower with `--moe-cache`. Something else on \
             the machine holding VRAM has the same effect; `moearc` with no arguments reports \
             what is free right now."
                .to_string(),
        );
    }
    if hay.contains("error while loading shared libraries") || hay.contains("libsycl") {
        return Some(
            "the engine could not load its SYCL runtime. That is a packaging or driver \
             problem rather than a tuning one: the Intel runtime has to be reachable by the \
             dynamic loader before llama.cpp's SYCL backend can start."
                .to_string(),
        );
    }
    None
}

/// Poll `GET /health` until the server answers 200, the child dies, or the deadline passes.
fn await_ready(sup: &mut Supervisor, plan: &Plan) -> Result<Duration> {
    let addr = resolve_addr(&plan.bind_host, plan.port)?;
    let started = Instant::now();
    let mut confirmed = false;
    let mut said_at = Duration::ZERO;

    loop {
        sup.drain(Duration::from_millis(250));

        // The device check first, and every pass until it succeeds: it becomes answerable long
        // before /health does, and refusing early is the whole point of it.
        if !confirmed {
            sup.confirm_device(plan)?;
            if let Some(row) = sup.seen.iter().find(|d| d.id == plan.llama_device.id) {
                confirmed = true;
                println!("  ✓ device confirmed — {}", row.describe());
                if sup.seen.len() > 1 {
                    println!(
                        "    ({} other device(s) were visible and are not being used: {})",
                        sup.seen.len() - 1,
                        sup.seen
                            .iter()
                            .filter(|d| d.id != plan.llama_device.id)
                            .map(|d| d.describe())
                            .collect::<Vec<_>>()
                            .join("; ")
                    );
                }
            }
        }

        if let Some(status) = sup.child.try_wait()? {
            let tail = sup.tail(30);
            let mut msg = format!(
                "{} exited {} before it began serving.",
                plan.binary.path.display(),
                status
            );
            if let Some(d) = diagnose(&sup.tail(LOG_RING)) {
                msg.push_str("\n\n  ");
                msg.push_str(&d);
            }
            msg.push_str("\n\n  the engine's last words:\n");
            for line in tail {
                msg.push_str("    ");
                msg.push_str(&line);
                msg.push('\n');
            }
            bail!("{msg}");
        }

        if health_ok(addr) {
            return Ok(started.elapsed());
        }

        let waited = started.elapsed();
        if waited > READY_TIMEOUT {
            bail!(
                "the engine has not answered /health after {}. It is still running; stopping it.",
                format::duration(waited.as_secs())
            );
        }
        // A progress line rather than a spinner: this path is also what a pipe and a CI job
        // see, and a carriage-returning spinner in a log file is unreadable.
        if waited.as_secs() / 15 > said_at.as_secs() / 15 {
            println!("  … loading ({})", format::duration(waited.as_secs()));
            let _ = std::io::stdout().flush();
        }
        said_at = waited;
    }
}

/// `0.0.0.0` is a bind address, not somewhere to send a request to.
fn resolve_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let dial = match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    (dial, port)
        .to_socket_addrs()
        .map_err(|e| anyhow!("cannot resolve {dial}:{port}: {e}"))?
        .next()
        .ok_or_else(|| anyhow!("{dial}:{port} resolved to no address"))
}

/// One `GET /health`, hand-rolled.
///
/// A hand-written request rather than an HTTP client because this is the only HTTP this binary
/// speaks, and the dependency would be several hundred kilobytes and a TLS stack to ask a
/// loopback socket one question.
fn health_ok(addr: SocketAddr) -> bool {
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let req = format!(
        "GET /health HTTP/1.1\r\nHost: {addr}\r\nUser-Agent: moearc\r\nConnection: close\r\n\r\n"
    );
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let Ok(n) = s.read(&mut buf) else { return false };
    std::str::from_utf8(&buf[..n]).is_ok_and(|head| head.starts_with("HTTP/1.1 200"))
}

// ---------------------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------------------

pub fn run(cli: &Cli, sources: &Sources, args: &ServeArgs) -> Result<ExitCode> {
    let plan = build_plan(cli, sources, args)?;

    if cli.global.json && (args.dry_run || plan.refusal.is_some()) {
        println!("{}", serde_json::to_string_pretty(&plan.json(sources))?);
        return Ok(if plan.refusal.is_some() { ExitCode::FAILURE } else { ExitCode::SUCCESS });
    }
    if !cli.global.json {
        render(&plan, cli, sources);
    }

    if plan.refusal.is_some() {
        // Already rendered above in the plain path, and a field in the JSON one.
        return Ok(ExitCode::FAILURE);
    }
    if args.dry_run {
        println!();
        println!("  --dry-run: nothing was started.");
        return Ok(ExitCode::SUCCESS);
    }

    println!();
    println!("  starting the engine…");
    let mut sup = Supervisor::spawn(&plan, cli.global.verbose)?;
    let took = await_ready(&mut sup, &plan)?;

    if cli.global.json {
        let mut v = plan.json(sources);
        if let Some(o) = v.as_object_mut() {
            o.insert("ready".into(), json!(true));
            o.insert("load_seconds".into(), json!(took.as_secs_f64()));
            o.insert("pid".into(), json!(sup.child.id()));
        }
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!("  ✓ model loaded in {}", format::duration(took.as_secs()));
        println!("  ✓ listening on {}", plan.endpoint());
        println!();
        println!("  try it:");
        println!(
            "    curl {}/chat/completions -H 'Content-Type: application/json' \\\n      \
             -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'",
            plan.endpoint(),
            plan.card.id
        );
        println!();
        println!("  Ctrl-C to stop.");
    }
    let _ = std::io::stdout().flush();

    // Serve. The child owns the terminal's process group with us, so an interactive Ctrl-C
    // reaches it directly; this loop exists to keep draining its log (an unread pipe fills and
    // blocks the writer) and to report how it ended.
    let status = loop {
        sup.drain(Duration::from_millis(500));
        if let Some(status) = sup.child.try_wait()? {
            break status;
        }
    };

    if status.success() {
        println!();
        println!("  the engine stopped.");
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!();
    eprintln!("moearc: the engine exited {status}.");
    if let Some(d) = diagnose(&sup.tail(LOG_RING)) {
        eprintln!("  {d}");
    }
    for line in sup.tail(20) {
        eprintln!("    {line}");
    }
    Ok(ExitCode::FAILURE)
}

// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact line `llama-server --list-devices` printed on the reference machine.
    const LIST_OUTPUT: &str = "\
Available devices:
  SYCL0: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)
  SYCL1: Intel(R) Graphics (76029 MiB, 23270 MiB free)
";

    /// The exact rows the child logs at `-lv 4`, prefix and all.
    const INFO_ROWS: &str = "\
0.00.104.582 I cmn  common_param: device_info:
0.00.104.673 I cmn  common_param:   - SYCL0   : Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)
0.00.104.676 I cmn  common_param:   - CPU     : Intel(R) Core(TM) Ultra 7 265K (93705 MiB, 93705 MiB free)
";

    fn device(name: &str, total: u64) -> DeviceRow {
        DeviceRow {
            name: name.to_string(),
            backend: crate::source::Backend::LevelZero,
            driver: "xe / L0 build 37020".to_string(),
            total_bytes: total,
            free_bytes: total,
            budget_source: Some("measured free VRAM"),
            unusable: None,
            driver_build: Some(37_020),
        }
    }

    #[test]
    fn the_reference_device_list_parses_verbatim() {
        let d = parse_list_devices(LIST_OUTPUT);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].id, "SYCL0");
        assert_eq!(d[0].name, "Intel(R) Arc(TM) B580 Graphics");
        assert_eq!(d[0].total_mib, 12216);
        assert_eq!(d[0].free_mib, 11959);
        assert_eq!(d[1].name, "Intel(R) Graphics");
    }

    #[test]
    fn a_logged_device_row_parses_through_its_prefix() {
        let rows: Vec<LlamaDevice> = INFO_ROWS.lines().filter_map(parse_device_info_row).collect();
        assert_eq!(rows.len(), 2, "the header line must not parse as a device");
        assert_eq!(rows[0].id, "SYCL0");
        // The parenthesised vendor marks in the name must survive: `rfind`, not `find`.
        assert_eq!(rows[0].name, "Intel(R) Arc(TM) B580 Graphics");
        assert_eq!(rows[1].id, "CPU");
    }

    #[test]
    fn the_integrated_gpu_is_never_selected_by_position() {
        // 🔴 The release-blocking case. The iGPU is present, enumerated, and reports six times
        // the memory of the card; nothing but the name distinguishes the right answer.
        let devices = parse_list_devices(LIST_OUTPUT);
        let want = device("Intel(R) Arc(TM) B580 Graphics", 12_168_933_376);
        let (picked, note) = select_device(&devices, &want).unwrap();
        assert_eq!(picked.id, "SYCL0");
        assert!(note.is_none());
    }

    #[test]
    fn a_device_llama_cannot_see_is_a_refusal_not_a_fallback() {
        let devices = parse_list_devices(LIST_OUTPUT);
        let want = device("Intel(R) Arc(TM) A770 Graphics", 16 << 30);
        let err = select_device(&devices, &want).unwrap_err().to_string();
        assert!(err.contains("A770"), "the refusal names what MoEArc chose: {err}");
        assert!(err.contains("integrated"), "and why falling through is unsafe: {err}");
    }

    #[test]
    fn a_build_with_no_sycl_device_is_refused() {
        let devices = parse_list_devices(
            "Available devices:\n  Vulkan0: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 11959 MiB free)\n",
        );
        let want = device("Intel(R) Arc(TM) B580 Graphics", 12_168_933_376);
        let err = select_device(&devices, &want).unwrap_err().to_string();
        assert!(err.contains("no SYCL device"), "{err}");
    }

    #[test]
    fn two_identical_cards_pick_by_size_and_say_so() {
        let devices = parse_list_devices(
            "Available devices:\n  SYCL0: Arc B580 (12216 MiB, 11959 MiB free)\n  \
             SYCL1: Arc B580 (24432 MiB, 24000 MiB free)\n",
        );
        let want = device("Arc B580", 24432 * 1024 * 1024);
        let (picked, note) = select_device(&devices, &want).unwrap();
        assert_eq!(picked.id, "SYCL1");
        assert!(note.unwrap().contains("closest match"));
    }

    #[test]
    fn slots_become_whole_blocks_and_round_toward_the_host() {
        // 4,608 slots over 36 blocks of 128. A budget covering 4.5 blocks keeps 4.
        assert_eq!(ncmoe_for_slots(4 * 128 + 64, 128, 36), 32);
        assert_eq!(ncmoe_for_slots(0, 128, 36), 36);
        assert_eq!(ncmoe_for_slots(36 * 128, 128, 36), 0);
        // More slots than the model has cannot produce a negative block count.
        assert_eq!(ncmoe_for_slots(u32::MAX, 128, 36), 0);
        // A model with no expert geometry to speak of leaves everything on the host.
        assert_eq!(ncmoe_for_slots(100, 0, 36), 36);
    }

    #[test]
    fn context_is_clamped_down_to_the_trained_length_and_never_up() {
        // The measured case: olmoe is a 4,096-token model and the B580 has room for 47,360.
        assert_eq!(clamp_to_trained_context(47_360, 4_096), Some(4_096));
        assert_eq!(clamp_to_trained_context(2_048, 4_096), None, "never raised");
        assert_eq!(clamp_to_trained_context(4_096, 4_096), None);
        // A model whose header does not say is left alone rather than clamped to zero.
        assert_eq!(clamp_to_trained_context(47_360, 0), None);
    }

    #[test]
    fn an_override_takes_the_resolver_s_prose_about_the_old_split_with_it() {
        // 🔴 Otherwise the screen carries two different `-ncmoe` values, one of them in a
        // paragraph explaining why it is right.
        let caveats = vec![
            "🔴 `-ncmoe 0` is the floor for the `-c 47,360` printed beside it".to_string(),
            "`-ngl 99` means every layer".to_string(),
            "this file quantises some blocks differently".to_string(),
        ];
        let kept = supersede_split_caveats(caveats);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|c| !c.contains("47,360")));
        assert!(kept.iter().any(|c| c.contains("-ngl 99")), "unrelated caveats survive");
    }

    #[test]
    fn residency_follows_the_split_that_will_actually_run() {
        let mut card = crate::source::testing::card("m", "mxfp4");
        card.moe_blocks = 36;
        card.experts_per_block = 128;
        card.expert_slots_total = 36 * 128;
        assert_eq!(bank_resident(&card, 36), Some(0.0), "every block on the host");
        assert_eq!(bank_resident(&card, 0), Some(1.0), "every block on the card");
        assert_eq!(bank_resident(&card, 18), Some(0.5));
        card.expert_slots_total = 0;
        assert_eq!(bank_resident(&card, 0), None, "a model with no bank has no fraction");
    }

    #[test]
    fn nothing_but_a_measured_profile_may_read_measured() {
        // 🔴 The whole-surface guard. `Provenance` has three shapes and only one of them can
        // ever produce the word, whatever it was computed from.
        assert!(Provenance::Tuned(Origin::Measured).is_measured());
        assert_eq!(Provenance::Tuned(Origin::Measured).label(), "measured");
        for p in [
            Provenance::Tuned(Origin::Extrapolated),
            Provenance::Tuned(Origin::Derived),
            Provenance::Tuned(Origin::Untuned),
            Provenance::Replanned("--moe-cache"),
            Provenance::Replanned("the model's trained context"),
            Provenance::Chosen("MoEArc's own device detection"),
        ] {
            assert!(!p.is_measured(), "{p:?} must not read as measured");
            assert_ne!(p.label(), "measured", "{p:?} must not be labelled measured");
        }
    }

    #[test]
    fn a_replanned_flag_weakens_the_summary() {
        // An override cannot leave the screen claiming the configuration was measured.
        assert_eq!(Provenance::Replanned("--moe-cache").as_origin(), Origin::Derived);
        assert_eq!(Provenance::Chosen("x").as_origin(), Origin::Derived);
        assert_eq!(Provenance::Tuned(Origin::Measured).as_origin(), Origin::Measured);
    }

    #[test]
    fn the_out_of_memory_death_is_explained_in_terms_of_the_two_flags() {
        let log = vec![
            "ggml_backend_sycl_buffer_type_alloc_buffer: failed to allocate 1024.00 MiB".into(),
            "UR error: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY".into(),
        ];
        let d = diagnose(&log).expect("this is the failure users actually hit");
        assert!(d.contains("-ncmoe") && d.contains("-c"), "{d}");
        assert!(d.contains("--ctx"), "and it must name the flag that fixes it: {d}");
        assert!(diagnose(&["all good".to_string()]).is_none());
    }

    #[test]
    fn a_loopback_dial_address_is_used_for_a_wildcard_bind() {
        // Polling 0.0.0.0 is not an error on Linux and is not a health check either.
        assert_eq!(resolve_addr("0.0.0.0", 8080).unwrap().to_string(), "127.0.0.1:8080");
        assert_eq!(resolve_addr("127.0.0.1", 8080).unwrap().to_string(), "127.0.0.1:8080");
    }

    #[test]
    fn a_pasteable_command_quotes_only_what_needs_it() {
        let argv: Vec<String> = ["llama-server", "-m", "/models/a b.gguf", "-ncmoe", "36"]
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(shell_line(&argv), "llama-server -m '/models/a b.gguf' -ncmoe 36");
    }

    #[test]
    fn the_binary_is_never_resolved_by_globbing_a_build_directory() {
        // PROTOCOL §2: a glob-ordered pick once selected a Vulkan build 4.8x slower than SYCL.
        // Every candidate is a fixed file name under a named directory.
        for (path, _) in candidates() {
            assert_eq!(
                path.file_name().and_then(|s| s.to_str()),
                Some("llama-server"),
                "{} is not an exact file name",
                path.display()
            );
        }
    }

    #[test]
    fn path_is_the_last_place_looked() {
        // 🔴 A packaged install must resolve the engine it shipped with, never whichever
        // llama-server happens to be earlier in the user's PATH. Read from the real
        // environment rather than set here: mutating PATH inside a test would be visible to
        // every other test in the process.
        let c = candidates();
        let first_path = c.iter().position(|(_, how)| *how == "found on $PATH");
        let last_fixed = c.iter().rposition(|(_, how)| *how != "found on $PATH");
        if let (Some(p), Some(f)) = (first_path, last_fixed) {
            assert!(f < p, "every fixed location must be tried before PATH");
        }
        assert!(
            c.iter().any(|(_, how)| *how == "shipped beside moearc"),
            "the packaged layout has to be a candidate at all"
        );
    }
}
