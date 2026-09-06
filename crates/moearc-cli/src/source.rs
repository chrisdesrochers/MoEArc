//! What the interface needs to know, and where it comes from.
//!
//! Device enumeration lives in `moearc-device` and model metadata in `moearc-model`. Neither
//! is depended on here. Everything this crate renders arrives through one of the traits
//! below, with a fixture implementation behind it.
//!
//! That is not only a scheduling convenience. A UI wired directly to hardware cannot be
//! snapshot-tested — the frames would change with the card, the driver and the free VRAM at
//! the moment the test ran. Going through a trait makes the screens deterministic, which is
//! what lets `view.rs` assert on rendered text at all.
//!
//! **Swap points.** Replacing the fixtures is a change to [`Sources::stub`] and to the four
//! `Stub*` types; no screen, reducer or formatter refers to them. The trait signatures are
//! the contract the real crates have to meet.

use serde::Serialize;

use crate::format;

// ---------------------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------------------

/// Which SYCL backend enumerated a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    LevelZero,
    /// An Arc card also enumerates through OpenCL. The fixture does not produce one, so the
    /// compiler cannot see this constructed from inside this crate — it is part of the
    /// contract `moearc-device` fills, not dead code.
    #[allow(dead_code)]
    OpenCl,
    Cpu,
}

impl Backend {
    /// The string `sycl-ls` uses, so a user can match what they see here to what they see
    /// there without translating.
    pub fn label(self) -> &'static str {
        match self {
            Self::LevelZero => "level_zero",
            Self::OpenCl => "opencl",
            Self::Cpu => "cpu",
        }
    }

    /// Whether we will schedule inference onto it. Level Zero only: an Arc card also
    /// enumerates through OpenCL, and treating that as a second usable device would double
    /// count the same VRAM.
    pub fn is_inference_target(self) -> bool {
        matches!(self, Self::LevelZero)
    }
}

/// One row of the device report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceRow {
    pub name: String,
    pub backend: Backend,
    /// Kernel driver and runtime version as a single field, so it can be pasted into a bug
    /// report whole. Splitting it into columns invites the user to reason about which half
    /// matters, and neither does on its own.
    pub driver: String,
    pub total_bytes: u64,
    /// Free bytes *now*. Every plan is built from this, never from `total_bytes` — a card
    /// with a compositor on it does not hand over its whole framebuffer.
    pub free_bytes: u64,
}

impl DeviceRow {
    pub fn is_inference_target(&self) -> bool {
        self.backend.is_inference_target()
    }
}

/// Why the tool found no GPU. "No GPU found" alone is precisely the message `docs/ux.md`
/// rules out, so the absence always carries a named cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuOnlyCause {
    /// The known-bad configuration recorded in `docs/ux.md`: the oneAPI environment was
    /// never sourced, so the Level Zero loader is invisible. The stack is intact and looks
    /// dead. This cost the project real time before the tool existed, which is exactly why
    /// it is a named variant rather than a guess in prose.
    RuntimeNotSourced,
    /// The kernel-side GPU driver is absent. Per `docs/ux.md` this is the one dependency we
    /// are allowed to ask the user for, so it is named exactly and nothing is listed beside
    /// it.
    ///
    /// Telling this apart from [`Self::RuntimeNotSourced`] requires looking at the kernel,
    /// which is `moearc-device`'s job; the fixture cannot produce it.
    #[allow(dead_code)]
    KernelDriverMissing,
    /// Enumeration worked and the device is simply not one we can target.
    #[allow(dead_code)]
    UnsupportedDevice,
}

impl CpuOnlyCause {
    pub fn headline(self) -> &'static str {
        match self {
            Self::RuntimeNotSourced => {
                "No Level Zero device — the oneAPI runtime is not on this shell's path."
            }
            Self::KernelDriverMissing => {
                "No Level Zero device — the kernel GPU driver is not loaded."
            }
            Self::UnsupportedDevice => {
                "A GPU was found, but it is not a supported inference target."
            }
        }
    }

    /// The single next action. One line, one thing to do: a remedy that offers a choice is a
    /// dependency complaint wearing a helpful tone.
    pub fn remedy(self) -> &'static str {
        match self {
            Self::RuntimeNotSourced => {
                "The CPU device in the table is the giveaway: an unsourced runtime enumerates CPU-only \
                 and looks identical to a dead GPU. Nothing is broken — re-run under the \
                 bundled launcher."
            }
            Self::KernelDriverMissing => {
                "Load the `xe` driver (kernel 6.8+) or `i915` on older kernels. This is the \
                 only component MoEArc cannot ship for you: it lives in the kernel."
            }
            Self::UnsupportedDevice => "MoEArc targets Intel Arc. Other vendors are not supported.",
        }
    }
}

/// The one-line answer to "does my card work", plus everything needed to say why not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Verdict {
    /// A Level Zero GPU is present and usable.
    Ready { device: String, free_bytes: u64 },
    /// Devices enumerated, but none is an inference target.
    CpuOnly { cause: CpuOnlyCause },
    /// Nothing enumerated at all — not even a CPU device, which means the loader itself is
    /// missing rather than merely unlit. Constructed by [`Verdict::for_devices`].
    NoDevice,
}

impl Verdict {
    /// Classify an enumerated device list.
    ///
    /// Lives here rather than in the detector so that the *interface's* notion of "ready" is
    /// the one that decides, and cannot drift from what the screens then render. A detector
    /// that also diagnoses can hand back a richer [`CpuOnlyCause`]; this is the floor.
    pub fn for_devices(devices: &[DeviceRow]) -> Self {
        match devices.iter().find(|d| d.is_inference_target()) {
            Some(d) => Self::Ready { device: d.name.clone(), free_bytes: d.free_bytes },
            // An empty list means the loader never answered. A list with only CPU devices
            // means it answered and found no GPU — which, per docs/ux.md, is most often an
            // unsourced runtime rather than broken hardware.
            None if devices.is_empty() => Self::NoDevice,
            None => Self::CpuOnly { cause: CpuOnlyCause::RuntimeNotSourced },
        }
    }

    pub fn headline(&self) -> String {
        match self {
            Self::Ready { device, free_bytes } => {
                format!("{device} is ready — {} free right now.", format::bytes(*free_bytes))
            }
            Self::CpuOnly { cause } => cause.headline().to_string(),
            Self::NoDevice => "No compute device enumerated at all.".to_string(),
        }
    }

    pub fn remedy(&self) -> Option<&'static str> {
        match self {
            Self::Ready { .. } => None,
            Self::CpuOnly { cause } => Some(cause.remedy()),
            Self::NoDevice => Some(
                "The SYCL runtime did not load. Re-run under the bundled launcher; if that \
                 fails the install is incomplete rather than misconfigured.",
            ),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }
}

/// Everything `moearc` with no arguments has to say about the hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceReport {
    pub devices: Vec<DeviceRow>,
    pub verdict: Verdict,
}

impl DeviceReport {
    /// The device plans are built against: the first inference target, by index order.
    pub fn primary(&self) -> Option<&DeviceRow> {
        self.devices.iter().find(|d| d.is_inference_target())
    }
}

// ---------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------

/// What we know about one model — and specifically, the numbers the cache planner needs.
///
/// Every field is read out of the GGUF file by `moearc-model`; none of it is asked of a user.
/// That is the point of the type: expert geometry is exactly the sort of number a person
/// should never have to look up, and every runtime that asks for one is asking them to be
/// wrong occasionally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    /// Short handle, the thing a user types.
    pub id: String,
    /// Hugging Face repo id, the thing a user pastes — when there is one.
    ///
    /// 🔴 `None` for a model found on disk, and that is the ordinary case rather than an edge
    /// one. A GGUF's `general.repo_url` is written by whoever quantised it, and across the
    /// files this was developed against it is either absent or names an *organisation*
    /// (`huggingface.co/unsloth`) rather than a repo. Rendering that in a field labelled
    /// "repo" would hand the user a string that does not work in `moearc pull`.
    pub repo: Option<String>,
    /// The file this card was read from, name only.
    ///
    /// The directory is the user's own and does not belong on every row; the name is what
    /// tells them which of several quantisations they are looking at.
    pub file: Option<String>,
    /// The ggml type holding most of the expert weights: `q4_K`, `mxfp4`.
    ///
    /// ggml's spelling, from the tensor index — not the filename's tag. A "Q4_K_M" file is a
    /// mixture, and saying so is more use than repeating a label nobody verified.
    pub quant: String,
    /// Size of the file on disk.
    pub file_bytes: u64,
    /// Total parameters, summed over the tensor index.
    ///
    /// 🔴 The total only, deliberately. An "active parameters" figure looks derivable from the
    /// same index and is not: it turns on whether the embedding matrix counts as active, which
    /// is a convention rather than a measurement, and publishers do not agree on it. The two
    /// conventions differ by enough to be visible in the first decimal place. The expert
    /// geometry below carries the same meaning and is exact.
    pub parameters: u64,
    /// Non-expert weights: embeddings, attention, norms, the output head. Resident
    /// unconditionally, so they come off the budget before any split is planned.
    pub dense_weights_bytes: u64,
    /// Bytes one resident expert *slot* occupies.
    pub per_expert_bytes: u64,
    /// Whether every MoE block agreed on [`Self::per_expert_bytes`].
    ///
    /// `false` means the file mixes quantisation types across blocks — three of the four real
    /// files here do — and the figure is a conservative maximum. The plan then holds slightly
    /// fewer slots than the card could, which is the safe direction to be wrong in, but it is
    /// worth saying rather than leaving as an unexplained few percent.
    pub per_expert_bytes_uniform: bool,
    /// Residency slots the model has: one per *(block, expert)* pair.
    ///
    /// 🔴 Not the expert count. A 128-expert model with 36 MoE blocks has **4,608** slots, and
    /// the cache pages slots. Putting 128 here would understate the model by the block count
    /// and make every residency figure on screen wrong by the same factor.
    pub expert_slots_total: u32,
    /// Slots one token touches: `active experts × MoE blocks`. The floor on residency.
    pub expert_slots_active: u32,
    /// Experts per MoE block, as the model's own metadata states it.
    pub experts_per_block: u32,
    /// Experts routed to per token, per block.
    pub active_experts_per_block: u32,
    /// Blocks carrying an expert bank. Not necessarily every block.
    pub moe_blocks: u32,
    /// KV cache bytes for one token, across every block that caches. Pages are the planner's
    /// granularity, not a property of the model, so the per-token figure is what belongs here.
    pub kv_bytes_per_token: u64,
    /// The longest context the model was trained for.
    ///
    /// 🔴 A plan is never reported above it. The card can hold more KV pages than the model
    /// can use — `olmoe` is a 4,096-token model and this B580 has room for eleven times that —
    /// and printing the page count as a context length would be a capability claim the model
    /// does not support.
    pub trained_context_tokens: u32,
    /// Present on this machine.
    pub local: bool,
    /// Whether the footprint above was *measured on an Arc card* rather than derived from the
    /// header. `docs/ux.md`: a model we have not run does not get a green checkmark.
    pub measured: bool,
}

impl ModelCard {
    /// "116.8B" — total parameters. See [`Self::parameters`] for why there is no second half.
    pub fn params(&self) -> String {
        let billions = self.parameters as f64 / 1e9;
        if billions >= 1.0 {
            format!("{billions:.1}B")
        } else {
            format!("{:.0}M", self.parameters as f64 / 1e6)
        }
    }

    /// "128 per block, 4 routed" — the model's own geometry, before it meets a card.
    pub fn experts(&self) -> String {
        format!("{} per block, {} routed", self.experts_per_block, self.active_experts_per_block)
    }

    /// "4,608 across 36 MoE blocks" — what residency is actually counted in.
    pub fn slots(&self) -> String {
        format!(
            "{} across {} blocks",
            format::count(self.expert_slots_total as i64),
            self.moe_blocks
        )
    }

    /// Where this model came from, for a single column: a repo id if we have one, else the
    /// file it was read from.
    pub fn origin(&self) -> &str {
        self.repo.as_deref().or(self.file.as_deref()).unwrap_or("—")
    }
}

// ---------------------------------------------------------------------------------------
// The host machine
// ---------------------------------------------------------------------------------------

/// What the machine underneath the card offers a memory-mapped model.
///
/// 🔴 This is a first-class part of the report, not a detail. MoEArc runs models several times
/// the size of the card's memory, so the question that decides whether a model is pleasant to
/// use is not how much VRAM there is — it is how much of the file the host can keep in the page
/// cache, and therefore whether a cache miss is a copy over PCIe or a read from a drive. See
/// `moearc_engine::host_budget`, which owns the reasoning this type feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HostReport {
    /// RAM fitted to the machine.
    pub total_bytes: u64,
    /// RAM the operating system says can be used without pushing something else out.
    pub available_bytes: u64,
    /// Free space on the filesystem holding the model directory.
    ///
    /// The models' own filesystem, not the root one. They differ by orders of magnitude on any
    /// machine with a dedicated pool for them, and the one that decides whether a download fits
    /// is the one the models live on.
    pub models_free_bytes: u64,
}

// ---------------------------------------------------------------------------------------
// Transfers and serving
// ---------------------------------------------------------------------------------------

/// A download, before it starts.
///
/// The interface integrates progress from this on its own clock rather than subscribing to a
/// byte stream, which keeps the reducer a pure function of `(state, tick)` and therefore
/// testable. A real downloader replaces `bytes_per_sec` with a measured rate; nothing else in
/// the download screen changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransferPlan {
    pub repo: String,
    pub total_bytes: u64,
    pub bytes_per_sec: u64,
    /// Bytes already on disk from an interrupted attempt. Resumable per `docs/ux.md`.
    pub resume_from: u64,
}

/// One sample of a running server's vitals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ServeSample {
    pub tokens_per_sec: f64,
    pub prompt_tokens_per_sec: f64,
    pub active_requests: u32,
    /// Fraction of the planned KV cache in use, 0.0..=1.0.
    ///
    /// A fraction rather than a page count: the interface would otherwise be holding two
    /// numbers from two sources and hoping they agreed. They did not — an early version
    /// rendered "kv 187 / 12 pages", because the sample knew nothing of the plan it was
    /// being drawn against.
    pub kv_utilisation: f64,
    pub expert_hit_rate: f64,
}

// ---------------------------------------------------------------------------------------
// The seams
// ---------------------------------------------------------------------------------------

/// Enumerate the machine's compute devices. Implemented by `moearc-device`.
pub trait DeviceSource {
    fn detect(&self) -> anyhow::Result<DeviceReport>;
}

/// Find models, locally and in the curated list. Implemented by [`crate::catalog`].
pub trait ModelCatalog {
    /// Models present on this machine.
    fn installed(&self) -> anyhow::Result<Vec<ModelCard>>;
    /// The curated known-good list.
    fn curated(&self) -> anyhow::Result<Vec<ModelCard>>;
    /// Resolve a handle or a Hugging Face repo id to a card.
    fn resolve(&self, id: &str) -> anyhow::Result<ModelCard>;

    /// Files that look like models and could not be read, each with its reason.
    ///
    /// Rendered, not swallowed. A truncated download and a model that was never there produce
    /// the same empty list, and "no models found" sends a user looking in the wrong place.
    fn skipped(&self) -> Vec<String> {
        Vec::new()
    }

    /// Where the catalogue looked, for the message it prints when it found nothing.
    fn location(&self) -> Option<String> {
        None
    }
}

/// Measure the host machine. Implemented by [`crate::host`].
pub trait HostSource {
    fn probe(&self) -> anyhow::Result<HostReport>;
}

/// Size a download before starting it. Implemented by `moearc-model`.
pub trait TransferSource {
    fn plan(&self, repo: &str) -> anyhow::Result<TransferPlan>;
}

/// Sample a running server. Implemented by the server half of `moearc-engine`.
///
/// Takes the tick rather than reading a clock so that a snapshot test can ask for frame 40
/// and get frame 40.
pub trait ServeStats {
    fn sample(&self, tick: u64) -> ServeSample;
}

/// Everything the interface talks to, in one place.
pub struct Sources {
    pub devices: Box<dyn DeviceSource>,
    pub models: Box<dyn ModelCatalog>,
    pub host: Box<dyn HostSource>,
    pub transfers: Box<dyn TransferSource>,
    pub serve: Box<dyn ServeStats>,
    /// True while any of the above is a fixture. Surfaced in the footer and in `--json`,
    /// because output that looks like a measurement and is not is worse than no output.
    pub stubbed: bool,
    /// Which parts are still fixtures, named.
    ///
    /// 🔴 This is a field rather than a constant because the first version was a hardcoded
    /// sentence, and the moment device detection was wired in it began telling users that
    /// `moearc-device` was not wired in. A provenance note that goes stale is worse than none:
    /// it is trusted precisely because it looks like bookkeeping nobody would get wrong.
    pub stub_note: &'static str,
    /// The same fact, short enough for a keybind footer.
    ///
    /// 🔴 These two fields say one thing at two lengths and must move together. They exist as
    /// a pair because the interface's footer had room for "stub data" and nothing else, and
    /// once devices and models became real that marker was wrong in the *other* direction —
    /// it labelled measurements as fixtures, which is its own kind of lie. A test asserts
    /// every word here also appears in [`Self::stub_note`], so editing one and forgetting the
    /// other fails rather than ships.
    pub stub_parts: &'static str,
}

impl Sources {
    /// The real backends, where they exist.
    ///
    /// `stubbed` stays true while ANY source is a fixture, so the footer and the `--json`
    /// payload keep saying so. Devices and models are real now; transfers and serving are not.
    /// Reporting "not stubbed" the moment a backend lands would be the more flattering claim
    /// and the false one — and the note has to move with the code, or it becomes a lie that
    /// looks like bookkeeping.
    pub fn real(models_dir: std::path::PathBuf) -> Self {
        Self {
            devices: Box::new(crate::detect::LevelZeroDevices),
            host: Box::new(crate::host::RealHost::new(models_dir.clone())),
            models: Box::new(crate::catalog::LocalCatalog::new(models_dir)),
            transfers: Box::new(StubTransfers),
            serve: Box::new(StubServeStats),
            stubbed: true,
            stub_note: "devices and models are read from this machine; downloads and serving \
                        stats are still fixtures",
            stub_parts: "fixtures: downloads, serving",
        }
    }

    /// Fixture data, for building and testing the interface ahead of the backends.
    pub fn stub() -> Self {
        Self {
            devices: Box::new(StubDeviceSource),
            host: Box::new(StubHost),
            models: Box::new(StubCatalog),
            transfers: Box::new(StubTransfers),
            serve: Box::new(StubServeStats),
            stubbed: true,
            stub_note: "every number is a fixture — no hardware or model was consulted",
            stub_parts: "fixture: every number",
        }
    }
}

// ---------------------------------------------------------------------------------------
// Fixtures
//
// The shapes are real — an Arc B580 alongside an Arrow Lake iGPU and a CPU device is what
// `sycl-ls` reports on the development machine — but every number below is a fixture, and
// `Sources::stubbed` says so wherever they are rendered.
// ---------------------------------------------------------------------------------------

pub struct StubDeviceSource;

impl DeviceSource for StubDeviceSource {
    fn detect(&self) -> anyhow::Result<DeviceReport> {
        let devices = vec![
            DeviceRow {
                name: "Intel Arc B580 Graphics".to_string(),
                backend: Backend::LevelZero,
                driver: "xe / Level Zero 1.6".to_string(),
                total_bytes: 12 * 1024 * 1024 * 1024,
                free_bytes: 12_241_698_816,
            },
            DeviceRow {
                name: "Intel Arc 140T Graphics (iGPU)".to_string(),
                backend: Backend::LevelZero,
                driver: "xe / Level Zero 1.6".to_string(),
                total_bytes: 16 * 1024 * 1024 * 1024,
                free_bytes: 9_663_676_416,
            },
            DeviceRow {
                name: "Intel OpenCL Runtime (CPU)".to_string(),
                backend: Backend::Cpu,
                driver: "OpenCL 3.0".to_string(),
                total_bytes: 96 * 1024 * 1024 * 1024,
                free_bytes: 74_088_284_160,
            },
        ];
        let verdict = Verdict::for_devices(&devices);
        Ok(DeviceReport { devices, verdict })
    }
}

pub struct StubCatalog;

impl StubCatalog {
    /// The interface's own fixture: four models with short handles and a single-block expert
    /// geometry, kept deliberately small so the frames it produces are easy to read.
    ///
    /// It is **not** the shape of a real file — see [`Self::as_measured`] for that. Both exist
    /// because they test different things: this one pins the screens, that one pins the widths.
    fn all() -> Vec<ModelCard> {
        vec![
            ModelCard {
                id: "qwen3-30b-a3b".to_string(),
                repo: Some("Qwen/Qwen3-30B-A3B-GGUF".to_string()),
                file: Some("Qwen3-30B-A3B-Q4_K_M.gguf".to_string()),
                quant: "Q4_K_M".to_string(),
                file_bytes: 18_600_000_000,
                parameters: 30_500_000_000,
                dense_weights_bytes: 1_700_000_000,
                per_expert_bytes: 137_000_000,
                per_expert_bytes_uniform: true,
                expert_slots_total: 128,
                expert_slots_active: 8,
                experts_per_block: 128,
                active_experts_per_block: 8,
                moe_blocks: 1,
                kv_bytes_per_token: 98_304,
                trained_context_tokens: 131_072,
                local: true,
                measured: true,
            },
            ModelCard {
                id: "gpt-oss-20b".to_string(),
                repo: Some("openai/gpt-oss-20b-GGUF".to_string()),
                file: Some("gpt-oss-20b-MXFP4.gguf".to_string()),
                quant: "MXFP4".to_string(),
                file_bytes: 12_100_000_000,
                parameters: 20_900_000_000,
                dense_weights_bytes: 1_100_000_000,
                per_expert_bytes: 320_000_000,
                per_expert_bytes_uniform: true,
                expert_slots_total: 32,
                expert_slots_active: 4,
                experts_per_block: 32,
                active_experts_per_block: 4,
                moe_blocks: 1,
                kv_bytes_per_token: 49_152,
                trained_context_tokens: 131_072,
                local: true,
                measured: true,
            },
            ModelCard {
                id: "mixtral-8x7b".to_string(),
                repo: Some("mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string()),
                file: None,
                quant: "Q4_K_M".to_string(),
                file_bytes: 26_400_000_000,
                parameters: 46_700_000_000,
                dense_weights_bytes: 1_300_000_000,
                per_expert_bytes: 3_100_000_000,
                per_expert_bytes_uniform: true,
                expert_slots_total: 8,
                expert_slots_active: 2,
                experts_per_block: 8,
                active_experts_per_block: 2,
                moe_blocks: 1,
                kv_bytes_per_token: 32_768,
                trained_context_tokens: 32_768,
                local: false,
                measured: false,
            },
            ModelCard {
                id: "qwen3-235b-a22b".to_string(),
                repo: Some("Qwen/Qwen3-235B-A22B-GGUF".to_string()),
                file: None,
                quant: "Q4_K_M".to_string(),
                file_bytes: 142_000_000_000,
                parameters: 235_000_000_000,
                dense_weights_bytes: 4_800_000_000,
                per_expert_bytes: 1_070_000_000,
                per_expert_bytes_uniform: true,
                expert_slots_total: 128,
                expert_slots_active: 8,
                experts_per_block: 128,
                active_experts_per_block: 8,
                moe_blocks: 1,
                kv_bytes_per_token: 98_304,
                trained_context_tokens: 131_072,
                local: false,
                measured: false,
            },
        ]
    }

    /// The four GGUF files this was developed against, as [`ModelCard`]s.
    ///
    /// 🔴 **Every number here was read out of the real file** by `moearc-model` on 2026-09-05 —
    /// geometry, byte counts, parameter totals and all. It is a fixture only in the sense that
    /// it is frozen: no header is parsed when it is used.
    ///
    /// It exists because the widths are the interesting part and the widths cannot be tested
    /// against a machine's contents. `olmoe-1b-7b-0924-instruct` is a 25-character handle,
    /// `gpt-oss-120b` is 59 GiB against an 11.3 GiB card, and `qwen3.6-35b-a3b-ud` has 10,240
    /// residency slots. A row sized for `qwen3-30b-a3b` and `128` fits none of them, and a
    /// column that overflows on the one screen anybody looks at is the defect this fixture is
    /// here to catch.
    #[cfg(test)]
    pub fn as_measured() -> Vec<ModelCard> {
        let card = |id: &str,
                    file: &str,
                    quant: &str,
                    file_bytes: u64,
                    parameters: u64,
                    dense_weights_bytes: u64,
                    per_expert_bytes: u64,
                    uniform: bool,
                    experts_per_block: u32,
                    active_experts_per_block: u32,
                    moe_blocks: u32,
                    kv_bytes_per_token: u64,
                    trained_context_tokens: u32| ModelCard {
            id: id.to_string(),
            repo: None,
            file: Some(file.to_string()),
            quant: quant.to_string(),
            file_bytes,
            parameters,
            dense_weights_bytes,
            per_expert_bytes,
            per_expert_bytes_uniform: uniform,
            expert_slots_total: moe_blocks * experts_per_block,
            expert_slots_active: moe_blocks * active_experts_per_block,
            experts_per_block,
            active_experts_per_block,
            moe_blocks,
            kv_bytes_per_token,
            trained_context_tokens,
            local: true,
            measured: false,
        };
        vec![
            card(
                "gpt-oss-120b",
                "gpt-oss-120b-MXFP4.gguf",
                "mxfp4",
                63_387_346_208,
                116_829_156_672,
                2_460_250_368,
                13_219_200,
                true,
                128,
                4,
                36,
                73_728,
                131_072,
            ),
            card(
                "qwen3.6-35b-a3b-ud",
                "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                "q4_K",
                22_134_528_992,
                34_660_610_688,
                2_555_013_632,
                2_039_808,
                false,
                256,
                8,
                40,
                20_480,
                262_144,
            ),
            card(
                "qwen3-30b-a3b",
                "Qwen3-30B-A3B-Q4_K_M.gguf",
                "q4_K",
                18_556_686_912,
                30_532_122_624,
                997_554_176,
                3_059_712,
                false,
                128,
                8,
                48,
                98_304,
                40_960,
            ),
            card(
                "olmoe-1b-7b-0924-instruct",
                "olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
                "q4_K",
                4_213_512_672,
                6_919_161_856,
                311_027_712,
                4_079_616,
                false,
                64,
                8,
                16,
                131_072,
                4_096,
            ),
        ]
    }
}

impl ModelCatalog for StubCatalog {
    fn installed(&self) -> anyhow::Result<Vec<ModelCard>> {
        Ok(Self::all().into_iter().filter(|m| m.local).collect())
    }

    fn curated(&self) -> anyhow::Result<Vec<ModelCard>> {
        Ok(Self::all())
    }

    fn resolve(&self, id: &str) -> anyhow::Result<ModelCard> {
        Self::all()
            .into_iter()
            .find(|m| m.id == id || m.repo.as_deref().is_some_and(|r| r.eq_ignore_ascii_case(id)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown model `{id}` — run `moearc ls --all` for the curated list, or \
                     pass a full Hugging Face repo id"
                )
            })
    }
}

pub struct StubHost;

impl HostSource for StubHost {
    fn probe(&self) -> anyhow::Result<HostReport> {
        Ok(HostReport {
            total_bytes: 96 * 1024 * 1024 * 1024,
            available_bytes: 74_088_284_160,
            models_free_bytes: 3_298_534_883_328,
        })
    }
}

pub struct StubTransfers;

impl TransferSource for StubTransfers {
    fn plan(&self, repo: &str) -> anyhow::Result<TransferPlan> {
        // A curated handle resolves to the repo it names. Echoing the handle back would print
        // `mixtral-8x7b` in a field labelled "repo", which is not a repo id and would not
        // work if a user copied it.
        let known = StubCatalog::all().into_iter().find(|m| {
            m.id == repo || m.repo.as_deref().is_some_and(|r| r.eq_ignore_ascii_case(repo))
        });
        Ok(TransferPlan {
            repo: known.as_ref().and_then(|m| m.repo.clone()).unwrap_or_else(|| repo.to_string()),
            total_bytes: known.as_ref().map_or(9_400_000_000, |m| m.file_bytes),
            bytes_per_sec: 92 * 1024 * 1024,
            resume_from: 0,
        })
    }
}

pub struct StubServeStats;

impl ServeStats for StubServeStats {
    fn sample(&self, tick: u64) -> ServeSample {
        // Deterministic wobble: a fixed-period sine, not a random walk, so a snapshot of
        // frame N is a snapshot of frame N forever.
        let phase = (tick as f64) * 0.31;
        ServeSample {
            tokens_per_sec: 61.4 + 4.0 * phase.sin(),
            prompt_tokens_per_sec: 1420.0 + 90.0 * (phase * 0.5).cos(),
            active_requests: 1 + (tick % 3) as u32,
            kv_utilisation: 0.62 + 0.22 * (phase * 0.4).sin(),
            expert_hit_rate: 0.87 + 0.04 * (phase * 0.7).sin(),
        }
    }
}

/// Bare cards for tests that only care about one or two fields.
#[cfg(test)]
pub mod testing {
    use super::ModelCard;

    /// A card with nothing in it but a handle and a quantisation.
    ///
    /// For tests about naming and layout, where every other field is noise. Anything that
    /// plans against a card wants [`super::StubCatalog`] instead: the numbers here are zero,
    /// and a zeroed footprint is rejected by the planner rather than mis-planned.
    pub fn card(id: &str, quant: &str) -> ModelCard {
        ModelCard {
            id: id.to_string(),
            repo: None,
            file: Some(format!("{id}.gguf")),
            quant: quant.to_string(),
            file_bytes: 0,
            parameters: 0,
            dense_weights_bytes: 0,
            per_expert_bytes: 0,
            per_expert_bytes_uniform: true,
            expert_slots_total: 0,
            expert_slots_active: 0,
            experts_per_block: 0,
            active_experts_per_block: 0,
            moe_blocks: 0,
            kv_bytes_per_token: 0,
            trained_context_tokens: 0,
            local: true,
            measured: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_footer_marker_cannot_drift_from_the_note_it_abbreviates() {
        // The failure this guards against already happened once: a hardcoded provenance
        // sentence outlived the thing it described and began telling users a real backend was
        // a fixture. Checking the short form against the long one makes the pair maintainable
        // by making it impossible to update only half.
        for sources in [Sources::real(std::path::PathBuf::new()), Sources::stub()] {
            assert!(sources.stubbed, "both of these still have a fixture in them");
            for word in
                sources.stub_parts.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 3)
            {
                assert!(
                    sources.stub_note.contains(word),
                    "the footer says `{word}` and the note does not: {:?}",
                    sources.stub_note
                );
            }
        }
    }
}
