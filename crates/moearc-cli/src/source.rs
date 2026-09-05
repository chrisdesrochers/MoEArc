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
/// The expert geometry fields exist so the user never types them. They come out of the GGUF
/// header, which is `moearc-model`'s job; nothing here asks a person for an expert count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    /// Short handle, the thing a user types.
    pub id: String,
    /// Hugging Face repo id, the thing a user pastes.
    pub repo: String,
    pub quant: String,
    /// Size of the download.
    pub file_bytes: u64,
    /// Total and active parameter counts in millions, kept integral so the type stays `Eq`.
    pub params_total_m: u32,
    pub params_active_m: u32,
    /// Non-expert weights: embeddings, attention, norms. Resident unconditionally, so they
    /// come off the budget before any split is planned.
    pub dense_weights_bytes: u64,
    /// Bytes one resident expert occupies.
    pub per_expert_bytes: u64,
    /// Experts the model has.
    pub experts_total: u32,
    /// Experts routed to per token — the floor on residency.
    pub experts_active: u32,
    /// KV cache bytes for one token, across all layers. Pages are the planner's granularity,
    /// not a property of the model, so the per-token figure is what belongs on the card.
    pub kv_bytes_per_token: u64,
    /// Present on this machine.
    pub local: bool,
    /// Whether the footprint above was *measured on an Arc card* rather than derived from the
    /// header. `docs/ux.md`: a model we have not run does not get a green checkmark.
    pub measured: bool,
}

impl ModelCard {
    /// "30.5B total / 3.3B active" — the pair, because for an MoE model either number alone
    /// misleads: total predicts the download, active predicts the speed.
    pub fn params(&self) -> String {
        format!(
            "{:.1}B / {:.1}B",
            self.params_total_m as f64 / 1000.0,
            self.params_active_m as f64 / 1000.0
        )
    }
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

/// Find models, locally and in the curated list. Implemented by `moearc-model`.
pub trait ModelCatalog {
    /// Models present on this machine.
    fn installed(&self) -> anyhow::Result<Vec<ModelCard>>;
    /// The curated known-good list.
    fn curated(&self) -> anyhow::Result<Vec<ModelCard>>;
    /// Resolve a handle or a Hugging Face repo id to a card.
    fn resolve(&self, id: &str) -> anyhow::Result<ModelCard>;
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
}

impl Sources {
    /// The real backends, where they exist.
    ///
    /// `stubbed` stays true while ANY source is a fixture, so the footer and the `--json`
    /// payload keep saying so. Devices are real now; models, transfers and serving are not.
    /// Reporting "not stubbed" the moment the first backend lands would be the more
    /// flattering claim and the false one.
    pub fn real() -> Self {
        Self {
            devices: Box::new(crate::detect::LevelZeroDevices),
            models: Box::new(StubCatalog),
            transfers: Box::new(StubTransfers),
            serve: Box::new(StubServeStats),
            stubbed: true,
            stub_note: "device detection is real; the model list, downloads and serving stats \
                        are still fixtures",
        }
    }

    /// Fixture data, for building and testing the interface ahead of the backends.
    pub fn stub() -> Self {
        Self {
            devices: Box::new(StubDeviceSource),
            models: Box::new(StubCatalog),
            transfers: Box::new(StubTransfers),
            serve: Box::new(StubServeStats),
            stubbed: true,
            stub_note: "every number is a fixture — no hardware or model was consulted",
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
    fn all() -> Vec<ModelCard> {
        vec![
            ModelCard {
                id: "qwen3-30b-a3b".to_string(),
                repo: "Qwen/Qwen3-30B-A3B-GGUF".to_string(),
                quant: "Q4_K_M".to_string(),
                file_bytes: 18_600_000_000,
                params_total_m: 30_500,
                params_active_m: 3_300,
                dense_weights_bytes: 1_700_000_000,
                per_expert_bytes: 137_000_000,
                experts_total: 128,
                experts_active: 8,
                kv_bytes_per_token: 98_304,
                local: true,
                measured: true,
            },
            ModelCard {
                id: "gpt-oss-20b".to_string(),
                repo: "openai/gpt-oss-20b-GGUF".to_string(),
                quant: "MXFP4".to_string(),
                file_bytes: 12_100_000_000,
                params_total_m: 20_900,
                params_active_m: 3_600,
                dense_weights_bytes: 1_100_000_000,
                per_expert_bytes: 320_000_000,
                experts_total: 32,
                experts_active: 4,
                kv_bytes_per_token: 49_152,
                local: true,
                measured: true,
            },
            ModelCard {
                id: "mixtral-8x7b".to_string(),
                repo: "mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string(),
                quant: "Q4_K_M".to_string(),
                file_bytes: 26_400_000_000,
                params_total_m: 46_700,
                params_active_m: 12_900,
                dense_weights_bytes: 1_300_000_000,
                per_expert_bytes: 3_100_000_000,
                experts_total: 8,
                experts_active: 2,
                kv_bytes_per_token: 32_768,
                local: false,
                measured: false,
            },
            ModelCard {
                id: "qwen3-235b-a22b".to_string(),
                repo: "Qwen/Qwen3-235B-A22B-GGUF".to_string(),
                quant: "Q4_K_M".to_string(),
                file_bytes: 142_000_000_000,
                params_total_m: 235_000,
                params_active_m: 22_000,
                dense_weights_bytes: 4_800_000_000,
                per_expert_bytes: 1_070_000_000,
                experts_total: 128,
                experts_active: 8,
                kv_bytes_per_token: 98_304,
                local: false,
                measured: false,
            },
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
        Self::all().into_iter().find(|m| m.id == id || m.repo.eq_ignore_ascii_case(id)).ok_or_else(
            || {
                anyhow::anyhow!(
                    "unknown model `{id}` — run `moearc ls --all` for the curated list, or \
                     pass a full Hugging Face repo id"
                )
            },
        )
    }
}

pub struct StubTransfers;

impl TransferSource for StubTransfers {
    fn plan(&self, repo: &str) -> anyhow::Result<TransferPlan> {
        // A curated handle resolves to the repo it names. Echoing the handle back would print
        // `mixtral-8x7b` in a field labelled "repo", which is not a repo id and would not
        // work if a user copied it.
        let known = StubCatalog::all()
            .into_iter()
            .find(|m| m.id == repo || m.repo.eq_ignore_ascii_case(repo));
        Ok(TransferPlan {
            repo: known.as_ref().map_or_else(|| repo.to_string(), |m| m.repo.clone()),
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
