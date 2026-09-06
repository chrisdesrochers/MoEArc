//! Whether a device Level Zero enumerated can actually serve a plan.
//!
//! [`crate::detect`] answers *what is in this machine*. This module answers the next question,
//! which is not the same one and which the project got wrong in a way worth recording:
//! **is the memory figure that device reports a VRAM budget at all?**
//!
//! ## The measurement this module exists for
//!
//! On the reference machine — an Arrow Lake host with a discrete Arc B580 — Level Zero
//! enumerates two GPUs, and the integrated one reports **91,890,372,608 bytes (85.58 GiB)** of
//! device memory. That figure is not video memory. It is a share of the machine's system RAM,
//! which `/proc/meminfo` puts at **98,257,694,720 bytes (91.51 GiB)**: 93.5% of it.
//!
//! Three independent sources were checked before this code was written, and they agree:
//!
//! | source | what it reports for `Intel(R) Graphics` |
//! | --- | --- |
//! | `zeDeviceGetMemoryProperties` (core Level Zero) | `totalSize` 91,890,372,608 |
//! | `clinfo` (OpenCL, a different API on the same runtime) | global memory size 91,890,372,608 |
//! | `zesMemoryGetProperties` (Sysman) | **one** module, `ZES_MEM_LOC_SYSTEM`, 98,257,694,720 |
//!
//! 🔴 **It is not a stale-driver artefact.** All three readings above were taken on Intel
//! compute runtime `26.05.37020.3` — the newest stack this project has, the one that loads and
//! runs a model. `docs/packaging.md` records the same 85.6 GiB coming out of Ubuntu 24.04's
//! much older build 27642, and the natural conclusion — "an old driver is reporting host
//! memory as device memory" — is wrong. Every driver does this for a device with no memory of
//! its own. What the old driver adds is that it does not enumerate the discrete card at all,
//! so the integrated device stops being the second row of a table and becomes *the* answer.
//!
//! Two defects compose into one confident wrong answer, and only one of them is about drivers:
//!
//! 1. a device with no VRAM reports host RAM where a VRAM figure is expected, always; and
//! 2. a runtime too old for the installed card leaves that device as the only candidate.
//!
//! ## The rule
//!
//! **A budget must come from a memory pool that is measurably on the device.** Sysman's
//! `free_device_memory_bytes` is that measurement, and it already excludes host modules. Where
//! there is no such pool, the core `totalSize` may be used *only* for a discrete card — where
//! it means installed VRAM that nothing has taken yet — and never for an integrated one, where
//! it means someone else's RAM. [`vram_budget`] is that rule and nothing else.
//!
//! This mirrors what the engine's planner does with a budget once it has one: refuse in
//! arithmetic, before an allocation, with a typed error that names the number.
//!
//! ## ⬜ For callers — this crate refuses; the CLI does not yet
//!
//! `moearc-device-report` (this crate's own binary) uses [`inference_target`] and exits 1 with
//! the refusal. **`moearc-cli` still does not**, so the packaged `moearc` command continues to
//! print `✓ Intel(R) Graphics is ready — 85.6 GiB free right now` on a stale stack. Two edits
//! close it, both in `crates/moearc-cli/src/detect.rs`, which is where the fallback lives:
//!
//! - Replace `let free = match live_free { Some(f) => f.min(d.total_memory_bytes), None =>
//!   d.total_memory_bytes };` with [`vram_budget`], and drop the row — or mark it refused —
//!   when it returns `Err`. The comment above that `match` already describes this exact
//!   hazard; the code then does it anyway.
//! - Have `Verdict::for_devices` take the refusals, so `CpuOnlyCause::UnsupportedDevice` (which
//!   exists, and is `#[allow(dead_code)]`) can carry [`Unusable`]'s sentence instead of a
//!   generic one.
//!
//! `driver_build` in that file duplicates [`crate::GpuDevice::driver_build`], with the same
//! corroboration written out twice; folding it onto the shared one would be a third edit.

use std::fmt::Write as _;
use std::path::Path;

use thiserror::Error;

use crate::sysman::DeviceTelemetry;
use crate::{DeviceReport, GpuDevice, PciDevice};

/// The Intel compute-runtime build this project has measured a **model load and decode** on.
///
/// 🔴 This is an observation, not a specification. The Level Zero spec assigns no encoding to
/// `driverVersion` and publishes no minimum, so there is no documented threshold to gate on and
/// none is invented here. What exists is `docs/packaging.md`'s measured table, taken on this
/// project's own B580 in a clean container:
///
/// | build | device report | SYCL queue | model load + decode |
/// | --- | --- | --- | --- |
/// | 27642 (Ubuntu 24.04 stock) | ❌ does not enumerate the card | — | — |
/// | 33578 (Intel client repo for noble) | ✅ | ✅ | ❌ fails, then SIGSEGV |
/// | 37020 (Ubuntu 26.04, `26.05.37020.3`) | ✅ | ✅ | ✅ |
///
/// So a build below this one is a **caution with provenance**, never a refusal: MoEArc has not
/// been seen to work there, which is a different claim from "it cannot work there".
pub const VERIFIED_RUNTIME_BUILD: u32 = 37_020;

/// Where a budget figure came from. Reported rather than flattened, because "measured on the
/// card just now" and "installed capacity, assumed idle" are different promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// Sysman measured free memory in a pool that is on the device.
    MeasuredFree,
    /// No live reading was available, so the card's installed VRAM is used. Only ever taken
    /// for a discrete card.
    InstalledVram,
}

impl BudgetSource {
    pub fn describe(self) -> &'static str {
        match self {
            Self::MeasuredFree => "measured free VRAM",
            Self::InstalledVram => "installed VRAM, assumed idle (no live reading available)",
        }
    }
}

/// How many bytes MoEArc may plan against on one device, and where the figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramBudget {
    pub bytes: u64,
    pub source: BudgetSource,
}

/// What Sysman contributed. Recorded on the refusal because the two sources are independent and
/// a message that can say "and the other API agrees" is worth more than one that cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalMemoryEvidence {
    /// Sysman was not available, so only core Level Zero's integrated flag is in evidence.
    NotMeasured,
    /// Sysman enumerated this device's memory modules and none of them is on the device.
    /// `host_module_bytes` is the capacity of the system-memory module it found instead.
    NoneOnDevice { host_module_bytes: Option<u64> },
}

/// The measured reason a device Level Zero was willing to enumerate must not be planned
/// against.
///
/// One variant today. It is an enum because the next reason will not be this one, and because
/// callers should match rather than string-search.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Unusable {
    #[error("{}", host_memory_sentence(.device, *.reported_bytes, *.host_ram_bytes, *.evidence))]
    SharesHostMemory {
        device: String,
        /// What the device reported as its device memory.
        reported_bytes: u64,
        /// This machine's RAM, when `/proc/meminfo` could be read. The comparison is the whole
        /// argument, so it is carried rather than recomputed at print time.
        host_ram_bytes: Option<u64>,
        evidence: LocalMemoryEvidence,
    },
}

/// The refusal message. Assembled rather than written as one format string because the two
/// halves that vary — whether host RAM is known, and whether Sysman corroborated — change the
/// strength of the claim, and overstating it here would be the same mistake in a new place.
fn host_memory_sentence(
    device: &str,
    reported_bytes: u64,
    host_ram_bytes: Option<u64>,
    evidence: LocalMemoryEvidence,
) -> String {
    let mut s = format!(
        "{device} reports {} of memory, and that figure is this machine's system RAM rather \
         than video memory",
        gib(reported_bytes)
    );
    match host_ram_bytes {
        Some(ram) => {
            let _ = write!(s, " — this machine has {} of RAM. ", gib(ram));
        }
        None => s.push_str(". "),
    }
    match evidence {
        LocalMemoryEvidence::NoneOnDevice { host_module_bytes } => {
            s.push_str(
                "Level Zero's Sysman API agrees, and independently: this device has no memory \
                 pool of its own at all",
            );
            match host_module_bytes {
                Some(bytes) => {
                    let _ = write!(
                        s,
                        " — the single module it exposes is system memory, {}. ",
                        gib(bytes)
                    );
                }
                None => s.push_str(" — every module it exposes is system memory. "),
            }
        }
        LocalMemoryEvidence::NotMeasured => s.push_str(
            "Level Zero flags it as an integrated GPU, and Sysman was not available to \
             cross-check, so the figure is not corroborated either way. ",
        ),
    }
    s.push_str(
        "MoEArc keeps experts resident in VRAM and streams the rest across PCIe, so planning \
         against an integrated device's memory figure does not fail — it fits, and is wrong. \
         MoEArc needs a discrete Intel Arc card.",
    );
    s
}

/// The budget MoEArc may plan against on `device`, or the measured reason there is none.
///
/// `live` is the Sysman reading **for this device**, joined by UUID by the caller — Sysman and
/// core Level Zero enumerate in different orders on the reference machine, so joining by index
/// silently attributes one card's memory to another. [`reading_for`] does the join.
///
/// 🔴 The `None` case is the whole point. Falling back to [`GpuDevice::total_memory_bytes`]
/// whenever the live figure is missing is the shape that produced "85.6 GiB free" on a device
/// with no VRAM; it is safe for a discrete card and never for an integrated one.
pub fn vram_budget(
    device: &GpuDevice,
    live: Option<&DeviceTelemetry>,
) -> Result<VramBudget, Unusable> {
    // Capped at what the core API says is allocatable. Sysman reports more free memory than
    // core Level Zero reports allocatable on the reference B580 — 12,567,810,048 against
    // 12,168,933,376 — and they answer different questions; the allocatable figure is the
    // ceiling, always.
    if let Some(free) = live.and_then(DeviceTelemetry::free_device_memory_bytes) {
        return Ok(VramBudget {
            bytes: free.min(device.total_memory_bytes),
            source: BudgetSource::MeasuredFree,
        });
    }

    if !device.is_integrated {
        return Ok(VramBudget {
            bytes: device.total_memory_bytes,
            source: BudgetSource::InstalledVram,
        });
    }

    Err(Unusable::SharesHostMemory {
        device: device.name.clone(),
        reported_bytes: device.total_memory_bytes,
        host_ram_bytes: host_ram_bytes(),
        evidence: match live {
            Some(reading) => {
                LocalMemoryEvidence::NoneOnDevice { host_module_bytes: reading.host_memory_bytes() }
            }
            None => LocalMemoryEvidence::NotMeasured,
        },
    })
}

/// The Sysman reading for a device, joined on UUID rather than on position.
pub fn reading_for<'a>(
    readings: &'a [DeviceTelemetry],
    device: &GpuDevice,
) -> Option<&'a DeviceTelemetry> {
    readings.iter().find(|r| r.uuid == device.uuid)
}

/// Why nothing in a report can be planned against.
///
/// Carries every device's refusal, so the message can say what was offered and rejected rather
/// than only that nothing was left, plus the physical evidence for the second defect — hardware
/// the kernel is driving that the compute runtime did not expose.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{}", no_usable_device_sentence(.refusals, .hardware_not_enumerated, *.runtime_build))]
pub struct NoUsableDevice {
    pub refusals: Vec<Unusable>,
    /// Display controllers the kernel has bound a driver to that Level Zero did not expose,
    /// and whose DRM render node exists on this machine.
    pub hardware_not_enumerated: Vec<PciDevice>,
    /// The Level Zero driver build that answered, when there was a device to read it from.
    pub runtime_build: Option<u32>,
}

fn no_usable_device_sentence(
    refusals: &[Unusable],
    hardware: &[PciDevice],
    runtime_build: Option<u32>,
) -> String {
    let mut s = String::from(
        "MoEArc found no device it can plan against. Level Zero enumerated hardware; none of it \
         has VRAM to plan with.\n",
    );
    for refusal in refusals {
        let _ = write!(s, "\n  - {refusal}\n");
    }
    if !hardware.is_empty() {
        let _ = write!(
            s,
            "\nThe kernel is driving hardware this compute runtime did not expose:\n{}\n\nThat is \
             the shape of a runtime older than the card. The kernel driver is bound and the DRM \
             render node exists, so the kernel side is fine; the user-space Intel compute \
             runtime behind Level Zero is what does not know this device.",
            hardware.iter().map(|d| format!("  {}", d.describe())).collect::<Vec<_>>().join("\n")
        );
    }
    if let Some(build) = runtime_build {
        let _ = write!(s, "\n\n{}", runtime_provenance(build));
    }
    s
}

/// The one paragraph in this crate that names a version, and the provenance is in it.
fn runtime_provenance(build: u32) -> String {
    format!(
        "The Level Zero driver that answered reports build {build}. MoEArc has only been \
         measured loading and running a model on build {VERIFIED_RUNTIME_BUILD} or newer \
         (Intel's `26.05.37020.3`); this project measured build 33578 detecting the card and \
         then failing at model load, and build 27642 — Ubuntu 24.04's stock \
         `libze-intel-gpu1` — not enumerating a B580 at all. Those are measurements on one \
         machine, not a published minimum, and no such minimum exists to quote. If your card is \
         missing or a load fails, install Intel's client GPU runtime for your distribution."
    )
}

/// A note about the runtime version. Never an error: MoEArc has no documented minimum to
/// enforce, only [`VERIFIED_RUNTIME_BUILD`]'s measurement, and a caution that stops a working
/// machine would be worse than the problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("{}", runtime_provenance(*.observed_build))]
pub struct RuntimeCaution {
    pub observed_build: u32,
}

/// A caution when this device's runtime is older than the one MoEArc has been measured on.
pub fn runtime_caution(device: &GpuDevice) -> Option<RuntimeCaution> {
    let observed_build = device.driver_build();
    (observed_build < VERIFIED_RUNTIME_BUILD).then_some(RuntimeCaution { observed_build })
}

/// The device MoEArc will run on and the budget to plan with, or every reason it will not.
///
/// Discrete first, then any device — [`DeviceReport::preferred`] picks by class alone and is
/// the right answer to "which card is this machine about"; this is the answer to "which card
/// can carry the plan", and a device that cannot is skipped rather than selected.
pub fn inference_target<'a>(
    report: &'a DeviceReport,
    readings: &[DeviceTelemetry],
) -> Result<(&'a GpuDevice, VramBudget), NoUsableDevice> {
    let mut order: Vec<&GpuDevice> = report.devices.iter().collect();
    order.sort_by_key(|d| d.is_integrated);

    let mut refusals = Vec::new();
    for device in order {
        match vram_budget(device, reading_for(readings, device)) {
            Ok(budget) => return Ok((device, budget)),
            Err(refusal) => refusals.push(refusal),
        }
    }

    Err(NoUsableDevice {
        refusals,
        hardware_not_enumerated: hardware_the_runtime_did_not_expose(report),
        runtime_build: report.devices.first().map(GpuDevice::driver_build),
    })
}

/// Cards the kernel is driving that Level Zero did not expose, filtered to the ones that are
/// evidence of a runtime problem.
///
/// 🔴 The render-node filter is load-bearing and not decoration. `unusable_hardware` alone is
/// routinely non-empty for a perfectly healthy reason: `packaging/verify-clean.sh` hands a
/// container **one** render node on purpose, so the iGPU is genuinely unavailable inside it and
/// saying "your runtime is too old" there would be a false alarm on the project's own release
/// gate. A card whose render node exists on this machine, with a kernel driver bound, that the
/// compute runtime still did not enumerate, is a different thing entirely.
pub fn hardware_the_runtime_did_not_expose(report: &DeviceReport) -> Vec<PciDevice> {
    report
        .unusable_hardware
        .iter()
        .filter(|card| card.kernel_driver.is_some() && card.drm_render_node.is_some())
        .cloned()
        .collect()
}

/// This machine's total RAM, from `/proc/meminfo`.
///
/// `None` on anything that does not have one, or that will not let us read it. It is used only
/// to make a message concrete, so its absence weakens the sentence and never the refusal.
pub fn host_ram_bytes() -> Option<u64> {
    host_ram_bytes_at(Path::new("/proc/meminfo"))
}

/// [`host_ram_bytes`] against an arbitrary file, so the parse can be tested without a machine
/// whose RAM is known.
pub fn host_ram_bytes_at(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    // `MemTotal:       95954780 kB` — the unit is always kB, and the kernel means KiB by it.
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sysman::{MemoryHealth, MemoryModule};
    use crate::{DEFAULT_LOADER_SONAME, PCI_VENDOR_ID_INTEL};

    /// Measured on the reference machine, and every number here is a reading rather than a
    /// plausible-looking constant. See the module docs for the three sources.
    const IGPU_REPORTED_BYTES: u64 = 91_890_372_608;
    const HOST_RAM_BYTES: u64 = 98_257_694_720;
    const B580_TOTAL_BYTES: u64 = 12_168_933_376;
    const B580_SYSMAN_FREE_BYTES: u64 = 12_567_810_048;
    /// `libze-intel-gpu1 26.05.37020.3-1`, as `zeDriverGetProperties` reports it.
    const REFERENCE_DRIVER_VERSION: u32 = 17_010_844;

    fn igpu() -> GpuDevice {
        GpuDevice {
            name: "Intel(R) Graphics".into(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: 0x7d67,
            uuid: [1; 16],
            is_integrated: true,
            total_memory_bytes: IGPU_REPORTED_BYTES,
            max_alloc_bytes: 4_294_959_104,
            driver_version: REFERENCE_DRIVER_VERSION,
            compute_units: 64,
            max_total_group_size: 1024,
            subgroup_sizes: vec![8, 16, 32],
            pci_address: Some("0000:00:02.0".into()),
        }
    }

    fn b580() -> GpuDevice {
        GpuDevice {
            name: "Intel(R) Arc(TM) B580 Graphics".into(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: 0xe20b,
            uuid: [2; 16],
            is_integrated: false,
            total_memory_bytes: B580_TOTAL_BYTES,
            max_alloc_bytes: B580_TOTAL_BYTES,
            driver_version: REFERENCE_DRIVER_VERSION,
            compute_units: 160,
            max_total_group_size: 1024,
            subgroup_sizes: vec![16, 32],
            pci_address: Some("0000:04:00.0".into()),
        }
    }

    fn module(physical: Option<u64>, free: u64, on_device: bool) -> MemoryModule {
        MemoryModule {
            physical_bytes: physical,
            free_bytes: free,
            on_device,
            health: MemoryHealth::Unknown,
        }
    }

    /// What Sysman actually returns for the iGPU: one system-memory module, no device pool.
    fn igpu_reading() -> DeviceTelemetry {
        DeviceTelemetry {
            uuid: [1; 16],
            memory: vec![module(Some(HOST_RAM_BYTES), 42_207_121_408, false)],
        }
    }

    /// What Sysman actually returns for the B580: free known, capacity not.
    fn b580_reading() -> DeviceTelemetry {
        DeviceTelemetry { uuid: [2; 16], memory: vec![module(None, B580_SYSMAN_FREE_BYTES, true)] }
    }

    fn report(devices: Vec<GpuDevice>, unusable: Vec<PciDevice>) -> DeviceReport {
        DeviceReport {
            devices,
            loader: DEFAULT_LOADER_SONAME.to_string(),
            driver_count: 1,
            non_gpu_devices: 0,
            pci_display_devices: vec![],
            unusable_hardware: unusable,
        }
    }

    fn pci(address: &str, device_id: u32, driver: Option<&str>, render: Option<&str>) -> PciDevice {
        PciDevice {
            address: address.into(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id,
            class_code: 0x03_0000,
            kernel_driver: driver.map(str::to_string),
            drm_card: None,
            drm_render_node: render.map(str::to_string),
            boot_vga: false,
        }
    }

    // The bug, in one test. 85.58 GiB was offered as a budget because the live figure was
    // absent and the code fell back to the reported total.
    #[test]
    fn the_integrated_gpus_85_gib_is_refused_rather_than_offered_as_a_budget() {
        let err = vram_budget(&igpu(), Some(&igpu_reading()))
            .expect_err("a device with no VRAM has no budget");
        let message = err.to_string();
        assert!(message.contains("85.58 GiB"), "{message}");
        assert!(message.contains("system RAM"), "{message}");
        assert!(message.contains("91.51 GiB of RAM"), "{message}");
        assert!(message.contains("no memory pool of its own"), "{message}");
        assert!(message.contains("discrete Intel Arc card"), "{message}");
    }

    // Sysman is optional everywhere else in this crate, so the refusal cannot depend on it —
    // but the message must stop claiming corroboration it does not have.
    #[test]
    fn the_refusal_survives_sysman_being_unavailable_and_says_so() {
        let err = vram_budget(&igpu(), None).expect_err("still no VRAM");
        let message = err.to_string();
        assert!(message.contains("85.58 GiB"), "{message}");
        assert!(message.contains("not corroborated"), "{message}");
        assert!(!message.contains("Sysman API agrees"), "{message}");
    }

    #[test]
    fn a_discrete_card_is_planned_against_its_measured_free_memory() {
        let budget = vram_budget(&b580(), Some(&b580_reading())).expect("a real budget");
        // Capped at the allocatable figure, which is the smaller of the two.
        assert_eq!(budget.bytes, B580_TOTAL_BYTES);
        assert_eq!(budget.source, BudgetSource::MeasuredFree);
    }

    #[test]
    fn a_discrete_card_with_no_live_reading_falls_back_to_installed_vram_and_labels_it() {
        let budget = vram_budget(&b580(), None).expect("installed VRAM is a safe fallback here");
        assert_eq!(budget.bytes, B580_TOTAL_BYTES);
        assert_eq!(budget.source, BudgetSource::InstalledVram);
        assert!(budget.source.describe().contains("assumed idle"));
    }

    // A busy card is a smaller budget, not a refused one: the planner's job is to fit what is
    // left, and this crate's job is to hand it a true number.
    #[test]
    fn a_partly_occupied_card_yields_the_smaller_measured_figure() {
        let busy =
            DeviceTelemetry { uuid: [2; 16], memory: vec![module(None, 1_478_893_568, true)] };
        let budget = vram_budget(&b580(), Some(&busy)).expect("a real budget");
        assert_eq!(budget.bytes, 1_478_893_568);
    }

    // The reference machine: both devices enumerate, and the choice must not be enumeration
    // order or class alone — it must be which one has memory to plan in.
    #[test]
    fn the_discrete_card_is_chosen_and_the_integrated_one_is_never_reached() {
        let readings = vec![igpu_reading(), b580_reading()];
        let both = report(vec![igpu(), b580()], vec![]);
        let (device, budget) =
            inference_target(&both, &readings).expect("the B580 can carry a plan");
        assert_eq!(device.name, "Intel(R) Arc(TM) B580 Graphics");
        assert_eq!(budget.bytes, B580_TOTAL_BYTES);
    }

    // The stale-stack machine, simulated: the runtime enumerated one integrated device and the
    // kernel is driving a card it did not expose. Exactly the state `docs/packaging.md`
    // measured on Ubuntu 24.04, and the state in which MoEArc used to answer "ready".
    #[test]
    fn a_stale_stack_is_refused_and_names_the_card_the_runtime_did_not_expose() {
        let readings = vec![igpu_reading()];
        let stale =
            report(vec![igpu()], vec![pci("0000:04:00.0", 0xe20b, Some("xe"), Some("renderD129"))]);
        let err = inference_target(&stale, &readings)
            .expect_err("an integrated-only machine cannot carry a plan");
        let message = err.to_string();

        assert_eq!(err.refusals.len(), 1);
        assert!(message.contains("no device it can plan against"), "{message}");
        assert!(message.contains("85.58 GiB"), "{message}");
        assert!(message.contains("0000:04:00.0"), "{message}");
        assert!(message.contains("8086:e20b"), "{message}");
        assert!(message.contains("runtime older than the card"), "{message}");
        assert!(message.contains("build 37020"), "{message}");
        assert!(message.contains("client GPU runtime"), "{message}");
    }

    // The false alarm this must not raise. verify-clean.sh gives the container one render node,
    // so the iGPU is legitimately absent from /dev/dri and is not evidence of anything.
    #[test]
    fn a_card_with_no_render_node_on_this_machine_is_not_evidence_of_a_stale_runtime() {
        let report = report(vec![b580()], vec![pci("0000:00:02.0", 0x7d67, Some("i915"), None)]);
        assert!(hardware_the_runtime_did_not_expose(&report).is_empty());
    }

    #[test]
    fn an_unbound_card_is_not_evidence_of_a_stale_runtime_either() {
        let report =
            report(vec![b580()], vec![pci("0000:04:00.0", 0xe20b, None, Some("renderD129"))]);
        assert!(hardware_the_runtime_did_not_expose(&report).is_empty());
    }

    // The version gate: a caution, with its provenance in the sentence, and never on the
    // runtime this project has actually measured.
    #[test]
    fn the_reference_runtime_raises_no_caution() {
        assert_eq!(b580().driver_build(), VERIFIED_RUNTIME_BUILD);
        assert!(runtime_caution(&b580()).is_none());
    }

    #[test]
    fn an_older_runtime_is_cautioned_with_the_measurement_that_justifies_it() {
        let mut old = b580();
        // Same high half, build 33578 — the stack that detects the card and then fails to load
        // a model. 259 << 16 is the high half the reference driver reports.
        old.driver_version = (259 << 16) | 33_578;
        let caution = runtime_caution(&old).expect("older than the measured-good build");
        let message = caution.to_string();
        assert_eq!(caution.observed_build, 33_578);
        assert!(message.contains("build 33578"), "{message}");
        assert!(message.contains("37020"), "{message}");
        // The claim is bounded: measured here, not a published minimum.
        assert!(message.contains("not a published minimum"), "{message}");
    }

    #[test]
    fn mem_total_is_read_in_kib_and_reported_in_bytes() {
        let dir = std::env::temp_dir().join(format!("moearc-meminfo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("meminfo");
        std::fs::write(&path, "MemFree:        44884103 kB\nMemTotal:       95954780 kB\n")
            .expect("write");
        assert_eq!(host_ram_bytes_at(&path), Some(HOST_RAM_BYTES));
        assert_eq!(host_ram_bytes_at(&dir.join("absent")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
