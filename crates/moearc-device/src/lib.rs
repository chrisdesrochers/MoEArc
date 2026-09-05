//! GPU device discovery.
//!
//! Answers the first question the tool asks on behalf of the user: what card is in this
//! machine, how much of it can we use, and if the answer is "none", exactly why.
//!
//! See `docs/ux.md` — failure here must be legible. A missing kernel driver is the one
//! dependency we are allowed to ask the user for, and it must be named precisely.
//!
//! Two consequences of that shape the design:
//!
//! - **The Level Zero loader is opened at runtime, not linked.** A binary linked against
//!   `libze_loader` cannot start on a machine that lacks it, so the user would get a dynamic
//!   linker message instead of ours. See [`ze`] for the details.
//! - **Every failure is a distinct typed variant carrying a sentence.** [`DetectError`] never
//!   surfaces a bare `ze_result_t`; the codes we cannot name individually still get a
//!   description and the call that produced them.
//!
//! This module reports what the hardware *is*. Deciding what will fit on it is
//! `moearc-engine`'s `cache_budget`, which takes the byte figures from here.

pub mod pci;
pub mod sysman;
mod ze;

use std::ffi::{OsStr, OsString};

pub use pci::PciDevice;

use thiserror::Error;

/// The loader MoEArc asks for by default. A soname, not a path: the dynamic loader's own
/// search rules should find a distribution's copy, and a release build that bundles its own
/// copy points at it with [`LOADER_PATH_ENV`] rather than by second-guessing those rules.
pub const DEFAULT_LOADER_SONAME: &str = "libze_loader.so.1";

/// Overrides which Level Zero loader is opened.
///
/// Exists for the packaged binary, which ships its own loader (`docs/ux.md`: "the binary
/// brings its own dependencies"). It doubles as the only way to exercise the
/// loader-is-missing path without uninstalling Level Zero, which is what the negative-control
/// test in `tests/` uses it for.
pub const LOADER_PATH_ENV: &str = "MOEARC_ZE_LOADER";

/// Intel's PCI vendor id, so a caller can tell an Arc card from some other Level Zero device
/// without string-matching the marketing name.
pub const PCI_VENDOR_ID_INTEL: u32 = 0x8086;

/// One GPU that Level Zero enumerated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// The driver's own name for the card, e.g. as printed on the box.
    pub name: String,
    /// PCI vendor id. [`PCI_VENDOR_ID_INTEL`] for Arc.
    pub vendor_id: u32,
    /// PCI device id.
    pub device_id: u32,
    /// Stable identity for this device. Survives reboots and enumeration order, so it is what
    /// a saved device selection or a calibration record should key on — not the index.
    pub uuid: [u8; 16],
    /// True for an iGPU. The distinguishing fact for MoEArc: an integrated device's "memory"
    /// is host RAM it shares with everything else on the machine, so its VRAM figure means
    /// something entirely different from a discrete card's.
    pub is_integrated: bool,
    /// Sum of every local memory region's `totalSize`. This is the card's *installed* memory,
    /// not what is free right now — nothing here allocates, so nothing here can measure free
    /// memory.
    pub total_memory_bytes: u64,
    /// Largest single allocation the driver will accept. Distinct from, and usually well
    /// below, `total_memory_bytes`: a plan that fits in total memory can still be rejected
    /// one buffer at a time.
    pub max_alloc_bytes: u64,
    /// The Level Zero driver's version, as a monotonically increasing opaque integer. Only
    /// ordering is meaningful; the spec assigns no structure to it.
    pub driver_version: u32,
    /// Execution units: slices x sub-slices per slice x EUs per sub-slice.
    pub compute_units: u32,
    /// Largest work-group the device accepts, for kernel launch geometry.
    pub max_total_group_size: u32,
    /// Sub-group (SIMD) widths the device supports. Kernel tuning picks from this list, so it
    /// is reported rather than reduced to one number here.
    pub subgroup_sizes: Vec<u32>,
    /// Where this device sits on the PCI bus, when the sysfs probe could match it.
    ///
    /// `None` means Level Zero exposes a GPU that has no display controller behind it in
    /// sysfs — virtualised or an unusual topology. That is noted, never treated as an error:
    /// Level Zero is the authority on what is *usable*, and this field is corroboration.
    pub pci_address: Option<String>,
}

impl GpuDevice {
    /// The UUID in the conventional 8-4-4-4-12 form, for display and for logs.
    pub fn uuid_string(&self) -> String {
        let hex: Vec<String> = self.uuid.iter().map(|b| format!("{b:02x}")).collect();
        let hex = hex.concat();
        format!("{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
    }
}

/// Everything a single detection run learned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceReport {
    /// GPUs found, in enumeration order across all drivers.
    pub devices: Vec<GpuDevice>,
    /// Which loader was actually opened — the soname or an override. Worth reporting: when a
    /// machine has several Level Zero installs, "which one did you load" is the first
    /// question worth asking.
    pub loader: String,
    /// Level Zero drivers the loader initialised.
    pub driver_count: u32,
    /// Non-GPU devices seen and skipped (a Level Zero driver may expose CPUs, FPGAs or NPUs).
    /// Reported so "we found nothing" can be distinguished from "we found nothing usable".
    pub non_gpu_devices: u32,
    /// Every display controller the kernel knows about, read from sysfs independently of
    /// Level Zero.
    pub pci_display_devices: Vec<PciDevice>,
    /// Display controllers that are physically present but that Level Zero did not expose.
    /// Non-empty here means a partial success: something usable was found, and something else
    /// in this machine was not.
    pub unusable_hardware: Vec<PciDevice>,
}

impl DeviceReport {
    /// The device MoEArc would run on: the first discrete GPU, or failing that the first
    /// device of any kind.
    ///
    /// Discrete wins unconditionally, and deliberately so — on this project's own reference
    /// machine an Arrow Lake iGPU enumerates alongside the Arc card, and picking by
    /// enumeration order would silently benchmark the wrong one.
    pub fn preferred(&self) -> Option<&GpuDevice> {
        self.devices.iter().find(|d| !d.is_integrated).or_else(|| self.devices.first())
    }
}

/// Why detection could not produce a report.
///
/// Each variant names a cause and, where the user can act, exactly one action. No variant
/// renders a bare error code.
#[derive(Debug, Error)]
pub enum DetectError {
    #[error(
        "could not load the Level Zero loader `{loader}`: {source}. MoEArc talks to Intel GPUs \
         through this library, so without it no device can be found. A packaged MoEArc ships \
         its own copy — if you are seeing this you are running a build that does not, so \
         install your distribution's Level Zero loader package, or set {env} to the full path \
         of a copy you already have."
    )]
    LoaderNotFound { loader: String, env: &'static str, source: libloading::Error },

    #[error(
        "`{loader}` loaded but is not a Level Zero loader: it does not export `{symbol}` \
         ({source}). Something else on this system is answering to that name; set {env} to the \
         full path of the real loader."
    )]
    NotALoader {
        loader: String,
        symbol: &'static str,
        env: &'static str,
        source: libloading::Error,
    },

    #[error(
        "the Level Zero loader started but found no usable driver (zeInit returned \
         ZE_RESULT_ERROR_UNINITIALIZED). This almost always means the kernel GPU driver is not \
         loaded: check that `xe` (Arc A-series and B-series) or `i915` (older Intel graphics) \
         is present with `lsmod`. That kernel module ships with your kernel and is the one \
         thing MoEArc cannot bring with it."
    )]
    DriverUninitialized,

    #[error(
        "the Level Zero loader started but zero drivers registered. A Level Zero *loader* is \
         installed while the Intel compute runtime behind it is not, or the kernel GPU driver \
         (`xe` for Arc, `i915` for older Intel graphics) is not loaded — check `lsmod` first, \
         since that is the one dependency MoEArc cannot ship."
    )]
    NoDrivers,

    #[error(
        "{driver_count} Level Zero driver(s) are loaded but they expose no devices at all. The \
         runtime is installed and working; nothing is attached to it. If the card is physically \
         present, the kernel driver (`xe` or `i915`) has most likely not bound to it — check \
         `dmesg` for the GPU."
    )]
    NoDevices { driver_count: u32 },

    #[error(
        "{driver_count} Level Zero driver(s) are loaded and expose {non_gpu} device(s), but none \
         of them is a GPU. MoEArc needs a GPU; a CPU or NPU device is not a substitute."
    )]
    NoGpuDevices { driver_count: u32, non_gpu: u32 },

    #[error(
        "Level Zero refused access to the GPU during {call} (ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS). \
         The device nodes are there but this user may not open them: add your user to the \
         `render` group (and `video` on some distributions), then log out and back in."
    )]
    PermissionDenied { call: &'static str },

    #[error("{call} failed: {meaning} (ze_result_t {code:#010x}).")]
    Ze { call: &'static str, code: ze::ZeResult, meaning: &'static str },

    #[error(
        "MoEArc found no usable GPU, but this machine physically has one: the kernel reports \
         {}. {} Level Zero's own report was: {underlying}",
        describe_hardware(.hardware),
        advise_on_hardware(.hardware, .underlying)
    )]
    PresentButUnusable { hardware: Vec<PciDevice>, underlying: Box<DetectError> },
}

/// List the display controllers sysfs found, in the kernel's own terms.
fn describe_hardware(hardware: &[PciDevice]) -> String {
    hardware.iter().map(PciDevice::describe).collect::<Vec<_>>().join("; ")
}

/// The single action to take, chosen by what the kernel is already doing with the card and by
/// which half of the stack failed.
///
/// The three cases need different answers and are routinely confused for each other. Nothing
/// bound means the *kernel* driver is absent, which `docs/ux.md` says is the one dependency we
/// are allowed to ask for. A driver bound but no Level Zero loader is MoEArc's own packaging
/// problem, not the user's — so we must not send them looking for a runtime. A driver bound
/// and a loader present means the kernel side is fine and the user-space compute runtime is
/// not.
fn advise_on_hardware(hardware: &[PciDevice], underlying: &DetectError) -> &'static str {
    if hardware.iter().any(|d| d.kernel_driver.is_none()) {
        return "No kernel driver is bound to it, so the GPU kernel module is not loaded: check \
                for `xe` (Arc A-series and B-series) or `i915` (older Intel graphics) with \
                `lsmod`. That module ships with your kernel and is the one thing MoEArc cannot \
                bring with it.";
    }
    match underlying {
        DetectError::LoaderNotFound { .. } | DetectError::NotALoader { .. } => {
            "A kernel driver is bound, so the kernel side is fine and nothing needs doing \
             there; the missing piece is on MoEArc's side of the line."
        }
        _ => {
            "A kernel driver is bound, so the kernel can see the card and the kernel side is \
             fine. What is missing or broken is the user-space Intel compute runtime that Level \
             Zero loads behind the loader. A packaged MoEArc ships that runtime with the \
             binary; if you built this yourself, install your distribution's Intel Level Zero \
             GPU runtime."
        }
    }
}

impl DetectError {
    /// The Level Zero failure underneath, unwrapping the physical-evidence wrapper.
    ///
    /// [`DetectError::PresentButUnusable`] adds context to another error rather than replacing
    /// it, so callers that switch on *what went wrong* should match on this instead of on the
    /// outer variant. The message a user sees is always the outer one.
    pub fn root_cause(&self) -> &DetectError {
        match self {
            Self::PresentButUnusable { underlying, .. } => underlying.root_cause(),
            other => other,
        }
    }
}

/// Which failures are improved by naming the hardware.
///
/// Only the "nothing to run on" family. A device that was lost mid-query, or a call MoEArc got
/// wrong, is already a precise statement; adding a PCI listing to it would bury the cause.
fn wants_hardware_evidence(err: &DetectError) -> bool {
    matches!(
        err,
        DetectError::LoaderNotFound { .. }
            | DetectError::NotALoader { .. }
            | DetectError::DriverUninitialized
            | DetectError::NoDrivers
            | DetectError::NoDevices { .. }
            | DetectError::NoGpuDevices { .. }
    )
}

/// A sentence for every `ze_result_t` this crate can plausibly surface.
///
/// The fallback is still a sentence. A user who hits an unmapped code should learn what kind
/// of thing went wrong and where, which is more than the number alone gives them.
pub(crate) fn meaning(code: ze::ZeResult) -> &'static str {
    match code {
        ze::ZE_RESULT_ERROR_DEVICE_LOST => {
            "the GPU stopped responding — it hung, was reset, or was removed"
        }
        ze::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY => "the system ran out of host memory",
        ze::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => "the GPU ran out of memory",
        ze::ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET => {
            "the GPU needs to be reset before it can be used"
        }
        ze::ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE => "the GPU is in a low power state",
        ze::ZE_RESULT_ERROR_NOT_AVAILABLE => {
            "the GPU is in use by something that will not share it"
        }
        ze::ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE => {
            "the driver is missing a library it depends on"
        }
        ze::ZE_RESULT_ERROR_UNINITIALIZED => "the Level Zero driver was not initialised",
        ze::ZE_RESULT_ERROR_UNSUPPORTED_VERSION => {
            "the installed driver is too old for the Level Zero API MoEArc uses"
        }
        ze::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE => "the driver does not support this query",
        ze::ZE_RESULT_ERROR_INVALID_ARGUMENT
        | ze::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
        | ze::ZE_RESULT_ERROR_INVALID_NULL_POINTER => {
            "MoEArc called Level Zero incorrectly — this is a bug in MoEArc, please report it"
        }
        ze::ZE_RESULT_ERROR_UNKNOWN => "the driver reported an unspecified failure",
        _ => "the Level Zero driver returned an error this build has no description for",
    }
}

/// Turn a non-success `ze_result_t` into an error that says something.
fn ze_err(call: &'static str, code: ze::ZeResult) -> DetectError {
    if code == ze::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS {
        return DetectError::PermissionDenied { call };
    }
    DetectError::Ze { call, code, meaning: meaning(code) }
}

/// Which loader to open. Split from the environment lookup so both branches are testable
/// without a test mutating process-wide state that its neighbours can see.
fn loader_or_default(configured: Option<OsString>) -> OsString {
    configured.unwrap_or_else(|| OsString::from(DEFAULT_LOADER_SONAME))
}

/// Enumerate the GPUs on this machine.
///
/// Opens the Level Zero loader named by [`LOADER_PATH_ENV`], or [`DEFAULT_LOADER_SONAME`].
pub fn detect() -> Result<DeviceReport, DetectError> {
    detect_with_loader(&loader_or_default(std::env::var_os(LOADER_PATH_ENV)))
}

/// [`detect`] against a named loader, for callers that already know where theirs lives.
pub fn detect_with_loader(loader: &OsStr) -> Result<DeviceReport, DetectError> {
    // Read the physical view first and unconditionally. Every Level Zero failure below is more
    // useful when it can say whether there is a card in the machine at all, and this probe
    // cannot fail in a way that should stop us trying.
    detect_against(loader, pci::scan())
}

fn detect_against(loader: &OsStr, hardware: Vec<PciDevice>) -> Result<DeviceReport, DetectError> {
    finish(level_zero_probe(loader), hardware)
}

/// The Level Zero half, with no knowledge of the physical probe.
fn level_zero_probe(loader: &OsStr) -> Result<DeviceReport, DetectError> {
    let shown = loader.to_string_lossy().into_owned();

    let api = ze::ZeApi::open(loader).map_err(|e| match e {
        ze::LoadError::Open(source) => {
            DetectError::LoaderNotFound { loader: shown.clone(), env: LOADER_PATH_ENV, source }
        }
        ze::LoadError::Symbol { name, source } => DetectError::NotALoader {
            loader: shown.clone(),
            symbol: name,
            env: LOADER_PATH_ENV,
            source,
        },
    })?;

    // GPU_ONLY keeps a CPU or NPU driver from being spun up for a query that could never want
    // one. `zeInit` is deprecated in favour of `zeInitDrivers` from Level Zero 1.10, but
    // `zeInit` is the entry point every shipping loader still exports, and MoEArc has to run
    // against whatever the user's distribution installed.
    //
    // SAFETY: `flags` is a valid `ze_init_flags_t` and the call takes no pointers.
    let result = unsafe { (api.ze_init)(ze::ZE_INIT_FLAG_GPU_ONLY) };
    if result == ze::ZE_RESULT_ERROR_UNINITIALIZED {
        return Err(DetectError::DriverUninitialized);
    }
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(ze_err("zeInit", result));
    }

    let report = enumerate(&api, shown);

    // Past this point the driver holds process-global state; see `ZeApi::leak`.
    api.leak();
    report
}

/// Cross-reference the two probes.
///
/// On success this records which PCI address each Level Zero device sits at and which cards
/// were left over. On failure it upgrades the error to name the hardware that is present but
/// unusable — the case a bare "no devices found" is useless for.
fn finish(
    report: Result<DeviceReport, DetectError>,
    hardware: Vec<PciDevice>,
) -> Result<DeviceReport, DetectError> {
    match report {
        Ok(mut report) => {
            // Matched on vendor:device with multiplicity, not on an address Level Zero never
            // gave us. Two identical cards are therefore paired by enumeration order, which is
            // a guess about ordering but never a guess about which model is present.
            let mut unmatched: Vec<&PciDevice> = hardware.iter().collect();
            for device in &mut report.devices {
                if let Some(index) = unmatched.iter().position(|p| {
                    p.vendor_id == device.vendor_id && p.device_id == device.device_id
                }) {
                    device.pci_address = Some(unmatched.remove(index).address.clone());
                }
            }
            report.unusable_hardware = unmatched.into_iter().cloned().collect();
            report.pci_display_devices = hardware;
            Ok(report)
        }
        Err(err) if !hardware.is_empty() && wants_hardware_evidence(&err) => {
            Err(DetectError::PresentButUnusable { hardware, underlying: Box::new(err) })
        }
        Err(err) => Err(err),
    }
}

fn enumerate(api: &ze::ZeApi, loader: String) -> Result<DeviceReport, DetectError> {
    let mut driver_count: u32 = 0;
    // SAFETY: the two-call count-then-fill idiom Level Zero specifies. A null array pointer
    // with a zero count asks only for the count.
    let result = unsafe { (api.ze_driver_get)(&mut driver_count, std::ptr::null_mut()) };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(ze_err("zeDriverGet", result));
    }
    if driver_count == 0 {
        return Err(DetectError::NoDrivers);
    }

    let mut drivers: Vec<ze::ZeDriverHandle> = vec![std::ptr::null_mut(); driver_count as usize];
    // SAFETY: `drivers` has room for exactly `driver_count` handles.
    let result = unsafe { (api.ze_driver_get)(&mut driver_count, drivers.as_mut_ptr()) };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(ze_err("zeDriverGet", result));
    }
    drivers.truncate(driver_count as usize);

    let mut devices = Vec::new();
    let mut non_gpu_devices = 0u32;
    let mut total_devices = 0u32;

    for &driver in &drivers {
        // The driver version lives on the driver, not the device, which is why this crate
        // resolves `zeDriverGetProperties` in addition to the device queries.
        let mut driver_props = ze::ZeDriverProperties::new();
        // SAFETY: `driver` came from `zeDriverGet`; the struct is zeroed with its stype set.
        let result = unsafe { (api.ze_driver_get_properties)(driver, &mut driver_props) };
        if result != ze::ZE_RESULT_SUCCESS {
            return Err(ze_err("zeDriverGetProperties", result));
        }

        let mut device_count: u32 = 0;
        // SAFETY: count-then-fill, as above.
        let result =
            unsafe { (api.ze_device_get)(driver, &mut device_count, std::ptr::null_mut()) };
        if result != ze::ZE_RESULT_SUCCESS {
            return Err(ze_err("zeDeviceGet", result));
        }
        if device_count == 0 {
            continue;
        }

        let mut handles: Vec<ze::ZeDeviceHandle> =
            vec![std::ptr::null_mut(); device_count as usize];
        // SAFETY: `handles` has room for exactly `device_count` handles.
        let result =
            unsafe { (api.ze_device_get)(driver, &mut device_count, handles.as_mut_ptr()) };
        if result != ze::ZE_RESULT_SUCCESS {
            return Err(ze_err("zeDeviceGet", result));
        }
        handles.truncate(device_count as usize);
        total_devices += device_count;

        for &handle in &handles {
            match describe(api, handle, driver_props.driver_version)? {
                Some(device) => devices.push(device),
                None => non_gpu_devices += 1,
            }
        }
    }

    if devices.is_empty() {
        return Err(if total_devices == 0 {
            DetectError::NoDevices { driver_count }
        } else {
            DetectError::NoGpuDevices { driver_count, non_gpu: non_gpu_devices }
        });
    }

    Ok(DeviceReport {
        devices,
        loader,
        driver_count,
        non_gpu_devices,
        // Filled in by `finish`, which owns the cross-reference.
        pci_display_devices: Vec::new(),
        unusable_hardware: Vec::new(),
    })
}

/// Describe one device handle, or `None` if it is not a GPU.
fn describe(
    api: &ze::ZeApi,
    handle: ze::ZeDeviceHandle,
    driver_version: u32,
) -> Result<Option<GpuDevice>, DetectError> {
    let mut props = ze::ZeDeviceProperties::new();
    // SAFETY: `handle` came from `zeDeviceGet`; the struct is zeroed with its stype set.
    let result = unsafe { (api.ze_device_get_properties)(handle, &mut props) };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(ze_err("zeDeviceGetProperties", result));
    }
    if props.device_type != ze::ZE_DEVICE_TYPE_GPU {
        return Ok(None);
    }

    let mut memory_count: u32 = 0;
    // SAFETY: count-then-fill.
    let result = unsafe {
        (api.ze_device_get_memory_properties)(handle, &mut memory_count, std::ptr::null_mut())
    };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(ze_err("zeDeviceGetMemoryProperties", result));
    }
    let mut memories: Vec<ze::ZeDeviceMemoryProperties> =
        (0..memory_count).map(|_| ze::ZeDeviceMemoryProperties::new()).collect();
    if memory_count > 0 {
        // SAFETY: `memories` holds exactly `memory_count` correctly-stamped structs.
        let result = unsafe {
            (api.ze_device_get_memory_properties)(handle, &mut memory_count, memories.as_mut_ptr())
        };
        if result != ze::ZE_RESULT_SUCCESS {
            return Err(ze_err("zeDeviceGetMemoryProperties", result));
        }
        memories.truncate(memory_count as usize);
    }
    // Summed, not maxed: a device may expose several memory ordinals and all of them are
    // allocatable. Saturating because this is a report, and a driver reporting nonsense should
    // not take the process down.
    let total_memory_bytes = memories.iter().fold(0u64, |acc, m| acc.saturating_add(m.total_size));

    let mut compute = ze::ZeDeviceComputeProperties::new();
    // SAFETY: `handle` is valid; the struct is zeroed with its stype set.
    let result = unsafe { (api.ze_device_get_compute_properties)(handle, &mut compute) };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(ze_err("zeDeviceGetComputeProperties", result));
    }
    let reported = (compute.num_sub_group_sizes as usize).min(ze::ZE_SUBGROUPSIZE_COUNT);
    let subgroup_sizes = compute.sub_group_sizes[..reported].to_vec();

    Ok(Some(GpuDevice {
        name: ze::c_str_field(&props.name),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        uuid: props.uuid,
        is_integrated: props.flags & ze::ZE_DEVICE_PROPERTY_FLAG_INTEGRATED != 0,
        total_memory_bytes,
        max_alloc_bytes: props.max_mem_alloc_size,
        driver_version,
        compute_units: props
            .num_slices
            .saturating_mul(props.num_subslices_per_slice)
            .saturating_mul(props.num_eus_per_subslice),
        max_total_group_size: compute.max_total_group_size,
        subgroup_sizes,
        pci_address: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    fn device(name: &str, integrated: bool) -> GpuDevice {
        GpuDevice {
            name: name.to_string(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: 0,
            uuid: [0; 16],
            is_integrated: integrated,
            total_memory_bytes: 0,
            max_alloc_bytes: 0,
            driver_version: 0,
            compute_units: 0,
            max_total_group_size: 0,
            subgroup_sizes: vec![],
            pci_address: None,
        }
    }

    fn pci(address: &str, device_id: u32, driver: Option<&str>) -> PciDevice {
        PciDevice {
            address: address.to_string(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id,
            class_code: 0x03_0000,
            kernel_driver: driver.map(str::to_string),
            drm_card: None,
            drm_render_node: None,
            boot_vga: false,
        }
    }

    fn report_of(devices: Vec<GpuDevice>) -> DeviceReport {
        DeviceReport {
            devices,
            loader: DEFAULT_LOADER_SONAME.to_string(),
            driver_count: 1,
            non_gpu_devices: 0,
            pci_display_devices: vec![],
            unusable_hardware: vec![],
        }
    }

    // The whole point of the second probe: "no devices" is true and useless when the card is
    // sitting in the slot.
    #[test]
    fn an_unbound_card_turns_a_useless_failure_into_the_kernel_module_to_load() {
        let err = finish(Err(DetectError::NoDrivers), vec![pci("0000:04:00.0", 0xe20b, None)])
            .expect_err("no drivers means no report");
        let message = err.to_string();
        assert!(message.contains("physically has one"), "{message}");
        assert!(message.contains("0000:04:00.0"), "{message}");
        assert!(message.contains("8086:e20b"), "{message}");
        assert!(message.contains("No kernel driver is bound"), "{message}");
        assert!(message.contains("xe") && message.contains("i915"), "{message}");
        // The original cause survives inside the message and inside the type.
        assert!(message.contains("zero drivers registered"), "{message}");
        assert!(matches!(err.root_cause(), DetectError::NoDrivers), "{err:?}");
    }

    // The other half of the same story, and the one that is easy to get wrong: the kernel is
    // fine, so telling the user to load `xe` would send them somewhere there is nothing to fix.
    // A missing loader is MoEArc's packaging problem. Telling the user to go install a
    // compute runtime here would send them to fix something that is not broken.
    #[test]
    fn a_missing_loader_is_not_blamed_on_the_user_space_runtime() {
        let err = finish(
            Err(DetectError::LoaderNotFound {
                loader: DEFAULT_LOADER_SONAME.to_string(),
                env: LOADER_PATH_ENV,
                source: unsafe { libloading::Library::new("libze_loader.so.absent") }
                    .expect_err("absent library"),
            }),
            vec![pci("0000:04:00.0", 0xe20b, Some("xe"))],
        )
        .expect_err("no loader means no report");
        let message = err.to_string();
        assert!(message.contains("MoEArc's side of the line"), "{message}");
        assert!(!message.contains("install your distribution's Intel Level Zero"), "{message}");
    }

    #[test]
    fn a_bound_card_that_level_zero_cannot_see_blames_the_user_space_runtime() {
        let err = finish(
            Err(DetectError::NoDevices { driver_count: 1 }),
            vec![pci("0000:04:00.0", 0xe20b, Some("xe"))],
        )
        .expect_err("no devices means no report");
        let message = err.to_string();
        assert!(message.contains("A kernel driver is bound"), "{message}");
        assert!(message.contains("compute runtime"), "{message}");
        assert!(!message.contains("No kernel driver is bound"), "{message}");
    }

    #[test]
    fn with_no_hardware_present_the_original_failure_is_left_alone() {
        let err = finish(Err(DetectError::NoDrivers), vec![]).expect_err("still a failure");
        assert!(matches!(err, DetectError::NoDrivers), "{err:?}");
    }

    // A device that was lost mid-query already names its cause; a PCI listing would bury it.
    #[test]
    fn a_precise_failure_is_not_padded_with_a_pci_listing() {
        let err = finish(
            Err(ze_err("zeDeviceGetProperties", ze::ZE_RESULT_ERROR_DEVICE_LOST)),
            vec![pci("0000:04:00.0", 0xe20b, Some("xe"))],
        )
        .expect_err("still a failure");
        assert!(matches!(err, DetectError::Ze { .. }), "{err:?}");
    }

    #[test]
    fn each_level_zero_device_is_matched_to_the_card_it_sits_on() {
        let mut igpu = device("iGPU", true);
        igpu.device_id = 0x7d67;
        let mut arc = device("discrete", false);
        arc.device_id = 0xe20b;

        let report = finish(
            Ok(report_of(vec![arc, igpu])),
            vec![
                pci("0000:00:02.0", 0x7d67, Some("i915")),
                pci("0000:04:00.0", 0xe20b, Some("xe")),
            ],
        )
        .expect("a report");

        assert_eq!(report.devices[0].pci_address.as_deref(), Some("0000:04:00.0"));
        assert_eq!(report.devices[1].pci_address.as_deref(), Some("0000:00:02.0"));
        assert!(report.unusable_hardware.is_empty());
        assert_eq!(report.pci_display_devices.len(), 2);
    }

    #[test]
    fn a_card_level_zero_did_not_expose_is_noted_alongside_a_working_one() {
        let mut arc = device("discrete", false);
        arc.device_id = 0xe20b;
        let report = finish(
            Ok(report_of(vec![arc])),
            vec![
                pci("0000:00:02.0", 0x7d67, Some("i915")),
                pci("0000:04:00.0", 0xe20b, Some("xe")),
            ],
        )
        .expect("a partial success is still a success");

        assert_eq!(report.devices.len(), 1);
        assert_eq!(report.unusable_hardware.len(), 1);
        assert_eq!(report.unusable_hardware[0].address, "0000:00:02.0");
    }

    // Virtualised or unusual topology: Level Zero is the authority on usability, so this is
    // recorded and never escalated.
    #[test]
    fn a_device_with_no_pci_match_is_reported_without_an_address_and_without_complaint() {
        let report = finish(Ok(report_of(vec![device("virtual", false)])), vec![])
            .expect("an unmatched device is not a failure");
        assert_eq!(report.devices.len(), 1);
        assert!(report.devices[0].pci_address.is_none());
    }

    // The FFI structs are hand-transcribed, so their layout is an assumption the driver will
    // not check for us. These pin it: a field inserted, reordered or mistyped moves an offset
    // and fails here, instead of surfacing as plausible-looking garbage from the driver.
    #[test]
    fn device_properties_layout_matches_the_header() {
        assert_eq!(offset_of!(ze::ZeDeviceProperties, stype), 0);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, p_next), 8);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, device_type), 16);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, vendor_id), 20);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, device_id), 24);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, flags), 28);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, max_mem_alloc_size), 40);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, num_slices), 72);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, timer_resolution), 80);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, uuid), 96);
        assert_eq!(offset_of!(ze::ZeDeviceProperties, name), 112);
        assert_eq!(size_of::<ze::ZeDeviceProperties>(), 368);
    }

    #[test]
    fn memory_and_compute_property_layouts_match_the_header() {
        assert_eq!(offset_of!(ze::ZeDeviceMemoryProperties, flags), 16);
        assert_eq!(offset_of!(ze::ZeDeviceMemoryProperties, total_size), 32);
        assert_eq!(offset_of!(ze::ZeDeviceMemoryProperties, name), 40);
        assert_eq!(size_of::<ze::ZeDeviceMemoryProperties>(), 296);

        assert_eq!(offset_of!(ze::ZeDeviceComputeProperties, max_total_group_size), 16);
        assert_eq!(offset_of!(ze::ZeDeviceComputeProperties, num_sub_group_sizes), 48);
        assert_eq!(offset_of!(ze::ZeDeviceComputeProperties, sub_group_sizes), 52);
        assert_eq!(size_of::<ze::ZeDeviceComputeProperties>(), 88);

        assert_eq!(offset_of!(ze::ZeDriverProperties, uuid), 16);
        assert_eq!(offset_of!(ze::ZeDriverProperties, driver_version), 32);
        assert_eq!(size_of::<ze::ZeDriverProperties>(), 40);
    }

    #[test]
    fn every_property_struct_carries_its_stype() {
        assert_eq!(ze::ZeDeviceProperties::new().stype, ze::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES);
        assert_eq!(
            ze::ZeDeviceMemoryProperties::new().stype,
            ze::ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
        );
        assert_eq!(
            ze::ZeDeviceComputeProperties::new().stype,
            ze::ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES
        );
        assert_eq!(ze::ZeDriverProperties::new().stype, ze::ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES);
    }

    #[test]
    fn discrete_wins_over_an_integrated_device_listed_first() {
        let report = DeviceReport {
            devices: vec![device("iGPU", true), device("discrete", false)],
            loader: DEFAULT_LOADER_SONAME.to_string(),
            driver_count: 1,
            non_gpu_devices: 0,
            pci_display_devices: vec![],
            unusable_hardware: vec![],
        };
        assert_eq!(report.preferred().unwrap().name, "discrete");
    }

    #[test]
    fn an_integrated_only_machine_still_gets_a_device() {
        let report = DeviceReport {
            devices: vec![device("iGPU", true)],
            loader: DEFAULT_LOADER_SONAME.to_string(),
            driver_count: 1,
            non_gpu_devices: 0,
            pci_display_devices: vec![],
            unusable_hardware: vec![],
        };
        assert_eq!(report.preferred().unwrap().name, "iGPU");
    }

    #[test]
    fn uuid_is_rendered_in_the_conventional_form() {
        let mut d = device("x", false);
        d.uuid = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
            0x66, 0x77,
        ];
        assert_eq!(d.uuid_string(), "01234567-89ab-cdef-0011-223344556677");
    }

    // A code with no dedicated sentence must still produce one, because `docs/ux.md` forbids
    // reporting a symptom without naming the cause.
    #[test]
    fn an_unmapped_result_code_still_gets_a_sentence_and_names_the_call() {
        let message = ze_err("zeDeviceGet", 0x7fff_0000).to_string();
        assert!(message.contains("zeDeviceGet"), "{message}");
        assert!(message.contains("no description for"), "{message}");
        assert!(message.contains("0x7fff0000"), "{message}");
    }

    #[test]
    fn insufficient_permissions_is_reported_as_a_permissions_problem() {
        let err = ze_err("zeDeviceGet", ze::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        assert!(matches!(err, DetectError::PermissionDenied { .. }));
        assert!(err.to_string().contains("render"), "{err}");
    }

    // The one dependency `docs/ux.md` permits us to ask for must be named exactly.
    #[test]
    fn the_no_driver_errors_name_the_kernel_module() {
        for message in [
            DetectError::DriverUninitialized.to_string(),
            DetectError::NoDrivers.to_string(),
            DetectError::NoDevices { driver_count: 1 }.to_string(),
        ] {
            assert!(message.contains("xe"), "{message}");
            assert!(message.contains("i915"), "{message}");
        }
    }

    #[test]
    fn no_gpu_devices_is_distinct_from_no_devices() {
        let none = DetectError::NoDevices { driver_count: 2 }.to_string();
        let non_gpu = DetectError::NoGpuDevices { driver_count: 2, non_gpu: 3 }.to_string();
        assert!(none.contains("no devices at all"), "{none}");
        assert!(non_gpu.contains("none of them is a GPU"), "{non_gpu}");
    }

    #[test]
    fn the_loader_override_replaces_the_default_soname() {
        assert_eq!(loader_or_default(None), OsString::from(DEFAULT_LOADER_SONAME));
        assert_eq!(
            loader_or_default(Some(OsString::from("/somewhere/else/libze_loader.so.1"))),
            OsString::from("/somewhere/else/libze_loader.so.1")
        );
    }
}
