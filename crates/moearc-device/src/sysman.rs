//! Live telemetry, from the Level Zero Sysman API (`zes_*`).
//!
//! Separate from device discovery on purpose. Core Level Zero (`ze_*`) answers what the
//! hardware *is* — a static fact, and the thing MoEArc cannot start without. Sysman answers
//! what it is *doing right now*, and every part of it is optional: a driver may withhold a
//! domain, and a user may lack the permission to read one. Nothing here may ever be the reason
//! detection fails, so it has its own entry point and its own error type.
//!
//! The scope is deliberately one number: **free device memory**. `docs/ux.md` promises the
//! expert/KV split is computed rather than configured, and `plan_cache_budget` needs a budget
//! measured on the card as it is at that moment — installed VRAM is not that number, because
//! a desktop compositor got there first. Temperature, power and frequency are the same API
//! shape and would be a natural extension; they are not implemented, because nothing in the
//! engine consumes them yet and an unused reading is a number nobody has checked.
//!
//! Same runtime-loading rule as [`crate::detect`]: the loader is `dlopen`ed, never linked.

use std::ffi::{OsStr, c_void};

use libloading::Library;
use thiserror::Error;

use crate::ze::{self, LoadError};
use crate::{LOADER_PATH_ENV, loader_or_default};

type ZesDriverHandle = *mut c_void;
type ZesDeviceHandle = *mut c_void;
type ZesMemHandle = *mut c_void;

const ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x1;
const ZES_STRUCTURE_TYPE_MEM_PROPERTIES: u32 = 0xb;
const ZES_STRUCTURE_TYPE_MEM_STATE: u32 = 0x1e;

const ZES_STRING_PROPERTY_SIZE: usize = 64;

/// `zes_mem_loc_t::ZES_MEM_LOC_DEVICE`. The other value is system memory an integrated device
/// borrows, which is not a VRAM budget.
const ZES_MEM_LOC_DEVICE: u32 = 1;
/// `zes_mem_health_t`. `UNKNOWN` is the value this project's own B580 reports, which is why
/// health is modelled as three states and not as a bool: treating "cannot be determined" as
/// "unhealthy" would raise a false alarm on the reference card.
const ZES_MEM_HEALTH_UNKNOWN: u32 = 0;
const ZES_MEM_HEALTH_OK: u32 = 1;
const ZES_MEM_HEALTH_DEGRADED: u32 = 2;
const ZES_MEM_HEALTH_CRITICAL: u32 = 3;
const ZES_MEM_HEALTH_REPLACE: u32 = 4;

/// `zes_device_properties_t`. Only `core.uuid` is read — it is how a sysman device is tied back
/// to the [`crate::GpuDevice`] it describes — but the whole struct must be laid out correctly,
/// because the driver writes all of it.
#[repr(C)]
struct ZesDeviceProperties {
    stype: u32,
    p_next: *mut c_void,
    core: ze::ZeDeviceProperties,
    num_subdevices: u32,
    serial_number: [u8; ZES_STRING_PROPERTY_SIZE],
    board_number: [u8; ZES_STRING_PROPERTY_SIZE],
    brand_name: [u8; ZES_STRING_PROPERTY_SIZE],
    model_name: [u8; ZES_STRING_PROPERTY_SIZE],
    vendor_name: [u8; ZES_STRING_PROPERTY_SIZE],
    driver_version: [u8; ZES_STRING_PROPERTY_SIZE],
}

impl ZesDeviceProperties {
    fn new() -> Self {
        Self {
            stype: ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES,
            p_next: std::ptr::null_mut(),
            // The nested core struct carries its own stype, which the driver also inspects.
            core: ze::ZeDeviceProperties::new(),
            num_subdevices: 0,
            serial_number: [0; ZES_STRING_PROPERTY_SIZE],
            board_number: [0; ZES_STRING_PROPERTY_SIZE],
            brand_name: [0; ZES_STRING_PROPERTY_SIZE],
            model_name: [0; ZES_STRING_PROPERTY_SIZE],
            vendor_name: [0; ZES_STRING_PROPERTY_SIZE],
            driver_version: [0; ZES_STRING_PROPERTY_SIZE],
        }
    }
}

/// `zes_mem_properties_t`.
#[repr(C)]
struct ZesMemProperties {
    stype: u32,
    p_next: *mut c_void,
    mem_type: u32,
    on_subdevice: u8,
    subdevice_id: u32,
    location: u32,
    physical_size: u64,
    bus_width: i32,
    num_channels: i32,
}

impl ZesMemProperties {
    fn new() -> Self {
        Self {
            stype: ZES_STRUCTURE_TYPE_MEM_PROPERTIES,
            p_next: std::ptr::null_mut(),
            mem_type: 0,
            on_subdevice: 0,
            subdevice_id: 0,
            location: 0,
            physical_size: 0,
            bus_width: 0,
            num_channels: 0,
        }
    }
}

/// `zes_mem_state_t`. Note `pNext` is `const void*` here, unlike the others; the layout is the
/// same and Rust has no const-pointer distinction to preserve at this level.
#[repr(C)]
struct ZesMemState {
    stype: u32,
    p_next: *const c_void,
    health: u32,
    free: u64,
    size: u64,
}

impl ZesMemState {
    fn new() -> Self {
        Self {
            stype: ZES_STRUCTURE_TYPE_MEM_STATE,
            p_next: std::ptr::null(),
            health: 0,
            free: 0,
            size: 0,
        }
    }
}

type PfnZesInit = unsafe extern "C" fn(u32) -> ze::ZeResult;
type PfnZesDriverGet = unsafe extern "C" fn(*mut u32, *mut ZesDriverHandle) -> ze::ZeResult;
type PfnZesDeviceGet =
    unsafe extern "C" fn(ZesDriverHandle, *mut u32, *mut ZesDeviceHandle) -> ze::ZeResult;
type PfnZesDeviceGetProperties =
    unsafe extern "C" fn(ZesDeviceHandle, *mut ZesDeviceProperties) -> ze::ZeResult;
type PfnZesDeviceEnumMemoryModules =
    unsafe extern "C" fn(ZesDeviceHandle, *mut u32, *mut ZesMemHandle) -> ze::ZeResult;
type PfnZesMemoryGetProperties =
    unsafe extern "C" fn(ZesMemHandle, *mut ZesMemProperties) -> ze::ZeResult;
type PfnZesMemoryGetState = unsafe extern "C" fn(ZesMemHandle, *mut ZesMemState) -> ze::ZeResult;

struct ZesApi {
    zes_init: PfnZesInit,
    zes_driver_get: PfnZesDriverGet,
    zes_device_get: PfnZesDeviceGet,
    zes_device_get_properties: PfnZesDeviceGetProperties,
    zes_device_enum_memory_modules: PfnZesDeviceEnumMemoryModules,
    zes_memory_get_properties: PfnZesMemoryGetProperties,
    zes_memory_get_state: PfnZesMemoryGetState,
    library: Library,
}

impl ZesApi {
    fn open(path: &OsStr) -> Result<Self, LoadError> {
        // SAFETY: as in `ze::ZeApi::open` — dlopen runs the library's initialisers.
        let library = unsafe { Library::new(path) }.map_err(LoadError::Open)?;

        macro_rules! sym {
            ($name:literal) => {{
                // SAFETY: signatures transcribed from `zes_api.h`; layouts pinned by tests.
                let symbol = unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|source| LoadError::Symbol { name: $name, source })?;
                *symbol
            }};
        }

        Ok(Self {
            zes_init: sym!("zesInit"),
            zes_driver_get: sym!("zesDriverGet"),
            zes_device_get: sym!("zesDeviceGet"),
            zes_device_get_properties: sym!("zesDeviceGetProperties"),
            zes_device_enum_memory_modules: sym!("zesDeviceEnumMemoryModules"),
            zes_memory_get_properties: sym!("zesMemoryGetProperties"),
            zes_memory_get_state: sym!("zesMemoryGetState"),
            library,
        })
    }

    /// As in `ze::ZeApi::leak`: once sysman has initialised, the loader owns process-global
    /// state and this crate does not want to depend on unmapping it being safe.
    fn leak(self) {
        std::mem::forget(self.library);
    }
}

/// What the driver says about a memory module's condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryHealth {
    /// The driver does not track it. **This is what an Arc B580 reports**, so it must not be
    /// presented as a problem.
    Unknown,
    /// All channels healthy.
    Ok,
    /// Excessive correctable errors; the driver wants the device reset.
    Degraded,
    /// Running with reduced memory because banks have uncorrectable errors.
    Critical,
    /// The driver considers the device due for replacement.
    ReplaceNeeded,
    /// A value this build does not recognise, kept rather than discarded.
    Other(u32),
}

impl MemoryHealth {
    fn from_raw(raw: u32) -> Self {
        match raw {
            ZES_MEM_HEALTH_UNKNOWN => Self::Unknown,
            ZES_MEM_HEALTH_OK => Self::Ok,
            ZES_MEM_HEALTH_DEGRADED => Self::Degraded,
            ZES_MEM_HEALTH_CRITICAL => Self::Critical,
            ZES_MEM_HEALTH_REPLACE => Self::ReplaceNeeded,
            other => Self::Other(other),
        }
    }

    /// Whether this is worth telling the user about. `Unknown` is not.
    pub fn is_a_problem(self) -> bool {
        matches!(self, Self::Degraded | Self::Critical | Self::ReplaceNeeded)
    }
}

/// One memory module on a device, as it stands right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryModule {
    /// Installed capacity, when the driver knows it.
    ///
    /// `None` where Level Zero reports 0, which the spec defines as "not known" — and which is
    /// what the Arc B580 reports for its own VRAM. Modelled as an `Option` because a 0 treated
    /// as a capacity silently turns into "the card is completely full", or into a used figure
    /// of zero, depending on which way it is subtracted.
    pub physical_bytes: Option<u64>,
    /// Free physical memory, measured now. This one the B580 does report.
    pub free_bytes: u64,
    /// True for memory on the card; false for host memory an integrated device borrows.
    pub on_device: bool,
    /// The driver's opinion of the module's condition.
    pub health: MemoryHealth,
}

/// A live reading for one GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceTelemetry {
    /// Matches [`crate::GpuDevice::uuid`]. Sysman enumerates devices separately from core
    /// Level Zero and the two orders are *observably* different — on this project's reference
    /// machine sysman lists the iGPU first and core Level Zero lists the discrete card first —
    /// so the two are joined on identity and never on index.
    pub uuid: [u8; 16],
    /// Every memory module the driver exposes.
    pub memory: Vec<MemoryModule>,
}

impl DeviceTelemetry {
    /// Free bytes across the card's own memory, or `None` if it has none of its own.
    ///
    /// This — not installed VRAM — is the figure a cache plan should be built from. Host memory
    /// modules are excluded: an integrated device reports system RAM there, which is not a
    /// budget this engine is entitled to. `None` rather than 0 for exactly that case, because
    /// a 0 would read as "the card is full" when the truth is "this device has no VRAM".
    pub fn free_device_memory_bytes(&self) -> Option<u64> {
        let mut modules = self.memory.iter().filter(|m| m.on_device).peekable();
        modules.peek()?;
        Some(modules.fold(0u64, |acc, m| acc.saturating_add(m.free_bytes)))
    }

    /// Bytes of the card's own memory in use, or `None` when the driver does not report a
    /// capacity to subtract from.
    ///
    /// Not derivable on every card: the B580's driver reports free memory but not physical
    /// size, so "in use" is genuinely unknown there rather than zero.
    pub fn used_device_memory_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for module in self.memory.iter().filter(|m| m.on_device) {
            let physical = module.physical_bytes?;
            total = total.saturating_add(physical.saturating_sub(module.free_bytes));
            any = true;
        }
        any.then_some(total)
    }

    /// Capacity of the host-memory modules this device exposes, when the driver reports one.
    ///
    /// **Never a budget.** It is here to make a refusal concrete: on the reference iGPU this
    /// reads 98,257,694,720 bytes, which is exactly `MemTotal` from `/proc/meminfo` — the fact
    /// that explains where the 85.58 GiB the same device reports as "device memory" comes from.
    /// See [`crate::fitness`].
    pub fn host_memory_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut any = false;
        for module in self.memory.iter().filter(|m| !m.on_device) {
            if let Some(physical) = module.physical_bytes {
                total = total.saturating_add(physical);
                any = true;
            }
        }
        any.then_some(total)
    }

    /// Modules the driver is actively complaining about.
    pub fn unhealthy_modules(&self) -> impl Iterator<Item = &MemoryModule> {
        self.memory.iter().filter(|m| m.health.is_a_problem())
    }
}

/// Why a live reading could not be taken.
///
/// Kept apart from [`crate::DetectError`] so that no caller can accidentally treat a missing
/// telemetry reading as a missing GPU.
#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error(
        "could not load `{loader}` to read GPU telemetry: {source}. The GPU itself is \
         unaffected; only live memory, and anything MoEArc would compute from it, is \
         unavailable."
    )]
    LoaderNotFound { loader: String, source: libloading::Error },

    #[error(
        "`{loader}` does not export `{symbol}`, so it has no Sysman support ({source}). This \
         Level Zero loader predates the Sysman API or was built without it; MoEArc can still \
         run, but it cannot measure free VRAM and will have to assume the card is idle."
    )]
    NoSysmanSupport { loader: String, symbol: &'static str, source: libloading::Error },

    #[error(
        "the Level Zero Sysman API is present but reported no devices to measure. Detection \
         and inference are unaffected; free-VRAM readings are not available on this system."
    )]
    NoDevices,

    #[error(
        "Level Zero refused access to GPU telemetry during {call} \
         (ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS). Some Sysman domains are readable only by \
         privileged users; MoEArc does not need them to run."
    )]
    PermissionDenied { call: &'static str },

    #[error("{call} failed: {meaning} (ze_result_t {code:#010x}).")]
    Zes { call: &'static str, code: ze::ZeResult, meaning: &'static str },
}

fn zes_err(call: &'static str, code: ze::ZeResult) -> TelemetryError {
    if code == ze::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS {
        return TelemetryError::PermissionDenied { call };
    }
    TelemetryError::Zes { call, code, meaning: crate::meaning(code) }
}

/// Take one live reading of every GPU sysman can see.
///
/// A snapshot, not a subscription: each call re-enters the API. That suits a stats panel
/// polling at human speed, and it keeps this out of the inference path entirely.
pub fn telemetry() -> Result<Vec<DeviceTelemetry>, TelemetryError> {
    telemetry_with_loader(&loader_or_default(std::env::var_os(LOADER_PATH_ENV)))
}

/// [`telemetry`] against a named loader. Defaults to [`DEFAULT_LOADER_SONAME`] like the rest of
/// the crate.
pub fn telemetry_with_loader(loader: &OsStr) -> Result<Vec<DeviceTelemetry>, TelemetryError> {
    let shown = loader.to_string_lossy().into_owned();
    let api = ZesApi::open(loader).map_err(|e| match e {
        LoadError::Open(source) => TelemetryError::LoaderNotFound { loader: shown, source },
        LoadError::Symbol { name, source } => {
            TelemetryError::NoSysmanSupport { loader: shown, symbol: name, source }
        }
    })?;

    let result = collect(&api);
    api.leak();
    result
}

fn collect(api: &ZesApi) -> Result<Vec<DeviceTelemetry>, TelemetryError> {
    // `zesInit` takes no flags today; the spec says it must be 0.
    // SAFETY: no pointers are passed.
    let result = unsafe { (api.zes_init)(0) };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(zes_err("zesInit", result));
    }

    let drivers = enumerate_handles(|count, out| {
        // SAFETY: count-then-fill, as specified.
        unsafe { (api.zes_driver_get)(count, out) }
    })
    .map_err(|code| zes_err("zesDriverGet", code))?;

    let mut telemetry = Vec::new();
    for driver in drivers {
        let devices = enumerate_handles(|count, out| {
            // SAFETY: `driver` came from `zesDriverGet`.
            unsafe { (api.zes_device_get)(driver, count, out) }
        })
        .map_err(|code| zes_err("zesDeviceGet", code))?;

        for device in devices {
            telemetry.push(read_device(api, device)?);
        }
    }

    if telemetry.is_empty() {
        return Err(TelemetryError::NoDevices);
    }
    Ok(telemetry)
}

fn read_device(api: &ZesApi, device: ZesDeviceHandle) -> Result<DeviceTelemetry, TelemetryError> {
    let mut props = ZesDeviceProperties::new();
    // SAFETY: `device` came from `zesDeviceGet`; the struct is zeroed with its stype set.
    let result = unsafe { (api.zes_device_get_properties)(device, &mut props) };
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(zes_err("zesDeviceGetProperties", result));
    }

    let modules = enumerate_handles(|count, out| {
        // SAFETY: count-then-fill on a valid device handle.
        unsafe { (api.zes_device_enum_memory_modules)(device, count, out) }
    })
    .map_err(|code| zes_err("zesDeviceEnumMemoryModules", code))?;

    let mut memory = Vec::with_capacity(modules.len());
    for module in modules {
        let mut mem_props = ZesMemProperties::new();
        // SAFETY: `module` came from `zesDeviceEnumMemoryModules`.
        let result = unsafe { (api.zes_memory_get_properties)(module, &mut mem_props) };
        if result != ze::ZE_RESULT_SUCCESS {
            return Err(zes_err("zesMemoryGetProperties", result));
        }

        let mut state = ZesMemState::new();
        // SAFETY: same handle, freshly stamped struct.
        let result = unsafe { (api.zes_memory_get_state)(module, &mut state) };
        if result != ze::ZE_RESULT_SUCCESS {
            return Err(zes_err("zesMemoryGetState", result));
        }

        memory.push(MemoryModule {
            // `zes_mem_state_t::size` is documented as deprecated and no longer a reliable
            // capacity, so capacity is taken from the properties and only `free` from the
            // state. The spec defines a physical size of 0 as "not known", not as "empty".
            physical_bytes: (mem_props.physical_size > 0).then_some(mem_props.physical_size),
            free_bytes: state.free,
            on_device: mem_props.location == ZES_MEM_LOC_DEVICE,
            health: MemoryHealth::from_raw(state.health),
        });
    }

    Ok(DeviceTelemetry { uuid: props.core.uuid, memory })
}

/// The count-then-fill idiom every Level Zero enumeration uses, once instead of six times.
///
/// Concrete in the handle type rather than generic: every Level Zero handle is an opaque
/// `void*`, so a pre-zeroed `Vec` is enough and no uninitialised memory is ever exposed.
fn enumerate_handles(
    mut call: impl FnMut(*mut u32, *mut *mut c_void) -> ze::ZeResult,
) -> Result<Vec<*mut c_void>, ze::ZeResult> {
    let mut count: u32 = 0;
    let result = call(&mut count, std::ptr::null_mut());
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(result);
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut handles: Vec<*mut c_void> = vec![std::ptr::null_mut(); count as usize];
    let result = call(&mut count, handles.as_mut_ptr());
    if result != ze::ZE_RESULT_SUCCESS {
        return Err(result);
    }
    // The driver may report fewer than it first said.
    handles.truncate(count as usize);
    Ok(handles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_LOADER_SONAME;
    use std::ffi::OsString;
    use std::mem::{offset_of, size_of};

    // Verified against `zes_api.h`, same as the core structs. `zes_device_properties_t` embeds
    // the core struct, so an error in either one shows up here.
    #[test]
    fn sysman_struct_layouts_match_the_header() {
        assert_eq!(offset_of!(ZesDeviceProperties, core), 16);
        assert_eq!(offset_of!(ZesDeviceProperties, num_subdevices), 384);
        assert_eq!(offset_of!(ZesDeviceProperties, serial_number), 388);
        assert_eq!(offset_of!(ZesDeviceProperties, driver_version), 708);
        assert_eq!(size_of::<ZesDeviceProperties>(), 776);

        assert_eq!(offset_of!(ZesMemProperties, location), 28);
        assert_eq!(offset_of!(ZesMemProperties, physical_size), 32);
        assert_eq!(size_of::<ZesMemProperties>(), 48);

        assert_eq!(offset_of!(ZesMemState, health), 16);
        assert_eq!(offset_of!(ZesMemState, free), 24);
        assert_eq!(size_of::<ZesMemState>(), 40);
    }

    fn module(physical: Option<u64>, free: u64, on_device: bool) -> MemoryModule {
        MemoryModule {
            physical_bytes: physical,
            free_bytes: free,
            on_device,
            health: MemoryHealth::Unknown,
        }
    }

    #[test]
    fn host_memory_modules_are_excluded_from_the_device_budget() {
        let reading = DeviceTelemetry {
            uuid: [0; 16],
            memory: vec![module(Some(12_000), 9_000, true), module(Some(64_000), 60_000, false)],
        };
        assert_eq!(reading.free_device_memory_bytes(), Some(9_000));
        assert_eq!(reading.used_device_memory_bytes(), Some(3_000));
    }

    /// The shape the Arc B580 actually reports: free memory known, capacity not. "In use" is
    /// unknown there, and reporting it as 0 would be a fabricated number.
    #[test]
    fn an_unknown_capacity_makes_used_memory_unknown_not_zero() {
        let reading =
            DeviceTelemetry { uuid: [0; 16], memory: vec![module(None, 12_567_810_048, true)] };
        assert_eq!(reading.free_device_memory_bytes(), Some(12_567_810_048));
        assert_eq!(reading.used_device_memory_bytes(), None);
    }

    /// The shape the Arrow Lake iGPU actually reports: one host memory module and no VRAM.
    /// A device budget of 0 would read as "full"; the truth is "there is no such pool".
    #[test]
    fn a_device_with_no_local_memory_reports_no_budget_rather_than_an_empty_one() {
        let reading = DeviceTelemetry {
            uuid: [0; 16],
            memory: vec![module(Some(98_257_694_720), 42_207_121_408, false)],
        };
        assert_eq!(reading.free_device_memory_bytes(), None);
        assert_eq!(reading.used_device_memory_bytes(), None);
    }

    #[test]
    fn the_host_pool_is_reported_separately_and_only_when_the_driver_gives_a_capacity() {
        let igpu = DeviceTelemetry {
            uuid: [0; 16],
            memory: vec![module(Some(98_257_694_720), 42_207_121_408, false)],
        };
        assert_eq!(igpu.host_memory_bytes(), Some(98_257_694_720));
        assert_eq!(igpu.free_device_memory_bytes(), None);

        let b580 =
            DeviceTelemetry { uuid: [0; 16], memory: vec![module(None, 12_567_810_048, true)] };
        assert_eq!(b580.host_memory_bytes(), None);

        // A host module the driver gave no capacity for is not a zero-byte pool.
        let unknown = DeviceTelemetry { uuid: [0; 16], memory: vec![module(None, 1_000, false)] };
        assert_eq!(unknown.host_memory_bytes(), None);
    }

    /// `UNKNOWN` is what the reference card reports, and it must never look like a fault.
    #[test]
    fn unknown_health_is_not_treated_as_a_fault() {
        assert!(!MemoryHealth::from_raw(0).is_a_problem());
        assert!(!MemoryHealth::from_raw(1).is_a_problem());
        assert!(MemoryHealth::from_raw(2).is_a_problem());
        assert!(MemoryHealth::from_raw(4).is_a_problem());
        assert_eq!(MemoryHealth::from_raw(99), MemoryHealth::Other(99));
        assert!(!MemoryHealth::from_raw(99).is_a_problem());
    }

    /// Telemetry must never be mistaken for a detection failure.
    #[test]
    fn a_loader_without_sysman_says_inference_is_unaffected() {
        let err = telemetry_with_loader(OsStr::new("libc.so.6"))
            .expect_err("libc exports no Sysman entry points");
        match &err {
            TelemetryError::NoSysmanSupport { symbol, .. } => assert_eq!(*symbol, "zesInit"),
            other => panic!("expected NoSysmanSupport, got {other:?}"),
        }
        assert!(err.to_string().contains("MoEArc can still run"), "{err}");
    }

    #[test]
    fn a_missing_loader_is_reported_without_implying_the_gpu_is_gone() {
        let err = telemetry_with_loader(OsStr::new("libze_loader.so.this-does-not-exist"))
            .expect_err("a nonexistent library cannot be loaded");
        assert!(matches!(err, TelemetryError::LoaderNotFound { .. }), "{err:?}");
        assert!(err.to_string().contains("GPU itself is unaffected"), "{err}");
    }

    #[test]
    fn the_default_loader_is_the_one_the_rest_of_the_crate_uses() {
        assert_eq!(loader_or_default(None), OsString::from(DEFAULT_LOADER_SONAME));
    }
}
