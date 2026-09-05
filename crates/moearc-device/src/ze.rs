//! Hand-written Level Zero FFI and a runtime loader for it.
//!
//! **Level Zero is opened with `dlopen`, never linked.** `docs/ux.md` makes the binary's
//! self-sufficiency a product requirement: MoEArc ships as one static binary that brings its
//! own dependencies, and the *only* thing it is allowed to ask a user to install is the
//! kernel-side GPU driver. Linking `libze_loader` at build time breaks that twice over — the
//! build would need Level Zero headers, and on a machine without the loader the binary would
//! fail in the dynamic linker before `main`, so the user gets `error while loading shared
//! libraries` instead of the sentence we wrote for exactly this case. With `dlopen` the
//! binary always starts and always gets to explain itself.
//!
//! The declarations below are transcribed by hand from `level_zero/ze_api.h` (Apache-2.0)
//! rather than generated, so the build needs no headers and no oneAPI installation. That
//! makes field order load-bearing: every one of these structs starts with `stype`/`pNext`,
//! and the driver reads `stype` to decide what it was handed. A struct that is not fully
//! zeroed with the right `stype` gets rejected or filled with garbage, so each one is built
//! by an explicit constructor here and never by `Default`. `tests` in `lib.rs` pin the
//! resulting sizes and field offsets.

use std::ffi::{OsStr, c_char, c_void};

use libloading::Library;

/// `ze_result_t`. The C enum's largest value is `0x7fffffff`, so it is an `int` on every
/// platform we care about; `u32` has the same ABI and no negative values to explain away.
pub type ZeResult = u32;

/// Opaque driver/device handles. Level Zero only ever hands these back to us.
pub type ZeDriverHandle = *mut c_void;
pub type ZeDeviceHandle = *mut c_void;

pub const ZE_RESULT_SUCCESS: ZeResult = 0;
pub const ZE_RESULT_ERROR_DEVICE_LOST: ZeResult = 0x7000_0001;
pub const ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: ZeResult = 0x7000_0002;
pub const ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: ZeResult = 0x7000_0003;
pub const ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET: ZeResult = 0x7000_0006;
pub const ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE: ZeResult = 0x7000_0007;
pub const ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: ZeResult = 0x7001_0000;
pub const ZE_RESULT_ERROR_NOT_AVAILABLE: ZeResult = 0x7001_0001;
pub const ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: ZeResult = 0x7002_0000;
pub const ZE_RESULT_ERROR_UNINITIALIZED: ZeResult = 0x7800_0001;
pub const ZE_RESULT_ERROR_UNSUPPORTED_VERSION: ZeResult = 0x7800_0002;
pub const ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: ZeResult = 0x7800_0003;
pub const ZE_RESULT_ERROR_INVALID_ARGUMENT: ZeResult = 0x7800_0004;
pub const ZE_RESULT_ERROR_INVALID_NULL_HANDLE: ZeResult = 0x7800_0005;
pub const ZE_RESULT_ERROR_INVALID_NULL_POINTER: ZeResult = 0x7800_0007;
pub const ZE_RESULT_ERROR_UNKNOWN: ZeResult = 0x7fff_fffe;

pub const ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES: u32 = 0x1;
pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
pub const ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES: u32 = 0x4;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES: u32 = 0x7;

pub const ZE_INIT_FLAG_GPU_ONLY: u32 = 1 << 0;

pub const ZE_DEVICE_TYPE_GPU: u32 = 1;

pub const ZE_DEVICE_PROPERTY_FLAG_INTEGRATED: u32 = 1 << 0;

/// `ZE_MAX_DEVICE_NAME`. Both device and memory names are this fixed-size array.
pub const ZE_MAX_DEVICE_NAME: usize = 256;
/// `ZE_MAX_DEVICE_UUID_SIZE` / `ZE_MAX_DRIVER_UUID_SIZE`, both 16.
pub const ZE_MAX_UUID_SIZE: usize = 16;
/// `ZE_SUBGROUPSIZE_COUNT`.
pub const ZE_SUBGROUPSIZE_COUNT: usize = 8;

/// `ze_driver_properties_t`.
#[repr(C)]
pub struct ZeDriverProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub uuid: [u8; ZE_MAX_UUID_SIZE],
    pub driver_version: u32,
}

impl ZeDriverProperties {
    pub fn new() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES,
            p_next: std::ptr::null_mut(),
            uuid: [0; ZE_MAX_UUID_SIZE],
            driver_version: 0,
        }
    }
}

/// `ze_device_properties_t`.
#[repr(C)]
pub struct ZeDeviceProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub device_type: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub flags: u32,
    pub subdevice_id: u32,
    pub core_clock_rate: u32,
    pub max_mem_alloc_size: u64,
    pub max_hardware_contexts: u32,
    pub max_command_queue_priority: u32,
    pub num_threads_per_eu: u32,
    pub physical_eu_simd_width: u32,
    pub num_eus_per_subslice: u32,
    pub num_subslices_per_slice: u32,
    pub num_slices: u32,
    pub timer_resolution: u64,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub uuid: [u8; ZE_MAX_UUID_SIZE],
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

impl ZeDeviceProperties {
    /// Deliberately `ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES` and not the `_1_2` variant: the two
    /// share this layout, and the only difference is that `_1_2` reports `timerResolution` in
    /// cycles/sec instead of nanoseconds. We do not read that field, and asking for the
    /// older stype is the one a 1.0 driver is guaranteed to understand.
    pub fn new() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
            p_next: std::ptr::null_mut(),
            device_type: 0,
            vendor_id: 0,
            device_id: 0,
            flags: 0,
            subdevice_id: 0,
            core_clock_rate: 0,
            max_mem_alloc_size: 0,
            max_hardware_contexts: 0,
            max_command_queue_priority: 0,
            num_threads_per_eu: 0,
            physical_eu_simd_width: 0,
            num_eus_per_subslice: 0,
            num_subslices_per_slice: 0,
            num_slices: 0,
            timer_resolution: 0,
            timestamp_valid_bits: 0,
            kernel_timestamp_valid_bits: 0,
            uuid: [0; ZE_MAX_UUID_SIZE],
            name: [0; ZE_MAX_DEVICE_NAME],
        }
    }
}

/// `ze_device_memory_properties_t`. One of these per memory ordinal.
#[repr(C)]
pub struct ZeDeviceMemoryProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub flags: u32,
    pub max_clock_rate: u32,
    pub max_bus_width: u32,
    pub total_size: u64,
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

impl ZeDeviceMemoryProperties {
    pub fn new() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES,
            p_next: std::ptr::null_mut(),
            flags: 0,
            max_clock_rate: 0,
            max_bus_width: 0,
            total_size: 0,
            name: [0; ZE_MAX_DEVICE_NAME],
        }
    }
}

/// `ze_device_compute_properties_t`.
#[repr(C)]
pub struct ZeDeviceComputeProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub max_total_group_size: u32,
    pub max_group_size_x: u32,
    pub max_group_size_y: u32,
    pub max_group_size_z: u32,
    pub max_group_count_x: u32,
    pub max_group_count_y: u32,
    pub max_group_count_z: u32,
    pub max_shared_local_memory: u32,
    pub num_sub_group_sizes: u32,
    pub sub_group_sizes: [u32; ZE_SUBGROUPSIZE_COUNT],
}

impl ZeDeviceComputeProperties {
    pub fn new() -> Self {
        Self {
            stype: ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
            p_next: std::ptr::null_mut(),
            max_total_group_size: 0,
            max_group_size_x: 0,
            max_group_size_y: 0,
            max_group_size_z: 0,
            max_group_count_x: 0,
            max_group_count_y: 0,
            max_group_count_z: 0,
            max_shared_local_memory: 0,
            num_sub_group_sizes: 0,
            sub_group_sizes: [0; ZE_SUBGROUPSIZE_COUNT],
        }
    }
}

// `ZE_APICALL` expands to nothing on Linux, so every entry point is plain `extern "C"`.
type PfnZeInit = unsafe extern "C" fn(flags: u32) -> ZeResult;
type PfnZeDriverGet = unsafe extern "C" fn(*mut u32, *mut ZeDriverHandle) -> ZeResult;
type PfnZeDriverGetProperties =
    unsafe extern "C" fn(ZeDriverHandle, *mut ZeDriverProperties) -> ZeResult;
type PfnZeDeviceGet =
    unsafe extern "C" fn(ZeDriverHandle, *mut u32, *mut ZeDeviceHandle) -> ZeResult;
type PfnZeDeviceGetProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut ZeDeviceProperties) -> ZeResult;
type PfnZeDeviceGetMemoryProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut u32, *mut ZeDeviceMemoryProperties) -> ZeResult;
type PfnZeDeviceGetComputeProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut ZeDeviceComputeProperties) -> ZeResult;

/// Why a Level Zero loader could not be opened. Kept separate from the crate's public
/// `DetectError` so this module stays free of product wording.
#[derive(Debug)]
pub enum LoadError {
    /// `dlopen` failed: the file is absent, unreadable, or not a shared object.
    Open(libloading::Error),
    /// The library opened but does not export an entry point we need — which means it is
    /// something other than a Level Zero loader, not a broken GPU.
    Symbol { name: &'static str, source: libloading::Error },
}

/// The subset of Level Zero MoEArc needs to answer "what card is in this machine".
///
/// The function pointers are only valid while `library` is mapped, so this struct owns it and
/// they are never handed out. Field order is the invariant: Rust drops fields in declaration
/// order, so `library` is declared last and therefore unmapped last.
pub struct ZeApi {
    pub ze_init: PfnZeInit,
    pub ze_driver_get: PfnZeDriverGet,
    pub ze_driver_get_properties: PfnZeDriverGetProperties,
    pub ze_device_get: PfnZeDeviceGet,
    pub ze_device_get_properties: PfnZeDeviceGetProperties,
    pub ze_device_get_memory_properties: PfnZeDeviceGetMemoryProperties,
    pub ze_device_get_compute_properties: PfnZeDeviceGetComputeProperties,
    library: Library,
}

impl ZeApi {
    /// `dlopen` the named loader and resolve every entry point up front.
    ///
    /// Resolving eagerly rather than lazily is intentional: a half-usable loader should be one
    /// legible error at startup, not a surprise partway through enumeration.
    pub fn open(path: &OsStr) -> Result<Self, LoadError> {
        // SAFETY: `dlopen` runs the library's initialisers, which is arbitrary code. We accept
        // that for a library the user (or our own installer) named; there is no way to load
        // Level Zero without it.
        let library = unsafe { Library::new(path) }.map_err(LoadError::Open)?;

        macro_rules! sym {
            ($name:literal) => {{
                // SAFETY: the signature is transcribed from `ze_api.h`; a mismatch here would
                // be undefined behaviour, which is why the layouts are pinned by tests.
                let symbol = unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|source| LoadError::Symbol { name: $name, source })?;
                // Copy the function pointer out so the borrow of `library` ends here and the
                // library can be moved into the struct below.
                *symbol
            }};
        }

        Ok(Self {
            ze_init: sym!("zeInit"),
            ze_driver_get: sym!("zeDriverGet"),
            ze_driver_get_properties: sym!("zeDriverGetProperties"),
            ze_device_get: sym!("zeDeviceGet"),
            ze_device_get_properties: sym!("zeDeviceGetProperties"),
            ze_device_get_memory_properties: sym!("zeDeviceGetMemoryProperties"),
            ze_device_get_compute_properties: sym!("zeDeviceGetComputeProperties"),
            library,
        })
    }

    /// Keep the loader mapped for the rest of the process.
    ///
    /// Called only after `zeInit` has succeeded. At that point the loader and the vendor
    /// driver behind it hold process-global state, and whether unmapping them is safe is not
    /// something this crate wants to depend on. Every *failure* path drops the handle
    /// normally, so the negative-control tests can run as often as they like.
    pub fn leak(self) {
        std::mem::forget(self.library);
    }
}

/// Read a fixed-size NUL-terminated C string field.
///
/// Level Zero specifies these as `char[256]`; a driver that fills all 256 bytes without a
/// terminator would still be handled here, and non-UTF-8 is replaced rather than rejected —
/// a device name is for a human to read, and is never a reason to fail detection.
pub fn c_str_field(field: &[c_char]) -> String {
    let bytes: Vec<u8> = field.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
