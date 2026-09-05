//! The positive control: detection against whatever GPU this machine actually has.
//!
//! It cannot assert a specific card — MoEArc has to build and test on machines that have no
//! Intel GPU at all — so it asserts the invariants a report must satisfy whenever one is
//! produced, and reports the reason it stood down when there is no hardware to look at. The
//! measured figures for a specific card belong in the calibration record, not in a test.

use moearc_device::{DetectError, detect};

#[test]
fn a_report_from_real_hardware_is_internally_consistent() {
    let report = match detect() {
        Ok(report) => report,
        Err(err) => {
            // Every one of these means "there is no GPU here to test against", which is a
            // legitimate state for a build machine. Anything else is a real failure.
            // Matched through `root_cause` because a machine that has a card but cannot use it
            // reports the same causes wrapped in physical evidence.
            let no_hardware = matches!(
                err.root_cause(),
                DetectError::LoaderNotFound { .. }
                    | DetectError::DriverUninitialized
                    | DetectError::NoDrivers
                    | DetectError::NoDevices { .. }
                    | DetectError::NoGpuDevices { .. }
                    | DetectError::PermissionDenied { .. }
            );
            assert!(no_hardware, "detection failed unexpectedly: {err}");
            eprintln!("no GPU available on this machine, skipping: {err}");
            return;
        }
    };

    assert!(!report.devices.is_empty(), "a successful report always has at least one GPU");
    assert!(report.driver_count >= 1);
    assert!(report.preferred().is_some());

    for device in &report.devices {
        assert!(!device.name.is_empty(), "a device must be nameable to the user");
        assert!(device.total_memory_bytes > 0, "{}: reported no memory", device.name);
        assert!(
            device.max_alloc_bytes > 0 && device.max_alloc_bytes <= device.total_memory_bytes,
            "{}: max allocation {} is not within its {} bytes of memory",
            device.name,
            device.max_alloc_bytes,
            device.total_memory_bytes
        );
        assert!(device.compute_units > 0, "{}: reported no execution units", device.name);
        assert!(device.uuid != [0u8; 16], "{}: reported a null UUID", device.name);
        assert!(!device.subgroup_sizes.is_empty(), "{}: reported no subgroup sizes", device.name);
        assert!(device.max_total_group_size > 0);
    }

    // The classification that `bench/README.md` calls the highest-risk step in the benchmark
    // protocol. On a machine with both kinds, exactly one discrete device must win.
    if report.devices.iter().any(|d| d.is_integrated)
        && report.devices.iter().any(|d| !d.is_integrated)
    {
        assert!(!report.preferred().expect("a device").is_integrated);
    }
}
