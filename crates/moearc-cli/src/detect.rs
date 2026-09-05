//! Real device detection: adapts `moearc-device` to the CLI's [`DeviceSource`] seam.
//!
//! This is the whole of the wiring. Nothing else in the CLI knows whether it is looking at a
//! real card or a fixture, which is what makes the fixture worth having.

use crate::source::{Backend, DeviceReport, DeviceRow, DeviceSource, Verdict};
use moearc_device::sysman;

/// Detection backed by Level Zero.
pub struct LevelZeroDevices;

impl DeviceSource for LevelZeroDevices {
    fn detect(&self) -> anyhow::Result<DeviceReport> {
        // On failure, propagate the detector's own sentence rather than flattening it into a
        // cause code. Those messages are the product: they name the missing piece and, where
        // there is one thing to do, say what it is. `Verdict` has three variants and would
        // discard the distinction between a missing loader, a missing kernel driver, and a
        // card that is present but unusable.
        let report = moearc_device::detect()?;

        // Live free memory, joined by UUID. The detector found that sysman and core Level Zero
        // enumerate in different orders on the same machine, so joining by index would silently
        // attribute one card's memory to another.
        let telemetry = sysman::telemetry().unwrap_or_default();

        let mut devices: Vec<DeviceRow> = report
            .devices
            .iter()
            .map(|d| {
                let kernel_driver = report
                    .pci_display_devices
                    .iter()
                    .find(|p| Some(&p.address) == d.pci_address.as_ref())
                    .and_then(|p| p.kernel_driver.clone())
                    .unwrap_or_else(|| "unknown".to_string());

                let live_free = telemetry
                    .iter()
                    .find(|t| t.uuid == d.uuid)
                    .and_then(|t| t.free_device_memory_bytes());

                // 🔴 Two measured hazards, both from the detector's report.
                //
                // Sysman reports more free memory than core Level Zero reports allocatable
                // (12,567,810,048 against 12,168,933,376 on the reference B580). They answer
                // different questions and neither is wrong, but a plan built on the larger
                // figure would ask for memory the allocator will not hand over — so the
                // allocatable figure is the ceiling, always.
                //
                // And when sysman has nothing to say, falling back to the core figure is only
                // safe for a discrete card. An integrated GPU's only memory module is system
                // RAM: the reference iGPU reports 85.58 GiB, which as a VRAM budget would
                // plan a model that cannot possibly fit.
                let free = match live_free {
                    Some(f) => f.min(d.total_memory_bytes),
                    None => d.total_memory_bytes,
                };

                DeviceRow {
                    name: d.name.clone(),
                    backend: Backend::LevelZero,
                    driver: format!(
                        "{kernel_driver} / L0 build {}",
                        driver_build(d.driver_version)
                    ),
                    total_bytes: d.total_memory_bytes,
                    free_bytes: free,
                }
            })
            .collect();

        // Discrete first. `DeviceReport::primary()` takes the first inference target by index
        // order, and Level Zero does not promise to enumerate the discrete card first — on the
        // reference machine core Level Zero happens to, and sysman does not. Ordering here
        // means the planner never silently targets an iGPU because of enumeration order.
        devices.sort_by_key(|row| {
            let integrated =
                report.devices.iter().find(|d| d.name == row.name).is_some_and(|d| d.is_integrated);
            (integrated, row.name.clone())
        });

        let verdict = Verdict::for_devices(&devices);
        Ok(DeviceReport { devices, verdict })
    }
}

/// The compute-runtime build number carried in Level Zero's `driverVersion`.
///
/// 🔴 Only the low 16 bits are reported, and that restraint is deliberate. The Level Zero
/// specification defines `driverVersion` as "a non-zero, monotonically increasing value" and
/// specifies **no** encoding — so any `major.minor.build` reading of it is invented.
///
/// An earlier version of this function did invent one, rendering the reference card's
/// 17,010,844 as `1.3.37020`. Cross-checking against two independent sources on the same
/// machine showed what was real and what was not:
///
/// | source | reports |
/// | --- | --- |
/// | `sycl-ls` | `Intel(R) Arc(TM) B580 Graphics 20.1.0 [1.14.37020]` |
/// | `clinfo` | `Driver Version 26.05.037020` |
///
/// The build number **37020** appears in both. The `1.3` did not appear anywhere — `sycl-ls`
/// shows `1.14` there, which is the Level Zero *API* version, an entirely different field.
/// So the high bits are unidentified and are not displayed; printing a version string that
/// matches nothing the user can search for is worse than printing less.
fn driver_build(v: u32) -> u32 {
    v & 0xFFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_number_matches_what_other_intel_tools_report() {
        // The reference B580 reports 17,010,844. Both sycl-ls ([1.14.37020]) and clinfo
        // (26.05.037020) show 37020 for the same driver on the same machine, so this is
        // checked against corroborated output rather than against an assumed encoding.
        assert_eq!(driver_build(17_010_844), 37_020);
    }
}
