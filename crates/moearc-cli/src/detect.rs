//! Real device detection: adapts `moearc-device` to the CLI's [`DeviceSource`] seam.
//!
//! This is the whole of the wiring. Nothing else in the CLI knows whether it is looking at a
//! real card or a fixture, which is what makes the fixture worth having.

use crate::source::{Backend, DeviceReport, DeviceRow, DeviceSource, Verdict};
use moearc_device::{fitness, sysman};

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

                // 🔴 The budget comes from `fitness::vram_budget` and nowhere else. The rule
                // is one sentence — *a budget must come from a memory pool that is measurably
                // on the device* — and the reason it is a shared function rather than a
                // `match` here is that the `match` that used to be here got it wrong in the
                // most expensive way available: it fell back to the device's reported total
                // whenever the live reading was missing, and on an integrated GPU that total
                // is a share of host RAM. The tool then printed `✓ … is ready — 85.6 GiB free
                // right now` and would have agreed that a 132 GiB model fits.
                //
                // Sysman is joined on UUID inside `reading_for`, not by index: Sysman and core
                // Level Zero enumerate in different orders on the reference machine, so an
                // index join silently attributes one card's memory to another.
                let budget = fitness::vram_budget(d, fitness::reading_for(&telemetry, d));

                DeviceRow {
                    name: d.name.clone(),
                    backend: Backend::LevelZero,
                    // `GpuDevice::driver_build` rather than a second copy of the same masking
                    // and the same corroboration table. There were two; only one of them could
                    // ever be the one a reader checks.
                    driver: format!("{kernel_driver} / L0 build {}", d.driver_build()),
                    total_bytes: d.total_memory_bytes,
                    // Zero for a device with no pool of its own. Literally true, and the row
                    // beside it says why in the device's own numbers.
                    free_bytes: budget.as_ref().map(|b| b.bytes).unwrap_or(0),
                    budget_source: budget.as_ref().ok().map(|b| b.source.describe()),
                    unusable: budget.as_ref().err().map(ToString::to_string),
                    driver_build: Some(d.driver_build()),
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

#[cfg(test)]
mod tests {
    use crate::source::{Backend, DeviceRow, Verdict};

    fn row(name: &str, free: u64, unusable: Option<&str>) -> DeviceRow {
        DeviceRow {
            name: name.to_string(),
            backend: Backend::LevelZero,
            driver: "xe / L0 build 27642".to_string(),
            total_bytes: 91_890_372_608,
            free_bytes: free,
            budget_source: None,
            unusable: unusable.map(str::to_string),
            driver_build: Some(27_642),
        }
    }

    /// 🔴 The shipped-binary defect, asserted at the seam that shipped it.
    ///
    /// `moearc-device` refuses correctly; what the packaged `moearc` command did with that
    /// refusal is this crate's problem, and for a while the answer was to ignore it and print
    /// `✓ Intel(R) Graphics is ready — 85.6 GiB free right now`.
    #[test]
    fn a_device_with_no_pool_of_its_own_is_never_ready() {
        let devices = vec![row(
            "Intel(R) Graphics",
            0,
            Some(
                "Intel(R) Graphics reports 85.58 GiB of memory, and that figure is this \
                  machine's system RAM rather than video memory",
            ),
        )];
        let verdict = Verdict::for_devices(&devices);
        assert!(!verdict.is_ready(), "{verdict:?}");
        assert!(!devices[0].is_inference_target());
        // And the sentence that reaches the screen is the measurement, not a category.
        let headline = verdict.headline();
        assert!(headline.contains("85.58 GiB"), "{headline}");
        assert!(headline.contains("system RAM"), "{headline}");
    }

    #[test]
    fn a_refusal_outranks_the_generic_no_gpu_message() {
        // A GPU *was* found and it was Level Zero. "The oneAPI runtime is not on this shell's
        // path" would send the user to fix a path that is already correct.
        let devices = vec![row("Intel(R) Graphics", 0, Some("no pool of its own"))];
        assert!(!Verdict::for_devices(&devices).headline().contains("path"));
    }

    #[test]
    fn a_usable_card_is_still_ready_and_plans_against_its_budget() {
        let mut d = row("Intel(R) Arc(TM) B580 Graphics", 12_168_933_376, None);
        d.total_bytes = 12_168_933_376;
        assert!(d.is_inference_target());
        let Verdict::Ready { free_bytes, .. } = Verdict::for_devices(&[d]) else {
            panic!("a card with a real pool is ready")
        };
        assert_eq!(free_bytes, 12_168_933_376);
    }
}
