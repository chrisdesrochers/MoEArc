//! Prints the device report — the zero-argument behaviour `docs/ux.md` describes, in the plain
//! form. It is what `moearc` with no arguments must be able to say before there is a TUI to
//! say it in, and it is the harness the crate's negative-control test drives.
//!
//! Exit code is the contract: 0 when a GPU was found, 1 when detection failed with a legible
//! explanation on stderr. Nothing here panics on a detection failure — a stack trace is the
//! failure mode this crate exists to avoid.

use std::process::ExitCode;

use moearc_device::sysman::{self, DeviceTelemetry};
use moearc_device::{DeviceReport, GpuDevice, detect};

fn main() -> ExitCode {
    match detect() {
        Ok(report) => {
            print_report(&report);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("moearc: no usable GPU.\n\n{err}");
            ExitCode::FAILURE
        }
    }
}

fn print_report(report: &DeviceReport) {
    let preferred = report.preferred();
    // Optional by design: a machine can run inference perfectly well without it, so a failure
    // here is a note, never an error.
    let live = sysman::telemetry();
    println!(
        "Level Zero loader : {}\nDrivers           : {}\nGPUs              : {}{}",
        report.loader,
        report.driver_count,
        report.devices.len(),
        if report.non_gpu_devices > 0 {
            format!(" ({} non-GPU device(s) skipped)", report.non_gpu_devices)
        } else {
            String::new()
        }
    );

    if !report.pci_display_devices.is_empty() {
        println!("\nDisplay controllers the kernel sees:");
        for card in &report.pci_display_devices {
            println!("  {}{}", card.describe(), drm_nodes(card));
        }
    }
    if !report.unusable_hardware.is_empty() {
        println!("\nPresent but NOT usable by Level Zero:");
        for card in &report.unusable_hardware {
            println!("  {}{}", card.describe(), drm_nodes(card));
        }
    }

    for (index, device) in report.devices.iter().enumerate() {
        let chosen = preferred.is_some_and(|p| std::ptr::eq(p, device));
        println!("\n[{index}] {}{}", device.name, if chosen { "   <- selected" } else { "" });
        print_device(device);
        match &live {
            Ok(readings) => print_live_memory(readings, device),
            Err(err) => println!("     live memory    : unavailable ({err})"),
        }
    }
}

fn print_device(device: &GpuDevice) {
    println!(
        "     class          : {}",
        if device.is_integrated { "integrated" } else { "discrete" }
    );
    println!("     pci id         : {:04x}:{:04x}", device.vendor_id, device.device_id);
    println!("     uuid           : {}", device.uuid_string());
    println!("     memory         : {}", bytes(device.total_memory_bytes));
    println!("     max allocation : {}", bytes(device.max_alloc_bytes));
    println!("     execution units: {}", device.compute_units);
    println!("     max group size : {}", device.max_total_group_size);
    println!(
        "     subgroup sizes : {}",
        device.subgroup_sizes.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
    );
    println!("     driver version : {}", device.driver_version);
    println!(
        "     pci address    : {}",
        device.pci_address.as_deref().unwrap_or("not matched in sysfs")
    );
}

/// The measured free VRAM, joined to the device by UUID rather than by position.
fn print_live_memory(readings: &[DeviceTelemetry], device: &GpuDevice) {
    match readings.iter().find(|r| r.uuid == device.uuid) {
        Some(reading) => {
            let Some(free) = reading.free_device_memory_bytes() else {
                println!("     live memory    : this device has no memory pool of its own");
                return;
            };
            let used = match reading.used_device_memory_bytes() {
                Some(used) => format!(", {} in use", bytes(used)),
                // The driver reports free memory but not capacity, so "in use" is genuinely
                // unknown. Saying 0 would be inventing a measurement.
                None => ", amount in use not reported by the driver".to_string(),
            };
            println!("     live memory    : {} free{used}", bytes(free));
            for module in reading.unhealthy_modules() {
                println!(
                    "     memory health  : {:?} — the driver is reporting a fault",
                    module.health
                );
            }
        }
        None => println!("     live memory    : no Sysman reading for this device"),
    }
}

/// Which DRM nodes the kernel gave this card. On a two-GPU machine the render node is the
/// handle that decides which GPU a process actually runs on, so it is worth printing next to
/// the card it belongs to.
fn drm_nodes(card: &moearc_device::PciDevice) -> String {
    let nodes: Vec<&str> =
        [card.drm_card.as_deref(), card.drm_render_node.as_deref()].into_iter().flatten().collect();
    if nodes.is_empty() { String::new() } else { format!(" [{}]", nodes.join(", ")) }
}

/// Bytes as GiB alongside the exact figure. Both, deliberately: GiB is what a person reasons
/// in, and the exact byte count is what a bug report needs.
fn bytes(value: u64) -> String {
    format!("{:.2} GiB ({value} bytes)", value as f64 / (1024.0 * 1024.0 * 1024.0))
}
