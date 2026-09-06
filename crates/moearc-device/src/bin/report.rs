//! Prints the device report — the zero-argument behaviour `docs/ux.md` describes, in the plain
//! form. It is what `moearc` with no arguments must be able to say before there is a TUI to
//! say it in, and it is the harness the crate's negative-control test drives.
//!
//! Exit code is the contract: 0 when there is a device MoEArc can plan against, 1 when there is
//! not — including the case where Level Zero happily enumerated something that has no VRAM.
//! Nothing here panics on a detection failure; a stack trace is the failure mode this crate
//! exists to avoid.

use std::process::ExitCode;

use moearc_device::fitness::{self, VramBudget};
use moearc_device::sysman::{self, DeviceTelemetry};
use moearc_device::{DeviceReport, GpuDevice, detect};

fn main() -> ExitCode {
    match detect() {
        Ok(report) => print_report(&report),
        Err(err) => {
            eprintln!("moearc: no usable GPU.\n\n{err}");
            ExitCode::FAILURE
        }
    }
}

fn print_report(report: &DeviceReport) -> ExitCode {
    // Optional by design: a machine can run inference perfectly well without it, so a failure
    // here is a note, never an error.
    let live = sysman::telemetry();
    let readings: Vec<DeviceTelemetry> = live.as_ref().map(Clone::clone).unwrap_or_default();
    let target = fitness::inference_target(report, &readings);
    let selected = target.as_ref().ok().map(|(device, _)| *device);

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
        let chosen = selected.is_some_and(|p| std::ptr::eq(p, device));
        println!("\n[{index}] {}{}", device.name, if chosen { "   <- selected" } else { "" });
        print_device(device);
        match &live {
            Ok(measured) => print_live_memory(measured, device),
            Err(err) => println!("     live memory    : unavailable ({err})"),
        }
        // The line that used to be missing. A device is now told apart by whether MoEArc will
        // plan on it, on the same row as the memory figure that would otherwise be read as a
        // budget.
        match fitness::vram_budget(device, fitness::reading_for(&readings, device)) {
            Ok(VramBudget { bytes, source }) => {
                println!("     plan budget    : {} ({})", bytes_str(bytes), source.describe())
            }
            Err(refusal) => println!("     plan budget    : NONE — {refusal}"),
        }
    }

    // A runtime older than the one this project has measured is worth saying even when
    // everything works, because what it breaks is the step after this one. Only when
    // everything works, though: the refusal below already carries the same paragraph, and
    // printing it twice makes a reader look for the difference between the two copies.
    if target.is_ok()
        && let Some(caution) = report.devices.first().and_then(fitness::runtime_caution)
    {
        println!("\nnote: {caution}");
    }

    match target {
        Ok((device, budget)) => {
            println!(
                "\n{} is ready — {} to plan with ({}).",
                device.name,
                bytes_str(budget.bytes),
                budget.source.describe()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("\nmoearc: no usable GPU.\n\n{err}");
            ExitCode::FAILURE
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
    println!("     memory         : {}", bytes_str(device.total_memory_bytes));
    println!("     max allocation : {}", bytes_str(device.max_alloc_bytes));
    println!("     execution units: {}", device.compute_units);
    println!("     max group size : {}", device.max_total_group_size);
    println!(
        "     subgroup sizes : {}",
        device.subgroup_sizes.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
    );
    println!("     driver version : {} (build {})", device.driver_version, device.driver_build());
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
                // Naming the host pool, rather than only its absence: on the reference iGPU it
                // is exactly `MemTotal`, which is the fact that explains the device's own
                // "memory" figure two lines above.
                match reading.host_memory_bytes() {
                    Some(host) => println!(
                        "     live memory    : no memory pool of its own — it exposes {} of \
                         system memory instead",
                        bytes_str(host)
                    ),
                    None => {
                        println!("     live memory    : this device has no memory pool of its own")
                    }
                }
                return;
            };
            let used = match reading.used_device_memory_bytes() {
                Some(used) => format!(", {} in use", bytes_str(used)),
                // The driver reports free memory but not capacity, so "in use" is genuinely
                // unknown. Saying 0 would be inventing a measurement.
                None => ", amount in use not reported by the driver".to_string(),
            };
            println!("     live memory    : {} free{used}", bytes_str(free));
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
fn bytes_str(value: u64) -> String {
    format!("{:.2} GiB ({value} bytes)", value as f64 / (1024.0 * 1024.0 * 1024.0))
}
