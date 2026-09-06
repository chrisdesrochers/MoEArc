//! The host machine: how much RAM there is, and how much room where models are kept.
//!
//! This is the third wiring module, beside [`crate::detect`] for devices and
//! [`crate::catalog`] for models, and it exists for the same reason they do: the interface
//! owns no facts. `moearc-engine`'s [`host_budget`] does the reasoning; everything here does
//! is read two numbers off the machine and parse one the user typed.
//!
//! **Why the numbers are read here rather than in the engine.** `moearc-engine` has no
//! dependencies at all — that is what makes the planner testable on any machine, in CI, with
//! no card and no oneAPI. Putting `sysinfo` in it to answer "how much RAM is there" would put
//! a platform crate in the way of arithmetic that has no platform in it. The split is the same
//! one `memory.rs` already draws: the engine decides, the CLI measures.
//!
//! [`host_budget`]: moearc_engine::host_budget

use std::path::{Path, PathBuf};

use moearc_engine::host_budget::{HostMemory, Storage};
use sysinfo::{Disks, System};

use crate::source::{HostReport, HostSource};

/// The environment variable that sets the host RAM budget, for a user who does not want to
/// type the flag every time. Same precedence rule as [`crate::catalog::MODELS_DIR_ENV`]: an
/// explicit flag beats an environment someone may have forgotten setting.
pub const HOST_BUDGET_ENV: &str = "MOEARC_HOST_BUDGET";

/// Host memory and free space, read off this machine.
pub struct RealHost {
    /// Where models are kept. Free space is reported for *this* filesystem, not for `/`: on the
    /// reference machine those are different pools by two orders of magnitude, and the one that
    /// decides whether a download fits is the one the models live on.
    models_dir: PathBuf,
}

impl RealHost {
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }
}

impl HostSource for RealHost {
    fn probe(&self) -> anyhow::Result<HostReport> {
        let mut sys = System::new();
        sys.refresh_memory();
        Ok(HostReport {
            total_bytes: sys.total_memory(),
            // 🔴 `available`, never `total - used`. Page cache counts as used and is
            // reclaimable, so on a machine that has already read a model once the subtraction
            // reports *less* memory the longer the tool has been useful. `MemAvailable` is the
            // kernel's own answer to the question we are actually asking.
            available_bytes: sys.available_memory(),
            models_free_bytes: free_space_for(&self.models_dir),
        })
    }
}

/// Free bytes on the filesystem holding `dir`.
///
/// The mount point with the longest matching prefix wins, which is what makes this correct on a
/// machine with nested mounts — the reference box has `/`, `/zfs/swift` and
/// `/zfs/swift/models` as three separate filesystems, and only the last one answers the
/// question.
///
/// 🔴 Unknown means [`u64::MAX`], not zero. This figure is consulted for one purpose: refusing
/// to download a model there is no room for. A measurement we could not take must not
/// manufacture that refusal — a download that fails on a real `ENOSPC` is a true error, and a
/// refusal derived from a missing number is a false one.
fn free_space_for(dir: &Path) -> u64 {
    let Some(probe) = existing_ancestor(dir) else {
        return u64::MAX;
    };
    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if probe.starts_with(mount) {
            let depth = mount.components().count();
            if best.is_none_or(|(d, _)| depth > d) {
                best = Some((depth, disk.available_space()));
            }
        }
    }
    best.map_or(u64::MAX, |(_, bytes)| bytes)
}

/// The nearest ancestor of `dir` that exists, canonicalised.
///
/// A first-run user's model directory has not been created yet, and `starts_with` against a
/// path with `..` or a symlink in it matches the wrong mount. Walking up finds the filesystem
/// the directory *would* be created on, which is the one that has to hold the download.
fn existing_ancestor(dir: &Path) -> Option<PathBuf> {
    dir.ancestors().find_map(|p| p.canonicalize().ok())
}

/// `"24G"`, `"1.5TiB"`, `"8192"` — a size in bytes.
///
/// Binary units throughout, matching [`crate::format::bytes`] and every other number this tool
/// prints. `G` and `GiB` are the same thing here; a tool that renders GiB and parses GB is off
/// by 7% in the direction the user will not check.
pub fn parse_size(text: &str) -> Result<u64, String> {
    let s = text.trim();
    if s.is_empty() {
        return Err("expected a size such as `24G`".to_string());
    }
    let digits = s.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let unit = s[digits.len()..].trim().to_ascii_lowercase();
    let value: f64 = digits
        .trim()
        .parse()
        .map_err(|_| format!("`{text}` is not a size — expected something like `24G`"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("`{text}` is not a size a machine can have"));
    }
    let scale: u64 = match unit.as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        "t" | "tb" | "tib" => 1u64 << 40,
        other => {
            return Err(format!(
                "`{other}` is not a unit — use K, M, G or T (binary, so G is GiB)"
            ));
        }
    };
    let bytes = value * scale as f64;
    if bytes > u64::MAX as f64 {
        return Err(format!("`{text}` is larger than any machine's address space"));
    }
    Ok(bytes as u64)
}

/// Turn a [`HostReport`] into the engine's view of the machine.
pub fn memory(report: &HostReport) -> HostMemory {
    HostMemory { total_bytes: report.total_bytes, available_bytes: report.available_bytes }
}

/// Turn a [`HostReport`] into the engine's view of the drive.
pub fn storage(report: &HostReport) -> Storage {
    Storage { free_bytes: report.models_free_bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_binary_and_case_insensitive() {
        assert_eq!(parse_size("24G").unwrap(), 24 << 30);
        assert_eq!(parse_size("24g").unwrap(), 24 << 30);
        assert_eq!(parse_size("24GiB").unwrap(), 24 << 30);
        // The one that would be a silent 7% lie if it disagreed with `format::bytes`.
        assert_eq!(parse_size("24GB").unwrap(), parse_size("24GiB").unwrap());
        assert_eq!(parse_size("512M").unwrap(), 512 << 20);
        assert_eq!(parse_size("2T").unwrap(), 2u64 << 40);
    }

    #[test]
    fn a_bare_number_is_bytes_and_a_fraction_is_allowed() {
        assert_eq!(parse_size("8192").unwrap(), 8192);
        assert_eq!(parse_size("1.5G").unwrap(), 1_610_612_736);
    }

    #[test]
    fn zero_is_a_setting_not_an_error() {
        // "keep nothing in RAM, page everything" is a legitimate thing to ask for, and it is
        // the most useful setting for showing what the disk tier costs.
        assert_eq!(parse_size("0").unwrap(), 0);
        assert_eq!(parse_size("0G").unwrap(), 0);
    }

    #[test]
    fn a_bad_size_names_what_was_expected() {
        for bad in ["", "lots", "24Q", "-4G"] {
            let err = parse_size(bad).unwrap_err();
            assert!(!err.is_empty(), "{bad} produced an empty message");
        }
        assert!(parse_size("24Q").unwrap_err().contains("unit"));
    }

    #[test]
    fn free_space_is_answered_for_a_directory_that_does_not_exist_yet() {
        // A first-run user has no model directory. The answer must be about the filesystem it
        // would be created on, not a failure.
        let n = free_space_for(Path::new("/nonexistent-moearc/models/deeper"));
        assert!(n > 0, "an unmeasurable filesystem must not read as a full one");
    }

    #[test]
    fn free_space_prefers_the_most_specific_mount() {
        // Rooted at the temp directory, which exists on every machine this runs on. The
        // assertion is only that a number comes back and it is not the "unknown" sentinel,
        // because the actual figure is a property of the machine running the test.
        let n = free_space_for(&std::env::temp_dir());
        assert_ne!(n, u64::MAX, "a real directory should resolve to a real filesystem");
    }
}
