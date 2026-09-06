//! Reading the machine. Everything in [`super::guard`] is a pure function of what this
//! produces, so this is the only place that touches `/proc`, `/sys` or another process.
//!
//! Nothing here decides anything. A probe that could not take a reading returns `None`, never
//! a substitute value — the guards treat "unknown" as its own answer, and in the load-average
//! case they treat it as a refusal.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------------------
// The machine
// ---------------------------------------------------------------------------------------

pub fn logical_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The kernel's 1-minute load average.
///
/// 🔴 Read **immediately before** each timed run, never once at the start of a sweep. §3 asks
/// for it per run, and `bench/baselines/gpt-oss-120b.md` §6.7 records why the *mid*-sweep value
/// is different again: a sweep that drives a host pool across 19 threads carries its own
/// previous row into the average. The pre-run figure is the one that says the box was quiet.
pub fn load1() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/loadavg").ok()?;
    text.split_whitespace().next()?.parse().ok()
}

/// `MemTotal` and `MemAvailable`, in bytes.
///
/// `MemAvailable`, never `MemTotal - used`: page cache counts as used and is reclaimable, so
/// the subtraction reports less memory the more useful the machine has been.
pub fn meminfo() -> (u64, u64) {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return (0, 0);
    };
    let field = |name: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    };
    (field("MemTotal:"), field("MemAvailable:"))
}

/// ZFS ARC counters, or `None` where ZFS is not loaded.
pub fn zfs_arc() -> Option<super::guard::ZfsArc> {
    let text = std::fs::read_to_string("/proc/spl/kstat/zfs/arcstats").ok()?;
    let field = |name: &str| -> u64 {
        text.lines()
            .find(|l| l.split_whitespace().next() == Some(name))
            .and_then(|l| l.split_whitespace().nth(2))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    Some(super::guard::ZfsArc {
        c_max_bytes: field("c_max"),
        size_bytes: field("size"),
        hits: field("hits"),
        misses: field("misses"),
    })
}

// ---------------------------------------------------------------------------------------
// Which filesystem, and which block device
// ---------------------------------------------------------------------------------------

/// The mount entry covering `path`: its filesystem type and its source.
///
/// The longest matching mount point wins. On the reference machine `/`, `/zfs/swift` and
/// `/zfs/swift/models` are three separate filesystems, and only the last answers the question.
pub fn mount_for(path: &Path) -> Option<(String, String)> {
    let target = path
        .canonicalize()
        .ok()
        .or_else(|| path.ancestors().find_map(|p| p.canonicalize().ok()))?;
    let text = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let mut best: Option<(usize, String, String)> = None;
    for line in text.lines() {
        let Some((left, right)) = line.split_once(" - ") else { continue };
        let mount_point = match left.split_whitespace().nth(4) {
            Some(m) => unescape_octal(m),
            None => continue,
        };
        let mut r = right.split_whitespace();
        let (Some(fstype), Some(source)) = (r.next(), r.next()) else { continue };
        if target.starts_with(&mount_point) {
            let depth = Path::new(&mount_point).components().count();
            if best.as_ref().is_none_or(|(d, _, _)| depth > *d) {
                best = Some((depth, fstype.to_string(), unescape_octal(source)));
            }
        }
    }
    best.map(|(_, fs, src)| (fs, src))
}

/// `\040` and friends, as the kernel writes them into `mountinfo`.
///
/// Not cosmetic: a mount point with a space in it truncates at the space without this, matches
/// far too many paths, and silently attributes the wrong filesystem.
fn unescape_octal(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() {
            if let Some(v) = std::str::from_utf8(&bytes[i + 1..i + 4])
                .ok()
                .and_then(|d| u8::from_str_radix(d, 8).ok())
            {
                out.push(v as char);
                i += 4;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// The block device whose `/proc/diskstats` row describes reads of `path`.
///
/// 🔴 `None` rather than "all devices". `/proc/diskstats` is machine-wide, so summing every
/// device would fold in an unrelated tenant's I/O and produce a number that cannot be
/// attributed — which is worse than no number, and is the same reasoning
/// `examples/ctx_curve.rs` gives for requiring `MOEARC_BENCH_DISK`.
///
/// Two resolutions are attempted, in order:
/// * an ordinary filesystem whose `mountinfo` source is a `/dev/` node — use that node;
/// * ZFS, where the source is `pool/dataset` — ask `zpool status -P` for the pool's vdevs and
///   use them if there is no ambiguity about which pool it is.
pub fn disk_device_for(path: &Path) -> Option<Vec<String>> {
    if let Ok(explicit) = std::env::var("MOEARC_BENCH_DISK") {
        if !explicit.is_empty() {
            return Some(explicit.split(',').map(|s| s.trim().to_string()).collect());
        }
    }
    let (fstype, source) = mount_for(path)?;
    if let Some(node) = source.strip_prefix("/dev/") {
        return Some(vec![node.to_string()]);
    }
    if fstype == "zfs" {
        let pool = source.split('/').next()?;
        return zpool_vdevs(pool);
    }
    None
}

/// Whole-disk and partition names backing a ZFS pool, from `zpool status -P`.
///
/// `-P` prints full paths, which is what makes the output parseable without guessing: any
/// whitespace-separated token starting with `/dev/` on a config line is a vdev.
fn zpool_vdevs(pool: &str) -> Option<Vec<String>> {
    let out =
        std::process::Command::new("zpool").arg("status").arg("-P").arg(pool).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let devs: Vec<String> = text
        .split_whitespace()
        .filter_map(|t| t.strip_prefix("/dev/"))
        .map(|s| s.to_string())
        .collect();
    if devs.is_empty() { None } else { Some(devs) }
}

// ---------------------------------------------------------------------------------------
// I/O counters, bracketed around a run
// ---------------------------------------------------------------------------------------

/// Cumulative counters. Only differences over a bracketed window mean anything, and any other
/// tenant's I/O lands in them too — which is the other reason the load average is read first.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Io {
    /// Bytes read from the devices backing the model, or `None` when they could not be
    /// attributed to a specific device.
    pub disk_read_bytes: Option<u64>,
    pub arc_hits: Option<u64>,
    pub arc_misses: Option<u64>,
}

impl Io {
    /// Snapshot the counters now.
    ///
    /// Only the timed worker brackets a run with these, so a build without the `gpu` feature
    /// has no caller for them outside the tests. Kept rather than gated: the counters are the
    /// evidence §4 asks for and their absence from a non-GPU build is a property of that
    /// build, not a reason to delete the code.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn now(devices: Option<&[String]>) -> Self {
        Self {
            disk_read_bytes: devices.and_then(read_bytes_for),
            arc_hits: zfs_arc().map(|a| a.hits),
            arc_misses: zfs_arc().map(|a| a.misses),
        }
    }

    /// `self - before`, saturating. A counter that went backwards means the device was reset
    /// or renumbered mid-run; zero is the honest answer, not a wrapped enormous number.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn since(&self, before: &Io) -> Io {
        fn d(a: Option<u64>, b: Option<u64>) -> Option<u64> {
            Some(a?.saturating_sub(b?))
        }
        Io {
            disk_read_bytes: d(self.disk_read_bytes, before.disk_read_bytes),
            arc_hits: d(self.arc_hits, before.arc_hits),
            arc_misses: d(self.arc_misses, before.arc_misses),
        }
    }
}

/// Sectors read, from `/proc/diskstats` field 6, times the fixed 512-byte sector the kernel
/// reports in regardless of the device's physical sector size.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn read_bytes_for(devices: &[String]) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/diskstats").ok()?;
    let mut total = 0u64;
    let mut matched = false;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 6 {
            continue;
        }
        if devices.iter().any(|d| d == f[2]) {
            matched = true;
            total += f[5].parse::<u64>().unwrap_or(0) * 512;
        }
    }
    if matched { Some(total) } else { None }
}

// ---------------------------------------------------------------------------------------
// This build
// ---------------------------------------------------------------------------------------

/// What `build.rs` recorded about the commit this binary was compiled from.
pub fn build_facts() -> super::guard::BuildFacts {
    let commit = option_env!("MOEARC_BUILD_COMMIT")
        .filter(|s| !s.is_empty() && *s != "unknown")
        .map(|s| s.to_string());
    let dirty = match option_env!("MOEARC_BUILD_DIRTY") {
        Some("yes") => Some(true),
        Some("no") => Some(false),
        _ => None,
    };
    let mut features = Vec::new();
    if cfg!(feature = "gpu") {
        features.push("gpu");
    }
    super::guard::BuildFacts {
        commit,
        dirty,
        profile: if cfg!(debug_assertions) { "debug" } else { "release" },
        target: option_env!("MOEARC_BUILD_TARGET").unwrap_or("unknown"),
        features,
    }
}

/// Whether the engine's device half is in this binary at all.
pub const GPU_COMPILED_IN: bool = cfg!(feature = "gpu");

/// The path a probe should be taken against, for a model handle that may or may not exist.
pub fn model_facts(path: &Path) -> Option<super::guard::ModelUnderTest> {
    let bytes = std::fs::metadata(path).ok()?.len();
    let filesystem = mount_for(path).map(|(fs, _)| fs).unwrap_or_else(|| "unknown".to_string());
    Some(super::guard::ModelUnderTest { path: path.display().to_string(), bytes, filesystem })
}

/// Absolute path of this binary, for spawning the per-invocation workers.
pub fn self_exe() -> Option<PathBuf> {
    std::env::current_exe().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machine_answers_the_basic_questions() {
        assert!(logical_cpus() >= 1);
        let (total, avail) = meminfo();
        // On any Linux box these are non-zero; the test asserts the parse, not the values.
        assert!(total > 0, "MemTotal parsed as zero");
        assert!(avail > 0, "MemAvailable parsed as zero");
        assert!(avail <= total, "available {avail} exceeds total {total}");
    }

    #[test]
    fn the_load_average_is_readable_on_this_platform() {
        // If this ever fails, the guard's refusal path is the one that fires, which is the
        // correct behaviour — but it should not fire on Linux.
        assert!(load1().is_some_and(|v| v >= 0.0));
    }

    #[test]
    fn octal_escapes_in_a_mount_point_are_decoded() {
        assert_eq!(unescape_octal("/mnt/my\\040disk"), "/mnt/my disk");
        assert_eq!(unescape_octal("/plain/path"), "/plain/path");
        // A trailing backslash must not panic or eat the string.
        assert_eq!(unescape_octal("/odd\\"), "/odd\\");
    }

    #[test]
    fn the_temp_directory_resolves_to_a_filesystem() {
        let (fs, _src) = mount_for(&std::env::temp_dir()).expect("temp dir has a mount");
        assert!(!fs.is_empty());
    }

    #[test]
    fn a_path_that_does_not_exist_resolves_to_its_nearest_ancestor() {
        let mut p = std::env::temp_dir();
        p.push("moearc-bench-nonexistent/deeper/still");
        assert!(mount_for(&p).is_some());
    }

    #[test]
    fn an_io_delta_never_wraps() {
        let before = Io { disk_read_bytes: Some(100), arc_hits: Some(5), arc_misses: Some(1) };
        let after = Io { disk_read_bytes: Some(40), arc_hits: Some(9), arc_misses: Some(1) };
        let d = after.since(&before);
        assert_eq!(d.disk_read_bytes, Some(0));
        assert_eq!(d.arc_hits, Some(4));
    }

    #[test]
    fn an_unattributable_counter_stays_unknown_rather_than_becoming_zero() {
        let before = Io::default();
        let after = Io::default();
        assert_eq!(after.since(&before).disk_read_bytes, None);
        assert_eq!(read_bytes_for(&["no-such-device-xyz".to_string()]), None);
    }
}
