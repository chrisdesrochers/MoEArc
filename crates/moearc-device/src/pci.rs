//! A second, independent view of the GPU situation, read from sysfs with no Level Zero
//! involved.
//!
//! Level Zero answers "what can the compute runtime use". It cannot answer "what is in this
//! machine" — if the kernel driver is not loaded, or the compute runtime is missing, an Arc
//! card sitting in the slot is invisible to it. Reporting "no devices" to someone who can see
//! the card is true and useless, and `docs/ux.md` makes naming the cause a requirement, so
//! this module supplies the physical evidence the Level Zero probe cannot.
//!
//! **No PCI-ID table.** Mapping device ids to marketing names is a table that goes stale, and
//! one that does not recognise a card reports nothing at all. Everything here is a fact the
//! kernel already knows: the address, the raw vendor/device ids, which driver is bound, and
//! which DRM nodes it owns. The card's *name* comes from Level Zero, which gets it from the
//! driver that actually owns the hardware.

use std::fs;
use std::path::{Path, PathBuf};

/// Where the kernel exposes every PCI function.
pub const SYSFS_PCI_DEVICES: &str = "/sys/bus/pci/devices";

/// PCI base class 0x03: display controller. The lower 16 bits are sub-class and programming
/// interface, which vary across VGA-compatible and non-VGA GPUs and are not worth filtering
/// on — a compute card with no display output is still a card we want.
const PCI_BASE_CLASS_DISPLAY: u32 = 0x03;

/// One display controller as the kernel sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciDevice {
    /// Domain:bus:device.function, e.g. `0000:04:00.0`.
    pub address: String,
    /// PCI vendor id.
    pub vendor_id: u32,
    /// PCI device id.
    pub device_id: u32,
    /// Full 24-bit class code, as reported (e.g. `0x030000`).
    pub class_code: u32,
    /// The kernel module bound to it, if any. `None` is the interesting case: the hardware is
    /// present and nothing is driving it.
    pub kernel_driver: Option<String>,
    /// The DRM card node, e.g. `card1`.
    pub drm_card: Option<String>,
    /// The DRM render node, e.g. `renderD129`. Reported because on a two-GPU machine this is
    /// the handle that says *which* GPU a process is actually using.
    pub drm_render_node: Option<String>,
    /// Whether firmware picked this device as the boot display.
    pub boot_vga: bool,
}

impl PciDevice {
    /// A one-line description in the terms the kernel uses, with no names invented.
    pub fn describe(&self) -> String {
        format!(
            "{} ({:04x}:{:04x}, kernel driver {})",
            self.address,
            self.vendor_id,
            self.device_id,
            match &self.kernel_driver {
                Some(driver) => format!("`{driver}`"),
                None => "none — nothing is bound to it".to_string(),
            }
        )
    }
}

/// Every display controller the kernel knows about.
///
/// Returns an empty list rather than an error when sysfs is absent or unreadable. This probe
/// exists to *improve* an explanation; it must never be the reason detection fails.
pub fn scan() -> Vec<PciDevice> {
    scan_at(Path::new(SYSFS_PCI_DEVICES))
}

/// [`scan`] against an arbitrary directory in the sysfs PCI layout.
///
/// Public so the parsing can be tested against a fixture tree. Real hardware cannot be asked
/// to unbind its driver on demand, and the unbound case is the one whose message matters most.
pub fn scan_at(root: &Path) -> Vec<PciDevice> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };

    let mut devices: Vec<PciDevice> = entries
        .flatten()
        .filter_map(|entry| read_device(&entry.path()))
        .filter(|device| device.class_code >> 16 == PCI_BASE_CLASS_DISPLAY)
        .collect();

    // read_dir order is unspecified; PCI address order is stable and is how every other tool
    // on the machine lists these, so a user can line our output up against theirs.
    devices.sort_by(|a, b| a.address.cmp(&b.address));
    devices
}

fn read_device(path: &Path) -> Option<PciDevice> {
    let address = path.file_name()?.to_str()?.to_string();

    Some(PciDevice {
        address,
        vendor_id: read_hex(path, "vendor")?,
        device_id: read_hex(path, "device")?,
        class_code: read_hex(path, "class")?,
        // The `driver` symlink is absent exactly when no module is bound.
        kernel_driver: fs::read_link(path.join("driver"))
            .ok()
            .and_then(|target| Some(target.file_name()?.to_str()?.to_string())),
        drm_card: drm_node(path, "card"),
        drm_render_node: drm_node(path, "renderD"),
        boot_vga: read_trimmed(path, "boot_vga").as_deref() == Some("1"),
    })
}

/// sysfs writes these as `0x8086`, with a trailing newline.
fn read_hex(path: &Path, file: &str) -> Option<u32> {
    let text = read_trimmed(path, file)?;
    u32::from_str_radix(text.strip_prefix("0x").unwrap_or(&text), 16).ok()
}

fn read_trimmed(path: &Path, file: &str) -> Option<String> {
    fs::read_to_string(path.join(file)).ok().map(|s| s.trim().to_string())
}

/// The device's `drm/` directory lists its nodes: `card0`, `renderD128`, and legacy
/// `controlD*` aliases. Matching on the prefix keeps `card0` from also matching `card0-DP-1`,
/// which exists in `/sys/class/drm` but not here.
fn drm_node(path: &Path, prefix: &str) -> Option<String> {
    let entries = fs::read_dir(PathBuf::from(path).join("drm")).ok()?;
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .find(|name| name.starts_with(prefix) && !name.contains('-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a fixture in the sysfs PCI layout. Only the files this module reads are created,
    /// so a test also documents exactly what the probe depends on.
    pub(crate) struct FakeSysfs {
        pub root: PathBuf,
    }

    impl FakeSysfs {
        pub fn new(tag: &str) -> Self {
            let root = std::env::temp_dir()
                .join(format!("moearc-device-pci-{tag}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).expect("fixture root");
            Self { root }
        }

        pub fn device(
            &self,
            address: &str,
            vendor: &str,
            device: &str,
            class: &str,
            driver: Option<&str>,
        ) -> &Self {
            let dir = self.root.join(address);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("vendor"), format!("{vendor}\n")).unwrap();
            fs::write(dir.join("device"), format!("{device}\n")).unwrap();
            fs::write(dir.join("class"), format!("{class}\n")).unwrap();
            fs::write(dir.join("boot_vga"), "0\n").unwrap();
            if let Some(driver) = driver {
                let module = self.root.join("_modules").join(driver);
                fs::create_dir_all(&module).unwrap();
                std::os::unix::fs::symlink(&module, dir.join("driver")).unwrap();
            }
            self
        }
    }

    impl Drop for FakeSysfs {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn only_display_controllers_are_reported() {
        let fake = FakeSysfs::new("classes");
        fake.device("0000:04:00.0", "0x8086", "0xe20b", "0x030000", Some("xe"))
            .device("0000:00:1f.3", "0x8086", "0x7d70", "0x040300", Some("snd_hda_intel"))
            .device("0000:05:00.0", "0x144d", "0xa80c", "0x010802", Some("nvme"));

        let found = scan_at(&fake.root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].device_id, 0xe20b);
        assert_eq!(found[0].kernel_driver.as_deref(), Some("xe"));
    }

    /// The state this whole module exists for: the card is in the slot and nothing is driving
    /// it. There is no way to produce it from real hardware without taking the display down.
    #[test]
    fn a_card_with_no_bound_driver_is_reported_as_unbound() {
        let fake = FakeSysfs::new("unbound");
        fake.device("0000:04:00.0", "0x8086", "0xe20b", "0x030000", None);

        let found = scan_at(&fake.root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kernel_driver, None);
        assert!(found[0].describe().contains("nothing is bound to it"), "{:?}", found[0]);
    }

    #[test]
    fn devices_are_listed_in_pci_address_order() {
        let fake = FakeSysfs::new("order");
        fake.device("0000:04:00.0", "0x8086", "0xe20b", "0x030000", Some("xe")).device(
            "0000:00:02.0",
            "0x8086",
            "0x7d67",
            "0x030000",
            Some("i915"),
        );

        let found = scan_at(&fake.root);
        assert_eq!(
            found.iter().map(|d| d.address.as_str()).collect::<Vec<_>>(),
            ["0000:00:02.0", "0000:04:00.0"]
        );
    }

    /// A machine with no sysfs — or a container that does not mount it — must degrade to "no
    /// evidence", never to a failure.
    #[test]
    fn an_absent_sysfs_yields_no_evidence_rather_than_an_error() {
        assert!(scan_at(Path::new("/this-path-does-not-exist/pci/devices")).is_empty());
    }
}
