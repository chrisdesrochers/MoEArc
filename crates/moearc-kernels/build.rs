//! Compile the SYCL kernels with Intel's DPC++ and link them in.
//!
//! 🔴 The toolchain requirement lands HERE, on whoever builds MoEArc, and never on whoever
//! runs it. `docs/packaging.md` treats that distinction as load-bearing: "requires oneAPI to
//! build" sounds disqualifying and is not, because the user receives a compiled artifact.
//!
//! Bindings are hand-written in `src/ffi.rs` rather than generated. bindgen would have to parse
//! `<sycl/sycl.hpp>`, which would put the oneAPI headers back in every consumer's build —
//! reintroducing the dependency this arrangement exists to remove.
//!
//! # Why a shared object rather than a static archive
//!
//! The first version built a `.a` with `ar` and let cargo link it with `cc`. It failed on
//! `undefined symbol: _intel_fast_memcpy` — a symbol from `libintlc`, one of several Intel
//! runtime libraries that `icpx` links automatically and `cc` knows nothing about. Chasing
//! them one at a time (`intlc`, `irc`, `imf`, `svml`, `irng`, ...) is the pitfall the
//! vendoring literature warns about, and the list is a property of the compiler version rather
//! than of our code.
//!
//! Letting `icpx` perform the link instead makes that its problem: the `.so` records its own
//! dependencies in `DT_NEEDED`, and cargo links one library. It is also the shape we ship
//! anyway — the SYCL runtime cannot be statically linked, so a packaged MoEArc embeds this
//! object and extracts it at run time (see `docs/packaging.md`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels.cpp");
    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let oneapi = std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| "/opt/intel/oneapi".into());

    let icpx = find_icpx(&oneapi).unwrap_or_else(|| {
        panic!(
            "could not find `icpx`. MoEArc's kernels are compiled with Intel's DPC++, which is \
             needed to BUILD but never to RUN. Install the oneAPI base toolkit, or set \
             ONEAPI_ROOT to an existing install (looked under {oneapi})."
        )
    });

    let so = out.join("libmoearc_kernels.so");
    let status = Command::new(&icpx)
        .args(["-fsycl", "-O2", "-fPIC", "-shared"])
        .arg("kernels.cpp")
        .arg("-o")
        .arg(&so)
        .status()
        .expect("failed to run icpx");
    assert!(status.success(), "icpx failed to build the kernel library");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=moearc_kernels");

    // An rpath so tests and dev builds find the object without LD_LIBRARY_PATH. A packaged
    // build extracts the embedded copy and does not rely on this.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());

    // The SYCL runtime lives in the oneAPI tree; the kernel object records it in DT_NEEDED,
    // but the dynamic loader still has to find it at run time.
    for dir in ["lib", "lib64", "compiler/latest/lib"] {
        let p = Path::new(&oneapi).join(dir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p.display());
        }
    }

    // Expose the built object so a packaging step can embed it.
    println!("cargo:rustc-env=MOEARC_KERNEL_LIB={}", so.display());
}

/// Locate `icpx`, preferring an explicit oneAPI root over whatever is on PATH.
fn find_icpx(oneapi: &str) -> Option<PathBuf> {
    for c in [
        format!("{oneapi}/compiler/latest/bin/icpx"),
        format!("{oneapi}/compiler/latest/linux/bin/icpx"),
    ] {
        let p = PathBuf::from(&c);
        if p.exists() {
            return Some(p);
        }
    }
    which_on_path("icpx")
}

fn which_on_path(bin: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(bin)).find(|p| p.exists()))
}
