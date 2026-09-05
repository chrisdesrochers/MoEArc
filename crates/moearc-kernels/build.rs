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
//!
//! # 🔴 How a downstream binary finds the object: the soname carries the path
//!
//! This is the part that was wrong for a while and is worth stating plainly, because the
//! obvious fix does not work.
//!
//! A build script's `cargo:rustc-link-arg` applies **only to the crate that emits it**. It is
//! not inherited by anything downstream. So an `-Wl,-rpath,$OUT_DIR` emitted here reaches this
//! crate's own tests and nothing else: `moearc-server` linked, and then died in the dynamic
//! loader with `libmoearc_kernels.so: cannot open shared object file`. Every test was green,
//! because the tests are the one target class that *does* inherit it.
//!
//! `cargo:rustc-link-search` and `cargo:rustc-link-lib` **do** propagate — that is why
//! downstream crates link at all — but `-L` is a *link*-time path and leaves no trace in the
//! executable. Emitting more link args (`-bins`, `-tests`, ...) cannot help: they are all
//! scoped to this crate too.
//!
//! So the path has to travel inside something that propagates, and there is exactly one such
//! thing here: the shared object itself. `ld` copies a library's `DT_SONAME` verbatim into the
//! `DT_NEEDED` entry of everything that links it, and glibc's loader treats a `DT_NEEDED`
//! string containing a slash as a **path** rather than a name to search for. Setting the
//! soname to the object's absolute location therefore reaches every consumer — binaries,
//! tests, examples, benches, in this crate and in any other — with no cooperation from any of
//! them and no environment variable.
//!
//! Two consequences worth knowing:
//!
//! - It is **not relocatable**. The path is this build tree's `OUT_DIR`, so the artifact is a
//!   development build, not something to copy to another machine. That is what the
//!   `include_bytes!` + extract + `dlopen` route in `docs/packaging.md` is for, and it is
//!   unaffected by this — a `dlopen`d copy never consults `DT_NEEDED`.
//! - It is **stale-proof by construction**. `OUT_DIR` is stable for a given crate, profile and
//!   feature set, and cargo rewrites the object in place; anything that moves it to a new
//!   `OUT_DIR` relinks the consumers in the same build. There is no cache to invalidate and no
//!   copy to race against, because nothing is copied.
//!
//! The object also carries its own `DT_RUNPATH` into the oneAPI runtime directories, so that
//! *its* dependencies (`libsycl`, `libsvml`, `libimf`, `libintlc`, `libirng`) resolve for
//! every consumer as well. Those used to be rpaths on this crate's targets, and so had exactly
//! the same non-propagation bug one level down.

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

    // Where the SYCL runtime lives. Used twice: as `-L` so the linker can resolve the object's
    // transitive dependencies, and as an rpath *on the object* so the loader can.
    let runtime_dirs: Vec<PathBuf> = ["lib", "lib64", "compiler/latest/lib"]
        .iter()
        .map(|d| Path::new(&oneapi).join(d))
        .filter(|p| p.exists())
        .collect();

    let so = out.join("libmoearc_kernels.so");
    let mut cmd = Command::new(&icpx);
    cmd.args(["-fsycl", "-O2", "-fPIC", "-shared"]).arg("kernels.cpp").arg("-o").arg(&so);

    // The soname IS the path. See the module comment: this is the only channel that reaches a
    // downstream binary, because `ld` copies it into their `DT_NEEDED` and a `DT_NEEDED` with a
    // slash in it is used as a path.
    cmd.arg(format!("-Wl,-soname,{}", so.display()));

    // `DT_RUNPATH` rather than `DT_RPATH`: it resolves this object's own dependencies and is
    // not inherited by theirs, which is correct here — `libsycl.so.9` carries `$ORIGIN` and
    // finds its Unified Runtime adapters itself.
    cmd.arg("-Wl,--enable-new-dtags");
    for dir in &runtime_dirs {
        cmd.arg(format!("-Wl,-rpath,{}", dir.display()));
    }

    let status = cmd.status().expect("failed to run icpx");
    assert!(status.success(), "icpx failed to build the kernel library");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=moearc_kernels");

    // Link-time only, and unlike link *args* these do propagate to dependents, which is what
    // lets `ld` resolve the object's own `DT_NEEDED` entries when linking a downstream binary.
    // 🔴 Deliberately no `cargo:rustc-link-arg=-Wl,-rpath,...` here. An rpath emitted from a
    // build script reaches this crate's targets and nothing else; relying on one is what let
    // `tests/clean_env_binary.rs`'s failure mode ship unnoticed behind 309 passing tests.
    for dir in &runtime_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
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
