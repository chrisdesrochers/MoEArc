//! Does a program that links the kernels actually *start*?
//!
//! 🔴 This is the test that was missing. `docs/packaging.md` records the failure: every crate's
//! tests were green and the workspace built clean, while `moearc-server` exited 127 with
//! `libmoearc_kernels.so: cannot open shared object file`. The suite could not see it because
//! the only things being executed were this crate's own test binaries, which inherit the build
//! script's `cargo:rustc-link-arg` rpath. Nothing downstream does.
//!
//! `src/bin/moearc-kernels-smoke.rs` is a stand-in for those downstream binaries, and it is a
//! fair one **only because the build script now emits no link args at all**. What it does emit
//! — `rustc-link-search` and `rustc-link-lib` — propagates to dependents unchanged, so this
//! binary is linked exactly the way `moearc-server` is. `resolves_without_this_crates_rpath`
//! below asserts that equivalence rather than trusting it, so the day someone reintroduces a
//! local rpath the test stops being a fair proxy *and says so*.

use std::process::Command;

/// Cargo builds the bin before the integration tests and hands us its path.
const SMOKE: &str = env!("CARGO_BIN_EXE_moearc-kernels-smoke");

const MARKER: &str = "moearc-kernels-smoke: ok";

/// The environment is not scrubbed with `env -u LD_LIBRARY_PATH` but emptied outright: no
/// `PATH`, no `HOME`, nothing. Anything that still resolves is resolving from the ELF headers.
fn run_with_empty_env(extra: &[(&str, &str)]) -> std::process::Output {
    let mut cmd = Command::new(SMOKE);
    cmd.env_clear();
    for (k, v) in extra {
        cmd.env(k, v);
    }
    cmd.output().unwrap_or_else(|e| panic!("could not execute {SMOKE}: {e}"))
}

fn assert_started(out: &std::process::Output) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stderr.contains("cannot open shared object file"),
        "the binary died in the dynamic loader — the kernel object does not follow the \
         binary. See the packaging note in `build.rs`.\nstderr: {stderr}"
    );
    assert!(
        out.status.success(),
        "binary exited with {:?}\nstdout: {stdout}\nstderr: {stderr}",
        out.status
    );
    assert!(stdout.contains(MARKER), "expected {MARKER:?} on stdout, got: {stdout:?}");
}

#[test]
fn starts_with_an_empty_environment() {
    assert_started(&run_with_empty_env(&[]));
}

/// A hostile `LD_LIBRARY_PATH` rather than an absent one. Passing this means the resolution is
/// not merely surviving without help, it is not *asking* for help: a `DT_NEEDED` entry that
/// contains a slash is used as a path and never searched for, so a wrong search path cannot
/// shadow it either.
#[test]
fn starts_with_a_hostile_library_path() {
    assert_started(&run_with_empty_env(&[("LD_LIBRARY_PATH", "/nonexistent/moearc")]));
}

/// The proxy check: prove this binary is not passing for a reason `moearc-server` would not get.
///
/// Two assertions, and the second is the real one. The kernel library must be named by an
/// absolute path in `DT_NEEDED` — that is the mechanism, and it is recorded in the object's
/// `DT_SONAME`, so it reaches every consumer. And no `RPATH`/`RUNPATH` on this binary may
/// mention the kernels build directory — if one did, this binary would be finding the library
/// the way only *this crate's* targets can, and the test would be vacuous.
#[test]
fn resolves_without_this_crates_rpath() {
    let Ok(out) = Command::new("readelf").arg("-d").arg(SMOKE).output() else {
        eprintln!("skipping: `readelf` is not available on this machine");
        return;
    };
    assert!(out.status.success(), "readelf -d failed on {SMOKE}");
    let dynamic = String::from_utf8_lossy(&out.stdout);

    let needed: Vec<&str> = dynamic
        .lines()
        .filter(|l| l.contains("(NEEDED)") && l.contains("moearc_kernels"))
        .collect();
    assert_eq!(
        needed.len(),
        1,
        "expected exactly one NEEDED entry for the kernel object in:\n{dynamic}"
    );
    assert!(
        needed[0].contains('/'),
        "the kernel object is named by bare soname, so the loader has to search for it and \
         only this crate's own targets carry a path to it. It must be named by an absolute \
         path (set via DT_SONAME in build.rs). Got: {}",
        needed[0].trim()
    );

    for line in dynamic.lines().filter(|l| l.contains("(RPATH)") || l.contains("(RUNPATH)")) {
        assert!(
            !line.contains("moearc-kernels"),
            "this binary carries an rpath into the kernels build directory, which a downstream \
             binary would not get — the test is no longer a fair proxy. Remove the \
             `cargo:rustc-link-arg` from build.rs.\n{}",
            line.trim()
        );
    }
}
