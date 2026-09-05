//! Negative controls for the failure paths, driven out-of-process.
//!
//! `docs/ux.md` treats a legible failure as a feature, and a feature nobody has watched fail
//! is not evidence of anything. Each test here forces one failure mode and asserts on the
//! message a user would actually see.
//!
//! They run the report binary as a subprocess rather than calling `detect()` in-process for
//! two reasons: the loader override is an environment variable, which is process-global and
//! would race the other tests, and the exit code is part of the contract.

use std::ffi::OsStr;
use std::process::Command;

use moearc_device::{DetectError, LOADER_PATH_ENV, detect_with_loader};

const REPORT_BIN: &str = env!("CARGO_BIN_EXE_moearc-device-report");

/// The headline case: MoEArc running on a machine with no Level Zero at all. The binary must
/// start, fail cleanly, and explain itself.
#[test]
fn a_missing_loader_produces_an_explanation_and_a_nonzero_exit() {
    let output = Command::new(REPORT_BIN)
        .env(LOADER_PATH_ENV, "libze_loader.so.this-does-not-exist")
        .output()
        .expect("report binary should start even with no Level Zero present");

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "expected a failure exit, got: {:?}", output.status);
    assert!(stderr.contains("Level Zero"), "{stderr}");
    assert!(stderr.contains("libze_loader.so.this-does-not-exist"), "{stderr}");
    assert!(stderr.contains(LOADER_PATH_ENV), "{stderr}");
    // The product promise: we never hand the user a bare dependency complaint.
    assert!(stderr.contains("ships its own copy"), "{stderr}");
    // A panic would mean an abort message and no explanation.
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn a_missing_loader_is_typed_as_such() {
    let err = detect_with_loader(OsStr::new("libze_loader.so.this-does-not-exist"))
        .expect_err("a nonexistent library cannot be loaded");
    // On a machine with a GPU present the error is wrapped with the physical evidence, so
    // the cause is asserted through `root_cause` rather than on the outer variant.
    assert!(matches!(err.root_cause(), DetectError::LoaderNotFound { .. }), "{err:?}");
}

/// A real, loadable library that is not Level Zero. Distinguishing this from "no library" is
/// worth a variant: the user's action is different, and a wrong `MOEARC_ZE_LOADER` is the
/// obvious way to reach it.
#[test]
fn a_library_that_is_not_a_loader_is_reported_as_the_wrong_library() {
    let err = detect_with_loader(OsStr::new("libc.so.6"))
        .expect_err("libc is loadable but exports no Level Zero entry points");
    match err.root_cause() {
        DetectError::NotALoader { symbol, .. } => assert_eq!(*symbol, "zeInit"),
        other => panic!("expected NotALoader, got {other:?}"),
    }
    assert!(err.to_string().contains("is not a Level Zero loader"), "{err}");
}
