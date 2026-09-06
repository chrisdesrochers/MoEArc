//! Records the commit this binary was built from, so `moearc bench` can print it.
//!
//! 🔴 `bench/PROTOCOL.md` §2: *assert the backend, device, and build commit in the output of
//! every run.* A benchmark artefact that cannot name the source it was compiled from is not
//! reproducible, and the failure §2 records — a Vulkan build 4.8x slower than SYCL, producing
//! real CSV and exit 0 — is precisely the kind that only a provenance field reveals.
//!
//! Everything here degrades to `unknown` rather than failing the build: a source tarball with
//! no `.git`, or a machine with no `git` on PATH, must still compile.

use std::process::Command;

fn main() {
    // A packager who builds from a tarball can supply the commit directly.
    println!("cargo::rerun-if-env-changed=MOEARC_BUILD_COMMIT");
    // Rebuild when HEAD moves. `.git/HEAD` covers a checkout; the ref file covers a commit on
    // the current branch, which is the common case and the one a naive HEAD-only watch misses.
    for path in git_watch_paths() {
        println!("cargo::rerun-if-changed={path}");
    }

    let commit = std::env::var("MOEARC_BUILD_COMMIT")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| git(&["rev-parse", "--short=9", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());

    // `--porcelain` is empty exactly when the tree is clean. `None` — git absent or not a
    // repository — is left as `unknown`, which the artefact prints rather than assuming clean.
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(out) if out.trim().is_empty() => "no",
        Some(_) => "yes",
        None => "unknown",
    };

    println!("cargo::rustc-env=MOEARC_BUILD_COMMIT={commit}");
    println!("cargo::rustc-env=MOEARC_BUILD_DIRTY={dirty}");
    println!(
        "cargo::rustc-env=MOEARC_BUILD_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
    );
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_watch_paths() -> Vec<String> {
    let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) else {
        return Vec::new();
    };
    let mut paths = vec![format!("{dir}/HEAD")];
    if let Ok(head) = std::fs::read_to_string(format!("{dir}/HEAD")) {
        if let Some(r) = head.strip_prefix("ref: ") {
            paths.push(format!("{dir}/{}", r.trim()));
        }
    }
    paths
}
