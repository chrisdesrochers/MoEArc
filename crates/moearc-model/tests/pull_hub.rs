//! Integration tests for `pull` against the live Hugging Face Hub.
//!
//! Skipped unless `MOEARC_TEST_HUB=1`. These make real network requests and download a real
//! (small) model, so they are opt-in: CI without egress must not fail on them, and no
//! contributor should pay a few hundred megabytes for `cargo test`.
//!
//! ```text
//! MOEARC_TEST_HUB=1 cargo test -p moearc-model --test pull_hub -- --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` matters: these share one download directory on purpose, because the
//! interesting behaviour is what a *second* run does with what a first one left behind.

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use moearc_model::pull::{FileSelect, ProgressSink, PullError, PullRequest, Verification, pull};

/// A public GGUF repository small enough to download in a test.
const REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
/// The smallest quantisation it publishes.
const QUANT: &str = "q2_k";

fn enabled() -> bool {
    std::env::var_os("MOEARC_TEST_HUB").is_some()
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("moearc-pull-hub-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn request(dir: &Path) -> PullRequest {
    PullRequest {
        repo: REPO.to_string(),
        select: FileSelect::Quant(QUANT.to_string()),
        revision: None,
        dest_dir: dir.to_path_buf(),
        token: None,
        verify: true,
        force: false,
    }
}

/// Records every callback so the sequence can be asserted, not just the outcome.
#[derive(Default)]
struct Recorder(Mutex<Vec<(u64, u64)>>);

impl ProgressSink for Recorder {
    fn on_progress(&self, downloaded: u64, total: u64) {
        self.0.lock().unwrap().push((downloaded, total));
    }
}

#[test]
fn a_real_download_completes_verifies_and_then_costs_nothing_to_repeat() {
    if !enabled() {
        eprintln!("skipped: set MOEARC_TEST_HUB=1 to run the live Hub tests");
        return;
    }
    let dir = scratch();
    let _ = std::fs::remove_dir_all(&dir);
    let rec = Recorder::default();

    let got = pull(&request(&dir), Some(&rec)).expect("a public model should download");
    assert!(got.path.is_file());
    assert_eq!(std::fs::metadata(&got.path).unwrap().len(), got.file_size);
    assert_eq!(got.bytes_transferred, got.file_size);
    assert!(!got.was_resumed());
    // This repo publishes a dense model. That is a *successful* download, not a failure — the
    // GGUF structure was checked, only the MoE metadata is absent.
    assert!(matches!(got.verification, Verification::Dense { .. }), "{:?}", got.verification);

    let seen = rec.0.lock().unwrap().clone();
    assert!(seen.len() > 2, "expected progress callbacks, saw {}", seen.len());
    assert_eq!(seen.first(), Some(&(0, got.file_size)));
    assert_eq!(seen.last(), Some(&(got.file_size, got.file_size)));
    // Monotonic: a gauge that goes backwards is a bug in the accounting, not a cosmetic one.
    for w in seen.windows(2) {
        assert!(w[1].0 >= w[0].0, "progress went backwards: {:?} -> {:?}", w[0], w[1]);
    }

    // Running the same command twice is the natural response to any hiccup, so it must be free.
    let again = pull(&request(&dir), None).expect("a present file should be accepted as-is");
    assert_eq!(again.bytes_transferred, 0);
    assert_eq!(again.file_size, got.file_size);
}

#[test]
fn an_interrupted_transfer_resumes_instead_of_restarting() {
    if !enabled() {
        return;
    }
    let dir = scratch().join("resume");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    /// Aborts the transfer partway by panicking out of the progress callback.
    ///
    /// A real interruption — `SIGKILL`, a lost link, a full disk — cannot be staged inside a
    /// test process, but unwinding out of the callback exercises the same thing that matters:
    /// the download stops with bytes on disk and no orderly shutdown. If the `.part` file were
    /// truncated on open, or the resume offset came from anywhere but the file's real length,
    /// this test would fail.
    struct AbortAfter {
        limit: u64,
        reached: AtomicU64,
    }
    impl ProgressSink for AbortAfter {
        fn on_progress(&self, downloaded: u64, _total: u64) {
            self.reached.store(downloaded, Ordering::SeqCst);
            assert!(downloaded < self.limit, "deliberate interruption at {downloaded} B");
        }
    }

    // Interrupt after a few megabytes: enough that the resume is unambiguous, small enough that
    // the test is quick.
    let aborter = AbortAfter { limit: 4 << 20, reached: AtomicU64::new(0) };
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // the interruption is expected; do not print it
    let interrupted =
        std::panic::catch_unwind(AssertUnwindSafe(|| pull(&request(&dir), Some(&aborter))));
    std::panic::set_hook(hook);
    assert!(interrupted.is_err(), "the callback panic should have aborted the pull");

    let part = dir.join(format!("qwen2.5-0.5b-instruct-{QUANT}.gguf.part"));
    let partial = std::fs::metadata(&part).expect("an interrupted pull must leave a .part").len();
    assert!(partial > 0, "nothing was written before the interruption");

    let rec = Recorder::default();
    let done = pull(&request(&dir), Some(&rec)).expect("the second attempt should finish");
    assert!(done.was_resumed(), "expected a resume, got resumed_from = {}", done.resumed_from);
    assert_eq!(done.resumed_from, partial, "resume must start at the .part file's real length");
    // The whole point: the resumed run transfers only what was missing.
    assert_eq!(done.resumed_from + done.bytes_transferred, done.file_size);
    assert!(done.bytes_transferred < done.file_size);
    // The first callback reports the inherited bytes, not zero — otherwise a gauge would jump
    // backwards on resume, which is the reason this crate does its own progress accounting
    // rather than using hf-hub's per-request deltas.
    assert_eq!(rec.0.lock().unwrap().first(), Some(&(partial, done.file_size)));
    // And the spliced file is a real model, which is the only proof that matters.
    assert!(matches!(done.verification, Verification::Dense { .. }));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_partial_from_different_content_is_discarded_not_spliced() {
    if !enabled() {
        return;
    }
    let dir = scratch().join("stale");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // A partial file whose stamp names content that is not what the repo holds now. Splicing it
    // would yield a correctly sized, silently corrupt model — the worst available outcome.
    let part = dir.join(format!("qwen2.5-0.5b-instruct-{QUANT}.gguf.part"));
    std::fs::write(&part, vec![0xAB; 8 << 20]).unwrap();
    std::fs::write(
        dir.join(format!("qwen2.5-0.5b-instruct-{QUANT}.gguf.part.id")),
        "not-the-real-oid\n1\n",
    )
    .unwrap();

    let got = pull(&request(&dir), None).expect("a stale partial should be replaced, not fail");
    assert_eq!(got.resumed_from, 0, "a mismatched stamp must force a restart");
    assert_eq!(got.bytes_transferred, got.file_size);
    assert!(matches!(got.verification, Verification::Dense { .. }));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_repo_that_does_not_exist_says_so_without_guessing() {
    if !enabled() {
        return;
    }
    let dir = scratch().join("missing");
    let req = PullRequest { repo: "moearc-test/no-such-repo-exists-here".into(), ..request(&dir) };
    match pull(&req, None) {
        // Anonymous requests cannot tell "absent" from "private"; the error must say both.
        Err(PullError::RepoNotFoundOrPrivate { .. }) | Err(PullError::RepoNotFound { .. }) => {}
        other => panic!("expected a not-found error, got {other:?}"),
    }
}

#[test]
fn an_ambiguous_quant_selector_is_refused_rather_than_resolved() {
    if !enabled() {
        return;
    }
    let dir = scratch().join("ambiguous");
    // This repo publishes both q5_0 and q5_k_m, so `q5` cannot mean one thing.
    let req = PullRequest { select: FileSelect::Quant("q5".into()), ..request(&dir) };
    match pull(&req, None) {
        Err(PullError::AmbiguousSelector { matches, .. }) => assert_eq!(matches.len(), 2),
        other => panic!("expected an ambiguity error, got {other:?}"),
    }
}
