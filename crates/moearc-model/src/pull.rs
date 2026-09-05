//! Model acquisition: fetching a GGUF from the Hugging Face Hub.
//!
//! This is the library half of `moearc pull`. It resolves a repo id and a file (or a quant
//! selector) to one GGUF, downloads it into a directory the caller names, and refuses to hand
//! back anything it has not verified.
//!
//! Three properties are load-bearing, and each one is a deliberate departure from the shortest
//! possible implementation:
//!
//! - **Resumable, crash-safe.** Bytes land in a `.part` file opened for *append*, and a resume
//!   starts from the length that file actually has on disk. Nothing is buffered across the
//!   restart boundary and nothing needs to be: an append-only file is a valid prefix of the
//!   final file at every instant, including halfway through a write, so `SIGKILL` at any point
//!   leaves a resumable state with no bookkeeping to replay.
//! - **Silent.** Not one byte goes to stdout or stderr. The CLI owns the display, and a library
//!   that prints cannot live under a TUI. Progress is a callback; see [`ProgressSink`].
//! - **Verified.** A download is not finished when the socket closes. The size is checked
//!   against what the Hub declared, then the file is parsed by this crate's own GGUF reader, so
//!   a truncated or corrupt transfer fails here with a typed error rather than at first
//!   inference.
//!
//! ## Why `hf-hub` and not raw HTTP
//!
//! The Hub's URL layout, revision resolution, CDN redirect behaviour, `X-Linked-Size` (an LFS
//! file's `Content-Length` is the size of the *pointer*, not the blob), Xet-backed storage and
//! retry semantics are all real and all easy to get subtly wrong. `hf-hub` is Hugging Face's
//! own client and already models them.
//!
//! 🔴 **What it does not do is resume.** `hf-hub`'s `download_file` opens its destination with
//! `File::create`, which truncates, and issues a plain `GET` with no `Range` header — verified
//! by reading `src/repository/download.rs` at version 1.0.0, not assumed from the docs. An
//! interrupted 20 GiB transfer would start again from zero. So this module uses the lower-level
//! `download_file_stream`, which *does* accept a byte range, and owns the file handling itself.
//! That is also why the progress accounting here is our own rather than `hf-hub`'s: its
//! `DownloadEvent::Progress` reports per-file deltas for the current request, so on a resumed
//! transfer it would count from zero at the resume point and a progress bar would jump
//! backwards.

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use hf_hub::HFError;
use hf_hub::repository::RepoTreeEntry;

use crate::{ModelError, ModelInfo, gguf};

// ---------------------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------------------

/// A Hugging Face API token.
///
/// 🔴 **This is a credential and the type is built to keep it one.** There is no `Display`, no
/// `Serialize`, and the [`fmt::Debug`] impl prints a fixed placeholder — so a token cannot reach
/// a log line, a `dbg!`, a panic message or a serialised struct by accident. The only way to
/// read the secret is [`HfToken::expose`], which is crate-private and called in exactly one
/// place: handing it to the HTTP client.
///
/// Nothing in this crate ever puts a token in an error. See [`PullError`] for the related rule
/// about URLs.
#[derive(Clone)]
pub struct HfToken(String);

impl HfToken {
    /// Wrap a token, rejecting whitespace-only input.
    ///
    /// Returns [`None`] for an empty or blank string, because an empty `HF_TOKEN` in the
    /// environment is far more often a broken shell profile than an intent to authenticate,
    /// and sending `Authorization: Bearer ` is worse than sending nothing.
    pub fn new(raw: impl AsRef<str>) -> Option<Self> {
        let t = raw.as_ref().trim();
        (!t.is_empty()).then(|| Self(t.to_string()))
    }

    /// The secret. Crate-private on purpose.
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for HfToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("HfToken(<redacted>)")
    }
}

/// Where a token came from — never what it is.
///
/// Reported so a caller can tell the user *that* they are authenticated and from which source,
/// which is the difference between "downloads are being throttled and I don't know why" and a
/// one-line explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSource {
    /// Passed directly by the caller, e.g. `--token`.
    Explicit,
    /// The `HF_TOKEN` environment variable.
    Env,
    /// A file named by `HF_TOKEN_PATH`.
    TokenPathFile,
    /// `<HF_HOME>/token`, which is where `hf auth login` writes it.
    HfHomeFile,
    /// No token found; requests will be anonymous.
    Absent,
    /// A token exists but `HF_HUB_DISABLE_IMPLICIT_TOKEN` forbade using it implicitly.
    DisabledByEnv,
}

impl fmt::Display for TokenSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Explicit => "supplied by the caller",
            Self::Env => "the HF_TOKEN environment variable",
            Self::TokenPathFile => "the file named by HF_TOKEN_PATH",
            Self::HfHomeFile => "the Hugging Face CLI login file",
            Self::Absent => "not found — downloading anonymously",
            Self::DisabledByEnv => "suppressed by HF_HUB_DISABLE_IMPLICIT_TOKEN",
        })
    }
}

/// Why supplying a token is worth doing, stated for a user who has not got one.
///
/// Two effects are documented by the Hub and are the reason this is plumbed through at all:
///
/// - **Rate limits.** Anonymous requests are limited more tightly than authenticated ones. On a
///   multi-gigabyte model this is the difference between a download that completes and one that
///   starts returning HTTP 429 partway through — see [`PullError::RateLimited`].
/// - **Access.** Gated and private repositories are simply invisible without one. A gated repo
///   answers an anonymous request with 401/403, which this crate surfaces as
///   [`PullError::AuthRequired`] rather than as a bare status code.
///
/// ⚠️ **A third claim is commonly made and is *not* asserted here: that authenticated downloads
/// are faster.** That may well be true — it is the project owner's experience — but no
/// throughput comparison has been run from this code, so this crate does not state it as fact.
/// If it is measured later, this is the place to record the number.
pub const WHY_A_TOKEN_HELPS: &str = "\
An optional Hugging Face token raises the Hub's rate limits and is required for gated or \
private repositories. Set HF_TOKEN, or run `hf auth login`.";

/// Resolve a token, reporting where it came from.
///
/// Order: explicit argument, then `HF_TOKEN`, then the file named by `HF_TOKEN_PATH`, then
/// `<HF_HOME>/token`. This mirrors `hf-hub`'s own `resolve_token`, deliberately — a tool that
/// disagreed with the official client about which credential is in force would be a debugging
/// nightmare. It is reimplemented rather than delegated only so the *source* can be reported;
/// `hf-hub` resolves internally and tells no one.
///
/// An explicit token overrides `HF_HUB_DISABLE_IMPLICIT_TOKEN`, since that variable exists to
/// stop credentials being picked up *implicitly*, not to veto one the user just typed.
pub fn resolve_token(explicit: Option<HfToken>) -> (Option<HfToken>, TokenSource) {
    if let Some(t) = explicit {
        return (Some(t), TokenSource::Explicit);
    }
    let implicit_disabled =
        std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|v| !v.is_empty());

    let env_token = std::env::var("HF_TOKEN").ok();
    let path_token = std::env::var("HF_TOKEN_PATH").ok().and_then(|p| fs::read_to_string(p).ok());
    let home_token = fs::read_to_string(hf_home().join("token")).ok();

    resolve_token_from(env_token, path_token, home_token, implicit_disabled)
}

/// The pure core of [`resolve_token`], split out so the precedence can be tested without
/// touching process-global environment variables — which no test can do safely in parallel.
fn resolve_token_from(
    env_token: Option<String>,
    path_token: Option<String>,
    home_token: Option<String>,
    implicit_disabled: bool,
) -> (Option<HfToken>, TokenSource) {
    let found = env_token
        .and_then(HfToken::new)
        .map(|t| (t, TokenSource::Env))
        .or_else(|| path_token.and_then(HfToken::new).map(|t| (t, TokenSource::TokenPathFile)))
        .or_else(|| home_token.and_then(HfToken::new).map(|t| (t, TokenSource::HfHomeFile)));

    match found {
        None => (None, TokenSource::Absent),
        Some(_) if implicit_disabled => (None, TokenSource::DisabledByEnv),
        Some((t, src)) => (Some(t), src),
    }
}

/// `HF_HOME`, resolved the way the Hugging Face tooling resolves it.
///
/// Order: `HF_HOME`, then `$XDG_CACHE_HOME/huggingface`, then `~/.cache/huggingface`. Mirrors
/// `hf_hub::hf_home`, which is public but `#[doc(hidden)]` upstream and therefore not something
/// to build on.
fn hf_home() -> PathBuf {
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    hf_home_from(var("HF_HOME"), var("XDG_CACHE_HOME"), var("HOME"))
}

/// The pure core of [`hf_home`]. See the note there on testing without env mutation.
fn hf_home_from(hf: Option<String>, xdg: Option<String>, home: Option<String>) -> PathBuf {
    if let Some(h) = hf {
        return PathBuf::from(h);
    }
    if let Some(x) = xdg {
        return PathBuf::from(x).join("huggingface");
    }
    PathBuf::from(home.unwrap_or_else(|| ".".to_string())).join(".cache").join("huggingface")
}

// ---------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------

/// Why a model could not be fetched.
///
/// Per `docs/ux.md` a user must never see a bare error code, so every variant names the thing
/// that went wrong and, where there is one, the action that fixes it.
///
/// 🔴 **No variant carries a URL.** The Hub redirects downloads to a CDN with a presigned query
/// string; echoing that into an error message or a log would publish a time-limited credential
/// and tell the user nothing they did not already know. Errors identify the file by repo and
/// path — which is what the user typed — and carry the server's own message, never the address.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PullError {
    #[error("could not reach the Hugging Face Hub: {detail} (check the network connection)")]
    Network { detail: String },

    #[error(
        "repository `{repo}` was not found on the Hub (check the spelling; private repos need a token)"
    )]
    RepoNotFound { repo: String },

    #[error("revision `{revision}` does not exist in `{repo}`")]
    RevisionNotFound { repo: String, revision: String },

    #[error("`{repo}` has no file `{file}`{}", list_hint(available))]
    FileNotFound { repo: String, file: String, available: Vec<String> },

    #[error("`{repo}` contains no .gguf files; MoEArc loads GGUF models only")]
    NoGgufFiles { repo: String },

    #[error("`{selector}` matches {} files in `{repo}`{}; pass an exact filename", matches.len(), list_hint(matches))]
    AmbiguousSelector { repo: String, selector: String, matches: Vec<String> },

    #[error("`{repo}` requires authentication: it is gated or private. {WHY_A_TOKEN_HELPS}")]
    AuthRequired { repo: String },

    #[error(
        "`{repo}` was not found. It does not exist, is private, or is gated — the Hub answers all \
         three the same way to an anonymous request, so that private repositories cannot be \
         enumerated. Check the spelling first. {WHY_A_TOKEN_HELPS}"
    )]
    RepoNotFoundOrPrivate { repo: String },

    #[error(
        "access to `{repo}` was refused with the token supplied: the token may be expired or \
         revoked, or the repository may be gated and still waiting for you to accept its licence \
         on huggingface.co"
    )]
    AccessDenied { repo: String },

    #[error("the Hub is rate-limiting this download. {WHY_A_TOKEN_HELPS}")]
    RateLimited,

    #[error(
        "not enough space in {}: `{file}` needs {needed} B and only {available} B are free",
        dir.display()
    )]
    InsufficientSpace { dir: PathBuf, file: String, needed: u64, available: u64 },

    #[error("the disk filled up while writing {}; free some space and run the same command again \
             — the partial download is kept and will resume", path.display())]
    DiskFull { path: PathBuf },

    #[error("the Hub reports no size for `{file}`, so the download cannot be verified or resumed")]
    UnknownRemoteSize { file: String },

    #[error(
        "download finished at {actual} B but the Hub declared {expected} B — the transfer was truncated"
    )]
    SizeMismatch { expected: u64, actual: u64 },

    #[error("the downloaded file is not a usable model: {0}")]
    Invalid(#[from] ModelError),

    #[error("could not write to {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("the Hub returned an unexpected response ({status}){}", opt_detail(server_message))]
    Unexpected { status: String, server_message: Option<String> },
}

/// Render a short list of candidates for an error message, or nothing if there are none.
fn list_hint(items: &[String]) -> String {
    if items.is_empty() {
        return String::new();
    }
    // Six is enough to recognise the right one in a quant-per-file GGUF repo without turning a
    // one-line error into a screenful.
    let shown: Vec<&str> = items.iter().take(6).map(String::as_str).collect();
    let more = items.len().saturating_sub(shown.len());
    let tail = if more > 0 { format!(", and {more} more") } else { String::new() };
    format!(" (available: {}{tail})", shown.join(", "))
}

/// Strip anything URL-shaped out of a message this crate did not write.
///
/// 🔴 **Added after a live run leaked one.** `map_hf`'s fall-through arm interpolates
/// `HFError`'s own `Display`, and several `hf-hub` variants append `(<url>)` — so a malformed
/// response surfaced the full resolve URL in a user-facing error, straight through a module
/// whose docs promise no URL ever appears in one. The unit test did not catch it because it
/// asserted on errors built by hand, never on the one arm that embeds a foreign error's text.
///
/// Filtering the rendered string is the only version-proof place to enforce the rule: auditing
/// upstream variant by variant is correct exactly until the next patch release adds one.
fn redact_urls(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let mut rest = msg;
    while let Some(i) = rest.find("http") {
        let tail = &rest[i..];
        if !(tail.starts_with("http://") || tail.starts_with("https://")) {
            // The word "http" on its own is not a URL; step over it and keep looking.
            out.push_str(&rest[..i + 4]);
            rest = &rest[i + 4..];
            continue;
        }
        out.push_str(&rest[..i]);
        out.push_str("<url elided>");
        // A URL ends at the first whitespace or closing bracket. Erring long is safe here:
        // over-eliding costs legibility, under-eliding publishes a presigned link.
        let end = tail
            .find(|c: char| c.is_whitespace() || c == ')' || c == ']' || c == '>' || c == ',')
            .unwrap_or(tail.len());
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

fn opt_detail(msg: &Option<String>) -> String {
    msg.as_deref().map(|m| format!(": {m}")).unwrap_or_default()
}

/// Translate an HTTP status into the error a user can act on, or [`None`] to fall through.
///
/// 🔴 **The anonymous 401 is the subtle one, and it is not a bug in the Hub.** A request for a
/// repository that does not exist and a request for one that exists but is private both answer
/// `401` to an unauthenticated caller — deliberately, so that private repository names cannot be
/// enumerated by probing. Reporting that as "not found" would be a guess, and reporting it as
/// "authentication required" would send someone hunting for a token when they have simply
/// mistyped a name. [`PullError::RepoNotFoundOrPrivate`] says both, which is all that is known.
/// Once a token *is* in play the ambiguity is gone, and the same status means the credential was
/// rejected.
fn map_status(status: u16, repo: &str, authenticated: bool) -> Option<PullError> {
    let repo = repo.to_string();
    Some(match status {
        401 if authenticated => PullError::AccessDenied { repo },
        401 => PullError::RepoNotFoundOrPrivate { repo },
        403 if authenticated => PullError::AccessDenied { repo },
        403 => PullError::AuthRequired { repo },
        404 => PullError::RepoNotFound { repo },
        429 => PullError::RateLimited,
        _ => return None,
    })
}

/// Translate an `hf-hub` error into one the user can act on.
///
/// The wildcard arm is required — `HFError` is `#[non_exhaustive]` — and is also the right
/// default: an unmapped variant becomes [`PullError::Unexpected`] carrying the server's own
/// message, which is more useful than a generic "download failed" and still leaks no URL.
///
/// 🔴 **`HFError::Http` is re-examined for a status rather than passed straight through, and it
/// has to be.** `hf-hub` maps status codes in `HFClient::check_response`, but its paginated
/// listing path — which `list_tree`, and therefore quant selection, goes through — does not call
/// it: `src/pagination.rs` builds `HFError::Http` directly for *any* failure. So a mistyped repo
/// id arrives here as a raw 401 rather than as `HFError::AuthRequired`, and a consumer matching
/// only the semantic variants would report "unexpected response (401)" to a user who had simply
/// made a typo. Found by running it, not by reading it.
fn map_hf(repo: &str, file: Option<&str>, authenticated: bool, e: HFError) -> PullError {
    match e {
        HFError::RepoNotFound { .. } | HFError::BucketNotFound { .. } => {
            PullError::RepoNotFound { repo: repo.to_string() }
        }
        HFError::RevisionNotFound { revision, .. } => {
            PullError::RevisionNotFound { repo: repo.to_string(), revision }
        }
        HFError::EntryNotFound { path, .. } => {
            PullError::FileNotFound { repo: repo.to_string(), file: path, available: Vec::new() }
        }
        HFError::AuthRequired { .. } => map_status(401, repo, authenticated)
            .unwrap_or_else(|| PullError::AuthRequired { repo: repo.to_string() }),
        HFError::Forbidden { .. } => map_status(403, repo, authenticated)
            .unwrap_or_else(|| PullError::AuthRequired { repo: repo.to_string() }),
        HFError::RateLimited { .. } => PullError::RateLimited,
        // A transport failure is the one case where the underlying message is worth quoting: it
        // distinguishes DNS from TLS from a refused connection. `source` is a reqwest error,
        // whose Display does not include the URL.
        HFError::Request { source, .. } => {
            PullError::Network { detail: redact_urls(&source.to_string()) }
        }
        HFError::Io(source) => {
            PullError::Io { path: PathBuf::from(file.unwrap_or_default()), source }
        }
        HFError::Http { context } => map_status(context.status.as_u16(), repo, authenticated)
            .unwrap_or_else(|| PullError::Unexpected {
                status: context.status.to_string(),
                server_message: context.server_message.as_deref().map(redact_urls),
            }),
        other => PullError::Unexpected {
            status: "error".to_string(),
            server_message: Some(redact_urls(&other.to_string())),
        },
    }
}

// ---------------------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------------------

/// Receives download progress.
///
/// `downloaded` counts every byte of the final file that is on disk, **including bytes carried
/// over from an interrupted attempt** — so it only ever moves forward, which is what a gauge
/// needs. `total` is the size the Hub declared.
///
/// Implemented for any `Fn(u64, u64)`, so a caller can pass a closure.
pub trait ProgressSink: Send + Sync {
    fn on_progress(&self, downloaded: u64, total: u64);
}

impl<F: Fn(u64, u64) + Send + Sync> ProgressSink for F {
    fn on_progress(&self, downloaded: u64, total: u64) {
        self(downloaded, total);
    }
}

/// Minimum gap between progress callbacks.
///
/// Throttled in the library rather than left to the caller: chunks arrive in the low tens of
/// kilobytes, so an unthrottled callback fires hundreds of thousands of times on a large model,
/// and every consumer would have to write the same rate limiter. 50 ms is 20 Hz — smooth to a
/// human eye and far below a terminal's useful redraw rate. The first and last callbacks are
/// always delivered regardless, so a gauge starts at the right place and ends at 100%.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(50);

/// Rate-limits calls to a [`ProgressSink`].
struct Throttle<'a> {
    sink: Option<&'a dyn ProgressSink>,
    last: Option<Instant>,
}

impl<'a> Throttle<'a> {
    fn new(sink: Option<&'a dyn ProgressSink>) -> Self {
        Self { sink, last: None }
    }

    fn tick(&mut self, downloaded: u64, total: u64) {
        let Some(sink) = self.sink else { return };
        let now = Instant::now();
        if self.last.is_some_and(|t| now.duration_since(t) < PROGRESS_INTERVAL) {
            return;
        }
        self.last = Some(now);
        sink.on_progress(downloaded, total);
    }

    /// Deliver unconditionally, for the first and final updates.
    fn force(&mut self, downloaded: u64, total: u64) {
        if let Some(sink) = self.sink {
            self.last = Some(Instant::now());
            sink.on_progress(downloaded, total);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Request and result
// ---------------------------------------------------------------------------------------

/// Which file to take from the repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileSelect {
    /// An exact repo-relative path, e.g. `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
    Exact(String),
    /// A case-insensitive substring of the filename, e.g. `Q4_K_M`.
    ///
    /// GGUF repositories publish one file per quantisation with the quant in the name, so this
    /// is how a user actually thinks about the choice. Matching must land on exactly one file;
    /// zero or several is an error, never a silent pick — see [`PullError::AmbiguousSelector`].
    Quant(String),
}

/// A request to fetch one model file.
#[derive(Debug, Clone)]
pub struct PullRequest {
    /// Hub repo id, `owner/name`.
    pub repo: String,
    /// Which file to take.
    pub select: FileSelect,
    /// Git revision; `None` means the repository's default branch.
    pub revision: Option<String>,
    /// Directory to place the model in. Created if absent.
    pub dest_dir: PathBuf,
    /// Optional API token. See [`WHY_A_TOKEN_HELPS`] and [`resolve_token`].
    pub token: Option<HfToken>,
    /// Parse the finished file before returning it. Leave this on.
    pub verify: bool,
    /// Re-download even if a correctly sized file is already present.
    pub force: bool,
}

/// What verification found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verification {
    /// [`PullRequest::verify`] was off. The size was still checked.
    Skipped,
    /// The file parses and is a mixture-of-experts model; here are the planner's inputs.
    MixtureOfExperts(Box<ModelInfo>),
    /// The file parses as valid GGUF but is a dense model.
    ///
    /// 🔴 Not an error, and the distinction matters. `inspect` rejects a dense model with
    /// [`ModelError::NotMixtureOfExperts`], which is correct for its job but would be a lie
    /// here: the *download* succeeded and the file is sound. Treating it as a failed transfer
    /// would send a user chasing a network problem that does not exist. The structural checks —
    /// magic, version, tensor index, and the span check that catches truncation — all ran and
    /// all passed; only the MoE-specific metadata is absent.
    Dense { architecture: String },
}

/// A completed download.
#[derive(Debug, Clone)]
pub struct PulledModel {
    /// Where the model now is.
    pub path: PathBuf,
    /// Its size, as declared by the Hub and confirmed on disk.
    pub file_size: u64,
    /// Bytes transferred by *this* call. Zero when the file was already present.
    pub bytes_transferred: u64,
    /// Bytes inherited from an interrupted earlier attempt. Non-zero means a resume happened.
    pub resumed_from: u64,
    /// What the file turned out to be.
    pub verification: Verification,
    /// Where the token came from, if any. Reportable; never the token itself.
    pub token_source: TokenSource,
}

impl PulledModel {
    /// Whether this call resumed a partial download rather than starting fresh.
    pub fn was_resumed(&self) -> bool {
        self.resumed_from > 0
    }
}

// ---------------------------------------------------------------------------------------
// The download
// ---------------------------------------------------------------------------------------

/// Fetch a model file, resuming if a partial one is present, and verify it.
///
/// Blocking. Safe to call from inside a tokio runtime: the async work runs on a scoped thread
/// with its own current-thread runtime, so no `block_on` ever executes on the caller's thread.
/// That costs one thread spawn per pull and removes a panic that would otherwise only appear
/// once the CLI grew an async runtime.
pub fn pull(
    req: &PullRequest,
    progress: Option<&dyn ProgressSink>,
) -> Result<PulledModel, PullError> {
    // A scoped thread lets the future borrow `req` and `progress` from the caller's stack, so
    // neither has to be `'static` — a `&dyn ProgressSink` would otherwise be impossible to pass.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|source| PullError::Io { path: req.dest_dir.clone(), source })?;
                rt.block_on(pull_async(req, progress))
            })
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

async fn pull_async(
    req: &PullRequest,
    progress: Option<&dyn ProgressSink>,
) -> Result<PulledModel, PullError> {
    let (token, token_source) = resolve_token(req.token.clone());

    let mut builder =
        hf_hub::HFClient::builder().user_agent(concat!("moearc/", env!("CARGO_PKG_VERSION")));
    if let Some(t) = &token {
        builder = builder.token(t.expose());
    }
    // Whether a credential is in play changes what a 401 *means*; see `map_status`.
    let authed = token.is_some();
    let client = builder.build().map_err(|e| map_hf(&req.repo, None, authed, e))?;

    let (owner, name) = hf_hub::split_id(&req.repo);
    let repo = client.model(owner, name);

    let filename = resolve_filename(&repo, req, authed).await?;

    // 🔴 Size and identity come from `paths-info`, not from `get_file_metadata`.
    //
    // The obvious choice is `get_file_metadata`, which HEADs the resolve URL. It does not work on
    // real GGUF repositories: the HEAD follows a redirect to the CDN, the CDN response carries no
    // `X-Repo-Commit` header, and `hf-hub` treats a missing one as a malformed response and
    // fails. That was invisible from reading the code — it took running it against
    // `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, where it fails every time.
    //
    // `paths-info` is one API call that answers both questions the download needs and never
    // touches the CDN: `size` is the real blob size (an LFS file's `Content-Length` is the size
    // of the *pointer* — 134 bytes for this model — which is exactly the trap this sidesteps),
    // and `oid` is the git object id, which changes whenever the content does. The resume stamp
    // needs nothing more.
    let infos = repo
        .get_paths_info()
        .paths(vec![filename.clone()])
        .maybe_revision(req.revision.clone())
        .send()
        .await
        .map_err(|e| map_hf(&req.repo, Some(&filename), authed, e))?;

    let (total, ident) = match infos.iter().find_map(|e| match e {
        RepoTreeEntry::File { path, size, oid, .. } if *path == filename => {
            Some((*size, oid.clone()))
        }
        _ => None,
    }) {
        Some(found) => found,
        // Only here — on the failure path — is a full listing worth its round trip, and here it
        // is worth a great deal: a mistyped quant name is the most likely reason to be standing
        // in this branch.
        None => {
            return Err(PullError::FileNotFound {
                repo: req.repo.clone(),
                file: filename.clone(),
                available: list_ggufs(&repo, req, authed).await.unwrap_or_default(),
            });
        }
    };

    // A zero size defeats both the space check and the post-download size check, so refuse
    // rather than proceed with verification silently disabled.
    if total == 0 {
        return Err(PullError::UnknownRemoteSize { file: filename });
    }

    fs::create_dir_all(&req.dest_dir)
        .map_err(|source| PullError::Io { path: req.dest_dir.clone(), source })?;

    // Flatten any repo subdirectory: a model directory is a flat set of GGUFs, not a mirror of
    // someone's repo layout.
    let base =
        Path::new(&filename).file_name().map_or_else(|| PathBuf::from(&filename), PathBuf::from);
    let dest = req.dest_dir.join(base);
    let part = with_suffix(&dest, ".part");
    let stamp = with_suffix(&dest, ".part.id");

    // Already here and the right size: nothing to do but verify. Makes `pull` idempotent, which
    // matters because the natural response to any failure is to run the same command again.
    if !req.force && fs::metadata(&dest).is_ok_and(|m| m.len() == total) {
        return Ok(PulledModel {
            verification: verify(&dest, req.verify)?,
            path: dest,
            file_size: total,
            bytes_transferred: 0,
            resumed_from: 0,
            token_source,
        });
    }

    // A partial file is only reusable if it belongs to the same remote content. The ETag pins
    // that. Without this check, a repo that republished the file between attempts would splice
    // two different files together and produce a plausible-sized, silently corrupt model — the
    // exact failure this crate exists to make impossible.
    let want_stamp = format!("{ident}\n{total}\n");
    let mut resumed_from =
        if req.force || fs::read_to_string(&stamp).ok().as_deref() != Some(want_stamp.as_str()) {
            let _ = fs::remove_file(&part);
            0
        } else {
            fs::metadata(&part).map(|m| m.len()).unwrap_or(0)
        };
    // A partial longer than the whole file is not a prefix of anything. Start over.
    if resumed_from > total {
        let _ = fs::remove_file(&part);
        resumed_from = 0;
    }

    let remaining = total - resumed_from;
    if let Some(free) = free_bytes(&req.dest_dir) {
        if free < remaining {
            return Err(PullError::InsufficientSpace {
                dir: req.dest_dir.clone(),
                file: filename,
                needed: remaining,
                available: free,
            });
        }
    }

    let mut throttle = Throttle::new(progress);
    throttle.force(resumed_from, total);

    if remaining > 0 {
        fs::write(&stamp, &want_stamp)
            .map_err(|source| PullError::Io { path: stamp.clone(), source })?;

        let (_, mut stream) = repo
            .download_file_stream()
            .filename(filename.clone())
            .maybe_revision(req.revision.as_deref())
            .maybe_range((resumed_from > 0).then_some(resumed_from..total))
            .send()
            .await
            .map_err(|e| map_hf(&req.repo, Some(&filename), authed, e))?;

        // Append, never truncate: this is the whole resume mechanism. Combined with resuming
        // from the file's real on-disk length, it needs no journal and survives a kill signal
        // mid-write, because a prefix of a prefix is still a prefix.
        let handle = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part)
            .map_err(|source| PullError::Io { path: part.clone(), source })?;
        let mut writer = BufWriter::with_capacity(1 << 20, handle);

        let mut have = resumed_from;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| map_hf(&req.repo, Some(&filename), authed, e))?;
            writer.write_all(&chunk).map_err(|source| io_err(&part, source))?;
            have += chunk.len() as u64;
            throttle.tick(have, total);
        }
        writer.flush().map_err(|source| io_err(&part, source))?;
        // fsync before the rename. Without it the rename can reach the disk before the data
        // does, and a power loss leaves a full-length file of zeroes under the final name —
        // which would then pass the size check and fail only at inference.
        writer.get_ref().sync_all().map_err(|source| io_err(&part, source))?;
    }

    let actual = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if actual != total {
        return Err(PullError::SizeMismatch { expected: total, actual });
    }

    fs::rename(&part, &dest).map_err(|source| PullError::Io { path: dest.clone(), source })?;
    let _ = fs::remove_file(&stamp);
    throttle.force(total, total);

    Ok(PulledModel {
        verification: verify(&dest, req.verify)?,
        path: dest,
        file_size: total,
        bytes_transferred: remaining,
        resumed_from,
        token_source,
    })
}

/// Map an IO error, promoting a full disk to its own variant.
///
/// `ENOSPC` is checked by number rather than through `io::ErrorKind::StorageFull`, which is
/// newer than this crate's declared MSRV. The distinction earns its keep: "the disk is full,
/// free some space and re-run — your partial download is kept" is actionable, where "os error
/// 28" is not.
fn io_err(path: &Path, source: std::io::Error) -> PullError {
    #[cfg(unix)]
    if source.raw_os_error() == Some(libc::ENOSPC) {
        return PullError::DiskFull { path: path.to_path_buf() };
    }
    PullError::Io { path: path.to_path_buf(), source }
}

/// `path` with `suffix` appended to its filename.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

/// Every `.gguf` path in the repository, recursively.
///
/// Split out because it serves two callers with opposite cost profiles: quant selection needs it
/// on the happy path, while an exact-filename miss needs it only to fill the "did you mean" list
/// in an error — so the ordinary download never pays for it.
async fn list_ggufs(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    req: &PullRequest,
    authed: bool,
) -> Result<Vec<String>, PullError> {
    let mut stream = Box::pin(
        repo.list_tree()
            .maybe_revision(req.revision.clone())
            .recursive(true)
            .send()
            .map_err(|e| map_hf(&req.repo, None, authed, e))?,
    );
    let mut ggufs = Vec::new();
    while let Some(entry) = stream.next().await {
        let entry = entry.map_err(|e| map_hf(&req.repo, None, authed, e))?;
        if let RepoTreeEntry::File { path, .. } = entry {
            if path.to_ascii_lowercase().ends_with(".gguf") {
                ggufs.push(path);
            }
        }
    }
    Ok(ggufs)
}

/// Turn a [`FileSelect`] into one repo-relative filename.
async fn resolve_filename(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    req: &PullRequest,
    authed: bool,
) -> Result<String, PullError> {
    let selector = match &req.select {
        // An exact name is taken at face value; a wrong one surfaces from the `paths-info` call
        // that follows, which is where the "did you mean" listing is fetched. The happy path
        // never lists the repository.
        FileSelect::Exact(name) => return Ok(name.clone()),
        FileSelect::Quant(q) => q,
    };

    let ggufs = list_ggufs(repo, req, authed).await?;
    if ggufs.is_empty() {
        return Err(PullError::NoGgufFiles { repo: req.repo.clone() });
    }

    let needle = selector.to_ascii_lowercase();
    let mut hits: Vec<String> =
        ggufs.iter().filter(|p| p.to_ascii_lowercase().contains(&needle)).cloned().collect();
    match hits.len() {
        1 => Ok(hits.remove(0)),
        0 => Err(PullError::FileNotFound {
            repo: req.repo.clone(),
            file: selector.clone(),
            available: ggufs,
        }),
        // Never guess. A multi-part GGUF or a near-miss quant name (`Q4_K` matching `Q4_K_M`
        // and `Q4_K_S`) both land here, and picking one would give the user a different model
        // than they asked for without saying so.
        _ => Err(PullError::AmbiguousSelector {
            repo: req.repo.clone(),
            selector: selector.clone(),
            matches: hits,
        }),
    }
}

/// Parse the finished file, so a corrupt transfer fails here rather than at first inference.
fn verify(path: &Path, enabled: bool) -> Result<Verification, PullError> {
    if !enabled {
        return Ok(Verification::Skipped);
    }
    // `gguf::read` is the structural pass: magic, version, every declared length bounded against
    // the real file size, and the tensor-span check that catches a truncated download. It runs
    // for every model, MoE or not.
    let header = gguf::read(path)?;
    match ModelInfo::from_header(&header) {
        Ok(info) => Ok(Verification::MixtureOfExperts(Box::new(info))),
        Err(ModelError::NotMixtureOfExperts { architecture }) => {
            Ok(Verification::Dense { architecture })
        }
        Err(e) => Err(PullError::Invalid(e)),
    }
}

/// Bytes an unprivileged process can still write into `dir`, or [`None`] if unknowable.
///
/// `f_bavail`, not `f_bfree`: the latter counts the filesystem's reserved blocks, which only
/// root can use, and would let this pass a space check that the actual write then fails. On a
/// 20 GiB model the reserve is easily large enough to matter.
#[cfg(unix)]
fn free_bytes(dir: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated path that outlives the call, and `stat` is
    // fully written by `statvfs` before it is read. A non-zero return means it was not written,
    // and that path returns without reading it.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return None;
        }
        Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
    }
}

/// No portable equivalent; the space check is skipped rather than guessed.
#[cfg(not(unix))]
fn free_bytes(_dir: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Obviously-fake tokens. Real ones start `hf_`; these deliberately do not, so no scanner
    /// or reader can mistake this file for a leak.
    const FAKE_ENV: &str = "NOT-A-REAL-TOKEN-env";
    const FAKE_PATH: &str = "NOT-A-REAL-TOKEN-path";
    const FAKE_HOME: &str = "NOT-A-REAL-TOKEN-home";

    #[test]
    fn a_token_never_prints_itself() {
        let t = HfToken::new(FAKE_ENV).unwrap();
        let rendered = format!("{t:?}");
        assert_eq!(rendered, "HfToken(<redacted>)");
        assert!(!rendered.contains(FAKE_ENV));
        // The secret is still reachable by the one caller that needs it.
        assert_eq!(t.expose(), FAKE_ENV);
    }

    #[test]
    fn blank_tokens_are_not_tokens() {
        assert!(HfToken::new("").is_none());
        assert!(HfToken::new("   \n").is_none());
        // Surrounding whitespace is stripped: a token file written by a shell ends in a newline.
        assert_eq!(HfToken::new(format!("  {FAKE_ENV}\n")).unwrap().expose(), FAKE_ENV);
    }

    #[test]
    fn token_precedence_is_env_then_path_then_login_file() {
        let all = || {
            (Some(FAKE_ENV.to_string()), Some(FAKE_PATH.to_string()), Some(FAKE_HOME.to_string()))
        };
        let (e, p, h) = all();
        let (t, src) = resolve_token_from(e, p, h, false);
        assert_eq!(t.unwrap().expose(), FAKE_ENV);
        assert_eq!(src, TokenSource::Env);

        let (_, p, h) = all();
        let (t, src) = resolve_token_from(None, p, h, false);
        assert_eq!(t.unwrap().expose(), FAKE_PATH);
        assert_eq!(src, TokenSource::TokenPathFile);

        let (_, _, h) = all();
        let (t, src) = resolve_token_from(None, None, h, false);
        assert_eq!(t.unwrap().expose(), FAKE_HOME);
        assert_eq!(src, TokenSource::HfHomeFile);

        let (t, src) = resolve_token_from(None, None, None, false);
        assert!(t.is_none());
        assert_eq!(src, TokenSource::Absent);
    }

    #[test]
    fn an_empty_env_var_does_not_shadow_a_real_token_file() {
        // A shell profile exporting an empty HF_TOKEN is common and must not silently disable
        // the login file — nor send `Authorization: Bearer ` with nothing after it.
        let (t, src) = resolve_token_from(Some(String::new()), None, Some(FAKE_HOME.into()), false);
        assert_eq!(t.unwrap().expose(), FAKE_HOME);
        assert_eq!(src, TokenSource::HfHomeFile);
    }

    #[test]
    fn implicit_tokens_can_be_disabled() {
        let (t, src) = resolve_token_from(Some(FAKE_ENV.into()), None, None, true);
        assert!(t.is_none());
        assert_eq!(src, TokenSource::DisabledByEnv);
        // But nothing is reported as suppressed when there was nothing to suppress.
        let (t, src) = resolve_token_from(None, None, None, true);
        assert!(t.is_none());
        assert_eq!(src, TokenSource::Absent);
    }

    #[test]
    fn an_explicit_token_beats_the_environment_and_the_disable_switch() {
        let explicit = HfToken::new("NOT-A-REAL-TOKEN-explicit").unwrap();
        let (t, src) = resolve_token(Some(explicit));
        assert_eq!(t.unwrap().expose(), "NOT-A-REAL-TOKEN-explicit");
        assert_eq!(src, TokenSource::Explicit);
    }

    #[test]
    fn hf_home_follows_the_documented_order() {
        assert_eq!(
            hf_home_from(Some("/hf".into()), Some("/xdg".into()), Some("/h".into())),
            PathBuf::from("/hf")
        );
        assert_eq!(
            hf_home_from(None, Some("/xdg".into()), Some("/h".into())),
            PathBuf::from("/xdg/huggingface")
        );
        assert_eq!(
            hf_home_from(None, None, Some("/h".into())),
            PathBuf::from("/h/.cache/huggingface")
        );
    }

    #[test]
    fn no_error_message_carries_a_url_or_a_token() {
        // The presigned CDN URL is the thing that must never be echoed; this asserts the shape
        // of the rendered messages, not merely the intent of the code.
        let errors: Vec<PullError> = vec![
            PullError::RepoNotFound { repo: "owner/repo".into() },
            PullError::AuthRequired { repo: "owner/repo".into() },
            PullError::RateLimited,
            PullError::SizeMismatch { expected: 10, actual: 4 },
            PullError::Network { detail: "connection refused".into() },
            PullError::UnknownRemoteSize { file: "m.gguf".into() },
            PullError::DiskFull { path: PathBuf::from("/tmp/m.gguf.part") },
            PullError::Unexpected {
                status: "500 Internal Server Error".into(),
                server_message: None,
            },
            PullError::RepoNotFoundOrPrivate { repo: "owner/repo".into() },
            PullError::AccessDenied { repo: "owner/repo".into() },
        ];
        for e in errors {
            let msg = e.to_string();
            assert!(!msg.contains("http://"), "{msg}");
            assert!(!msg.contains("https://"), "{msg}");
            assert!(!msg.to_ascii_lowercase().contains("bearer"), "{msg}");
            assert!(!msg.contains("hf_"), "{msg}");
        }
    }

    #[test]
    fn an_anonymous_401_is_reported_as_ambiguous_not_guessed() {
        // The Hub answers 401 for both "no such repo" and "private repo" when anonymous. Saying
        // either one outright would be a guess; this asserts we say both.
        let e = map_status(401, "owner/typo", false).unwrap();
        let msg = e.to_string();
        assert!(matches!(e, PullError::RepoNotFoundOrPrivate { .. }));
        assert!(msg.contains("does not exist"), "{msg}");
        assert!(msg.contains("private"), "{msg}");

        // With a token the ambiguity is gone: the credential was rejected.
        assert!(matches!(map_status(401, "owner/x", true), Some(PullError::AccessDenied { .. })));
        // 403 anonymous is a gate to pass, not a typo to fix.
        assert!(matches!(map_status(403, "owner/x", false), Some(PullError::AuthRequired { .. })));
        assert!(matches!(map_status(404, "owner/x", false), Some(PullError::RepoNotFound { .. })));
        assert!(matches!(map_status(429, "owner/x", false), Some(PullError::RateLimited)));
        // Anything unrecognised falls through rather than being forced into a wrong shape.
        assert!(map_status(500, "owner/x", false).is_none());
        assert!(map_status(200, "owner/x", false).is_none());
    }

    #[test]
    fn a_foreign_error_message_cannot_smuggle_a_url_through() {
        // The exact string that leaked in a live run, before `redact_urls` existed.
        let real = "Hub response missing required data: missing X-Repo-Commit header for a.gguf \
                    (https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/a.gguf)";
        let clean = redact_urls(real);
        assert!(!clean.contains("https://"), "{clean}");
        assert!(clean.contains("<url elided>"), "{clean}");
        // The useful half of the message survives.
        assert!(clean.contains("missing X-Repo-Commit header"), "{clean}");

        // A presigned CDN link — the case that actually matters.
        let signed = "failed at https://cdn-lfs.hf.co/repos/ab/blob?X-Amz-Signature=deadbeef now";
        assert_eq!(redact_urls(signed), "failed at <url elided> now");

        // The bare word "http" is not a URL and must not be eaten.
        assert_eq!(redact_urls("plain http talk"), "plain http talk");
        assert_eq!(redact_urls("no urls here"), "no urls here");

        // And the guarantee is asserted through `map_hf`, not only through the helper — that is
        // the arm that leaked.
        let e = map_hf("o/r", None, false, hf_hub::HFError::Other(signed.to_string()));
        assert!(!e.to_string().contains("https://"), "{e}");
        assert!(!e.to_string().contains("Signature"), "{e}");
    }

    #[test]
    fn errors_name_the_fix_not_just_the_fault() {
        // `docs/ux.md`: never a bare code. Each of these must tell the user what to do.
        assert!(PullError::RateLimited.to_string().contains("HF_TOKEN"));
        assert!(PullError::AuthRequired { repo: "a/b".into() }.to_string().contains("HF_TOKEN"));
        assert!(PullError::DiskFull { path: "x".into() }.to_string().contains("resume"));
        assert!(PullError::RepoNotFound { repo: "a/b".into() }.to_string().contains("spelling"),);
    }

    #[test]
    fn a_file_list_hint_is_bounded() {
        assert_eq!(list_hint(&[]), "");
        let many: Vec<String> = (0..10).map(|i| format!("f{i}.gguf")).collect();
        let hint = list_hint(&many);
        assert!(hint.contains("f0.gguf"));
        assert!(hint.contains("and 4 more"));
        assert!(!hint.contains("f9.gguf"));
    }

    #[test]
    fn progress_is_throttled_but_always_starts_and_ends() {
        use std::sync::Mutex;
        struct Rec(Mutex<Vec<(u64, u64)>>);
        impl ProgressSink for Rec {
            fn on_progress(&self, d: u64, t: u64) {
                self.0.lock().unwrap().push((d, t));
            }
        }
        let rec = Rec(Mutex::new(Vec::new()));
        let mut th = Throttle::new(Some(&rec));
        th.force(0, 100);
        // A burst inside one interval collapses to (at most) the first of them.
        for i in 1..50 {
            th.tick(i, 100);
        }
        th.force(100, 100);
        let seen = rec.0.lock().unwrap().clone();
        assert_eq!(seen.first(), Some(&(0, 100)));
        assert_eq!(seen.last(), Some(&(100, 100)));
        assert!(seen.len() < 10, "expected throttling, saw {} calls", seen.len());
    }

    #[test]
    fn a_null_sink_is_free_and_silent() {
        let mut th = Throttle::new(None);
        th.force(0, 10);
        th.tick(5, 10);
    }

    #[test]
    fn suffixes_attach_to_the_filename_not_the_extension() {
        assert_eq!(with_suffix(Path::new("/m/a.gguf"), ".part"), PathBuf::from("/m/a.gguf.part"));
        assert_eq!(with_suffix(Path::new("a.gguf"), ".part.id"), PathBuf::from("a.gguf.part.id"));
    }

    #[test]
    fn free_space_is_reported_for_a_real_directory() {
        // Not a fixed number — only that the syscall path works and returns something usable.
        let free = free_bytes(&std::env::temp_dir());
        #[cfg(unix)]
        assert!(free.is_some_and(|b| b > 0), "statvfs returned {free:?} for the temp dir");
        #[cfg(not(unix))]
        assert!(free.is_none());
    }

    #[test]
    fn a_dense_model_verifies_as_dense_rather_than_failing() {
        // Reuses the synthetic GGUF builder from the crate's own tests via a real file: a
        // download of a dense model is a *successful* download.
        let bytes = crate::tests::synthetic_gguf_without_experts();
        let dir = std::env::temp_dir().join(format!("moearc-pull-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dense.gguf");
        fs::write(&path, &bytes).unwrap();
        match verify(&path, true) {
            Ok(Verification::Dense { architecture }) => assert_eq!(architecture, "testmoe"),
            other => panic!("expected Dense, got {other:?}"),
        }
        // And a truncated file of the same model is still an error, so the exemption above is
        // narrow: it forgives a missing expert count, not a broken file.
        let cut = dir.join("cut.gguf");
        fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
        assert!(matches!(verify(&cut, true), Err(PullError::Invalid(_))));
        assert!(matches!(verify(&cut, false), Ok(Verification::Skipped)));
        let _ = fs::remove_dir_all(&dir);
    }
}
