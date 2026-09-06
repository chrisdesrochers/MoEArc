//! The real model catalogue: every GGUF in the models directory, read from its own header.
//!
//! This is the model half of the wiring `detect.rs` did for devices, and it is deliberately
//! thin. `moearc-model` already answers every question the planner asks — how many experts a
//! file has, how big one resident slot is, what the dense weights cost — so this module's only
//! job is to find the files and translate one vocabulary into the other.
//!
//! Three decisions are worth stating, because each removes a way of being wrong on screen:
//!
//! * **Nothing is parsed out of a filename.** The quantisation, the geometry and the size all
//!   come from the tensor index. A file renamed `-Q8_0.gguf` still reports what it actually
//!   contains — and the one place the filename *is* trimmed, [`without_quant_tag`], only trims
//!   a tag it has confirmed against the measured type.
//! * **A residency slot is a *(block, expert)* pair, not an expert.** A 128-expert model with
//!   36 MoE blocks has 4,608 slots. The cache pages slots, so slots are what the plan is
//!   expressed in; reporting 128 would understate the model by a factor of the block count and
//!   make every residency percentage wrong.
//! * **A file that cannot be read is named, not skipped silently.** A half-finished download
//!   looks exactly like a model that is not there, and "your model is missing" is a worse
//!   message than "this file is truncated".
//!
//! **Cost, measured.** The header is read and the tensor blob is never touched, so the scan is
//! bounded by the tensor *count*, not by the file size. The two are easy to confuse and the
//! numbers separate them: a release build reads `gpt-oss-120b` (687 tensors, 59.0 GiB) in
//! **6.8 ms** and `Qwen3.6-35B-A3B` (733 tensors, 20.6 GiB) in **6.9 ms** — the larger file is
//! the faster read. All four models together, 103 GiB on disk, cost **12.6 ms** of a 13.5 ms
//! `moearc ls --json` against a 0.9 ms empty-directory baseline. Unoptimised the same scan is
//! 56 ms, which is still under a frame.
//!
//! (Warm page cache, which is the case that matters: the header is a few MiB — 12.4 of
//! `gpt-oss-120b`'s 59 GiB, 0.02% of the file — and the OS has it after the first read.)
//!
//! So there is no cache here for speed, and none is needed. The [`OnceCell`] exists for a
//! different reason: [`ModelCatalog::skipped`] has to describe the same scan the cards came
//! from, and reading the directory twice would let the two disagree.

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use moearc_model::gguf::{self, GgufHeader, TensorInfo};
use moearc_model::{ModelInfo, quant};

use crate::source::{ModelCard, ModelCatalog};

/// The environment variable that names the model directory.
pub const MODELS_DIR_ENV: &str = "MOEARC_MODELS";

/// Where to look for models: the flag, then [`MODELS_DIR_ENV`], then the default.
///
/// A stranger's models are not where ours are, so the path is never compiled in. The order is
/// the usual one — an explicit flag beats an environment the user may have forgotten setting.
pub fn models_dir(flag: Option<&Path>) -> PathBuf {
    resolve_dir(
        flag.map(Path::to_path_buf),
        std::env::var_os(MODELS_DIR_ENV),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// The resolution rule, with its inputs passed in so it can be tested.
///
/// The default is a *cache* directory rather than a data one. A GGUF is re-downloadable and
/// tens of gigabytes; putting it where backup tools sweep by default is a surprise the user
/// discovers from their bandwidth bill.
fn resolve_dir(
    flag: Option<PathBuf>,
    env: Option<OsString>,
    xdg_cache: Option<OsString>,
    home: Option<OsString>,
) -> PathBuf {
    fn set(v: Option<OsString>) -> Option<PathBuf> {
        v.filter(|s| !s.is_empty()).map(PathBuf::from)
    }
    if let Some(p) = flag {
        return p;
    }
    if let Some(p) = set(env) {
        return p;
    }
    let base = set(xdg_cache)
        .or_else(|| set(home).map(|h| h.join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("moearc").join("models")
}

/// The catalogue backed by a directory of GGUF files.
pub struct LocalCatalog {
    dir: PathBuf,
    /// The scan, done once per process.
    ///
    /// Not an optimisation — see the module note. Reading the directory twice would let
    /// [`ModelCatalog::skipped`] describe a different set of files from the one the cards came
    /// from, which is the sort of inconsistency a user reads as a bug in their disk.
    scan: OnceCell<Scan>,
}

struct Scan {
    cards: Vec<ModelCard>,
    skipped: Vec<String>,
}

impl LocalCatalog {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, scan: OnceCell::new() }
    }

    fn scan(&self) -> &Scan {
        self.scan.get_or_init(|| scan_dir(&self.dir))
    }
}

impl ModelCatalog for LocalCatalog {
    fn installed(&self) -> anyhow::Result<Vec<ModelCard>> {
        Ok(self.scan().cards.clone())
    }

    /// The same list. There is no curated remote registry yet, and inventing one would mean
    /// printing footprints for models nobody here has read — which is the failure the
    /// provenance note in [`crate::source::Sources`] exists to prevent.
    fn curated(&self) -> anyhow::Result<Vec<ModelCard>> {
        self.installed()
    }

    fn resolve(&self, id: &str) -> anyhow::Result<ModelCard> {
        let cards = &self.scan().cards;
        let want = id.trim();
        if let Some(c) = cards.iter().find(|c| {
            c.id.eq_ignore_ascii_case(want)
                || c.file.as_deref().is_some_and(|f| f.eq_ignore_ascii_case(want))
        }) {
            return Ok(c.clone());
        }
        // A partial handle is how people actually type: `moearc serve gpt-oss` for a file
        // called `gpt-oss-120b-mxfp4.gguf`. It must land on exactly one model — several is an
        // error, never a silent pick.
        let needle = want.to_lowercase();
        let hits: Vec<&ModelCard> = cards.iter().filter(|c| c.id.contains(&needle)).collect();
        match hits.as_slice() {
            [only] => Ok((*only).clone()),
            [] => anyhow::bail!(
                "no model matching `{want}` in {} — `moearc ls` lists what is there",
                self.dir.display()
            ),
            several => anyhow::bail!(
                "`{want}` matches {} models — {}",
                several.len(),
                several.iter().map(|c| c.id.as_str()).collect::<Vec<_>>().join(", ")
            ),
        }
    }

    fn skipped(&self) -> Vec<String> {
        self.scan().skipped.clone()
    }

    fn location(&self) -> Option<String> {
        Some(self.dir.display().to_string())
    }
}

/// Read every GGUF in `dir`.
fn scan_dir(dir: &Path) -> Scan {
    let mut cards = Vec::new();
    let mut skipped = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            // Not an error worth failing the whole command over: a first-run user has no model
            // directory yet, and the renderers say where they looked.
            if e.kind() != std::io::ErrorKind::NotFound {
                skipped.push(format!("{}: {e}", dir.display()));
            }
            return Scan { cards, skipped };
        }
    };

    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
        .collect();
    // Sorted before reading so the "could not read" list is stable between runs.
    paths.sort();

    for path in paths {
        let name = file_name(&path);
        match read_card(&path) {
            Ok(card) => cards.push(card),
            // A dense GGUF in the directory is not a broken file, it is a model this tool does
            // not run. Reported all the same, because a user who put it there expects to see
            // it and would otherwise conclude the scan missed it.
            Err(why) => skipped.push(format!("{name}: {why}")),
        }
    }

    // Largest first. "Will the big one run" is the question this tool exists to answer, and
    // the model the user is least sure about is the one that should not need scrolling to.
    cards.sort_by(|a, b| b.file_bytes.cmp(&a.file_bytes).then_with(|| a.id.cmp(&b.id)));
    shorten_handles(&mut cards);
    Scan { cards, skipped }
}

/// Drop the trailing quantisation tag from each handle, where doing so stays unambiguous.
///
/// `gpt-oss-120b-mxfp4` in a column beside another column reading `mxfp4` says the same thing
/// twice and costs six characters of a row that has to hold a residency figure and a context
/// length. Trimming it is worth doing — but only as a check, never as a guess: the tag is
/// removed only when it matches the type actually measured out of the tensor index, and only
/// when the shorter handle collides with nothing else in the directory. A user keeping two
/// quantisations of one model side by side keeps two full handles and can still tell them
/// apart, which is the entire reason the tag is in the filename.
fn shorten_handles(cards: &mut [ModelCard]) {
    let proposed: Vec<Option<String>> =
        cards.iter().map(|c| without_quant_tag(&c.id, &c.quant)).collect();
    let mut claimed: BTreeMap<String, usize> = BTreeMap::new();
    for (card, short) in cards.iter().zip(&proposed) {
        *claimed.entry(short.clone().unwrap_or_else(|| card.id.clone())).or_default() += 1;
    }
    for (card, short) in cards.iter_mut().zip(proposed) {
        if let Some(short) = short
            && claimed.get(&short) == Some(&1)
        {
            card.id = short;
        }
    }
}

/// `("qwen3-30b-a3b-q4_k_m", "q4_K")` -> `Some("qwen3-30b-a3b")`.
///
/// The last dash-separated segment is removed only if, normalised, it *starts with* the ggml
/// type name we measured. That is what makes it a check rather than a heuristic: `-q4_k_m`
/// goes because the experts really are `q4_K`, `-ud` stays because it is not a type name, and
/// a file mislabelled `-Q8_0` that turns out to hold `q4_K` keeps its whole name — the
/// mismatch is information, and hiding it would be the one thing worse than a long handle.
fn without_quant_tag(handle: &str, quant: &str) -> Option<String> {
    let (head, tag) = handle.rsplit_once('-')?;
    if head.is_empty() || quant.is_empty() {
        return None;
    }
    let normalise = |s: &str| s.to_ascii_uppercase().replace('-', "_");
    normalise(tag).starts_with(&normalise(quant)).then(|| head.to_string())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// One file, read once.
///
/// The header is parsed a single time and used twice: `moearc-model` derives the planner's
/// inputs from it, and the two figures this crate needs on top — the parameter count and the
/// quantisation actually in use — are summed from the same tensor index. Calling
/// [`moearc_model::inspect`] and then re-reading the file would parse it twice.
fn read_card(path: &Path) -> Result<ModelCard, moearc_model::ModelError> {
    let header = gguf::read(path)?;
    let info = ModelInfo::from_header(&header)?;
    Ok(card_from(path, &header, &info))
}

fn card_from(path: &Path, header: &GgufHeader, info: &ModelInfo) -> ModelCard {
    ModelCard {
        // The file's own stem, lowercased. `shorten_handles` may trim a confirmed quantisation
        // tag off the end afterwards; nothing else is invented.
        id: path
            .file_stem()
            .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().to_lowercase()),
        repo: None,
        file: path.file_name().map(|n| n.to_string_lossy().into_owned()),
        quant: dominant_expert_quant(header),
        file_bytes: info.file_size,
        parameters: header.tensors.iter().map(TensorInfo::n_elements).sum(),
        dense_weights_bytes: info.dense_weights_bytes,
        per_expert_bytes: info.per_expert_bytes,
        per_expert_bytes_uniform: info.per_expert_bytes_uniform,
        expert_slots_total: info.moe_block_count.saturating_mul(info.total_experts),
        expert_slots_active: info.moe_block_count.saturating_mul(info.active_experts),
        experts_per_block: info.total_experts,
        active_experts_per_block: info.active_experts,
        moe_blocks: info.moe_block_count,
        kv_bytes_per_token: info.kv_bytes_per_token,
        trained_context_tokens: info.context_length,
        local: true,
        // 🔴 Never true from a file. `docs/ux.md`: a model we have not run does not get a green
        // checkmark. Everything above is read out of a header; none of it has been on the card.
        measured: false,
    }
}

/// The ggml type holding most of the expert weights.
///
/// The expert banks are 88–96% of every MoE file measured here, so the type that covers most
/// of them is the type that describes the file. It is taken from the tensor index rather than
/// from `general.file_type`, which would need a table of llama.cpp's `LLAMA_FTYPE` enum
/// transcribed from another project — a table that goes stale silently and reports a name for
/// a file it never looked inside.
///
/// The name is ggml's own spelling (`q4_K`, `mxfp4`), not the filename's marketing one
/// (`Q4_K_M`). Those differ for a reason: a "Q4_K_M" file is a *mixture*, and real ones here
/// carry q6_K alongside the q4_K. Printing the dominant type says something true about the
/// bytes; printing the filename's tag would repeat a claim nobody checked.
fn dominant_expert_quant(header: &GgufHeader) -> String {
    let mut bytes: BTreeMap<&'static str, u64> = BTreeMap::new();
    for t in &header.tensors {
        // The same predicate `moearc-model` partitions the weights with.
        if !t.name.ends_with("_exps.weight") {
            continue;
        }
        if let Some(q) = quant::lookup(t.type_id) {
            *bytes.entry(q.name).or_default() += t.nbytes().unwrap_or(0);
        }
    }
    bytes
        .into_iter()
        .max_by_key(|(_, b)| *b)
        .map_or_else(|| "unknown".to_string(), |(name, _)| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(s: &str) -> Option<OsString> {
        Some(OsString::from(s))
    }

    #[test]
    fn an_explicit_directory_beats_everything_else() {
        let d = resolve_dir(Some(PathBuf::from("/models")), os("/env"), os("/xdg"), os("/home"));
        assert_eq!(d, PathBuf::from("/models"));
    }

    #[test]
    fn the_environment_is_consulted_before_the_default() {
        assert_eq!(resolve_dir(None, os("/env"), os("/xdg"), os("/home")), PathBuf::from("/env"));
    }

    #[test]
    fn an_empty_variable_is_not_a_setting() {
        // `MOEARC_MODELS=` in a stale shell profile should fall through, not point the scan at
        // the current directory.
        assert_eq!(
            resolve_dir(None, os(""), os("/xdg"), os("/home")),
            PathBuf::from("/xdg/moearc/models")
        );
    }

    #[test]
    fn the_default_lives_under_the_cache_directory() {
        assert_eq!(
            resolve_dir(None, None, None, os("/home/u")),
            PathBuf::from("/home/u/.cache/moearc/models")
        );
    }

    #[test]
    fn a_confirmed_quantisation_tag_comes_off_the_handle() {
        assert_eq!(
            without_quant_tag("qwen3-30b-a3b-q4_k_m", "q4_K").as_deref(),
            Some("qwen3-30b-a3b")
        );
        assert_eq!(
            without_quant_tag("gpt-oss-120b-mxfp4", "mxfp4").as_deref(),
            Some("gpt-oss-120b")
        );
    }

    #[test]
    fn a_tag_that_is_not_the_measured_type_stays_put() {
        // The filename says Q8_0 and the tensors say q4_K. Trimming would hide the
        // contradiction; keeping it puts both on screen, in adjacent columns.
        assert_eq!(without_quant_tag("some-model-q8_0", "q4_K"), None);
        // `UD` is Unsloth's dynamic-quant marker, not a type name.
        assert_eq!(without_quant_tag("qwen3.6-35b-a3b-ud", "q4_K"), None);
        assert_eq!(without_quant_tag("nodashes", "q4_K"), None);
    }

    #[test]
    fn two_quantisations_of_one_model_keep_their_full_handles() {
        let mut cards = vec![
            crate::source::testing::card("model-q4_k_m", "q4_K"),
            crate::source::testing::card("model-q8_0", "q8_0"),
            crate::source::testing::card("other-mxfp4", "mxfp4"),
        ];
        shorten_handles(&mut cards);
        // Both would shorten to `model`, so neither does — the tag is the only thing that
        // tells them apart.
        assert_eq!(cards[0].id, "model-q4_k_m");
        assert_eq!(cards[1].id, "model-q8_0");
        assert_eq!(cards[2].id, "other", "an unambiguous one still shortens");
    }

    #[test]
    fn a_directory_that_is_not_there_is_an_empty_catalogue_not_a_failure() {
        let c = LocalCatalog::new(PathBuf::from("/nonexistent-moearc-models"));
        assert!(c.installed().unwrap().is_empty());
        // Absence is the first-run state, not a fault: nothing to report.
        assert!(c.skipped().is_empty());
        assert!(c.resolve("anything").is_err());
    }
}
