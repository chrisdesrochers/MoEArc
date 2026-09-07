//! Finding and loading `bench/tuning-profiles.json`.
//!
//! 🔴 **This half does not own that file.** It is produced by the benchmark harness from real
//! runs on real hardware; everything here does is find it, check it is internally consistent,
//! and hand it to the resolver. The contract between the two halves is `docs/tuning.md`.
//!
//! **Absence is the normal case and must be silent.** A machine that has never seen the file —
//! a fresh install, a `cargo test` in CI, a user who cloned without the bench data — resolves
//! every model to [`Origin::Derived`](super::Origin::Derived) and says so on screen. It does
//! not warn, because there is nothing wrong: derived settings are computed from the model's own
//! geometry and this card's free VRAM, which is a real answer.
//!
//! **A malformed file is a different thing entirely** and is never silent. Reading half a file
//! and tuning from what parsed is how a user ends up running a configuration nobody wrote.

use std::path::{Path, PathBuf};

use super::schema::{ProfileFile, SCHEMA_VERSION, TuningProfile};

/// Overrides the search below. A path to a file, or to a directory containing
/// `tuning-profiles.json`.
pub const PROFILES_ENV: &str = "MOEARC_PROFILES";

/// The file's name wherever it is looked for.
pub const PROFILES_FILE: &str = "tuning-profiles.json";

/// Profiles compiled into this binary, so a shipped `moearc` carries its measurements without
/// needing the repository beside it. A file on disk always wins over this, so a user can
/// replace our measurements with their own.
///
/// # 🔴 Why this is not `include_str!("../../../../bench/tuning-profiles.json")`
///
/// That was the plan, and it does not work: **`bench/tuning-profiles.json` is not written to
/// the schema `docs/tuning.md` specifies.** It is a report — one top-level `hardware` block, a
/// `models` array keyed by GGUF filename, `recommended` settings, `ncmoe_floor_by_ctx`,
/// `knob_value` — where this half expects `schema`, `profiles[]`, and a `model.id` matching
/// what `moearc ls` prints. `serde` refuses it at the first field, so `include_str!`ing it
/// would compile and then load nothing.
///
/// The file beside this one is a transcription of the same measurements into the documented
/// shape. Every number in it comes from `bench/tuning-profiles.md` (settings, scores, notes) or
/// from `moearc ls --json` run against the same GGUF files (ids, quantisations, geometry,
/// parameter and byte counts) — nothing was measured, inferred or rounded here. It is a
/// **transcription and it will drift**; the fix is for the producer to emit the contract shape,
/// after which this becomes the one-line `include_str!` it was meant to be.
///
/// ⚠️ Until then, note that `bench/tuning-profiles.json` still sits on the search path below and
/// is found first in a repository checkout — where it produces a parse error and suppresses
/// this set, exactly as `from_json` promises for a file it cannot read. That is a producer bug
/// and is deliberately not papered over here: `validate`'s own comment is the rule, and it
/// applies to files as well as to profiles.
pub(crate) const BUILTIN: Option<&str> = Some(include_str!("builtin-profiles.json"));

/// Every profile this machine can see, and where they came from.
#[derive(Debug, Clone, Default)]
pub struct Store {
    profiles: Vec<TuningProfile>,
    /// The file the profiles were read from. `None` for the built-in set or an empty store.
    source: Option<PathBuf>,
    /// A file that was found and could not be used. 🔴 Held rather than logged: this has to
    /// reach the screen, and a `tracing` line in a TUI reaches nobody.
    load_error: Option<String>,
    /// Profiles that parsed but failed validation, with the reason. Named individually so a
    /// producer can fix them.
    rejected: Vec<String>,
}

impl Store {
    /// An empty store. Every lookup falls through to derivation.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Search the usual places and load the first file that exists.
    pub fn load() -> Self {
        for path in search_paths() {
            if path.is_file() {
                return Self::from_path(&path);
            }
        }
        match BUILTIN {
            Some(text) => Self::from_json(text, None),
            None => Self::empty(),
        }
    }

    pub fn from_path(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_json(&text, Some(path.to_path_buf())),
            Err(e) => Self {
                load_error: Some(format!("{} could not be read: {e}", path.display())),
                ..Self::empty()
            },
        }
    }

    /// Parse and validate. An inherent method rather than a `FromStr` impl because it
    /// never fails: a bad file produces an *empty store carrying an explanation*, which is
    /// what the screens need. A `Result` here would put the explanation in an error the TUI
    /// would have to invent a place for.
    pub fn from_json(text: &str, source: Option<PathBuf>) -> Self {
        let file: ProfileFile = match serde_json::from_str(text) {
            Ok(f) => f,
            Err(e) => {
                return Self {
                    load_error: Some(format!(
                        "{} is not a valid tuning-profiles file: {e}",
                        describe(source.as_deref())
                    )),
                    source,
                    ..Self::empty()
                };
            }
        };
        if file.schema != SCHEMA_VERSION {
            return Self {
                load_error: Some(format!(
                    "{} declares schema {} and this build understands {SCHEMA_VERSION}. \
                     Nothing was loaded from it — a file from a newer producer may mean \
                     something different by the same field names.",
                    describe(source.as_deref()),
                    file.schema
                )),
                source,
                ..Self::empty()
            };
        }

        let mut profiles = Vec::new();
        let mut rejected = Vec::new();
        for p in file.profiles {
            match validate(&p) {
                Ok(()) => profiles.push(p),
                Err(why) => rejected.push(format!("{}: {why}", p.id)),
            }
        }
        Self { profiles, source, load_error: None, rejected }
    }

    pub fn profiles(&self) -> &[TuningProfile] {
        &self.profiles
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn source(&self) -> Option<&Path> {
        self.source.as_deref()
    }

    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    pub fn rejected(&self) -> &[String] {
        &self.rejected
    }

    /// One line for a footer: where the numbers came from, or that there are none.
    pub fn provenance(&self) -> String {
        if let Some(e) = self.load_error() {
            return e.to_string();
        }
        match (self.source(), self.len()) {
            (_, 0) => "no measured tuning profiles on this machine — every setting below is \
                       derived from the model's geometry"
                .to_string(),
            (Some(p), n) => {
                format!("{n} measured tuning profile{} from {}", plural(n), p.display())
            }
            (None, n) => format!("{n} measured tuning profile{}, built in", plural(n)),
        }
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Where to look, in order. Earlier wins, so a user's own file beats ours.
fn search_paths() -> Vec<PathBuf> {
    search_paths_with(std::env::var_os(PROFILES_ENV).map(PathBuf::from))
}

/// The search, with the environment passed in.
///
/// Split out so the precedence rule can be asserted without writing to the process
/// environment — which is `unsafe` under edition 2024 and is a data race against every other
/// test thread besides.
fn search_paths_with(override_path: Option<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = override_path {
        // A directory is accepted as well as a file: pointing an environment variable at a
        // checkout's `bench/` is what a person actually does.
        out.push(if p.is_dir() { p.join(PROFILES_FILE) } else { p });
    }
    // A repository checkout, run from its root. This is how the benchmark half and this half
    // meet during development, before anything is packaged.
    out.push(PathBuf::from("bench").join(PROFILES_FILE));
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push(dir.join(PROFILES_FILE));
        // The installed layout: `<prefix>/bin/moearc` beside `<prefix>/share/moearc/`.
        if let Some(prefix) = dir.parent() {
            out.push(prefix.join("share").join("moearc").join(PROFILES_FILE));
        }
    }
    if let Some(data) = std::env::var_os("XDG_DATA_HOME") {
        out.push(PathBuf::from(data).join("moearc").join(PROFILES_FILE));
    } else if let Some(home) = std::env::var_os("HOME") {
        out.push(PathBuf::from(home).join(".local/share/moearc").join(PROFILES_FILE));
    }
    out
}

fn describe(path: Option<&Path>) -> String {
    path.map_or_else(|| "the built-in profile set".to_string(), |p| p.display().to_string())
}

/// Reject a profile whose own numbers contradict each other.
///
/// 🔴 Rejected, not repaired. A geometry that does not multiply out is the 36× bug — the one
/// that makes every residency figure wrong by the block count while still looking entirely
/// plausible — and silently "fixing" it would hide a producer bug behind a consumer patch.
fn validate(p: &TuningProfile) -> Result<(), String> {
    if p.id.trim().is_empty() {
        return Err("no id".to_string());
    }
    if p.hardware.gpu_key.trim().is_empty() {
        return Err("no hardware.gpu_key to match on".to_string());
    }
    if p.hardware.vram_bytes == 0 {
        return Err("hardware.vram_bytes is zero".to_string());
    }
    if !p.model.is_consistent() {
        return Err(format!(
            "model geometry does not multiply out: {} blocks x {} experts is not {} slots",
            p.model.moe_blocks, p.model.experts_per_block, p.model.expert_slots_total
        ));
    }
    if let Some(n) = p.settings.n_cpu_moe
        && n > p.model.moe_blocks
    {
        return Err(format!(
            "settings.n_cpu_moe is {n}, past the model's {} MoE blocks",
            p.model.moe_blocks
        ));
    }
    if let Some(c) = &p.coverage
        && c.points.is_empty()
    {
        return Err("coverage is present but has no points".to_string());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod fixture {
    /// A two-profile file in the exact shape `docs/tuning.md` asks the benchmark harness for.
    ///
    /// It is the contract, written down as data. If the producer and this consumer ever
    /// disagree, this string is where the disagreement shows up as a failing test rather than
    /// as an empty screen.
    pub const FILE: &str = r#"{
  "schema": 1,
  "generated_at": "2026-09-06T21:00:00Z",
  "profiles": [
    {
      "id": "arc-b580/gpt-oss-120b/mxfp4",
      "hardware": {
        "gpu": "Intel Arc B580 Graphics",
        "gpu_key": "arc-b580",
        "vram_bytes": 12884901888,
        "driver": "xe / L0 build 37020",
        "cpu": "Intel Core Ultra 9 285K",
        "physical_cores": 20,
        "ram_bytes": 98257694720
      },
      "model": {
        "id": "gpt-oss-120b",
        "quant": "mxfp4",
        "file_bytes": 63377342464,
        "moe_blocks": 36,
        "experts_per_block": 128,
        "active_experts_per_block": 4,
        "expert_slots_total": 4608,
        "parameters": 116800000000
      },
      "engine": { "name": "llama.cpp", "version": "b7xxx", "backend": "SYCL" },
      "settings": {
        "threads": 16,
        "n_cpu_moe": 31,
        "n_gpu_layers": 99,
        "ctx_size": 4096,
        "kv_cache_type": "f16",
        "flash_attn": true,
        "extra_args": []
      },
      "score": {
        "metric": "decode_tokens_per_second",
        "depth_tokens": 0,
        "mean": 28.5,
        "stddev": 0.2,
        "runs": 5
      },
      "coverage": {
        "model": "gpt-oss-120b",
        "trace": "bench/traces/",
        "points": [
          { "bank_resident": 0.05, "prose": 0.508, "code": 0.312 },
          { "bank_resident": 0.13, "prose": 0.788, "code": 0.533 },
          { "bank_resident": 0.35, "prose": 0.978, "code": 0.842 },
          { "bank_resident": 0.50, "prose": 0.998, "code": 0.939 }
        ]
      },
      "measured_at": "2026-09-06",
      "protocol": "bench/PROTOCOL.md",
      "notes": "thread count pinned and read back from llama-bench -o csv"
    },
    {
      "id": "arc-b580/qwen3-30b-a3b/q4_K",
      "hardware": {
        "gpu": "Intel Arc B580 Graphics",
        "gpu_key": "arc-b580",
        "vram_bytes": 12884901888,
        "cpu": "Intel Core Ultra 9 285K",
        "physical_cores": 20
      },
      "model": {
        "id": "qwen3-30b-a3b",
        "quant": "q4_K",
        "moe_blocks": 48,
        "experts_per_block": 128,
        "active_experts_per_block": 8,
        "expert_slots_total": 6144,
        "parameters": 30500000000
      },
      "settings": { "threads": 16, "n_cpu_moe": 22, "n_gpu_layers": 99 },
      "measured_at": "2026-09-06",
      "protocol": "bench/PROTOCOL.md"
    }
  ]
}"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_fixture_loads_whole() {
        let s = Store::from_json(fixture::FILE, None);
        assert!(s.load_error().is_none(), "{:?}", s.load_error());
        assert!(s.rejected().is_empty(), "{:?}", s.rejected());
        assert_eq!(s.len(), 2);
        let p = &s.profiles()[0];
        assert_eq!(p.settings.n_cpu_moe, Some(31));
        assert_eq!(p.settings.threads, Some(16));
        assert_eq!(p.baseline().unwrap().mean, 28.5);
        assert!(p.coverage.is_some());
        // The second profile has no score. That is legal and it still tunes.
        assert!(s.profiles()[1].score.is_none());
        assert_eq!(s.profiles()[1].settings.n_cpu_moe, Some(22));
    }

    #[test]
    fn the_built_in_set_loads_whole_and_every_profile_in_it_is_usable() {
        // 🔴 A built-in file that fails `validate` is worse than no built-in file: it ships a
        // silent hole in the tuning that only shows up as a `derived` badge on a model the
        // project claims to have measured. Asserted here so a bad transcription cannot be
        // committed.
        let text = BUILTIN.expect("the built-in set is compiled in");
        let s = Store::from_json(text, None);
        assert!(s.load_error().is_none(), "{:?}", s.load_error());
        assert!(s.rejected().is_empty(), "{:?}", s.rejected());
        assert_eq!(s.len(), 7, "the seven models bench/tuning-profiles.md measured");
        assert!(s.provenance().contains("built in"), "{}", s.provenance());

        for p in s.profiles() {
            assert!(p.settings.threads.is_some(), "{}: -t is the whole point", p.id);
            assert!(p.settings.n_cpu_moe.is_some(), "{}: -ncmoe is the other half", p.id);
            assert_eq!(p.hardware.physical_cores, Some(20), "{}: -t must be transferable", p.id);
            assert_eq!(p.settings.kv_cache_type.as_deref(), Some("f16"), "{}", p.id);
            assert_eq!(p.settings.flash_attn, Some(true), "{}", p.id);
            // A score is optional, but one that is present has to survive PROTOCOL 5 --
            // otherwise it is in the file as a number nobody may climb from, and that is a
            // transcription mistake rather than a retraction.
            if let Some(score) = &p.score {
                assert!(score.is_trustworthy(), "{}: {}", p.id, score.describe());
            }
            // 🔴 `-ngl` and `-c` are absent on purpose: bench left `-ngl` at -1 and never
            // tuned it, and llama-bench's context is not the context a server runs at.
            // Writing a default into a field nobody measured is the one thing the file
            // contract forbids.
            assert!(p.settings.n_gpu_layers.is_none(), "{}: -ngl was never a measured lever", p.id);
            assert!(p.settings.ctx_size.is_none(), "{}: no context was measured", p.id);
        }
    }

    #[test]
    fn an_absent_file_is_an_empty_store_and_not_an_error() {
        let s = Store::from_path(Path::new("/nonexistent/moearc/tuning-profiles.json"));
        assert!(s.is_empty());
        // A path that was named and could not be read *is* worth reporting; a search that
        // simply found nothing is not, and `load()` never reaches this branch for one.
        assert!(s.load_error().is_some());
        let quiet = Store::empty();
        assert!(quiet.load_error().is_none());
        assert!(quiet.provenance().contains("derived from the model's geometry"));
    }

    #[test]
    fn a_malformed_file_loads_nothing_rather_than_half_of_itself() {
        let s = Store::from_json("{ not json", None);
        assert!(s.is_empty());
        assert!(s.load_error().unwrap().contains("not a valid"));
    }

    #[test]
    fn a_newer_schema_is_refused_outright() {
        let s = Store::from_json(r#"{"schema": 99, "profiles": []}"#, None);
        assert!(s.is_empty());
        let e = s.load_error().unwrap();
        assert!(e.contains("schema 99"), "{e}");
        assert!(e.contains("Nothing was loaded"), "{e}");
    }

    #[test]
    fn an_inconsistent_geometry_is_named_and_dropped_while_its_neighbours_survive() {
        let broken =
            fixture::FILE.replace("\"expert_slots_total\": 4608", "\"expert_slots_total\": 128");
        let s = Store::from_json(&broken, None);
        assert_eq!(s.len(), 1, "the good profile is still usable");
        assert_eq!(s.rejected().len(), 1);
        assert!(s.rejected()[0].contains("does not multiply out"), "{:?}", s.rejected());
    }

    #[test]
    fn an_impossible_ncmoe_is_rejected_rather_than_clamped() {
        let broken = fixture::FILE.replace("\"n_cpu_moe\": 31", "\"n_cpu_moe\": 99");
        let s = Store::from_json(&broken, None);
        assert_eq!(s.len(), 1);
        assert!(s.rejected()[0].contains("past the model's 36 MoE blocks"), "{:?}", s.rejected());
    }

    #[test]
    fn the_search_prefers_an_explicit_path_to_a_repository_checkout() {
        // Order only -- the paths need not exist. What is asserted is the precedence rule:
        // a user's own file must be reachable without deleting ours.
        let paths = search_paths_with(Some(PathBuf::from("/tmp/mine.json")));
        assert_eq!(paths[0], PathBuf::from("/tmp/mine.json"));
        assert!(paths.iter().any(|p| p.ends_with("bench/tuning-profiles.json")));
        assert!(!search_paths_with(None).is_empty(), "there is always somewhere to look");
    }
}
