//! What a tuning profile is, and where every number in one came from.
//!
//! # The rule this file exists to enforce
//!
//! 🔴 **A setting with a measurement behind it must never look like one without.** This
//! project has already reported an invented constant to its owner as a result, and the fix is
//! not care — it is a type. Every value that reaches a screen is a [`Setting<T>`], which
//! cannot be constructed without stating its [`Origin`], and every renderer prints that origin
//! beside the value.
//!
//! # The division of labour with the benchmark half
//!
//! The file on disk (`bench/tuning-profiles.json`) is written by the benchmark harness and
//! contains **only measurements**. It carries no origin field, and there is deliberately no
//! way for it to declare one: anything in that file was measured, by definition, and anything
//! absent from it is [`Origin::Untuned`]. Downgrades — extrapolated, derived — are decided
//! *here*, at resolution time, from what actually matched. A producer that could label its own
//! output "measured" would be a producer that could lie in one field.

use serde::{Deserialize, Serialize};

/// The version of the on-disk file this build understands.
///
/// A file declaring a newer schema is refused rather than parsed leniently. Silently ignoring
/// fields we do not understand is how a tool ends up running a configuration nobody wrote.
pub const SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------------------

/// Where a value came from. The whole point of this module.
///
/// Ordered weakest-first in [`Self::rank`] so a profile assembled from several sources can be
/// summarised by its weakest field without anyone having to remember the ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Benchmarked on this hardware, with this model, under `bench/PROTOCOL.md`.
    Measured,
    /// Carried across from a profile measured on *different* hardware or a *different* model,
    /// with the differences reasoned about. A starting point, not a measurement.
    Extrapolated,
    /// Computed from first principles by `moearc_engine::memory` — real arithmetic over the
    /// model's geometry and this card's free VRAM, and nothing was ever run.
    Derived,
    /// Not tuned at all. llama.cpp's own default stands, and we make no claim about it.
    Untuned,
}

impl Origin {
    /// Lower is stronger.
    pub fn rank(self) -> u8 {
        match self {
            Self::Measured => 0,
            Self::Extrapolated => 1,
            Self::Derived => 2,
            Self::Untuned => 3,
        }
    }

    /// The weaker of two origins. A profile is only as good as its worst field.
    pub fn weakest(self, other: Self) -> Self {
        if other.rank() > self.rank() { other } else { self }
    }

    /// The word. Spelled out, never abbreviated: the distinction this carries is the one the
    /// user most needs to read, and an abbreviation is a distinction they have to decode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Extrapolated => "extrapolated",
            Self::Derived => "derived",
            Self::Untuned => "untuned",
        }
    }

    /// One character, for a table cell that has no room for a word.
    ///
    /// Distinguished by *shape* as well as colour, so the badge survives a pipe, a
    /// monochrome terminal and a colour-blind reader — the same rule the model list already
    /// follows with `✓ ~ ·`.
    pub fn glyph(self) -> &'static str {
        match self {
            Self::Measured => "◆",
            Self::Extrapolated => "◇",
            Self::Derived => "·",
            Self::Untuned => " ",
        }
    }

    /// The sentence that goes under a table using [`Self::glyph`].
    pub fn legend() -> &'static str {
        "badges:  ◆ measured on this card   ◇ extrapolated from a nearby measurement   \
         · derived from the model's geometry, never run"
    }

    pub fn is_measured(self) -> bool {
        self == Self::Measured
    }
}

/// A value and its provenance, inseparably.
///
/// 🔴 There is no `Deref` and no `From<T>`. Both would let a bare number flow into a place
/// that renders it, and the entire mechanism is that it cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Setting<T> {
    pub value: T,
    pub origin: Origin,
}

impl<T> Setting<T> {
    pub fn measured(value: T) -> Self {
        Self { value, origin: Origin::Measured }
    }

    pub fn extrapolated(value: T) -> Self {
        Self { value, origin: Origin::Extrapolated }
    }

    pub fn derived(value: T) -> Self {
        Self { value, origin: Origin::Derived }
    }
}

// ---------------------------------------------------------------------------------------
// The on-disk file
// ---------------------------------------------------------------------------------------

/// `bench/tuning-profiles.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileFile {
    pub schema: u32,
    /// When the file itself was written. Distinct from a profile's own `measured_at`, which
    /// is when *that measurement* was taken.
    #[serde(default)]
    pub generated_at: Option<String>,
    #[serde(default)]
    pub profiles: Vec<TuningProfile>,
}

/// One measured (hardware, model) pairing and the settings that won on it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TuningProfile {
    /// A stable handle for this row, so a report can cite it.
    pub id: String,
    pub hardware: Hardware,
    pub model: ModelIdentity,
    #[serde(default)]
    pub engine: Engine,
    pub settings: Settings,
    /// What this configuration scored, with its error bar.
    ///
    /// `None` is allowed and means the settings were established without a headline number —
    /// a `-ncmoe` floor found by bisecting until llama.cpp stopped crashing is a real
    /// measurement with no tok/s attached. Such a profile still tunes; it just cannot be a
    /// baseline to climb from, and [`Self::baseline`] says so by returning `None`.
    #[serde(default)]
    pub score: Option<Score>,
    /// The expert-coverage curve measured on **this model's own routing trace**.
    ///
    /// 🔴 Optional, and per-model rather than global, because `bench/PROTOCOL.md` §9 records
    /// what happens otherwise: a curve taken from Qwen3-30B (8 of 128 experts active) was
    /// applied to gpt-oss (4 of 128) and called conservative when it was optimistic. A curve
    /// is a property of one model's routing and is never carried to another.
    #[serde(default)]
    pub coverage: Option<crate::tuning::coverage::Curve>,
    /// Date the measurement was taken, `YYYY-MM-DD`.
    pub measured_at: String,
    /// The protocol it was taken under. Recorded so a stranger can audit the number.
    #[serde(default)]
    pub protocol: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

impl TuningProfile {
    /// The score, if it is one a candidate can be compared against.
    ///
    /// A score whose own error bar disqualifies it under `PROTOCOL` §5 is **kept** — a
    /// retraction stays in the tree with its evidence — but it is not handed out as a
    /// baseline, because comparing against it would produce a verdict nobody should act on.
    pub fn baseline(&self) -> Option<&Score> {
        self.score.as_ref().filter(|s| s.is_trustworthy())
    }
}

/// The machine a profile was measured on.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hardware {
    /// The device string, as the driver reports it.
    pub gpu: String,
    /// A normalised key for matching: `arc-b580`, `arc-pro-b60`.
    ///
    /// Separate from `gpu` because driver strings are not stable across driver versions and
    /// a profile must not stop matching because someone updated a package.
    pub gpu_key: String,
    pub vram_bytes: u64,
    #[serde(default)]
    pub driver: Option<String>,
    /// The host CPU. 🔴 Load-bearing for `threads` and for nothing else — a thread count
    /// measured on a 20-core box is not a measurement about an 8-core one, and carrying it
    /// across without saying so is the same class of error as carrying a coverage curve
    /// between models.
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub physical_cores: Option<u32>,
    #[serde(default)]
    pub ram_bytes: Option<u64>,
}

/// Enough of a model to know whether a profile is about it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelIdentity {
    pub id: String,
    /// The ggml type holding most of the expert weights. Part of the identity: the same
    /// weights at two quantisations have different footprints and different floors.
    pub quant: String,
    #[serde(default)]
    pub file_bytes: Option<u64>,
    pub moe_blocks: u32,
    pub experts_per_block: u32,
    pub active_experts_per_block: u32,
    /// `moe_blocks × experts_per_block`. Carried rather than computed so a file with an
    /// inconsistent geometry is caught at load rather than believed.
    pub expert_slots_total: u32,
    #[serde(default)]
    pub parameters: Option<u64>,
}

impl ModelIdentity {
    /// Whether the three numbers agree with each other.
    pub fn is_consistent(&self) -> bool {
        self.moe_blocks > 0
            && self.experts_per_block > 0
            && self.active_experts_per_block > 0
            && self.active_experts_per_block <= self.experts_per_block
            && self.moe_blocks as u64 * self.experts_per_block as u64
                == self.expert_slots_total as u64
    }

    /// Whether two models are the same *architecture at the same scale* — the condition
    /// under which one's settings are a defensible starting point for the other.
    ///
    /// 🔴 Structural, not nominal. `Qwen3-30B-A3B` and `Qwen3-Coder-30B-A3B` share a name
    /// prefix *and* a geometry, and it is the geometry that makes the transfer defensible;
    /// a name prefix on its own would happily match `Qwen3-4B` to `Qwen3-235B`. Names are
    /// used only to rank among several structural matches, never to qualify one.
    pub fn is_same_family(&self, other: &Self) -> bool {
        self.moe_blocks == other.moe_blocks
            && self.experts_per_block == other.experts_per_block
            && self.active_experts_per_block == other.active_experts_per_block
            && self.quant == other.quant
            && match (self.parameters, other.parameters) {
                (Some(a), Some(b)) if a > 0 && b > 0 => {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    // Within 5%: same architecture at the same scale. The threshold is a
                    // stated judgement, not a measurement, and it is only ever used to
                    // decide whether to label something *extrapolated*.
                    (hi - lo) * 20 <= hi
                }
                // No parameter count on one side: the geometry already matched on four
                // fields, which is a stronger test than a parameter count would add.
                _ => true,
            }
    }

    /// How alike two handles read, `0..=1`. Ranking only.
    pub fn name_affinity(&self, other: &Self) -> f64 {
        let (a, b) = (self.id.to_ascii_lowercase(), other.id.to_ascii_lowercase());
        let shared = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
        let longest = a.chars().count().max(b.chars().count()).max(1);
        shared as f64 / longest as f64
    }
}

/// Which engine, at which build.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Engine {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

impl Default for Engine {
    fn default() -> Self {
        Self { name: "llama.cpp".to_string(), version: None, backend: None }
    }
}

/// The settings themselves. Every field optional; absent means [`Origin::Untuned`].
///
/// 🔴 `Option` is the whole encoding of "we did not tune this". A default value here would be
/// indistinguishable from a measured one the moment it reached a screen, which is precisely
/// the failure this module exists to prevent.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Settings {
    /// `-t`. 🔴 The single most valuable field in the file: `llama-bench` defaults to **4**
    /// threads, and on this project's 20-core box that default cost **2.1×** — 13.6 tok/s
    /// against 28.5 on a 59 GiB model. Nobody guesses their way to that.
    #[serde(default)]
    pub threads: Option<u32>,
    /// `-ncmoe` / `--n-cpu-moe`. 🔴 Has a model-specific floor below which llama.cpp aborts
    /// with `OUT_OF_DEVICE_MEMORY`, documented nowhere.
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    /// `-ngl`.
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,
    /// `-c`.
    #[serde(default)]
    pub ctx_size: Option<u32>,
    /// `-b`.
    #[serde(default)]
    pub batch_size: Option<u32>,
    /// `-ub`.
    #[serde(default)]
    pub ubatch_size: Option<u32>,
    /// `--cache-type-k` / `--cache-type-v`, one value for both.
    #[serde(default)]
    pub kv_cache_type: Option<String>,
    /// `-fa`.
    #[serde(default)]
    pub flash_attn: Option<bool>,
    /// Anything else that was in force. Recorded verbatim and reproduced verbatim; never
    /// parsed, because a flag we do not understand is still a flag that was measured.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

// ---------------------------------------------------------------------------------------
// Scores and the noise floor
// ---------------------------------------------------------------------------------------

/// The relative standard deviation at which `bench/PROTOCOL.md` §5 stops calling a run a
/// measurement.
///
/// 🔴 **A stated threshold, not a measured one**, and it is quoted wherever it is applied so
/// nobody mistakes it for a property of the machine. It comes straight from the rule: *"a run
/// whose stddev is 20–30% of its mean is not a measurement."* 20% is the strict end of that
/// range.
pub const UNTRUSTWORTHY_RSD: f64 = 0.20;

/// What a configuration scored, with the error bar that decides whether the number means
/// anything.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Score {
    /// What was measured. Compared as a string: two different metrics are not comparable, and
    /// this is how that is caught rather than assumed.
    #[serde(default = "default_metric")]
    pub metric: String,
    /// Prompt depth the measurement was taken at. Part of the identity of the number:
    /// throughput on this engine falls 5.9× with depth, so a figure at depth 0 and a figure at
    /// depth 8192 are answers to different questions.
    #[serde(default)]
    pub depth_tokens: u32,
    pub mean: f64,
    pub stddev: f64,
    /// **Independent invocations**, not iterations inside one. PROTOCOL §5: process-level
    /// variance is the variance that bit this project, and `-r` inside one process cannot
    /// see it.
    pub runs: u32,
}

fn default_metric() -> String {
    "decode_tokens_per_second".to_string()
}

impl Score {
    /// Relative standard deviation — the error bar as a fraction of the value.
    pub fn rsd(&self) -> f64 {
        if self.mean.abs() < f64::EPSILON { f64::INFINITY } else { self.stddev.abs() / self.mean }
    }

    /// The standard error of the mean.
    pub fn standard_error(&self) -> f64 {
        if self.runs == 0 { f64::INFINITY } else { self.stddev.abs() / (self.runs as f64).sqrt() }
    }

    /// Whether this is a measurement at all, by PROTOCOL §5's rule.
    pub fn is_trustworthy(&self) -> bool {
        self.runs >= 2
            && self.mean > 0.0
            && self.stddev.is_finite()
            && self.rsd() < UNTRUSTWORTHY_RSD
    }

    /// `"28.5 ± 0.2 tok/s over 5 runs"`.
    pub fn describe(&self) -> String {
        format!(
            "{:.2} ± {:.2} {} over {} run{}{}",
            self.mean,
            self.stddev,
            unit_of(&self.metric),
            self.runs,
            if self.runs == 1 { "" } else { "s" },
            if self.depth_tokens > 0 {
                format!(" at depth {}", crate::format::count(self.depth_tokens as i64))
            } else {
                String::new()
            }
        )
    }
}

/// The unit a metric is quoted in. Unknown metrics keep their own name rather than being
/// given one — inventing a unit is how a proxy becomes a headline.
pub fn unit_of(metric: &str) -> &str {
    match metric {
        "decode_tokens_per_second" | "prompt_tokens_per_second" => "tok/s",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(id: &str, blocks: u32, per: u32, active: u32, params: u64) -> ModelIdentity {
        ModelIdentity {
            id: id.to_string(),
            quant: "q4_K".to_string(),
            file_bytes: None,
            moe_blocks: blocks,
            experts_per_block: per,
            active_experts_per_block: active,
            expert_slots_total: blocks * per,
            parameters: Some(params),
        }
    }

    #[test]
    fn a_weaker_origin_always_wins() {
        assert_eq!(Origin::Measured.weakest(Origin::Derived), Origin::Derived);
        assert_eq!(Origin::Derived.weakest(Origin::Measured), Origin::Derived);
        assert_eq!(Origin::Extrapolated.weakest(Origin::Untuned), Origin::Untuned);
        assert_eq!(Origin::Measured.weakest(Origin::Measured), Origin::Measured);
    }

    #[test]
    fn every_origin_prints_a_distinct_word_and_glyph() {
        let all = [Origin::Measured, Origin::Extrapolated, Origin::Derived, Origin::Untuned];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.label(), b.label());
                assert_ne!(a.glyph(), b.glyph());
            }
        }
    }

    #[test]
    fn the_family_test_is_structural_rather_than_nominal() {
        // The case the feature exists for.
        let base = identity("qwen3-30b-a3b", 48, 128, 8, 30_000_000_000);
        let coder = identity("qwen3-coder-30b-a3b", 48, 128, 8, 30_500_000_000);
        assert!(base.is_same_family(&coder));

        // Same name family, wildly different scale: refused.
        let big = identity("qwen3-235b-a22b", 94, 128, 8, 235_000_000_000);
        assert!(!base.is_same_family(&big));

        // Different routing width is a different cache, whatever the name says. This is the
        // PROTOCOL §9 failure encoded as a test.
        let four_routed = identity("qwen3-30b-a3b-x", 48, 128, 4, 30_000_000_000);
        assert!(!base.is_same_family(&four_routed));
    }

    #[test]
    fn an_inconsistent_geometry_is_detectable() {
        let mut m = identity("m", 36, 128, 4, 120_000_000_000);
        assert!(m.is_consistent());
        // The 36x bug: the per-block count where the slot count belongs.
        m.expert_slots_total = 128;
        assert!(!m.is_consistent());
    }

    #[test]
    fn protocol_five_decides_what_counts_as_a_measurement() {
        // The measured pair from bench/PROTOCOL.md §4, both kept in the record.
        let good =
            Score { metric: default_metric(), depth_tokens: 0, mean: 28.5, stddev: 0.2, runs: 5 };
        let bad =
            Score { metric: default_metric(), depth_tokens: 0, mean: 17.59, stddev: 5.56, runs: 2 };
        assert!(good.is_trustworthy());
        assert!(!bad.is_trustworthy(), "31.6% of its mean is not a measurement");
        assert!(bad.rsd() > UNTRUSTWORTHY_RSD);
    }

    #[test]
    fn a_single_run_has_no_error_bar_and_is_not_a_measurement() {
        let one =
            Score { metric: default_metric(), depth_tokens: 0, mean: 114.0, stddev: 0.0, runs: 1 };
        assert!(!one.is_trustworthy(), "a stddev of zero over one run is an absence, not a zero");
    }

    #[test]
    fn a_profile_with_an_untrustworthy_score_is_kept_but_is_not_a_baseline() {
        let p = TuningProfile {
            id: "x".into(),
            hardware: Hardware {
                gpu: "Intel Arc B580 Graphics".into(),
                gpu_key: "arc-b580".into(),
                vram_bytes: 12_884_901_888,
                driver: None,
                cpu: None,
                physical_cores: None,
                ram_bytes: None,
            },
            model: identity("m", 36, 128, 4, 120_000_000_000),
            engine: Engine::default(),
            settings: Settings::default(),
            score: Some(Score {
                metric: default_metric(),
                depth_tokens: 0,
                mean: 17.59,
                stddev: 5.56,
                runs: 2,
            }),
            coverage: None,
            measured_at: "2026-09-06".into(),
            protocol: None,
            notes: None,
        };
        assert!(p.score.is_some(), "the discard stays in the record with its error bars");
        assert!(p.baseline().is_none(), "and it is never climbed from");
    }
}
