//! Is this tweak a win?
//!
//! The optimisation loop the owner described — *"if we hit 114 and things are solid, we apply
//! some tweaks; if that gets it to 128, that's a win. If we hit 40, we back up."* — is only
//! sound if "a win" means *beat the noise*, not *beat the mean*. Two means are two samples
//! from two distributions, and comparing them by eye is how a project keeps changes that hurt.
//!
//! # The number that matters
//!
//! 🔴 The headline output here is not the verdict. It is
//! [`Comparison::minimum_detectable_pct`] — **the smallest improvement this pair of
//! measurements could possibly have resolved.** At the ±32% error bars some of this project's
//! runs carried, that figure is around 60%: a real 12% gain is *invisible*, and a loop run on
//! means alone would have accepted it, then accepted the next regression with equal
//! confidence. Printing it turns "we cannot tell" into a number instead of a shrug.
//!
//! # The statistics, stated plainly
//!
//! Welch's standard error over two independent samples,
//! `se = sqrt(sd_b²/n_b + sd_c²/n_c)`, and a difference counts only if it exceeds
//! [`CONFIDENCE_K`] standard errors. That is a normal approximation and it is **optimistic at
//! the run counts this project uses** — with n = 3 a t-distribution would demand roughly twice
//! the margin. It is documented rather than corrected because the correction needs a t-table
//! this crate has no business carrying, and because being explicit about erring toward
//! "indistinguishable" is the safe direction: the failure this guards against is calling a
//! non-result a win.

use serde::Serialize;

use super::schema::{Score, UNTRUSTWORTHY_RSD, unit_of};

/// Standard errors a difference must clear to count.
///
/// 🔴 **A chosen constant, not a measured one.** 2.0 is the conventional ~95% two-sided
/// interval under a normal approximation. It is named, exported and printed with every
/// verdict so that moving it is a visible act.
pub const CONFIDENCE_K: f64 = 2.0;

/// What a comparison concluded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// The candidate is faster by more than the noise floor. Keep it.
    Win,
    /// The candidate is slower by more than the noise floor. Back up.
    Regression,
    /// The difference is inside the noise floor. 🔴 **Not "no change"** — it is *"this
    /// experiment cannot tell"*, and the two are different instructions: the first says stop
    /// looking, the second says take more runs.
    Indistinguishable,
    /// One side is not a measurement, so no verdict is available at any margin.
    Untrustworthy {
        /// `"baseline"` or `"candidate"` — or both, named in one string.
        which: String,
        reason: String,
    },
    /// The two numbers are answers to different questions.
    NotComparable { reason: String },
}

impl Verdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Win => "win",
            Self::Regression => "regression",
            Self::Indistinguishable => "indistinguishable",
            Self::Untrustworthy { .. } => "untrustworthy",
            Self::NotComparable { .. } => "not comparable",
        }
    }

    /// Whether this verdict justifies keeping the candidate.
    pub fn is_win(&self) -> bool {
        matches!(self, Self::Win)
    }

    /// Whether a person should act on this at all.
    pub fn is_actionable(&self) -> bool {
        matches!(self, Self::Win | Self::Regression)
    }
}

/// A candidate measured against a baseline, with the noise floor made explicit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub baseline: Score,
    pub candidate: Score,
    /// `candidate.mean - baseline.mean`, in the metric's own unit.
    pub delta: f64,
    /// The same, as a percentage of the baseline.
    pub delta_pct: f64,
    /// [`CONFIDENCE_K`] × the combined standard error, in the metric's unit. A difference
    /// smaller than this is not visible to these two measurements.
    pub noise_floor: f64,
    /// The noise floor as a percentage of the baseline mean — the minimum detectable effect.
    pub noise_floor_pct: f64,
    /// How many standard errors apart the two means are. Reported so a near-miss reads as a
    /// near-miss rather than as a flat "no".
    pub sigmas: f64,
    pub verdict: Verdict,
    pub confidence_k: f64,
}

impl Comparison {
    /// 🔴 The headline. The smallest improvement this experiment could have seen.
    pub fn minimum_detectable_pct(&self) -> f64 {
        self.noise_floor_pct
    }

    /// One sentence, for a terminal.
    pub fn summary(&self) -> String {
        let unit = unit_of(&self.baseline.metric);
        match &self.verdict {
            Verdict::Untrustworthy { which, reason } => {
                format!("no verdict — the {which} is not a measurement: {reason}")
            }
            Verdict::NotComparable { reason } => format!("no verdict — {reason}"),
            Verdict::Win => format!(
                "win: {:+.2} {unit} ({:+.1}%), clear of a ±{:.1}% noise floor at {:.1}σ",
                self.delta, self.delta_pct, self.noise_floor_pct, self.sigmas
            ),
            Verdict::Regression => format!(
                "regression: {:+.2} {unit} ({:+.1}%), clear of a ±{:.1}% noise floor at {:.1}σ \
                 — back up",
                self.delta, self.delta_pct, self.noise_floor_pct, self.sigmas
            ),
            Verdict::Indistinguishable => format!(
                "indistinguishable: {:+.2} {unit} ({:+.1}%) is inside a ±{:.1}% noise floor. \
                 This experiment could not have seen an improvement smaller than {:.1}% — take \
                 more runs before deciding",
                self.delta, self.delta_pct, self.noise_floor_pct, self.noise_floor_pct
            ),
        }
    }
}

/// Compare a candidate against a baseline, honestly.
///
/// The order of the checks is the argument. Trust is decided **before** the difference is
/// looked at, because a difference computed from an untrustworthy sample is not a small
/// finding — it is not a finding.
pub fn compare(baseline: &Score, candidate: &Score) -> Comparison {
    let delta = candidate.mean - baseline.mean;
    let delta_pct =
        if baseline.mean.abs() < f64::EPSILON { 0.0 } else { delta / baseline.mean * 100.0 };
    let se = (baseline.standard_error().powi(2) + candidate.standard_error().powi(2)).sqrt();
    let noise_floor = CONFIDENCE_K * se;
    let noise_floor_pct = if baseline.mean.abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        noise_floor / baseline.mean * 100.0
    };
    let sigmas = if se.abs() < f64::EPSILON || !se.is_finite() {
        // Two zero-variance samples. Real only for a degenerate input, and calling that
        // infinitely significant would be the most confident possible wrong answer.
        0.0
    } else {
        delta.abs() / se
    };

    let verdict = verdict_for(baseline, candidate, delta, noise_floor);

    Comparison {
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        delta,
        delta_pct,
        noise_floor,
        noise_floor_pct,
        sigmas,
        verdict,
        confidence_k: CONFIDENCE_K,
    }
}

fn verdict_for(baseline: &Score, candidate: &Score, delta: f64, noise_floor: f64) -> Verdict {
    if baseline.metric != candidate.metric {
        return Verdict::NotComparable {
            reason: format!(
                "the baseline measures `{}` and the candidate measures `{}`",
                baseline.metric, candidate.metric
            ),
        };
    }
    if baseline.depth_tokens != candidate.depth_tokens {
        // Throughput on this engine falls 5.9x with prompt depth. Two depths are two
        // questions, and averaging them into a verdict would be the tidiest possible lie.
        return Verdict::NotComparable {
            reason: format!(
                "the baseline was taken at depth {} and the candidate at depth {}; throughput \
                 falls with depth, so these are different questions",
                crate::format::count(baseline.depth_tokens as i64),
                crate::format::count(candidate.depth_tokens as i64)
            ),
        };
    }

    let untrusted: Vec<&str> = [("baseline", baseline), ("candidate", candidate)]
        .into_iter()
        .filter(|(_, s)| !s.is_trustworthy())
        .map(|(n, _)| n)
        .collect();
    if !untrusted.is_empty() {
        let worst = if baseline.is_trustworthy() { candidate } else { baseline };
        return Verdict::Untrustworthy {
            which: untrusted.join(" and the "),
            reason: if worst.runs < 2 {
                format!(
                    "{} independent run{} cannot carry an error bar; PROTOCOL §5 asks for at \
                     least 3",
                    worst.runs,
                    if worst.runs == 1 { "" } else { "s" }
                )
            } else {
                format!(
                    "its stddev is {:.0}% of its mean, past the {:.0}% at which PROTOCOL §5 \
                     stops calling a run a measurement",
                    worst.rsd() * 100.0,
                    UNTRUSTWORTHY_RSD * 100.0
                )
            },
        };
    }

    if delta > noise_floor {
        Verdict::Win
    } else if delta < -noise_floor {
        Verdict::Regression
    } else {
        Verdict::Indistinguishable
    }
}

/// Parse a candidate typed on the command line: `128.4`, `128.4±3.1`, `128.4+-3.1/5`.
///
/// Deliberately terse, because it is typed by hand between two benchmark runs. The stddev and
/// run count are optional in the *grammar* and not in the *result*: omitting them produces a
/// score that [`Score::is_trustworthy`] rejects, so a bare mean gets a verdict of
/// "untrustworthy" rather than a confident comparison against nothing.
pub fn parse_candidate(spec: &str, metric: &str, depth_tokens: u32) -> Result<Score, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("expected a number, e.g. `128.4±3.1/5`".to_string());
    }
    let (value, runs) = match spec.split_once('/') {
        Some((v, r)) => (
            v,
            r.trim().parse::<u32>().map_err(|_| format!("`{}` is not a run count", r.trim()))?,
        ),
        None => (spec, 1),
    };
    let (mean, stddev) = match value.split_once('±').or_else(|| value.split_once("+-")) {
        Some((m, s)) => (m, Some(s)),
        None => (value, None),
    };
    let mean =
        mean.trim().parse::<f64>().map_err(|_| format!("`{}` is not a number", mean.trim()))?;
    let stddev = match stddev {
        Some(s) => {
            s.trim().parse::<f64>().map_err(|_| format!("`{}` is not a stddev", s.trim()))?
        }
        None => 0.0,
    };
    if !mean.is_finite() || mean <= 0.0 {
        return Err(format!("`{mean}` is not a throughput"));
    }
    if !stddev.is_finite() || stddev < 0.0 {
        return Err(format!("`{stddev}` is not a standard deviation"));
    }
    Ok(Score { metric: metric.to_string(), depth_tokens, mean, stddev, runs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(mean: f64, stddev: f64, runs: u32) -> Score {
        Score {
            metric: "decode_tokens_per_second".to_string(),
            depth_tokens: 0,
            mean,
            stddev,
            runs,
        }
    }

    #[test]
    fn the_owners_loop_reads_correctly_when_the_box_is_quiet() {
        // "If we hit 114 and things are solid, we apply some tweaks -- if that gets it to
        // 128, that's a win."
        let c = compare(&score(114.0, 1.2, 5), &score(128.0, 1.4, 5));
        assert_eq!(c.verdict, Verdict::Win);
        assert!(c.delta_pct > 12.0);
        assert!(c.noise_floor_pct < 2.0, "a quiet box resolves a 12% move easily");
    }

    #[test]
    fn and_backing_up_is_detected_as_such() {
        // "If we hit 40, we back up."
        let c = compare(&score(114.0, 1.2, 5), &score(40.0, 1.4, 5));
        assert_eq!(c.verdict, Verdict::Regression);
        assert!(c.summary().contains("back up"));
    }

    #[test]
    fn a_twelve_percent_gain_is_invisible_at_thirty_two_percent_error_bars() {
        // 🔴 The exact failure this module exists to prevent. Means alone say "+12%, keep
        // it". The error bars say the experiment could not have seen it.
        let baseline = score(114.0, 36.5, 3); // 32% RSD
        let candidate = score(127.7, 40.9, 3);
        let c = compare(&baseline, &candidate);
        assert!(c.delta_pct > 11.0, "the mean did move: {:.1}%", c.delta_pct);
        assert!(
            matches!(c.verdict, Verdict::Untrustworthy { .. }),
            "at 32% RSD neither side is a measurement: {:?}",
            c.verdict
        );
    }

    #[test]
    fn a_small_gain_inside_the_noise_floor_names_what_it_could_have_seen() {
        // Both sides trustworthy (below 20% RSD) but not precise enough to resolve 4%.
        let c = compare(&score(114.0, 17.0, 3), &score(118.5, 17.0, 3));
        assert_eq!(c.verdict, Verdict::Indistinguishable);
        assert!(c.minimum_detectable_pct() > c.delta_pct.abs());
        let s = c.summary();
        assert!(s.contains("could not have seen"), "{s}");
        assert!(s.contains("take more runs"), "{s}");
    }

    #[test]
    fn a_bare_mean_gets_no_verdict_rather_than_a_confident_one() {
        let candidate = parse_candidate("128.4", "decode_tokens_per_second", 0).unwrap();
        assert_eq!(candidate.runs, 1);
        let c = compare(&score(114.0, 1.2, 5), &candidate);
        assert!(matches!(c.verdict, Verdict::Untrustworthy { .. }));
        assert!(c.summary().contains("cannot carry an error bar"));
    }

    #[test]
    fn two_depths_are_two_questions() {
        let mut deep = score(160.0, 1.0, 5);
        deep.depth_tokens = 8_192;
        let c = compare(&score(114.0, 1.2, 5), &deep);
        assert!(matches!(c.verdict, Verdict::NotComparable { .. }));
        assert!(!c.verdict.is_actionable());
    }

    #[test]
    fn two_metrics_are_never_averaged_into_a_verdict() {
        let mut other = score(200.0, 1.0, 5);
        other.metric = "prompt_tokens_per_second".to_string();
        assert!(matches!(
            compare(&score(114.0, 1.2, 5), &other).verdict,
            Verdict::NotComparable { .. }
        ));
    }

    #[test]
    fn the_candidate_grammar_accepts_what_a_person_types() {
        let m = "decode_tokens_per_second";
        assert_eq!(parse_candidate("128.4±3.1/5", m, 0).unwrap().stddev, 3.1);
        assert_eq!(parse_candidate("128.4+-3.1/5", m, 0).unwrap().runs, 5);
        assert_eq!(parse_candidate(" 128.4 ", m, 0).unwrap().mean, 128.4);
        assert_eq!(parse_candidate("128.4/3", m, 0).unwrap().runs, 3);
        assert!(parse_candidate("fast", m, 0).is_err());
        assert!(parse_candidate("-3", m, 0).is_err());
        assert!(parse_candidate("", m, 0).is_err());
    }

    #[test]
    fn more_runs_lower_the_floor_and_nothing_else_does() {
        // The actionable half of the advice: the fix for "indistinguishable" is repeats.
        let wide = compare(&score(114.0, 17.0, 3), &score(118.5, 17.0, 3));
        let narrow = compare(&score(114.0, 17.0, 200), &score(118.5, 17.0, 200));
        assert!(narrow.noise_floor_pct < wide.noise_floor_pct);
        assert_eq!(narrow.verdict, Verdict::Win, "the same means, now resolvable");
        assert!((narrow.delta_pct - wide.delta_pct).abs() < 1e-9, "the means did not move");
    }
}
