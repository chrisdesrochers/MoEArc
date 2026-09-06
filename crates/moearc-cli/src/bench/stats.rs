//! Aggregating repeated measurements, and deciding when the spread means there is no result.
//!
//! `bench/PROTOCOL.md` §5: report mean ± stddev, never a single run; prefer several
//! **independent invocations** over more iterations inside one; and a run whose stddev is
//! 20–30% of its mean *is not a measurement*.

use serde::Serialize;

use super::guard::{Finding, Level, Thresholds};

/// A set of independent measurements of one quantity.
///
/// 🔴 `values` is kept in full and serialised into the artefact, not just its summary. §5 asks
/// that a discarded run stay "in the record with its error bars visible, so the discard is
/// auditable rather than convenient" — which is only possible if the individual figures
/// survive into the output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sample {
    pub label: String,
    pub unit: &'static str,
    pub values: Vec<f64>,
}

impl Sample {
    pub fn new(label: impl Into<String>, unit: &'static str, values: Vec<f64>) -> Self {
        Self { label: label.into(), unit, values }
    }

    pub fn n(&self) -> usize {
        self.values.len()
    }

    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return f64::NAN;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// **Sample** standard deviation (`n - 1`), because these are draws from a process whose
    /// variance is being estimated, not the whole population. Undefined below two values, and
    /// reported as such rather than as zero — a single run with "± 0.00" beside it reads as
    /// the most precise number in the table and is the least.
    pub fn stddev(&self) -> Option<f64> {
        if self.values.len() < 2 {
            return None;
        }
        let mean = self.mean();
        let var = self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.values.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Coefficient of variation: stddev as a fraction of the mean. The quantity §5 states its
    /// rule in.
    pub fn cv(&self) -> Option<f64> {
        let mean = self.mean();
        if mean.abs() < f64::EPSILON {
            return None;
        }
        self.stddev().map(|s| s / mean.abs())
    }

    /// `28.46 ± 0.31` — or `28.46 (1 run, no error bar)`.
    pub fn render(&self) -> String {
        match self.stddev() {
            Some(sd) => format!("{:.2} ± {:.2}", self.mean(), sd),
            None if self.values.len() == 1 => format!("{:.2} (1 run, no error bar)", self.mean()),
            None => "—".to_string(),
        }
    }

    /// §5's rule, as a finding.
    ///
    /// Two independent reasons to withhold a headline, kept separate because they call for
    /// different fixes: too few invocations (run it again) and too much spread between them
    /// (the machine is not in a state where this can be measured).
    pub fn dispersion(&self, t: &Thresholds) -> Finding {
        let code = "dispersion";
        let rule = "§5";
        if self.n() < t.min_invocations {
            return Finding {
                level: Level::Refuse,
                code,
                rule,
                headline: format!(
                    "{}: {} invocation(s), below the {} required to headline",
                    self.label,
                    self.n(),
                    t.min_invocations
                ),
                detail: format!(
                    "Values so far: {}. §5 asks for several *independent invocations* rather \
                     than more iterations inside one, because process-level variance is the \
                     variance that bit this project. The figures are kept, but nothing is \
                     headlined from them.",
                    self.list()
                ),
            };
        }
        let Some(cv) = self.cv() else {
            return Finding {
                level: Level::Refuse,
                code,
                rule,
                headline: format!("{}: the mean is zero, so the spread has no scale", self.label),
                detail: format!("Values: {}.", self.list()),
            };
        };
        let detail = format!(
            "{} over {} independent invocations, stddev {:.1}% of the mean. Warn at {:.0}%, \
             refuse to headline at {:.0}% — §5 states that a run whose stddev is 20–30% of its \
             mean is not a measurement, and the good triplicate it is contrasted against sat \
             at 0.7%. Individual values: {}.",
            self.render(),
            self.n(),
            cv * 100.0,
            t.cv_warn * 100.0,
            t.cv_refuse * 100.0,
            self.list(),
        );
        if cv >= t.cv_refuse {
            Finding {
                level: Level::Refuse,
                code,
                rule,
                headline: format!(
                    "{}: stddev is {:.0}% of the mean — this is not a measurement",
                    self.label,
                    cv * 100.0
                ),
                detail,
            }
        } else if cv >= t.cv_warn {
            Finding {
                level: Level::Warn,
                code,
                rule,
                headline: format!("{}: stddev is {:.0}% of the mean", self.label, cv * 100.0),
                detail,
            }
        } else {
            Finding {
                level: Level::Pass,
                code,
                rule,
                headline: format!("{}: {} {}", self.label, self.render(), self.unit),
                detail,
            }
        }
    }

    fn list(&self) -> String {
        self.values.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Thresholds {
        Thresholds::default()
    }

    #[test]
    fn the_stddev_is_the_sample_one() {
        // 1,2,3,4 -> mean 2.5, sample sd 1.290994, population sd 1.118034. The distinction is
        // visible in the second decimal, which is where these are quoted.
        let s = Sample::new("x", "tok/s", vec![1.0, 2.0, 3.0, 4.0]);
        assert!((s.mean() - 2.5).abs() < 1e-12);
        assert!((s.stddev().unwrap() - 1.290_994_448_735_806).abs() < 1e-9);
    }

    #[test]
    fn one_run_has_no_error_bar_and_says_so() {
        let s = Sample::new("x", "tok/s", vec![17.90]);
        assert_eq!(s.stddev(), None);
        assert!(s.render().contains("no error bar"), "{}", s.render());
    }

    #[test]
    fn the_documented_good_triplicate_headlines() {
        // bench/baselines/gpt-oss-120b.md §6.2, -t 16: three invocations at 28.39/28.42/28.58.
        let s = Sample::new("llama.cpp tg128", "tok/s", vec![28.39, 28.42, 28.58]);
        assert!(s.cv().unwrap() < 0.01);
        assert_eq!(s.dispersion(&t()).level, Level::Pass);
    }

    #[test]
    fn the_documented_bad_run_is_refused_and_kept() {
        // §6.2.1: 17.59 ± 5.56 — "its own error bars say it is not a measurement".
        let s = Sample::new("llama.cpp tg128", "tok/s", vec![11.5, 17.6, 23.7]);
        let cv = s.cv().unwrap();
        assert!(cv > 0.20, "cv {cv}");
        let f = s.dispersion(&t());
        assert_eq!(f.level, Level::Refuse);
        // Kept in the record with its error bars visible, per §5.
        assert!(f.detail.contains("11.50"), "{}", f.detail);
        assert!(f.detail.contains("23.70"), "{}", f.detail);
    }

    #[test]
    fn two_invocations_cannot_headline_however_close_they_are() {
        // §5 prefers independent invocations; two of them still under-describes the spread.
        let s = Sample::new("x", "tok/s", vec![28.40, 28.41]);
        assert_eq!(s.dispersion(&t()).level, Level::Refuse);
    }

    #[test]
    fn the_warn_band_sits_between_the_good_and_the_bad_case() {
        let s = Sample::new("x", "tok/s", vec![10.0, 11.5, 8.5]);
        let cv = s.cv().unwrap();
        assert!((0.10..0.20).contains(&cv), "cv {cv}");
        assert_eq!(s.dispersion(&t()).level, Level::Warn);
    }
}
