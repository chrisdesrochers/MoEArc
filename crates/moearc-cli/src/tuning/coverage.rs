//! The expert-coverage curve — the quantitative basis for moving a profile between cards.
//!
//! `docs/hardware-sizing.md` measures it: on a real gpt-oss-120B routing trace, holding **13%**
//! of the expert bank resident intercepts **53–79%** of expert touches, and **35%** intercepts
//! **84–98%**. That skew is why a bigger card helps far more than its capacity ratio suggests,
//! and it is what lets a profile measured on one Arc card say something defensible about
//! another.
//!
//! 🔴 **A curve belongs to one model and is never carried to another.** `bench/PROTOCOL.md` §9
//! records the cost of the alternative: a curve taken from Qwen3-30B (8 of 128 experts active)
//! was applied to gpt-oss (4 of 128) and called conservative. It was optimistic — 40.6% / 7.3%
//! / 1.2% predicted miss against a measured 46.4% / 15.4% / 6.3%. So there is no built-in
//! curve in this crate at all. A curve arrives attached to the profile of the model it was
//! measured on, or the tool makes no coverage claim.
//!
//! ⚠️ **Coverage is not throughput.** It predicts *staged bytes*, which
//! `bench/baselines/gpt-oss-120b.md` §6.4 measures as 80% of the throughput lost to prompt
//! depth. There is no validated model between staged bytes and tok/s in this project, so none
//! is published — an estimate here is never converted into a predicted speed.

use serde::{Deserialize, Serialize};

/// One measured point on the curve.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct Point {
    /// Fraction of the expert bank held resident, `0..=1`.
    pub bank_resident: f64,
    /// Fraction of expert touches intercepted, `0..=1`, on a prose workload.
    pub prose: f64,
    /// The same on a code workload. 🔴 Reliably the worst of the three: code revisits experts
    /// less, and at 13% residency the spread against prose is 25 points on the same hardware.
    pub code: f64,
    /// The same on a reasoning workload.
    #[serde(default)]
    pub reasoning: Option<f64>,
}

/// A model's measured coverage curve.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Curve {
    /// The model this was measured on. Carried inside the curve as well as beside it, so a
    /// curve that is copied out of its profile still says what it describes.
    pub model: String,
    /// Where the trace came from, e.g. `bench/traces/gpt-oss-120b-prose.jsonl`.
    #[serde(default)]
    pub trace: Option<String>,
    /// Measured points, in any order; sorted on read.
    pub points: Vec<Point>,
}

/// What a curve says about a residency fraction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Estimate {
    pub bank_resident: f64,
    pub prose: f64,
    pub code: f64,
    /// `1 - code`, the miss rate on the workload that punishes a small pool hardest.
    ///
    /// The pessimistic end is reported rather than the average, because `docs/hardware-sizing.md`
    /// says to size for the workload you actually have and a mean across three workloads is a
    /// workload nobody runs.
    pub worst_case_miss: f64,
    /// 🔴 False when the fraction falls outside the span that was actually measured, in which
    /// case the figures above are the nearest measured endpoint rather than an extrapolation.
    /// Every renderer must say so; a curve read past its own data is a guess wearing a
    /// measurement's clothes.
    pub within_measured_range: bool,
}

impl Curve {
    /// The measured span, as fractions of the bank.
    pub fn measured_span(&self) -> Option<(f64, f64)> {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for p in &self.points {
            lo = lo.min(p.bank_resident);
            hi = hi.max(p.bank_resident);
        }
        (lo <= hi).then_some((lo, hi))
    }

    /// Read the curve at a residency fraction, interpolating linearly between measured points.
    ///
    /// Linear, not fitted. A fitted curve would put a model of the routing skew between the
    /// data and the reader, and this project has been burned specifically by a simulation
    /// calibrated against nothing; a straight line between two measured points at least
    /// cannot claim anything the measurements do not bracket.
    pub fn at(&self, bank_resident: f64) -> Option<Estimate> {
        if self.points.is_empty() || !bank_resident.is_finite() {
            return None;
        }
        let mut points = self.points.clone();
        points.sort_by(|a, b| a.bank_resident.total_cmp(&b.bank_resident));
        let (lo, hi) = (points[0], points[points.len() - 1]);

        let (prose, code, within) = if bank_resident <= lo.bank_resident {
            (lo.prose, lo.code, bank_resident >= lo.bank_resident)
        } else if bank_resident >= hi.bank_resident {
            (hi.prose, hi.code, bank_resident <= hi.bank_resident)
        } else {
            let i = points.iter().position(|p| p.bank_resident >= bank_resident)?;
            let (a, b) = (points[i - 1], points[i]);
            let span = b.bank_resident - a.bank_resident;
            let t = if span.abs() < f64::EPSILON {
                0.0
            } else {
                (bank_resident - a.bank_resident) / span
            };
            (a.prose + (b.prose - a.prose) * t, a.code + (b.code - a.code) * t, true)
        };
        Some(Estimate {
            bank_resident,
            prose,
            code,
            worst_case_miss: (1.0 - code).clamp(0.0, 1.0),
            within_measured_range: within,
        })
    }
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    /// The measured gpt-oss-120B curve from `docs/hardware-sizing.md`.
    ///
    /// 🔴 A **test fixture**, and it lives under `cfg(test)` for that reason: shipping it as a
    /// built-in would let it be read for a model it was not measured on, which is the exact
    /// mistake PROTOCOL §9 records. The real one arrives in `bench/tuning-profiles.json`
    /// attached to gpt-oss's own profile.
    pub fn gpt_oss_120b() -> Curve {
        Curve {
            model: "gpt-oss-120b".to_string(),
            trace: Some("bench/traces/".to_string()),
            points: vec![
                Point { bank_resident: 0.05, prose: 0.508, code: 0.312, reasoning: Some(0.324) },
                Point { bank_resident: 0.10, prose: 0.710, code: 0.463, reasoning: Some(0.491) },
                Point { bank_resident: 0.13, prose: 0.788, code: 0.533, reasoning: Some(0.569) },
                Point { bank_resident: 0.20, prose: 0.896, code: 0.662, reasoning: Some(0.708) },
                Point { bank_resident: 0.25, prose: 0.937, code: 0.734, reasoning: Some(0.780) },
                Point { bank_resident: 0.35, prose: 0.978, code: 0.842, reasoning: Some(0.879) },
                Point { bank_resident: 0.40, prose: 0.989, code: 0.882, reasoning: Some(0.913) },
                Point { bank_resident: 0.50, prose: 0.998, code: 0.939, reasoning: Some(0.959) },
                Point { bank_resident: 0.60, prose: 1.000, code: 0.974, reasoning: Some(0.984) },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_cards_in_the_sizing_doc_come_back_out() {
        // docs/hardware-sizing.md: B580 13.1% -> 46.4% worst-case miss, 24 GB 35.5% -> 15.4%,
        // 32 GB 49.6% -> 6.3%. Interpolated between measured rows, so a point or two of
        // slack is expected; more than that means the curve or the table has drifted.
        let c = fixture::gpt_oss_120b();
        for (resident, expected_miss) in [(0.131, 0.464), (0.355, 0.154), (0.496, 0.063)] {
            let e = c.at(resident).unwrap();
            assert!(e.within_measured_range);
            assert!(
                (e.worst_case_miss - expected_miss).abs() < 0.02,
                "at {resident} the curve gives {:.3} miss, the doc says {expected_miss}",
                e.worst_case_miss
            );
        }
    }

    #[test]
    fn reading_past_the_measured_span_says_so_rather_than_extrapolating() {
        let c = fixture::gpt_oss_120b();
        let below = c.at(0.01).unwrap();
        assert!(!below.within_measured_range);
        assert_eq!(below.prose, 0.508, "clamped to the nearest measured point, not projected");
        let above = c.at(0.95).unwrap();
        assert!(!above.within_measured_range);
        assert_eq!(above.code, 0.974);
    }

    #[test]
    fn code_is_never_flattered_by_an_average() {
        let c = fixture::gpt_oss_120b();
        let e = c.at(0.13).unwrap();
        assert!(e.code < e.prose, "the 25-point spread is the finding, not an inconvenience");
        assert!((e.worst_case_miss - (1.0 - e.code)).abs() < 1e-9);
    }

    #[test]
    fn an_empty_curve_answers_nothing_rather_than_zero() {
        let c = Curve { model: "m".into(), trace: None, points: Vec::new() };
        assert!(c.at(0.3).is_none());
        assert!(c.measured_span().is_none());
    }

    #[test]
    fn points_need_not_arrive_sorted() {
        let mut c = fixture::gpt_oss_120b();
        c.points.reverse();
        assert!((c.at(0.13).unwrap().code - 0.533).abs() < 1e-9);
    }
}
