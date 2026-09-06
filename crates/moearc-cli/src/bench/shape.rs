//! The shape results — the part of a benchmark that is supposed to reproduce.
//!
//! `bench/PROTOCOL.md` §0 is blunt about it: *absolute throughput does not reproduce and we
//! should stop implying it does.* What must reproduce on any Arc box is the shape, and two of
//! its three claims are answered here:
//!
//! 1. dynamic residency beats a static split **at matched capacity**;
//! 3. the hit-rate-versus-slots curve has the same knee.
//!
//! Claim 2 — staging rather than attention dominating the cost of prompt depth — is a timed
//! device measurement and lives in [`super::timed`].
//!
//! # 🔴 Why these two reproduce *exactly*, and what that costs
//!
//! Both are a replay of a committed routing trace through
//! [`moearc_engine::residency::simulate`]. No clock is read, no device is touched, and the
//! simulation is deterministic — so a user on a different card, a busier machine or a slower
//! disk gets **the same numbers to the last digit**, not merely the same trend. That is the
//! strongest form of reproducibility this project can offer and it is the reason these are the
//! headline.
//!
//! The cost is stated rather than hidden: a hit rate is a property of the *trace*, so it
//! transfers to another machine and does **not** transfer to another model. §9's last rule was
//! learned that way — a coverage curve taken from Qwen3-30B (8 of 128 active) was applied to
//! gpt-oss (4 of 128) and called conservative when it was optimistic. Every table below is
//! therefore labelled with the trace it came from and never averaged across models.
//!
//! # And what is deliberately not derived from it
//!
//! 🔴 §9: *never convert a proxy into a headline unit you have not validated.* A hit rate
//! predicts **staged bytes** — that conversion is exact, one miss is one expert's bytes across
//! the bus — and it does not predict tok/s, because nothing here models overlap, host offload,
//! or the drain that follows a transfer. This module emits hit rates and bytes. It emits no
//! predicted throughput of any kind, and there is a test that says so.

use std::path::{Path, PathBuf};

use moearc_engine::residency::{Policy, Trace, simulate};
use serde::Serialize;

/// One capacity point on one trace.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CapacityRow {
    pub slots: u32,
    /// Hit rate of the dynamic policy under test.
    pub dynamic_hit: f64,
    /// Hit rate of the most generous static split that fits in the same `slots`.
    pub static_hit: f64,
    /// How many leading blocks that static split could keep resident.
    pub static_layers: u16,
    /// Dynamic minus static, in percentage points. The §0 claim, per row.
    pub gap_points: f64,
    /// Belady's optimal, when asked for: the ceiling any online policy could reach here.
    pub optimal_hit: Option<f64>,
    /// Bytes the dynamic policy would move across the bus. `None` unless the slot size is
    /// known — §9 forbids inventing the conversion.
    pub dynamic_staged_bytes: Option<u64>,
    pub static_staged_bytes: Option<u64>,
    /// Points of hit rate gained per doubling of capacity, against the previous row. The
    /// quantity the knee is defined on.
    pub points_per_doubling: Option<f64>,
    /// Set when the whole working set fits, so every policy ties and the row proves nothing.
    pub trivial: bool,
}

/// One trace, replayed across the capacity ladder.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TraceShape {
    pub name: String,
    /// The capture's own header line, verbatim. A result is only citable if the provenance
    /// that came with it survives intact.
    pub provenance: String,
    pub steps: usize,
    pub demands: u64,
    pub working_set: usize,
    pub peak_step_demand: usize,
    pub rows: Vec<CapacityRow>,
    /// The knee, per [`ShapeResult::knee_definition`]. `None` when the ladder never flattens
    /// enough to have one, which is a finding rather than a failure.
    pub knee_slots: Option<u32>,
    /// Where this machine's card would sit on the curve, when a model and a device were both
    /// resolvable. Machine-specific by construction, and labelled as such.
    pub card_slots: Option<u32>,
    /// Whether §0's first claim held at every non-trivial capacity on this trace.
    ///
    /// 🔴 Reported either way. A benchmark that only prints the claim when the claim holds is
    /// not a benchmark.
    pub dynamic_beats_static: bool,
}

/// A model whose per-slot size and planned residency are known, so byte columns and a
/// "this card sits here" marker can be attached — **to its own traces and no others.**
///
/// 🔴 This is §9's last rule made structural. *A measurement transferred from one model is not
/// a measurement of another*: a coverage curve taken from Qwen3-30B (8 of 128 experts active)
/// was applied to gpt-oss (4 of 128) and called conservative when it was optimistic. A slot
/// size is exactly that kind of number — gpt-oss-120B's slot is 12.607 MiB against
/// Qwen3-30B-A3B's 2.92 MiB, a factor of 4.3 — so it is carried with the file name it came
/// from and applied only where the capture's own header names that file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelContext {
    /// The GGUF's file name, matched against the capture header's `model_file`.
    pub file: String,
    pub slot_bytes: Option<u64>,
    pub card_slots: Option<u32>,
}

impl ModelContext {
    /// Whether this capture came from this model.
    ///
    /// Matched on the header's verbatim text rather than on a parsed field, for the same
    /// reason `LoadedTrace` keeps the header verbatim: a capture tool may add fields, and a
    /// loader that broke when it did would be worse than one that looks for a file name.
    fn covers(&self, header: &str) -> bool {
        !self.file.is_empty() && header.contains(&self.file)
    }
}

/// The whole shape section.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ShapeResult {
    pub policy: String,
    pub knee_definition: String,
    pub ladder_definition: String,
    pub model_context: Option<ModelContext>,
    pub traces: Vec<TraceShape>,
    /// Files that were found but not replayed, each with the reason.
    pub skipped: Vec<(String, String)>,
}

impl ShapeResult {
    /// The one-sentence answer to §0's first claim, across every trace replayed.
    pub fn verdict_line(&self) -> String {
        let total = self.traces.len();
        let held = self.traces.iter().filter(|t| t.dynamic_beats_static).count();
        if total == 0 {
            return "no trace was replayed".to_string();
        }
        let worst = self
            .traces
            .iter()
            .flat_map(|t| t.rows.iter().filter(|r| !r.trivial))
            .map(|r| r.gap_points)
            .fold(f64::INFINITY, f64::min);
        let best = self
            .traces
            .iter()
            .flat_map(|t| t.rows.iter().filter(|r| !r.trivial))
            .map(|r| r.gap_points)
            .fold(f64::NEG_INFINITY, f64::max);
        if held == total {
            format!(
                "dynamic residency beat the widest matched-capacity static split on {held}/{total} \
                 traces, by {worst:.1} to {best:.1} points"
            )
        } else {
            format!(
                "dynamic residency beat the widest matched-capacity static split on only \
                 {held}/{total} traces; the gap ranged {worst:.1} to {best:.1} points"
            )
        }
    }
}

/// How many points the default capacity ladder has.
///
/// The ladder is geometric and runs from the trace's **peak single-step demand** — the least
/// capacity at which the trace is servable at all — to its **working set**, where every policy
/// holds everything and ties. Both ends are properties of the trace, so the same ladder means
/// the same thing on a model that activates 144 experts a step and one that activates 320,
/// and it spans the whole range in which a residency policy can matter.
///
/// 🔴 An earlier version used fixed multiples of the peak demand, topping out at 8x. On these
/// captures the working set is 17-24x the peak, so the curve had not begun to flatten by the
/// last row and the tool reported "no knee on this ladder" for every trace — truthfully, and
/// uselessly, since the knee is one of the three things §0 says has to reproduce.
const LADDER_POINTS: usize = 9;

/// How the knee is located, in one sentence, because "the knee" is otherwise a matter of
/// eyesight and the artefact has to be able to say what it means.
///
/// The **elbow**: the rung whose distance from the straight line joining the first and last
/// rungs is greatest, measured in normalised log2(slots) x hit-rate space. That is the standard
/// construction for a saturating curve, and unlike the rule it replaced it needs no judgement
/// about how many percentage points are "enough" — only [`MIN_ELBOW_PROMINENCE`], which decides
/// whether there is a knee at all rather than where it is.
///
/// 🔴 The threshold it replaced was *"the first rung where the previous doubling bought fewer
/// than five percentage points"*, and on real captures it only ever fired at the last rung —
/// where the whole working set is resident and the curve has necessarily gone flat. It
/// reported the working set as the knee, which is true, useless, and indistinguishable from a
/// tool that has not looked. The elbow lands where the marginal return actually turns over,
/// which is the capacity a person sizing a card wants to know.
const KNEE_DEFINITION: &str = "the rung farthest from the straight line joining the first and \
     last rungs, in normalised log2(slots) x hit-rate space — the standard elbow of a \
     saturating curve, and the capacity past which each further doubling buys visibly less";

/// Replay every trace in `traces` across the ladder.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    traces: &[PathBuf],
    policy: Policy,
    slots: Option<&[u32]>,
    model: Option<&ModelContext>,
    restrict_to_model: bool,
    with_optimal: bool,
) -> anyhow::Result<ShapeResult> {
    let mut out = ShapeResult {
        policy: policy.name().to_string(),
        knee_definition: KNEE_DEFINITION.to_string(),
        ladder_definition: match slots {
            Some(_) => "given explicitly with --slots".to_string(),
            None => format!(
                "{LADDER_POINTS} geometric points from the trace's peak single-step demand (the \
                 least capacity at which it is servable) to its working set (where every policy \
                 ties) — both ends are properties of the trace, so the ladder means the same \
                 thing on a model with 144 activations per step and one with 320"
            ),
        },
        model_context: model.cloned(),
        traces: Vec::new(),
        skipped: Vec::new(),
    };

    for path in traces {
        let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if let Some(reason) = skip_reason(path) {
            out.skipped.push((name, reason));
            continue;
        }
        let loaded = match Trace::from_ndjson_file(path) {
            Ok(l) => l,
            Err(e) => {
                out.skipped.push((name, format!("could not be read: {e}")));
                continue;
            }
        };
        // 🔴 The slot size and the card's operating point are attached only where the
        // capture's own header names the model they were read from. Everywhere else the byte
        // columns are omitted rather than computed from another model's geometry.
        let covers = model.is_some_and(|m| m.covers(&loaded.header));
        if restrict_to_model && !covers {
            if let Some(m) = model {
                out.skipped.push((
                    name,
                    format!(
                        "not captured from `{}` — a hit rate describes one model's routing and \
                         does not transfer to another (PROTOCOL §9). Pass --all-traces to \
                         replay it anyway.",
                        m.file
                    ),
                ));
                continue;
            }
        }
        let (slot_bytes, card_slots) = match (model, covers) {
            (Some(m), true) => (m.slot_bytes, m.card_slots),
            _ => (None, None),
        };
        out.traces.push(replay(
            &name,
            &loaded.header,
            &loaded.trace,
            policy,
            slots,
            slot_bytes,
            card_slots,
            with_optimal,
        )?);
    }
    Ok(out)
}

/// Why a file in the trace directory is not replayed.
///
/// 🔴 Prefill traces are excluded by default rather than merged in. `bench/traces/README.md`:
/// residency is a decode-time problem, and the prefill captures are 102–143 steps and
/// therefore dominated by compulsory misses. Averaging them into a decode curve would move
/// every number in the safe-looking direction.
fn skip_reason(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().to_string();
    if name.contains(".prefill.") {
        return Some(
            "prefill capture — residency is a decode-time question, and a prefill trace is \
             short enough to be dominated by compulsory misses (bench/traces/README.md)"
                .to_string(),
        );
    }
    if !name.ends_with(".ndjson") {
        return Some("not an ndjson capture".to_string());
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn replay(
    name: &str,
    header: &str,
    trace: &Trace,
    policy: Policy,
    slots: Option<&[u32]>,
    slot_bytes: Option<u64>,
    card_slots: Option<u32>,
    with_optimal: bool,
) -> anyhow::Result<TraceShape> {
    let peak = trace.peak_step_demand();
    let ladder = ladder_for(peak, trace.working_set(), slots);

    let mut rows: Vec<CapacityRow> = Vec::new();
    for &cap in &ladder {
        let dynamic = simulate(trace, cap, policy, slot_bytes.unwrap_or(0))
            .map_err(|e| anyhow::anyhow!("{name} at {cap} slots: {e}"))?;
        let static_policy = trace.widest_static_split(cap);
        let statik = simulate(trace, cap, static_policy, slot_bytes.unwrap_or(0))
            .map_err(|e| anyhow::anyhow!("{name} static split at {cap} slots: {e}"))?;
        let optimal = if with_optimal {
            Some(
                simulate(trace, cap, Policy::Optimal, 0)
                    .map_err(|e| anyhow::anyhow!("{name} optimal at {cap} slots: {e}"))?
                    .hit_rate(),
            )
        } else {
            None
        };
        let Policy::StaticSplit { resident_layers } = static_policy else {
            unreachable!("widest_static_split returns a StaticSplit");
        };

        let points_per_doubling = rows.last().map(|prev: &CapacityRow| {
            let ratio = cap as f64 / prev.slots as f64;
            if ratio <= 1.0 {
                0.0
            } else {
                (dynamic.hit_rate() - prev.dynamic_hit) * 100.0 / ratio.log2()
            }
        });

        rows.push(CapacityRow {
            slots: cap,
            dynamic_hit: dynamic.hit_rate(),
            static_hit: statik.hit_rate(),
            static_layers: resident_layers,
            gap_points: (dynamic.hit_rate() - statik.hit_rate()) * 100.0,
            optimal_hit: optimal,
            dynamic_staged_bytes: slot_bytes.map(|_| dynamic.bytes_fetched),
            static_staged_bytes: slot_bytes.map(|_| statik.bytes_fetched),
            points_per_doubling,
            // 🔴 Once capacity holds the whole working set, this row stops being a
            // comparison. The dynamic policy never evicts, so every miss it takes is
            // compulsory — it is paying warm-up and nothing else — while the static split is
            // modelled as resident from step zero and is charged no warm-up at all. The gap
            // therefore goes *negative* by exactly the dynamic policy's compulsory misses,
            // which is a statement about the two models and not about residency. Marked, kept
            // visible, and excluded from the claim.
            trivial: trace.working_set() <= cap as usize,
        });
    }

    let knee = elbow(&rows);

    let non_trivial: Vec<&CapacityRow> = rows.iter().filter(|r| !r.trivial).collect();
    let holds = !non_trivial.is_empty() && non_trivial.iter().all(|r| r.gap_points > 0.0);

    Ok(TraceShape {
        name: name.to_string(),
        provenance: header.to_string(),
        steps: trace.steps.len(),
        demands: trace.demands(),
        working_set: trace.working_set(),
        peak_step_demand: peak,
        rows,
        knee_slots: knee,
        card_slots,
        dynamic_beats_static: holds,
    })
}

/// The elbow of the hit-rate curve: see [`KNEE_DEFINITION`].
///
/// Both axes are normalised to `0..=1` over the rungs present, so the answer does not depend on
/// the units either axis happens to be in. `None` below three rungs, where "farthest from the
/// chord" has no interior point to name.
fn elbow(rows: &[CapacityRow]) -> Option<u32> {
    if rows.len() < 3 {
        return None;
    }
    let x: Vec<f64> = rows.iter().map(|r| (r.slots as f64).log2()).collect();
    let y: Vec<f64> = rows.iter().map(|r| r.dynamic_hit).collect();
    let (x0, x1) = (*x.first()?, *x.last()?);
    let (y0, y1) = (*y.first()?, *y.last()?);
    let (dx, dy) = (x1 - x0, y1 - y0);
    if dx.abs() < f64::EPSILON || dy.abs() < f64::EPSILON {
        return None;
    }
    // Normalised, then the perpendicular distance to the chord — which after normalisation is
    // the unit diagonal, so the distance reduces to |u - v| up to a constant factor.
    let mut best: Option<(f64, u32)> = None;
    for (i, row) in rows.iter().enumerate() {
        let u = (x[i] - x0) / dx;
        let v = (y[i] - y0) / dy;
        let d = (v - u).abs();
        if best.is_none_or(|(bd, _)| d > bd) {
            best = Some((d, row.slots));
        }
    }
    // 🔴 A curve that is already a straight line has no elbow, and the rung that happens to sit
    // a rounding error off the chord is not one. Below this the answer would be noise dressed
    // as a finding, so it is reported as absent instead — which is also the honest answer for a
    // ladder too short, or a trace too uniform, to have a knee.
    best.filter(|(d, _)| *d >= MIN_ELBOW_PROMINENCE).map(|(_, slots)| slots)
}

/// How far off the chord a rung has to sit, as a fraction of the normalised span, before it is
/// called a knee.
///
/// Two percent, chosen to sit far above floating-point noise on a genuinely straight line while
/// admitting every knee the committed gpt-oss captures actually show — which
/// [`tests::the_knee_is_the_elbow_and_not_the_saturation_point`] asserts against one of those
/// curves rather than against a plausible-looking shape.
const MIN_ELBOW_PROMINENCE: f64 = 0.02;

/// The capacity ladder for a trace, from `peak` single-step demand to `working_set`.
fn ladder_for(peak: usize, working_set: usize, explicit: Option<&[u32]>) -> Vec<u32> {
    let lo = peak.max(1);
    if let Some(s) = explicit {
        let mut v: Vec<u32> = s.iter().copied().filter(|c| *c as usize >= lo).collect();
        v.sort_unstable();
        v.dedup();
        return v;
    }
    let hi = working_set.max(lo);
    if hi == lo {
        return vec![lo as u32];
    }
    // Geometric rather than linear: hit rate moves with the *ratio* of capacity to working
    // set, so evenly spaced slot counts would crowd the flat end of the curve and skip the
    // part that decides anything.
    let ratio = hi as f64 / lo as f64;
    let mut v: Vec<u32> = (0..LADDER_POINTS)
        .map(|i| {
            let t = i as f64 / (LADDER_POINTS - 1) as f64;
            (lo as f64 * ratio.powf(t)).round() as u32
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Every decode capture in `dir`, sorted so the artefact's row order is stable.
pub fn discover(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("reading trace directory {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ndjson"))
        .collect();
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moearc_engine::residency::synthetic_trace;

    /// A trace with real locality, so a cache has something to exploit.
    ///
    /// ⚠️ Synthetic, and used **only** to exercise this module's arithmetic — never as
    /// evidence about a model. `synthetic_trace`'s own documentation makes that a rule, and
    /// the numbers a test asserts on are structural (ordering, monotonicity, arithmetic
    /// identities) rather than values.
    fn trace() -> Trace {
        synthetic_trace(200, 8, 32, 4, 0.8, 42)
    }

    #[test]
    fn the_ladder_runs_from_the_servable_minimum_to_the_whole_working_set() {
        // gpt-oss-120B prose, measured: 144 activations a step, 2,442 distinct experts.
        let l = ladder_for(144, 2_442, None);
        assert_eq!(*l.first().unwrap(), 144, "below this the trace is unservable at any rate");
        assert_eq!(*l.last().unwrap(), 2_442, "and here every policy holds everything");
        assert!(l.windows(2).all(|w| w[0] < w[1]), "{l:?}");
    }

    #[test]
    fn the_ladder_scales_with_the_model_not_with_a_constant() {
        // 144 activations a step (gpt-oss-120B) and 320 (Qwen3.6) must get ladders that mean
        // the same thing, which a fixed list of slot counts would not.
        let a = ladder_for(144, 2_442, None);
        let b = ladder_for(320, 5_427, None);
        assert_eq!(a.len(), b.len());
        // Same shape: each ladder covers the same ratio of its own trace's range.
        for (x, y) in a.iter().zip(&b) {
            let ra = *x as f64 / 144.0;
            let rb = *y as f64 / 320.0;
            assert!((ra - rb).abs() / ra < 0.02, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn a_trace_whose_working_set_is_one_step_gets_a_single_rung() {
        // Degenerate but real: a trace that touches the same experts every step. A geometric
        // sequence from n to n is n, and dividing by log2(1.0) must not appear anywhere.
        assert_eq!(ladder_for(32, 32, None), vec![32]);
        assert_eq!(ladder_for(32, 8, None), vec![32], "a working set cannot be below the peak");
    }

    #[test]
    fn an_explicit_ladder_drops_capacities_the_trace_cannot_be_served_at() {
        // simulate() errors below peak step demand; silently passing those through would turn
        // a user's typo into a failed run rather than a shorter table.
        let l = ladder_for(144, 2_442, Some(&[64, 144, 300, 300, 600]));
        assert_eq!(l, vec![144, 300, 600]);
    }

    #[test]
    fn the_static_baseline_is_held_to_the_same_budget() {
        // The failure recorded in SimError::StaticSplitExceedsCapacity: a static policy
        // holding twice the slots of the dynamic one, invisible in the output.
        let t = trace();
        let s = replay("t", "{}", &t, Policy::Lru, None, None, None, false).unwrap();
        for row in &s.rows {
            let needed = t.experts_in_layers_below(row.static_layers);
            assert!(needed <= row.slots as usize, "{row:?} needs {needed}");
        }
    }

    #[test]
    fn the_curve_is_monotonic_in_capacity() {
        let s = replay("t", "{}", &trace(), Policy::Lru, None, None, None, false).unwrap();
        for w in s.rows.windows(2) {
            assert!(
                w[1].dynamic_hit >= w[0].dynamic_hit - 1e-12,
                "hit rate fell from {} to {}",
                w[0].dynamic_hit,
                w[1].dynamic_hit
            );
        }
    }

    #[test]
    fn the_knee_is_the_elbow_and_not_the_saturation_point() {
        // Real shape, from gptoss120b-prose: a curve that rises steeply and then saturates.
        // The threshold definition this replaced returned the *last* rung — the working set —
        // for every trace, which is true and tells a person sizing a card nothing.
        let row = |slots: u32, hit: f64| CapacityRow {
            slots,
            dynamic_hit: hit,
            static_hit: 0.0,
            static_layers: 0,
            gap_points: 0.0,
            optimal_hit: None,
            dynamic_staged_bytes: None,
            static_staged_bytes: None,
            points_per_doubling: None,
            trivial: false,
        };
        let rows = vec![
            row(144, 0.342),
            row(205, 0.413),
            row(292, 0.505),
            row(416, 0.606),
            row(593, 0.723),
            row(845, 0.823),
            row(1_203, 0.906),
            row(1_714, 0.952),
            row(2_442, 0.967),
        ];
        let knee = elbow(&rows).expect("a saturating curve has an elbow");
        assert!((593..=1_203).contains(&knee), "elbow landed at {knee}");
        assert_ne!(knee, 2_442, "the last rung is saturation, not the knee");

        // A straight line has no elbow, and the rung that lands a rounding error off the
        // chord must not be reported as one.
        let straight: Vec<CapacityRow> = [(100u32, 0.1), (200, 0.2), (400, 0.3), (800, 0.4)]
            .into_iter()
            .map(|(s, h)| row(s, h))
            .collect();
        assert_eq!(elbow(&straight), None, "a straight line has no knee to report");
        assert_eq!(elbow(&rows[..2]), None, "two rungs have no interior point");
    }

    #[test]
    fn a_capacity_holding_the_whole_working_set_is_marked_trivial() {
        let t = trace();
        let ws = t.working_set() as u32;
        let s = replay("t", "{}", &t, Policy::Lru, Some(&[ws + 10]), None, None, false).unwrap();
        assert!(s.rows[0].trivial);
        // And a trivial row is excluded from the claim, because it proves nothing.
        assert!(!s.dynamic_beats_static, "a trivial-only ladder must not assert the claim");
    }

    #[test]
    fn belady_bounds_the_online_policy() {
        let s = replay("t", "{}", &trace(), Policy::Lru, None, None, None, true).unwrap();
        for row in &s.rows {
            let opt = row.optimal_hit.expect("asked for optimal");
            assert!(opt >= row.dynamic_hit - 1e-12, "{row:?}");
        }
    }

    #[test]
    fn a_slot_size_is_applied_only_to_captures_from_its_own_model() {
        // 🔴 §9's last rule: a measurement transferred from one model is not a measurement of
        // another. gpt-oss's 12.607 MiB slot against Qwen3-30B's 2.92 MiB is a factor of 4.3,
        // so a byte column computed across models would be wrong by that much and look fine.
        let m = ModelContext {
            file: "gpt-oss-120b-MXFP4.gguf".to_string(),
            slot_bytes: Some(13_220_000),
            card_slots: Some(600),
        };
        assert!(m.covers("{\"model_file\":\"gpt-oss-120b-MXFP4.gguf\",\"n_expert\":128}"));
        assert!(!m.covers("{\"model_file\":\"Qwen3-30B-A3B-Q4_K_M.gguf\"}"));

        let dir = std::env::temp_dir().join("moearc-shape-crossmodel");
        std::fs::create_dir_all(&dir).unwrap();
        let mine = dir.join("mine.decode.ndjson");
        let theirs = dir.join("theirs.decode.ndjson");
        let body = "{\"step\":0,\"e\":[0,1,0,2,1,1,1,2]}\n{\"step\":1,\"e\":[0,1,0,3,1,1,1,4]}\n";
        std::fs::write(
            &mine,
            format!(
                "{{\"format\":\"moearc-trace-v1\",\"model_file\":\"gpt-oss-120b-MXFP4.gguf\"}}\n{body}"
            ),
        )
        .unwrap();
        std::fs::write(
            &theirs,
            format!(
                "{{\"format\":\"moearc-trace-v1\",\"model_file\":\"Qwen3-30B-A3B-Q4_K_M.gguf\"}}\n{body}"
            ),
        )
        .unwrap();

        let r = measure(
            &[mine.clone(), theirs.clone()],
            Policy::Lru,
            Some(&[4]),
            Some(&m),
            false,
            false,
        )
        .unwrap();
        assert!(r.traces[0].rows[0].dynamic_staged_bytes.is_some(), "its own model");
        assert_eq!(r.traces[0].card_slots, Some(600));
        assert!(r.traces[1].rows[0].dynamic_staged_bytes.is_none(), "a different model");
        assert_eq!(r.traces[1].card_slots, None);

        // And by default the other model's capture is not replayed at all, with the reason
        // named rather than the file silently missing.
        let r = measure(&[mine, theirs], Policy::Lru, Some(&[4]), Some(&m), true, false).unwrap();
        assert_eq!(r.traces.len(), 1);
        assert_eq!(r.skipped.len(), 1);
        assert!(r.skipped[0].1.contains("does not transfer"), "{:?}", r.skipped);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn staged_bytes_are_emitted_only_when_the_slot_size_is_known() {
        let none = replay("t", "{}", &trace(), Policy::Lru, None, None, None, false).unwrap();
        assert!(none.rows.iter().all(|r| r.dynamic_staged_bytes.is_none()));
        let some =
            replay("t", "{}", &trace(), Policy::Lru, None, Some(13_220_000), None, false).unwrap();
        assert!(some.rows.iter().all(|r| r.dynamic_staged_bytes.is_some()));
    }

    #[test]
    fn a_prefill_capture_is_skipped_with_its_reason() {
        let r = skip_reason(Path::new("/x/gptoss120b-prose.prefill.ndjson")).unwrap();
        assert!(r.contains("decode-time"), "{r}");
        assert!(skip_reason(Path::new("/x/gptoss120b-prose.decode.ndjson")).is_none());
    }

    #[test]
    fn the_provenance_header_survives_verbatim() {
        // A result is only citable if the model, quantisation, prompt and llama.cpp commit
        // that came with the capture travel with it.
        let text = "{\"format\":\"moearc-trace-v1\",\"llama_cpp_commit\":\"e107984\"}\n\
                    {\"step\":0,\"e\":[0,1,0,2,1,1,1,2]}\n\
                    {\"step\":1,\"e\":[0,1,0,3,1,1,1,4]}\n";
        let loaded = Trace::from_ndjson_str(text).unwrap();
        let s =
            replay("t", &loaded.header, &loaded.trace, Policy::Lru, Some(&[4]), None, None, false)
                .unwrap();
        assert!(s.provenance.contains("e107984"));
    }

    /// 🔴 PROTOCOL §9. Hit rate predicts staged bytes; it does not predict tok/s, and this
    /// project has no validated model between them. If a throughput field is ever added to
    /// this module's output, this test is the one that should stop it.
    #[test]
    fn no_shape_field_is_a_throughput() {
        let s = replay("t", "{}", &trace(), Policy::Lru, None, Some(1024), None, true).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        for banned in ["tok", "tps", "throughput", "per_second", "seconds"] {
            assert!(!json.contains(banned), "shape output leaked a `{banned}` field: {json}");
        }
    }
}
