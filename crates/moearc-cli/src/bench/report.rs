//! The artefact: one file a user can paste into a GitHub issue.
//!
//! `bench/PROTOCOL.md`'s last requirement is the hardest one: *write a single self-describing
//! artefact … one that contains enough context for a stranger to tell whether the number is
//! trustworthy.* Three things follow, and they shape this whole module:
//!
//! * **The checks are part of the artefact, not part of the console output.** A reader who was
//!   not there has to be able to see the load average, the cache ratio, the thread counts and
//!   the build commit that the numbers were taken under — and the thresholds they were judged
//!   against, so they can disagree with the judgement rather than only with the result.
//! * **A refusal is published, not swallowed.** §5 asks that a discarded run stay "in the
//!   record with its error bars visible, so the discard is auditable rather than convenient",
//!   and §9 that "retractions stay in the tree with their evidence". A refused run therefore
//!   still writes a full artefact; what it does not write is a headline.
//! * **One file carries both forms.** The human tables and the machine-readable JSON are the
//!   same data rendered twice, with the JSON in a collapsed block at the end, so pasting the
//!   artefact somewhere does not silently drop half of it.

use serde::Serialize;

use super::guard::{Finding, Level, Reading, Thresholds, Verdict};
use super::incumbent::IncumbentResult;
use super::shape::ShapeResult;
use super::timed::TimedPoint;
use crate::format;

/// Everything one `moearc bench` run produced.
#[derive(Debug, Clone, Serialize)]
pub struct Artefact {
    pub tool: &'static str,
    pub tool_version: &'static str,
    pub generated_utc: String,
    /// The exact command line, so the run can be repeated without reading the prose.
    pub command: Vec<String>,
    pub verdict: Verdict,
    /// `None` under [`Verdict::Refused`]. The absence is the point: this tool does not print a
    /// headline it does not stand behind.
    pub headline: Option<Vec<String>>,
    pub reading: Reading,
    pub thresholds: Thresholds,
    pub findings: Vec<Finding>,
    pub shape: Option<ShapeResult>,
    pub absolutes: Vec<TimedPoint>,
    pub incumbent: Option<IncumbentResult>,
    /// Named things the run did not do, so a gap reads as a gap rather than as a zero.
    pub not_measured: Vec<String>,
}

impl Artefact {
    pub fn exit_code(&self) -> std::process::ExitCode {
        match self.verdict {
            Verdict::Trusted | Verdict::Qualified => std::process::ExitCode::SUCCESS,
            Verdict::Refused => std::process::ExitCode::from(super::EXIT_REFUSED),
        }
    }
}

/// The human form, with the machine form folded in at the end.
pub fn render(a: &Artefact) -> String {
    let mut s = String::new();
    header(&mut s, a);
    checks(&mut s, a);
    machine(&mut s, a);
    shape(&mut s, a);
    absolutes(&mut s, a);
    incumbent(&mut s, a);
    gaps(&mut s, a);
    reproduce(&mut s, a);
    embedded_json(&mut s, a);
    s
}

fn header(s: &mut String, a: &Artefact) {
    s.push_str("# moearc bench\n\n");
    s.push_str(&format!(
        "**{}** · {} · generated {} UTC\n\n",
        match a.verdict {
            Verdict::Trusted => "VERDICT: trusted",
            Verdict::Qualified => "VERDICT: qualified — every number below carries a caveat",
            Verdict::Refused => "VERDICT: REFUSED — no number below is a measurement",
        },
        a.tool_version,
        a.generated_utc,
    ));

    match (&a.headline, a.verdict) {
        (Some(lines), _) if !lines.is_empty() => {
            s.push_str("## Result\n\n");
            for l in lines {
                s.push_str(&format!("- {l}\n"));
            }
            s.push('\n');
        }
        (_, Verdict::Refused) => {
            s.push_str("## Result\n\n");
            s.push_str(
                "**There is none.** At least one check below refused, so anything this run \
                 produced describes the state of the machine rather than the engine. The \
                 figures are kept in place, unheadlined, so the refusal is auditable rather \
                 than convenient — see `bench/PROTOCOL.md` §5 and §9.\n\n",
            );
        }
        _ => {}
    }

    s.push_str(
        "> **What reproduces and what does not.** Absolute throughput depends on CPU, memory \
         bandwidth, PCIe generation, filesystem and whether the model fits in page cache; it \
         is reported below as an artefact of *this* machine and should not be compared across \
         machines. The **shape** results are a deterministic replay of committed routing \
         traces — no clock is read and no device is touched — so they should come out \
         identical on yours, to the last digit. Those are the result.\n\n",
    );
}

fn checks(s: &mut String, a: &Artefact) {
    s.push_str("## Checks\n\n");
    s.push_str(&format!(
        "Thresholds in force: refuse above load **{:.2}** ({:.0}% of {} logical CPUs, floor \
         {:.1}), warn above **{:.2}**; a model above **{:.0}%** of the cache ceiling is \
         flagged; a result needs **{}** independent invocations and is refused a headline at a \
         stddev of **{:.0}%** of its mean (warned at {:.0}%).\n\n",
        a.thresholds.load_refuse(a.reading.logical_cpus),
        a.thresholds.load_fraction * 100.0,
        a.reading.logical_cpus,
        a.thresholds.load_floor,
        a.thresholds.load_warn(a.reading.logical_cpus),
        a.thresholds.cache_tight_ratio * 100.0,
        a.thresholds.min_invocations,
        a.thresholds.cv_refuse * 100.0,
        a.thresholds.cv_warn * 100.0,
    ));
    s.push_str("| | rule | check |\n|---|---|---|\n");
    for f in &a.findings {
        s.push_str(&format!(
            "| {} | {} | {} |\n",
            match f.level {
                Level::Pass => "ok",
                Level::Warn => "⚠️ warn",
                Level::Refuse => "🔴 **REFUSE**",
            },
            f.rule,
            f.headline
        ));
    }
    s.push('\n');
    for f in a.findings.iter().filter(|f| f.level != Level::Pass) {
        s.push_str(&format!("**{} — {}**\n\n{}\n\n", f.rule, f.headline, f.detail));
    }
}

fn machine(s: &mut String, a: &Artefact) {
    let r = &a.reading;
    s.push_str("## Machine\n\n");
    // Collected first rather than pushed as they are built: a closure that borrows `s`
    // mutably cannot coexist with the pushes around it, and threading the string through
    // every branch reads worse than one table built as data.
    let mut rows: Vec<(&str, String)> = Vec::new();
    let mut row = |k: &'static str, v: String| rows.push((k, v));
    match &r.device {
        Some(d) => {
            row("device", format!("{} ({})", d.name, d.backend));
            row("driver", d.driver.clone());
            row(
                "Level Zero build",
                match d.driver_build {
                    Some(b) => format!("{b}"),
                    None => "unknown".to_string(),
                },
            );
            if let Some(src) = &d.budget_source {
                row("free VRAM figure", src.clone());
            }
        }
        None => row("device", "none detected".to_string()),
    }
    row("logical CPUs", r.logical_cpus.to_string());
    row(
        "load average (1 min)",
        r.load1.map(|v| format!("{v:.2}")).unwrap_or_else(|| "unreadable".to_string()),
    );
    row(
        "memory",
        format!(
            "{} available of {}",
            format::bytes(r.mem_available_bytes),
            format::bytes(r.mem_total_bytes)
        ),
    );
    if let Some(arc) = r.zfs_arc {
        row(
            "ZFS ARC",
            format!(
                "{} in use, cap {} (`zfs_arc_max`)",
                format::bytes(arc.size_bytes),
                format::bytes(arc.c_max_bytes)
            ),
        );
    }
    if let Some(m) = &r.model {
        row("model", format!("{} — {} on {}", m.path, format::bytes(m.bytes), m.filesystem));
    }
    if let Some(t) = r.engine_threads {
        row(
            "moearc host threads",
            format!(
                "{} requested, {} reported by the engine",
                t.requested,
                t.reported.map(|n| n.to_string()).unwrap_or_else(|| "—".into())
            ),
        );
    }
    let b = &a.reading.build;
    row(
        "build",
        format!(
            "{} · {} · features [{}] · commit {} ({})",
            b.profile,
            b.target,
            if b.features.is_empty() { "none".to_string() } else { b.features.join(",") },
            b.commit.as_deref().unwrap_or("unknown"),
            match b.dirty {
                Some(true) => "dirty tree",
                Some(false) => "clean tree",
                None => "tree state unknown",
            }
        ),
    );
    s.push_str("| | |\n|---|---|\n");
    for (k, v) in rows {
        s.push_str(&format!("| {k} | {v} |\n"));
    }
    s.push('\n');
}

fn shape(s: &mut String, a: &Artefact) {
    let Some(shape) = &a.shape else { return };
    s.push_str("## Shape — the result that should reproduce on your machine\n\n");
    s.push_str(&format!(
        "Policy under test: **{}**, against the widest static split that fits the *same* \
         capacity. Ladder: {}. Knee: {}.\n\n",
        shape.policy, shape.ladder_definition, shape.knee_definition
    ));
    s.push_str(&format!("**{}**\n\n", shape.verdict_line()));
    match &shape.model_context {
        Some(m) => s.push_str(&format!(
            "Byte columns and the card marker are attached only to captures whose own header \
             names `{}` — a slot size belongs to one model (gpt-oss-120B's is 12.607 MiB, \
             Qwen3-30B-A3B's is 2.92 MiB) and carrying one onto another model's trace is \
             PROTOCOL §9's last failure.\n\n",
            m.file
        )),
        None => s.push_str(
            "No `--model` was given, so hit rates are reported without their byte equivalents \
             rather than with an invented slot size.\n\n",
        ),
    }

    for t in &shape.traces {
        s.push_str(&format!(
            "### {}\n\n{} decode steps · {} activations · working set {} experts · peak \
             {} per step{}\n\n",
            t.name,
            format::count(t.steps as i64),
            format::count(t.demands as i64),
            format::count(t.working_set as i64),
            t.peak_step_demand,
            match t.knee_slots {
                Some(k) => format!(" · knee at **{}** slots", format::count(k as i64)),
                None => " · no knee on this ladder".to_string(),
            }
        ));
        let bytes = t.rows.first().is_some_and(|r| r.dynamic_staged_bytes.is_some());
        let optimal = t.rows.first().is_some_and(|r| r.optimal_hit.is_some());
        s.push_str("| slots | dynamic hit | static hit (blocks) | gap (points) |");
        if optimal {
            s.push_str(" belady |");
        }
        if bytes {
            s.push_str(" dynamic staged | static staged |");
        }
        s.push_str(" pts/doubling |\n|---:|---:|---:|---:|");
        if optimal {
            s.push_str("---:|");
        }
        if bytes {
            s.push_str("---:|---:|");
        }
        s.push_str("---:|\n");
        for r in &t.rows {
            s.push_str(&format!(
                "| {}{} | {:.1}% | {:.1}% ({}) | {:+.1} |",
                format::count(r.slots as i64),
                if t.card_slots == Some(r.slots) { " ←this card" } else { "" },
                r.dynamic_hit * 100.0,
                r.static_hit * 100.0,
                r.static_layers,
                r.gap_points,
            ));
            if optimal {
                s.push_str(&format!(
                    " {:.1}% |",
                    r.optimal_hit.map(|v| v * 100.0).unwrap_or(f64::NAN)
                ));
            }
            if bytes {
                s.push_str(&format!(
                    " {} | {} |",
                    r.dynamic_staged_bytes.map(format::bytes).unwrap_or_else(|| "—".into()),
                    r.static_staged_bytes.map(format::bytes).unwrap_or_else(|| "—".into()),
                ));
            }
            s.push_str(&format!(
                " {} |",
                r.points_per_doubling.map(|v| format!("{v:+.1}")).unwrap_or_else(|| "—".into())
            ));
            if r.trivial {
                s.push_str(
                    " *(whole working set resident: the dynamic policy never evicts, so its \
                     misses here are all compulsory, while the static split is modelled as \
                     resident from step zero and pays no warm-up. Not a comparison — excluded \
                     from the claim.)*",
                );
            }
            s.push('\n');
        }
        if let Some(c) = t.card_slots {
            s.push_str(&format!(
                "\n⚠️ **{} slots** is where *this* card lands on the curve, which is a fact \
                 about this machine and not part of the shape.\n",
                format::count(c as i64)
            ));
        }
        s.push_str(&format!(
            "\nProvenance of the capture, verbatim:\n\n```json\n{}\n```\n\n",
            t.provenance
        ));
    }

    if !shape.skipped.is_empty() {
        s.push_str("Not replayed:\n\n");
        for (name, why) in &shape.skipped {
            s.push_str(&format!("- `{name}` — {why}\n"));
        }
        s.push('\n');
    }

    s.push_str(
        "🔴 A hit rate predicts **staged bytes** and nothing else. It does not predict tok/s: \
         there is no validated model between them in this project, so none is published here \
         and no throughput figure anywhere in this section is derived from one.\n\n",
    );
}

fn absolutes(s: &mut String, a: &Artefact) {
    if a.absolutes.is_empty() {
        return;
    }
    s.push_str("## Absolutes — this machine only\n\n");
    s.push_str(
        "Decode-only throughput: the timer starts after prefill, so these are steady-state \
         decode figures at the stated depth and not an average over the prompt. Cold and warm \
         are separate questions and are never averaged together. Each figure is the mean ± \
         sample stddev over **independent invocations of this binary**, not iterations inside \
         one process.\n\n",
    );
    s.push_str(
        "| depth | tokens | residency | host | threads | cold tok/s | warm tok/s | warm/cold | \
         cold hit | warm hit | disk read | ARC miss |\n\
         |---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for p in &a.absolutes {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.1}% | {:.1}% | {} | {} |\n",
            format::count(p.depth as i64),
            p.generated_tokens,
            p.residency,
            p.host_policy,
            p.threads_reported()
                .map(|n| n.to_string())
                .unwrap_or_else(|| format!("{}?", p.threads_requested)),
            p.cold.render(),
            p.warm.render(),
            p.warm_over_cold.map(|v| format!("{v:.2}x")).unwrap_or_else(|| "—".into()),
            p.cold_hit_rate.mean() * 100.0,
            p.warm_hit_rate.mean() * 100.0,
            p.disk_read_bytes().map(format::bytes).unwrap_or_else(|| "unattributed".into()),
            match p.arc() {
                Some((h, m)) if h + m > 0 => format!("{:.1}%", m as f64 / (h + m) as f64 * 100.0),
                Some(_) => "0%".to_string(),
                None => "—".to_string(),
            },
        ));
    }
    s.push_str(
        "\nA large `disk read` column means the run faulted the model off the drive and \
         measured the storage rather than the engine (PROTOCOL §4). `ARC miss` is the ZFS \
         cache's own miss rate over the same window, machine-wide — the second, independent \
         reading §4 asks for. Both coming back near zero is a result, not an absence: it says \
         staging read from RAM, and that this run measured the engine.\n\n",
    );

    s.push_str("Every individual invocation, so the spread is visible:\n\n");
    for p in &a.absolutes {
        s.push_str(&format!(
            "- depth {}: cold {} · warm {} · load before each child {}\n",
            p.depth,
            p.cold.values.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(", "),
            p.warm.values.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(", "),
            p.invocations
                .iter()
                .map(|r| r.load1_before.map(|v| format!("{v:.2}")).unwrap_or_else(|| "?".into()))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    s.push('\n');
}

fn incumbent(s: &mut String, a: &Artefact) {
    let Some(inc) = &a.incumbent else { return };
    s.push_str("## Incumbent — llama.cpp\n\n");
    s.push_str(&format!(
        "`{}` · build `{}` · backends `{}` · model `{}`\n\n",
        inc.binary,
        inc.facts.build_commit.as_deref().unwrap_or("unknown"),
        inc.facts.backends.as_deref().unwrap_or("unreported"),
        inc.facts.model_filename.as_deref().unwrap_or("unreported"),
    ));
    s.push_str(
        "🔴 The thread count in the `threads` column is read out of `llama-bench -o csv`'s own \
         `n_threads` field, never inferred and never assumed to have followed `-t`. The \
         binary's path was given explicitly; it was not found by glob.\n\n",
    );
    s.push_str("| depth | -t asked | n_threads reported | decode tok/s |\n|---:|---:|---:|---:|\n");
    for p in &inc.points {
        s.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            p.depth,
            p.threads_requested,
            p.threads_reported.iter().map(|n| n.to_string()).collect::<Vec<_>>().join("/"),
            p.decode.render(),
        ));
    }
    if let Some(best) = inc.best_threads {
        s.push_str(&format!(
            "\nQuoted at its best configuration, **-t {best}** — PROTOCOL §1 requires the \
             baseline be swept and quoted at its best, not its first.\n"
        ));
    }
    s.push('\n');
}

fn gaps(s: &mut String, a: &Artefact) {
    if a.not_measured.is_empty() {
        return;
    }
    s.push_str("## Not measured\n\n");
    s.push_str("Named so a gap reads as a gap rather than as a zero.\n\n");
    for g in &a.not_measured {
        s.push_str(&format!("- {g}\n"));
    }
    s.push('\n');
}

fn reproduce(s: &mut String, a: &Artefact) {
    s.push_str("## Reproduce\n\n```sh\n");
    s.push_str(&a.command.join(" "));
    s.push_str("\n```\n\n");
}

fn embedded_json(s: &mut String, a: &Artefact) {
    let json = serde_json::to_string_pretty(a).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
    s.push_str("<details>\n<summary>machine-readable</summary>\n\n```json\n");
    s.push_str(&json);
    s.push_str("\n```\n\n</details>\n");
}

// ---------------------------------------------------------------------------------------
// Time, without a dependency
// ---------------------------------------------------------------------------------------

/// `2026-09-06T18:04:11Z`, from the system clock.
///
/// Hand-rolled rather than adding `chrono` or `time` to a binary that needs one timestamp.
/// The civil-date conversion is Howard Hinnant's `civil_from_days`, which is exact for every
/// date this program will ever see and is small enough to test outright.
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    iso8601(secs)
}

fn iso8601(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let rem = unix_secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timestamp_is_a_real_iso_date() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        // 2026-09-06T18:04:11Z
        assert_eq!(iso8601(1_788_717_851), "2026-09-06T18:04:11Z");
        // A leap day, the case a hand-rolled converter gets wrong.
        assert_eq!(iso8601(1_709_164_800), "2024-02-29T00:00:00Z");
        assert!(now_utc().ends_with('Z'));
    }

    fn artefact(verdict: Verdict, findings: Vec<Finding>) -> Artefact {
        Artefact {
            tool: "moearc bench",
            tool_version: "test",
            generated_utc: "2026-09-06T00:00:00Z".to_string(),
            command: vec!["moearc".into(), "bench".into()],
            verdict,
            headline: match verdict {
                Verdict::Refused => None,
                _ => Some(vec!["something held".to_string()]),
            },
            reading: super::super::guard::quiet_reading(),
            thresholds: Thresholds::default(),
            findings,
            shape: None,
            absolutes: Vec::new(),
            incumbent: None,
            not_measured: vec!["staging-versus-attention attribution".to_string()],
        }
    }

    #[test]
    fn a_refused_run_prints_no_headline_and_says_why() {
        let f = Finding {
            level: Level::Refuse,
            code: "load",
            rule: "§3",
            headline: "the box is busy — load 21.69, refusing above 2.50".to_string(),
            detail: "…".to_string(),
        };
        let text = render(&artefact(Verdict::Refused, vec![f]));
        assert!(text.contains("VERDICT: REFUSED"));
        assert!(text.contains("**There is none.**"));
        assert!(text.contains("21.69"));
        // The refusal is in the artefact, not only on the console.
        assert!(text.contains("🔴 **REFUSE**"));
    }

    #[test]
    fn a_trusted_run_leads_with_the_result() {
        let text = render(&artefact(Verdict::Trusted, Vec::new()));
        assert!(text.contains("VERDICT: trusted"));
        assert!(text.contains("something held"));
    }

    #[test]
    fn the_artefact_states_the_thresholds_it_judged_against() {
        // A refusal a reader cannot argue with is not auditable.
        let text = render(&artefact(Verdict::Trusted, Vec::new()));
        assert!(text.contains("refuse above load **2.50**"), "{text}");
        assert!(text.contains("**20%** of its mean"), "{text}");
    }

    #[test]
    fn the_artefact_carries_its_own_machine_readable_form() {
        let text = render(&artefact(Verdict::Qualified, Vec::new()));
        assert!(text.contains("<details>"));
        assert!(text.contains("\"verdict\": \"qualified\""), "{text}");
        // And the command line that made it.
        assert!(text.contains("moearc bench"));
    }

    #[test]
    fn a_gap_is_named_rather_than_left_as_a_zero() {
        let text = render(&artefact(Verdict::Trusted, Vec::new()));
        assert!(text.contains("Not measured"));
        assert!(text.contains("staging-versus-attention"));
    }
}
