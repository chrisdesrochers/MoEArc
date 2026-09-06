//! Running the incumbent, `llama-bench`, and reading its answers back out of its own output.
//!
//! `bench/PROTOCOL.md` §1 and §2 both exist because of this program, and both failures were
//! invisible in every field except the one that carried the answer:
//!
//! * §1 — `llama-bench`'s default `n_threads` on a 20-core box is **4**, and no invocation in
//!   this repository passed `-t`. Every "beats llama.cpp" claim had to be withdrawn. The fix
//!   is not to pass `-t` and hope: it is to pass `-t` and then read `n_threads` back out of
//!   `-o csv`.
//! * §2 — `ls build*/bin/llama-bench | head -1` selected a **Vulkan** build, 4.8x slower than
//!   SYCL. Exit 0, real CSV, plausible numbers. Only the `backends` column revealed it.
//!
//! So this module does exactly two unusual things. It **never** looks for the binary; the path
//! is given or the incumbent is not run. And it treats the CSV header as the schema, indexing
//! every field by name, so a column added or reordered upstream cannot silently shift which
//! number is read as the thread count.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use super::guard::{IncumbentFacts, ThreadPin};
use super::stats::Sample;

/// One row of `llama-bench -o csv`, keyed by the header.
#[derive(Debug, Clone, PartialEq)]
pub struct Row(HashMap<String, String>);

impl Row {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn num<T: std::str::FromStr>(&self, key: &str) -> Option<T> {
        self.get(key)?.trim().parse().ok()
    }
}

/// Parse `llama-bench -o csv`.
///
/// RFC 4180 quoting: fields may be quoted, a quoted field may contain commas and newlines, and
/// `""` inside a quoted field is a literal quote. `cpu_info` on this machine contains a comma,
/// so a naive `split(',')` shifts every subsequent column by one — which would read the wrong
/// field as `n_threads` and produce exactly the kind of confident wrong number §1 is about.
pub fn parse_csv(text: &str) -> Result<Vec<Row>> {
    let records = records(text);
    let mut it = records.into_iter();
    let header = it.next().context("llama-bench produced no CSV header")?;
    let mut out = Vec::new();
    for rec in it {
        if rec.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        if rec.len() != header.len() {
            bail!(
                "llama-bench CSV row has {} fields against a {}-field header — refusing to \
                 guess which column is which",
                rec.len(),
                header.len()
            );
        }
        out.push(Row(header.iter().cloned().zip(rec).collect()));
    }
    Ok(out)
}

fn records(text: &str) -> Vec<Vec<String>> {
    let mut recs = Vec::new();
    let mut rec: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            }
            (true, _) => field.push(c),
            (false, '"') => quoted = true,
            (false, ',') => rec.push(std::mem::take(&mut field)),
            (false, '\n') => {
                rec.push(std::mem::take(&mut field));
                recs.push(std::mem::take(&mut rec));
            }
            (false, '\r') => {}
            (false, _) => field.push(c),
        }
    }
    if !field.is_empty() || !rec.is_empty() {
        rec.push(field);
        recs.push(rec);
    }
    recs
}

/// One `(threads, depth)` configuration of the incumbent, measured over independent
/// invocations of the binary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IncumbentPoint {
    pub threads_requested: usize,
    /// Read back from the tool's own `n_threads` column, per invocation. A set with more than
    /// one member means the flag did not take on at least one of them.
    pub threads_reported: Vec<usize>,
    pub depth: u32,
    pub generated_tokens: u32,
    pub decode: Sample,
    /// True when the first test in each process was discarded as warm-up, per §6.
    pub warmup_discarded: bool,
}

/// Everything the incumbent contributed to the artefact.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IncumbentResult {
    pub binary: String,
    pub command: Vec<String>,
    pub facts: IncumbentFacts,
    pub points: Vec<IncumbentPoint>,
    /// The best configuration, per §1: the incumbent is quoted at its best, never its first.
    pub best_threads: Option<usize>,
}

/// How to invoke it.
#[derive(Debug, Clone)]
pub struct Invocation {
    pub binary: PathBuf,
    pub model: PathBuf,
    /// Thread counts to sweep. §1: *sweep the baseline's tuning knobs and quote its best
    /// configuration, not its first.*
    pub threads: Vec<usize>,
    pub depths: Vec<u32>,
    pub generated_tokens: u32,
    /// Repetitions inside one process, passed as `-r`.
    pub inner_repeats: u32,
    /// Independent invocations of the binary, per §5.
    pub invocations: usize,
    /// Extra arguments the caller needs, e.g. `-ncmoe 31`. Passed through verbatim and
    /// recorded in the artefact, because a tuning flag that is not in the record makes the
    /// number unreproducible.
    pub extra: Vec<String>,
    /// Discard the first test in each process, per §6.
    pub discard_warmup: bool,
}

impl Invocation {
    /// The argument vector for one `(threads, depth)` point.
    ///
    /// `-p 0` because prefill is not the question — `llama-bench` starts its timer after the
    /// `-d` prefill, which is what makes `tg` decode-only at depth and what makes it the same
    /// question MoEArc's harness is asked.
    pub fn args(&self, threads: usize, depth: u32) -> Vec<String> {
        let mut v = vec![
            "-m".into(),
            self.model.display().to_string(),
            "-p".into(),
            "0".into(),
            "-n".into(),
            self.generated_tokens.to_string(),
            "-t".into(),
            threads.to_string(),
            "-r".into(),
            self.inner_repeats.to_string(),
            "-o".into(),
            "csv".into(),
        ];
        // §6: the first test in a process pays warm-up. Asking for the same depth twice makes
        // the first row a throwaway that can be discarded deliberately rather than averaged in.
        if self.discard_warmup {
            v.push("-d".into());
            v.push(format!("{depth},{depth}"));
        } else {
            v.push("-d".into());
            v.push(depth.to_string());
        }
        v.extend(self.extra.iter().cloned());
        v
    }
}

/// Run the incumbent across its sweep. Returns the parsed result and the raw stdout of every
/// invocation, which the artefact keeps verbatim.
pub fn run(inv: &Invocation) -> Result<(IncumbentResult, String)> {
    if !inv.binary.exists() {
        bail!(
            "no llama-bench at {} — the incumbent's binary is never searched for. PROTOCOL §2: \
             a glob-ordered pick once selected a Vulkan build 4.8x slower than SYCL and it \
             looked fine.",
            inv.binary.display()
        );
    }
    let mut raw = String::new();
    let mut facts: Option<IncumbentFacts> = None;
    let mut points = Vec::new();
    let mut first_command = Vec::new();

    for &threads in &inv.threads {
        for &depth in &inv.depths {
            let args = inv.args(threads, depth);
            if first_command.is_empty() {
                first_command = args.clone();
            }
            let mut values = Vec::new();
            let mut reported: Vec<usize> = Vec::new();
            for i in 0..inv.invocations.max(1) {
                let out = Command::new(&inv.binary)
                    .args(&args)
                    .output()
                    .with_context(|| format!("running {}", inv.binary.display()))?;
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                raw.push_str(&format!(
                    "\n$ {} {}\n[invocation {}/{}]\n{}",
                    inv.binary.display(),
                    args.join(" "),
                    i + 1,
                    inv.invocations.max(1),
                    stdout
                ));
                if !out.status.success() {
                    bail!("llama-bench exited {} — refusing to use a partial result", out.status);
                }
                let rows = parse_csv(&stdout)?;
                let rows = discard_warmup_row(rows, inv.discard_warmup);
                if rows.is_empty() {
                    bail!("llama-bench produced no usable rows for -t {threads} -d {depth}");
                }
                if facts.is_none() {
                    facts = Some(facts_from(&inv.binary, &rows[0], threads));
                }
                for r in &rows {
                    if let Some(n) = r.num::<usize>("n_threads") {
                        if !reported.contains(&n) {
                            reported.push(n);
                        }
                    }
                    if let Some(ts) = r.num::<f64>("avg_ts") {
                        values.push(ts);
                    }
                }
            }
            points.push(IncumbentPoint {
                threads_requested: threads,
                threads_reported: reported,
                depth,
                generated_tokens: inv.generated_tokens,
                decode: Sample::new(
                    format!("llama.cpp decode, -t {threads}, depth {depth}"),
                    "tok/s",
                    values,
                ),
                warmup_discarded: inv.discard_warmup,
            });
        }
    }

    let best_threads = points
        .iter()
        .filter(|p| p.decode.n() > 0)
        .max_by(|a, b| a.decode.mean().total_cmp(&b.decode.mean()))
        .map(|p| p.threads_requested);

    let mut facts = facts.context("llama-bench produced no rows at all")?;
    // The pin is judged against every invocation, not the first: one process that ignored -t
    // is enough to invalidate the comparison.
    facts.threads.reported = single_reported(&points);

    Ok((
        IncumbentResult {
            binary: inv.binary.display().to_string(),
            command: first_command,
            facts,
            points,
            best_threads,
        },
        raw,
    ))
}

/// §6 — a tool's first test in a process pays warm-up.
fn discard_warmup_row(rows: Vec<Row>, discard: bool) -> Vec<Row> {
    // Only when there is something left afterwards. If llama-bench collapsed the duplicated
    // depth into one test, dropping it would leave nothing, and reporting nothing is worse
    // than reporting a warm-up-contaminated row that the artefact labels as such.
    if discard && rows.len() > 1 { rows[1..].to_vec() } else { rows }
}

/// The one thread count every invocation agreed on, or `None` if they disagreed.
///
/// `None` is a refusal in [`super::guard::threads`], which is the point: a sweep in which one
/// process silently ran on a different number of cores is not a comparison.
fn single_reported(points: &[IncumbentPoint]) -> Option<usize> {
    let mut all: Vec<usize> = Vec::new();
    for p in points {
        if p.threads_reported.len() != 1 || p.threads_reported[0] != p.threads_requested {
            return None;
        }
        all.push(p.threads_requested);
    }
    // A sweep over several thread counts legitimately reports several. The per-point check
    // above is what enforces §1; this returns the value only when there is a single one to
    // report, and the per-point table carries the rest.
    all.dedup();
    if all.len() == 1 { all.first().copied() } else { all.last().copied() }
}

fn facts_from(binary: &Path, row: &Row, requested: usize) -> IncumbentFacts {
    IncumbentFacts {
        binary: binary.display().to_string(),
        build_commit: row.get("build_commit").map(str::to_string),
        backends: row.get("backends").map(str::to_string),
        threads: ThreadPin { requested, reported: row.num::<usize>("n_threads") },
        model_filename: row.get("model_filename").map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `llama-bench -o csv` header and two rows, with the shapes that matter: a quoted field
    /// containing a comma (`cpu_info` really does on this box) and a `backends` column.
    const CSV: &str = "build_commit,build_number,cpu_info,gpu_info,backends,model_filename,\
model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,n_gpu_layers,n_prompt,n_gen,\
n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n\
\"e107984b\",\"6543\",\"Intel(R) Core(TM) Ultra 7 265K, 20 threads\",\"Intel(R) Arc(TM) B580\",\
\"SYCL\",\"gpt-oss-120b-MXFP4.gguf\",\"gptoss 120B MXFP4\",\"63387346208\",\"116829344256\",\
\"2048\",\"512\",\"16\",\"99\",\"0\",\"64\",\"512\",\"2026-09-06T10:00:00Z\",\"2402000000\",\
\"31000000\",\"26.650000\",\"0.330000\"\n\
\"e107984b\",\"6543\",\"Intel(R) Core(TM) Ultra 7 265K, 20 threads\",\"Intel(R) Arc(TM) B580\",\
\"SYCL\",\"gpt-oss-120b-MXFP4.gguf\",\"gptoss 120B MXFP4\",\"63387346208\",\"116829344256\",\
\"2048\",\"512\",\"16\",\"99\",\"0\",\"64\",\"512\",\"2026-09-06T10:01:00Z\",\"2260000000\",\
\"28000000\",\"28.240000\",\"0.560000\"\n";

    #[test]
    fn a_quoted_comma_does_not_shift_the_thread_column() {
        // The failure this parser exists to avoid: `cpu_info` contains ", 20 threads", so a
        // naive split reads `n_gpu_layers` where `n_threads` should be.
        let rows = parse_csv(CSV).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("n_threads"), Some("16"));
        assert_eq!(rows[0].get("backends"), Some("SYCL"));
        assert_eq!(rows[0].get("cpu_info").unwrap(), "Intel(R) Core(TM) Ultra 7 265K, 20 threads");
    }

    #[test]
    fn fields_are_indexed_by_name_not_by_position() {
        // A column added upstream must not silently change which number is the thread count.
        let reordered = "n_threads,backends,avg_ts\n\"4\",\"Vulkan\",\"3.20\"\n";
        let rows = parse_csv(reordered).unwrap();
        assert_eq!(rows[0].get("n_threads"), Some("4"));
        assert_eq!(rows[0].get("backends"), Some("Vulkan"));
    }

    #[test]
    fn a_row_that_does_not_match_the_header_is_an_error_not_a_guess() {
        let bad = "a,b,c\n1,2\n";
        assert!(parse_csv(bad).is_err());
    }

    #[test]
    fn the_thread_flag_and_the_read_back_are_kept_apart() {
        let rows = parse_csv(CSV).unwrap();
        // Ask for 16, and the CSV agrees.
        let f = facts_from(Path::new("/opt/llama-bench"), &rows[0], 16);
        assert_eq!(f.threads, ThreadPin { requested: 16, reported: Some(16) });
        // Ask for 16 against a process that ran 4 — §1's failure, and it is visible here.
        let f = facts_from(Path::new("/opt/llama-bench"), &rows[0], 4);
        assert_eq!(f.threads.requested, 4);
        assert_eq!(f.threads.reported, Some(16));
        assert_ne!(f.threads.requested, f.threads.reported.unwrap());
    }

    #[test]
    fn the_backend_is_taken_from_the_csv_and_not_from_the_path() {
        // §2: a Vulkan build in a directory named build-sycl is exactly the case that bit.
        let rows = parse_csv("backends,n_threads,avg_ts\n\"Vulkan\",\"16\",\"3.2\"\n").unwrap();
        let f = facts_from(Path::new("/opt/llama.cpp/build-sycl/bin/llama-bench"), &rows[0], 16);
        assert_eq!(f.backends.as_deref(), Some("Vulkan"));
    }

    #[test]
    fn the_command_line_always_pins_threads_and_never_prefills() {
        let inv = Invocation {
            binary: PathBuf::from("/opt/llama-bench"),
            model: PathBuf::from("/m.gguf"),
            threads: vec![16],
            depths: vec![512],
            generated_tokens: 64,
            inner_repeats: 5,
            invocations: 3,
            extra: vec!["-ncmoe".into(), "31".into()],
            discard_warmup: true,
        };
        let args = inv.args(16, 512);
        let joined = args.join(" ");
        assert!(joined.contains("-t 16"), "{joined}");
        // -p 0: prefill is excluded from the timer on both sides.
        assert!(joined.contains("-p 0"), "{joined}");
        // The warm-up row is created deliberately so it can be discarded deliberately.
        assert!(joined.contains("-d 512,512"), "{joined}");
        assert!(joined.contains("-ncmoe 31"), "{joined}");
        assert!(joined.contains("-o csv"), "{joined}");
    }

    #[test]
    fn the_warmup_row_is_dropped_only_when_something_survives() {
        let rows = parse_csv(CSV).unwrap();
        assert_eq!(discard_warmup_row(rows.clone(), true).len(), 1);
        assert_eq!(discard_warmup_row(rows.clone(), false).len(), 2);
        // One row and a request to discard: keep it rather than report nothing.
        assert_eq!(discard_warmup_row(vec![rows[0].clone()], true).len(), 1);
    }

    #[test]
    fn a_point_whose_processes_disagreed_about_threads_reports_no_single_value() {
        let p = IncumbentPoint {
            threads_requested: 16,
            threads_reported: vec![16, 4],
            depth: 0,
            generated_tokens: 64,
            decode: Sample::new("x", "tok/s", vec![1.0]),
            warmup_discarded: false,
        };
        assert_eq!(single_reported(&[p]), None);
    }
}
