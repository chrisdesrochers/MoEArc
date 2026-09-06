//! The plain-text and JSON renderers — the same information as the interface, in a form a
//! script can consume.
//!
//! This exists because of one line in `docs/ux.md`: *nothing may be reachable only through
//! the TUI*. Treating that as a real constraint means this file has to stay at parity with
//! `tui/view.rs`, so the two are deliberately written from the same data and the same
//! formatters — a fact rendered here that the interface does not show (or the reverse) is a
//! defect, not a feature.
//!
//! It also carries the honesty rules. Where a subsystem is not wired up, these commands say
//! so and exit non-zero rather than printing a plausible success.

use std::process::ExitCode;

use anyhow::Result;
use moearc_engine::host_budget::{
    BudgetPolicy, BudgetSource, HostBudget, ModelBytes, Placement, Tier, place,
};
use serde_json::{Value, json};

use crate::cli::{Cli, Command, InfoArgs, LsArgs, PullArgs, ServeArgs};
use crate::fit::{self, Fit, FitOutcome};
use crate::format;
use crate::source::{DeviceRow, HostReport, ModelCard, Sources};

/// Exit code for a command whose backend does not exist yet.
///
/// Distinct from a plain failure so a script can tell "you asked for something impossible"
/// from "this part of MoEArc is not built". Both are non-zero, because a command that did
/// nothing must never look like a command that worked.
const EXIT_NOT_WIRED: u8 = 2;

pub fn run(cli: &Cli, sources: &Sources) -> Result<ExitCode> {
    match &cli.command {
        None => report(cli, sources),
        Some(Command::Ls(args)) => ls(cli, sources, args),
        Some(Command::Pull(args)) => pull(cli, sources, args),
        Some(Command::Serve(args)) => serve(cli, sources, args),
        Some(Command::Info(args)) => info(cli, sources, args),
    }
}

// ---------------------------------------------------------------------------------------
// moearc
// ---------------------------------------------------------------------------------------

fn report(cli: &Cli, sources: &Sources) -> Result<ExitCode> {
    let devices = sources.devices.detect()?;
    let models = sources.models.curated()?;
    let fits: Vec<Fit> = match devices.primary() {
        Some(d) => models.iter().map(|m| fit::plan(d, m, cli.ctx)).collect(),
        None => Vec::new(),
    };
    let host = sources.host.probe()?;
    let budget = budget_for(cli, &host);
    let placements = placements_for(&models, &host, budget);

    if cli.global.json {
        return emit(json!({
            "source": provenance(sources),
            "devices": devices.devices,
            "verdict": devices.verdict,
            "requested_ctx": cli.ctx,
            "calibrated": false,
            "host": host_json(&host, budget),
            "models": models,
            "fits": fits,
            "placements": models
                .iter()
                .zip(&placements)
                .map(|(m, p)| placement_json(&m.id, p))
                .collect::<Vec<_>>(),
            "unreadable": sources.models.skipped(),
        }));
    }

    section("Devices");
    for (i, d) in devices.devices.iter().enumerate() {
        let marker = if Some(i) == devices.devices.iter().position(DeviceRow::is_inference_target) {
            "▸"
        } else {
            " "
        };
        println!(
            "  {marker} {:<32} {:<12} {:<22} {:>10} / {:<10}",
            d.name,
            d.backend.label(),
            d.driver,
            format::bytes(d.free_bytes),
            format::bytes(d.total_bytes)
        );
        if cli.global.verbose >= 2 {
            println!("      free={} total={} bytes", d.free_bytes, d.total_bytes);
        }
    }

    println!();
    let mark = if devices.verdict.is_ready() { "✓" } else { "✗" };
    println!("  {mark} {}", devices.verdict.headline());
    if let Some(remedy) = devices.verdict.remedy() {
        println!("    {remedy}");
    }

    print_host(&host, budget);

    if models.is_empty() {
        no_models(sources);
    } else if !fits.is_empty() {
        section(&match cli.ctx {
            Some(ctx) => format!("What will fit at {} ctx", format::count(ctx as i64)),
            None => "What will fit".to_string(),
        });
        let cols = fit::Columns::of(&models);
        print_fit_header(cols);
        for ((card, f), p) in models.iter().zip(&fits).zip(&placements) {
            print_fit_row(card, f, Some(p), cols, cli.global.verbose);
        }
        println!();
        if placements.iter().any(|p| p.tier == Tier::RunsPagesFromDisk) {
            println!(
                "  note: past the budget is not past the machine — the excess is paged in from \
                 the drive as it is needed, which is slower and is not a failure."
            );
        }
        println!(
            "  note: residency and context are computed from this card's free VRAM. The \
             headroom behind them is provisional, not measured on Arc."
        );
        if fits.iter().any(Fit::context_at_floor) {
            println!(
                "  note: a context at the minimum is not the card's limit — experts took \
                 everything above it. `--ctx <tokens>` trades slots back for context."
            );
            // Measured; see the same note in `tui::view::fit_panel` for the trace it comes from.
            println!(
                "  note: measured on a real Qwen3-30B routing trace, only 1.4% of its 6,144 \
                 expert slots are touched on more than half of tokens, and every KV byte is \
                 touched on all of them."
            );
        }
    }
    print_unreadable(sources);
    print_provenance(sources);
    Ok(ExitCode::SUCCESS)
}

/// What to say when the model directory is empty.
///
/// Naming the directory is the whole point. "No models found" is the error `docs/ux.md` rules
/// out — a symptom with no cause — and the cause here is almost always that the files are
/// somewhere else.
fn no_models(sources: &Sources) {
    section("What will fit");
    match sources.models.location() {
        Some(dir) => {
            println!("  Nothing to plan: no GGUF files in {dir}");
            println!("  Point moearc at them with --models-dir <DIR>, or set MOEARC_MODELS.");
        }
        None => println!("  No models in the catalog."),
    }
}

/// The table's heading row. Present here for the same reason it is present in the interface:
/// the cells hold bare numbers, so something has to say what they are.
fn print_fit_header(cols: fit::Columns) {
    println!(
        "    {:<id$} {:<quant$} {:>size$}  {:<tier$}  {:<res$} {:>ctx$}",
        "model",
        "quant",
        "size",
        "host",
        "residency",
        "ctx",
        id = cols.id,
        quant = cols.quant,
        size = cols.size,
        tier = fit::Columns::TIER,
        res = cols.residency,
        ctx = cols.ctx
    );
}

fn print_fit_row(
    card: &ModelCard,
    f: &Fit,
    placement: Option<&Placement>,
    cols: fit::Columns,
    verbose: u8,
) {
    let mark = if f.fits() { "✓" } else { "·" };
    println!(
        "  {mark} {:<id$} {:<quant$} {:>size$}  {:<tier$}  {:<res$} {:>ctx$}",
        card.id,
        card.quant,
        format::bytes(card.file_bytes),
        placement.map_or("", |p| p.tier.label()),
        f.residency_cell(),
        f.context_cell(),
        id = cols.id,
        quant = cols.quant,
        size = cols.size,
        tier = fit::Columns::TIER,
        res = cols.residency,
        ctx = cols.ctx
    );
    if let Some(p) = placement
        && verbose >= 1
    {
        println!("      {}", p.reason);
    }
    if verbose >= 1 {
        match &f.outcome {
            FitOutcome::Fits { rationale, ceiling_tokens, .. } => {
                for step in rationale {
                    println!("      {step}");
                }
                if let Some(c) = ceiling_tokens {
                    println!("      {} tokens at minimum residency", format::count(*c as i64));
                }
            }
            FitOutcome::DoesNotFit { reason, .. } => println!("      {reason}"),
        }
    }
}

// ---------------------------------------------------------------------------------------
// moearc ls
// ---------------------------------------------------------------------------------------

fn ls(cli: &Cli, sources: &Sources, args: &LsArgs) -> Result<ExitCode> {
    let mut models =
        if args.all { sources.models.curated()? } else { sources.models.installed()? };
    if args.measured {
        models.retain(|m| m.measured);
    }

    if cli.global.json {
        return emit(json!({
            "source": provenance(sources),
            "models": models,
            "unreadable": sources.models.skipped(),
        }));
    }

    if models.is_empty() {
        if args.measured {
            // Not the same as an empty directory, and saying so matters: every model here is
            // read from a header, so `--measured` is empty until something has actually run.
            println!("No model here has been run on an Arc card yet.");
        } else {
            match sources.models.location() {
                Some(dir) => println!("No GGUF files in {dir}. Try --models-dir <DIR>."),
                None => println!("No models here yet."),
            }
        }
        print_unreadable(sources);
        print_provenance(sources);
        return Ok(ExitCode::SUCCESS);
    }

    // "Curated" only when the list actually reaches past this machine. There is no curated
    // remote registry yet, so with `--all` the honest heading is still the local one.
    section(if models.iter().any(|m| !m.local) {
        "Curated models"
    } else {
        "Models on this machine"
    });
    let cols = fit::Columns::of(&models);
    for m in &models {
        // Three states, three glyphs: run on Arc, present but never run, and not here at all.
        // docs/ux.md is explicit that the second must not look like the first.
        let mark = match (m.local, m.measured) {
            (_, true) => "✓",
            (true, false) => "~",
            (false, false) => "·",
        };
        println!(
            "  {mark} {:<id$} {:<quant$} {:>size$}  {:<7} {}",
            m.id,
            m.quant,
            format::bytes(m.file_bytes),
            if m.local { "local" } else { "remote" },
            m.origin(),
            id = cols.id,
            quant = cols.quant,
            size = cols.size
        );
    }
    println!();
    println!("  ✓ measured on an Arc card   ~ present, never run   · not downloaded");
    print_unreadable(sources);
    print_provenance(sources);
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------------------
// moearc pull
// ---------------------------------------------------------------------------------------

fn pull(cli: &Cli, sources: &Sources, args: &PullArgs) -> Result<ExitCode> {
    let plan = sources.transfers.plan(&args.model)?;

    if cli.global.json {
        return emit(json!({
            "source": provenance(sources),
            "transfer": plan,
            "started": false,
            "reason": "the downloader is not wired up yet",
        }))
        .map(|_| ExitCode::from(EXIT_NOT_WIRED));
    }

    section("Pull");
    println!("  {:<18}{}", "model", args.model);
    println!("  {:<18}{}", "repo", plan.repo);
    println!("  {:<18}{}", "download", format::bytes(plan.total_bytes));
    if plan.resume_from > 0 {
        println!("  {:<18}{}", "already here", format::bytes(plan.resume_from));
    }
    println!();
    // Saying "would download" and exiting 0 would be a lie a CI job cannot detect.
    println!("  not wired yet: downloads arrive with moearc-model. Nothing was fetched.");
    print_provenance(sources);
    Ok(ExitCode::from(EXIT_NOT_WIRED))
}

// ---------------------------------------------------------------------------------------
// moearc serve
// ---------------------------------------------------------------------------------------

fn serve(cli: &Cli, sources: &Sources, args: &ServeArgs) -> Result<ExitCode> {
    let card = sources.models.resolve(&args.model)?;
    let devices = sources.devices.detect()?;
    let Some(device) = devices.primary() else {
        anyhow::bail!("{}", devices.verdict.headline());
    };
    let plan = match args.moe_cache {
        Some(slots) => fit::plan_with_slot_override(device, &card, args.ctx, slots),
        None => fit::plan(device, &card, args.ctx),
    };

    if cli.global.json {
        let code = if plan.fits() && !args.dry_run {
            ExitCode::from(EXIT_NOT_WIRED)
        } else {
            ExitCode::SUCCESS
        };
        emit(json!({
            "source": provenance(sources),
            "model": card,
            "device": device,
            "host": args.host,
            "port": args.port,
            "plan": plan,
            "started": false,
        }))?;
        return Ok(code);
    }

    section("Serve");
    println!("  {:<18}{}", "model", card.id);
    println!("  {:<18}{}", "device", device.name);
    println!("  {:<18}http://{}:{}/v1", "endpoint", args.host, args.port);
    println!("  {:<18}{}", "requested ctx", requested(args.ctx));
    println!();
    print_plan(&plan, device, cli.global.verbose);

    if !plan.fits() {
        return Ok(ExitCode::from(EXIT_NOT_WIRED));
    }
    if args.dry_run {
        println!();
        println!("  --dry-run: nothing was started.");
        print_provenance(sources);
        return Ok(ExitCode::SUCCESS);
    }
    println!();
    println!(
        "  not wired yet: the inference server arrives with the engine. Nothing is listening."
    );
    print_provenance(sources);
    Ok(ExitCode::from(EXIT_NOT_WIRED))
}

/// The "what it decided" block. `docs/ux.md`: startup prints its reasoning, so a user can see
/// it without turning on debug logging.
fn print_plan(plan: &Fit, device: &DeviceRow, verbose: u8) {
    match &plan.outcome {
        FitOutcome::Fits {
            resident_experts,
            total_experts,
            context_tokens,
            kv_pages,
            expert_bytes,
            kv_bytes,
            headroom_bytes,
            ceiling_tokens,
            context_at_floor,
            // Already the last line of `rationale` below, in the planner's own words. Pulled
            // out for the interface's table, where there is no room for the full reasoning.
            yield_note: _,
            rationale,
        } => {
            println!("  Planned split");
            println!(
                "  {:<18}{} / {}  ({})",
                "residency",
                format::count(*resident_experts as i64),
                format::count(*total_experts as i64),
                format::percent(*resident_experts as i64, *total_experts as i64)
            );
            println!(
                "  {:<18}{} tokens · {} pages{}",
                "context",
                format::count(*context_tokens as i64),
                format::count(*kv_pages as i64),
                // Said on the line itself, not only in the reasoning below it. This is the
                // number a user copies into a client config.
                if *context_at_floor {
                    "   (the configured minimum, not this card's limit)"
                } else {
                    ""
                }
            );
            if let Some(c) = ceiling_tokens {
                println!(
                    "  {:<18}{} tokens at minimum residency",
                    "ceiling",
                    format::count(*c as i64)
                );
            }
            if let Some(slots) = plan.slot_override {
                println!("  {:<18}--moe-cache {slots} (computed value overridden)", "override");
            }
            println!();
            // Unconditional, not behind -v. docs/ux.md: startup prints what it decided, so a
            // user can see the reasoning without turning on debug logging.
            println!("  Why");
            for step in rationale {
                println!("  · {step}");
            }
            if verbose >= 1 {
                println!();
                println!("  {:<18}{} B free on {}", "measured", device.free_bytes, device.name);
                println!("  {:<18}{} B", "experts", expert_bytes);
                println!("  {:<18}{} B", "kv cache", kv_bytes);
                println!(
                    "  {:<18}{} B (provisional, not measured on Arc)",
                    "headroom", headroom_bytes
                );
            }
        }
        FitOutcome::DoesNotFit { headline, reason } => {
            println!("  ✗ {headline}");
            println!("    {reason}");
        }
    }
}

// ---------------------------------------------------------------------------------------
// moearc info
// ---------------------------------------------------------------------------------------

fn info(cli: &Cli, sources: &Sources, args: &InfoArgs) -> Result<ExitCode> {
    let card = sources.models.resolve(&args.model)?;
    let devices = sources.devices.detect()?;
    let plan = devices.primary().map(|d| fit::plan(d, &card, args.ctx));
    let host = sources.host.probe()?;
    let budget = budget_for(cli, &host);
    let placement = placements_for(std::slice::from_ref(&card), &host, budget).remove(0);

    if cli.global.json {
        return emit(json!({
            "source": provenance(sources),
            "model": card,
            "requested_ctx": args.ctx,
            "host": host_json(&host, budget),
            "placement": placement_json(&card.id, &placement),
            "plan": plan,
        }));
    }

    section(&card.id);
    if let Some(repo) = &card.repo {
        println!("  {:<18}{repo}", "repo");
    }
    if let Some(file) = &card.file {
        println!("  {:<18}{file}", "file");
    }
    println!("  {:<18}{}", "quantisation", card.quant);
    println!("  {:<18}{} total", "parameters", card.params());
    println!("  {:<18}{}", "download", format::bytes(card.file_bytes));
    println!("  {:<18}{}", "experts", card.experts());
    println!("  {:<18}{}", "residency slots", card.slots());
    println!(
        "  {:<18}{}{}",
        "per slot",
        format::bytes(card.per_expert_bytes),
        if card.per_expert_bytes_uniform {
            ""
        } else {
            "  (a maximum — this file quantises some blocks' experts differently)"
        }
    );
    println!(
        "  {:<18}{} tokens",
        "trained context",
        format::count(card.trained_context_tokens as i64)
    );
    println!(
        "  {:<18}{}",
        "footprint",
        if card.measured { "measured on Arc" } else { "from header, never run" }
    );
    println!("  {:<18}{}", "present here", if card.local { "yes" } else { "no" });
    if cli.global.verbose >= 1 {
        println!("  {:<18}{} B", "dense weights", card.dense_weights_bytes);
        println!("  {:<18}{} B", "per slot", card.per_expert_bytes);
        println!("  {:<18}{} B/token", "kv cache", card.kv_bytes_per_token);
    }
    println!("  {:<18}{}", "requested ctx", requested(args.ctx));
    println!();
    print_host(&host, budget);
    println!("  {:<18}{}", "host tier", placement.tier.label());
    println!("  {:<18}{}", "", placement.reason);
    println!();
    match (&plan, devices.primary()) {
        (Some(p), Some(d)) => print_plan(p, d, cli.global.verbose),
        _ => println!("  no device to plan against — {}", devices.verdict.headline()),
    }
    print_provenance(sources);
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------------------

/// How a context request reads when the user did not make one.
///
/// "largest that fits" rather than a number, because printing a default here would make an
/// assumption look like a request the user typed.
fn requested(ctx: Option<u32>) -> String {
    match ctx {
        Some(c) => format!("{} tokens", format::count(c as i64)),
        None => "largest that fits".to_string(),
    }
}

/// The budget this invocation is planning against.
///
/// One place, so the plain renderer and the interface cannot answer the same question
/// differently. The clamp is the engine's, not this crate's.
fn budget_for(cli: &Cli, host: &HostReport) -> HostBudget {
    let memory = crate::host::memory(host);
    let policy = BudgetPolicy::default();
    match cli.host_budget() {
        Some(want) => HostBudget::requested(memory, &policy, want),
        None => HostBudget::default_for(memory, &policy),
    }
}

fn placements_for(models: &[ModelCard], host: &HostReport, budget: HostBudget) -> Vec<Placement> {
    let storage = crate::host::storage(host);
    models
        .iter()
        .map(|c| {
            place(ModelBytes { weights_bytes: c.file_bytes, on_disk: c.local }, budget, storage)
        })
        .collect()
}

/// The host tier, printed before the card's plan.
///
/// 🔴 First, deliberately. On this engine the card is not what decides whether a model can run
/// — a 59 GiB model runs on an 11.33 GiB card — so the number that actually predicts how a
/// model will feel is how much of it the host can hold.
fn print_host(host: &HostReport, budget: HostBudget) {
    section("Host RAM");
    println!(
        "  {:<18}{} of {} usable",
        "budget",
        format::bytes(budget.bytes()),
        format::bytes(budget.max_bytes())
    );
    println!(
        "  {:<18}{} available of {} fitted",
        "memory",
        format::bytes(host.available_bytes),
        format::bytes(host.total_bytes)
    );
    println!("  {:<18}{} kept for the machine", "reserved", format::bytes(budget.reserved_bytes()));
    if let BudgetSource::Clamped { asked } = budget.source() {
        println!(
            "  {:<18}{} was asked for; this machine does not have it available",
            "clamped",
            format::bytes(asked)
        );
    }
    if host.models_free_bytes != u64::MAX {
        println!("  {:<18}{} free", "model storage", format::bytes(host.models_free_bytes));
    }
}

fn host_json(host: &HostReport, budget: HostBudget) -> Value {
    let (source, asked) = match budget.source() {
        BudgetSource::Default => ("default", None),
        BudgetSource::Requested => ("requested", None),
        BudgetSource::Clamped { asked } => ("clamped", Some(asked)),
    };
    json!({
        "total_bytes": host.total_bytes,
        "available_bytes": host.available_bytes,
        // Absent rather than `u64::MAX` when it could not be measured: a sentinel in a JSON
        // payload is a number to a script, and this one would read as an infinite drive.
        "models_free_bytes": (host.models_free_bytes != u64::MAX).then_some(host.models_free_bytes),
        "budget_bytes": budget.bytes(),
        "budget_max_bytes": budget.max_bytes(),
        "reserved_bytes": budget.reserved_bytes(),
        "budget_source": source,
        "budget_asked_bytes": asked,
    })
}

fn placement_json(model: &str, p: &Placement) -> Value {
    json!({
        "model": model,
        "tier": match p.tier {
            Tier::RunsFromRam => "runs_from_ram",
            Tier::RunsPagesFromDisk => "runs_pages_from_disk",
            Tier::WillNotFit => "will_not_fit",
        },
        "runs": p.tier.runs(),
        "ram_bytes": p.ram_bytes,
        "disk_bytes": p.disk_bytes,
        "reason": p.reason.to_string(),
    })
}

fn section(title: &str) {
    println!();
    println!("{title}");
    println!();
}

fn emit(value: Value) -> Result<ExitCode> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(ExitCode::SUCCESS)
}

/// Where the numbers in this payload came from.
///
/// 🔴 An object rather than the word `"stub"`, for the same reason the interface footer stopped
/// saying "stub data": with devices and models read off the machine, a single flag covering the
/// whole payload labels measurements as fixtures. A script deciding whether to trust a number
/// needs to know *which* number, and the shape below is the smallest thing that answers it.
fn provenance(sources: &Sources) -> Value {
    if sources.stubbed {
        json!({ "measured": false, "fixtures": sources.stub_parts, "note": sources.stub_note })
    } else {
        json!({ "measured": true })
    }
}

/// Files in the model directory that could not be read.
///
/// Printed rather than dropped. A truncated download and a model that was never fetched leave
/// the same empty row, and "no models found" would send a user looking in the wrong place.
fn print_unreadable(sources: &Sources) {
    let skipped = sources.models.skipped();
    if skipped.is_empty() {
        return;
    }
    println!();
    println!("  could not read {} file(s) in the model directory:", skipped.len());
    for line in &skipped {
        println!("    {line}");
    }
}

/// Repeated at the end of every command on purpose. A reader who only sees the last screen of
/// a long output still learns which numbers are fixtures.
fn print_provenance(sources: &Sources) {
    if sources.stubbed {
        println!();
        println!("  note: {}.", sources.stub_note);
    }
}
