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
use serde_json::{Value, json};

use crate::cli::{Cli, Command, InfoArgs, LsArgs, PullArgs, ServeArgs};
use crate::fit::{self, Fit, FitOutcome};
use crate::format;
use crate::source::{DeviceRow, ModelCard, Sources};

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

    if cli.global.json {
        return emit(json!({
            "source": provenance(sources),
            "devices": devices.devices,
            "verdict": devices.verdict,
            "requested_ctx": cli.ctx,
            "calibrated": false,
            "fits": fits,
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

    if !fits.is_empty() {
        section(&match cli.ctx {
            Some(ctx) => format!("What will fit at {} ctx", format::count(ctx as i64)),
            None => "What will fit".to_string(),
        });
        for (card, f) in models.iter().zip(&fits) {
            print_fit_row(card, f, cli.global.verbose);
        }
        println!();
        println!(
            "  note: residency and context are computed from this card's free VRAM. The \
             headroom behind them is provisional, not measured on Arc."
        );
    }
    print_provenance(sources);
    Ok(ExitCode::SUCCESS)
}

fn print_fit_row(card: &ModelCard, f: &Fit, verbose: u8) {
    let mark = if f.fits() { "✓" } else { "·" };
    println!(
        "  {mark} {:<22} {:<9} {:>10}   {}",
        card.id,
        card.quant,
        format::bytes(card.file_bytes),
        f.summary()
    );
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
            FitOutcome::DoesNotFit { reason } => println!("      {reason}"),
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
        return emit(json!({ "source": provenance(sources), "models": models }));
    }

    if models.is_empty() {
        println!("No models here yet. `moearc ls --all` lists the curated ones.");
        return Ok(ExitCode::SUCCESS);
    }

    section(if args.all { "Curated models" } else { "Models on this machine" });
    for m in &models {
        // Three states, three glyphs: run on Arc, present but never run, and not here at all.
        // docs/ux.md is explicit that the second must not look like the first.
        let mark = match (m.local, m.measured) {
            (_, true) => "✓",
            (true, false) => "~",
            (false, false) => "·",
        };
        println!(
            "  {mark} {:<22} {:<9} {:>10}  {:<9} {}",
            m.id,
            m.quant,
            format::bytes(m.file_bytes),
            if m.local { "local" } else { "remote" },
            m.repo
        );
    }
    println!();
    println!("  ✓ measured on an Arc card   ~ present, never run   · not downloaded");
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
            rationale,
        } => {
            println!("  Planned split");
            println!(
                "  {:<18}{resident_experts} / {total_experts}  ({})",
                "residency",
                format::percent(*resident_experts as i64, *total_experts as i64)
            );
            println!(
                "  {:<18}{} tokens · {} pages",
                "context",
                format::count(*context_tokens as i64),
                format::count(*kv_pages as i64)
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
        FitOutcome::DoesNotFit { reason } => {
            println!("  ✗ will not fit on this card");
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

    if cli.global.json {
        return emit(json!({
            "source": provenance(sources),
            "model": card,
            "requested_ctx": args.ctx,
            "plan": plan,
        }));
    }

    section(&card.id);
    println!("  {:<18}{}", "repo", card.repo);
    println!("  {:<18}{}", "quantisation", card.quant);
    println!("  {:<18}{} (total / active)", "parameters", card.params());
    println!("  {:<18}{}", "download", format::bytes(card.file_bytes));
    println!(
        "  {:<18}{} of {} routed per token",
        "experts", card.experts_active, card.experts_total
    );
    println!(
        "  {:<18}{}",
        "footprint",
        if card.measured { "measured on Arc" } else { "from header, never run" }
    );
    println!("  {:<18}{}", "present here", if card.local { "yes" } else { "no" });
    if cli.global.verbose >= 1 {
        println!("  {:<18}{} B", "dense weights", card.dense_weights_bytes);
        println!("  {:<18}{} B", "per expert", card.per_expert_bytes);
        println!("  {:<18}{} B/token", "kv cache", card.kv_bytes_per_token);
    }
    println!("  {:<18}{}", "requested ctx", requested(args.ctx));
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

fn section(title: &str) {
    println!();
    println!("{title}");
    println!();
}

fn emit(value: Value) -> Result<ExitCode> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(ExitCode::SUCCESS)
}

fn provenance(sources: &Sources) -> &'static str {
    if sources.stubbed { "stub" } else { "measured" }
}

/// Repeated at the end of every command on purpose. A reader who only sees the last screen of
/// a long output still learns the numbers are fixtures.
fn print_provenance(sources: &Sources) {
    if sources.stubbed {
        println!();
        println!("  note: {}.", sources.stub_note);
    }
}
