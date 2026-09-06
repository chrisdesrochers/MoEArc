//! Rendering. A pure function of [`Model`] — it reads no state of its own and mutates
//! nothing, which is what lets the tests at the bottom render every screen into a
//! `TestBackend` and assert on the text that comes out.
//!
//! Those snapshots are the closest thing a terminal interface has to a screenshot, and they
//! are the reason this file has no hidden animation state: anything that varied with wall
//! time would make them flap.

use moearc_engine::host_budget::{BudgetSource, HostBudget, Placement, Tier};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Cell, Clear, Gauge, LineGauge, List, ListItem, ListState, Paragraph, Row, Sparkline, Table,
    Wrap,
};
use throbber_widgets_tui::{Throbber, symbols::throbber};

use super::model::{Model, Screen};
use crate::fit::{CONTEXT_LADDER, Columns, Fit, FitOutcome, KvPrecision, ladder_index};
use crate::format;
use crate::source::{DeviceRow, ModelCard, Verdict};
use crate::theme;

/// Label column width. One value for the whole interface, so fields stacked in different
/// panels still line up when the panels sit side by side.
const LABEL_W: usize = 16;

pub fn view(m: &Model, f: &mut Frame) {
    let [header, body, footer] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0), Constraint::Length(2)])
        .areas(f.area());

    f.render_widget(header_line(m), header);
    match m.screen {
        Screen::Devices => devices_screen(m, f, body),
        Screen::Models => models_screen(m, f, body),
        Screen::Download => download_screen(m, f, body),
        Screen::Serving => serving_screen(m, f, body),
    }
    f.render_widget(footer_line(m), footer);

    if m.help {
        help_overlay(f, body);
    }
}

// ---------------------------------------------------------------------------------------
// Chrome
// ---------------------------------------------------------------------------------------

fn header_line(m: &Model) -> Paragraph<'_> {
    let mut spans = vec![
        Span::styled("  moearc", Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("   ", theme::subtle()),
    ];
    for (i, s) in m.screens().iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", Style::new().fg(theme::FAINT)));
        }
        spans.push(if *s == m.screen {
            Span::styled(s.title(), Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD))
        } else {
            Span::styled(s.title(), theme::subtle())
        });
    }
    Paragraph::new(Line::from(spans))
}

fn footer_line(m: &Model) -> Paragraph<'_> {
    let keys: &[(&str, &str)] = match m.screen {
        Screen::Devices => &[
            ("↑↓", "select"),
            ("-/+", "RAM"),
            ("[/]", "context"),
            ("r", "rescan"),
            ("⏎", "models"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Screen::Models if m.editing => &[("⏎", "pull"), ("esc", "cancel")],
        Screen::Models => &[
            ("↑↓", "select"),
            ("-/+", "RAM"),
            ("[/]", "context"),
            ("/", "repo id"),
            ("d", "pull"),
            ("s", "serve"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Screen::Download => &[("x", "cancel"), ("esc", "back"), ("?", "help"), ("q", "quit")],
        Screen::Serving => &[("s", "stop"), ("esc", "back"), ("?", "help"), ("q", "quit")],
    };

    let mut spans = vec![Span::raw("  ")];
    for (i, (key, what)) in keys.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", theme::subtle()));
        }
        spans.push(Span::styled(*key, Style::new().fg(theme::ACCENT)));
        spans.push(Span::styled(format!(" {what}"), theme::subtle()));
    }
    if let Some(status) = &m.status {
        spans.push(Span::styled(format!("   ⚠ {status}"), Style::new().fg(theme::BAD)));
    } else if let Some(parts) = m.provenance {
        // Provenance, permanently visible — and *specific*. It used to read "stub data" for
        // everything, which was right when everything was. Now that the device table and the
        // model list are read off this machine, a blanket marker would label measurements as
        // fixtures, and under-claiming is no more honest than over-claiming.
        spans.push(Span::styled(format!("   {parts}"), Style::new().fg(theme::FAINT)));
    }
    Paragraph::new(Line::from(spans))
}

fn spinner<'a>(m: &Model, label: &'a str) -> Line<'a> {
    Throbber::default()
        .label(label)
        .style(theme::subtle())
        .throbber_style(theme::accent())
        .throbber_set(throbber::BRAILLE_SIX)
        .to_line(&m.throbber)
}

// ---------------------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------------------

fn devices_screen(m: &Model, f: &mut Frame, area: Rect) {
    let Some(report) = &m.report else {
        let block = theme::panel("Devices");
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(vec![
                spinner(m, " Looking for a Level Zero device…"),
                Line::raw(""),
                Line::styled(
                    "Enumerating through the bundled runtime, so a shell that never sourced \
                     oneAPI is not mistaken for a dead GPU.",
                    theme::subtle(),
                ),
            ])
            .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    };

    // Chrome around a panel's content: two border rows and two padding rows.
    const CHROME: u16 = 4;
    // The table also spends a row on its header and one on the header's bottom margin. Sizing
    // this by hand is how the CPU row got clipped, and that row is the evidence that tells an
    // unsourced runtime apart from a dead card — losing it removes a diagnosis, not a line.
    let table_height = report.devices.len() as u16 + CHROME + 2;
    let [table_area, verdict_area, fit_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(table_height),
            Constraint::Length(verdict_height(&report.verdict, area.width) + CHROME),
            Constraint::Min(0),
        ])
        .areas(area);

    let header = Row::new(["", "Device", "Backend", "Driver", "VRAM free / total"])
        .style(theme::subtle())
        .bottom_margin(1);
    let rows = report.devices.iter().enumerate().map(|(i, d)| device_row(d, i == m.device_row));
    let widths = [
        Constraint::Length(2),
        Constraint::Min(24),
        Constraint::Length(12),
        Constraint::Length(22),
        Constraint::Length(22),
    ];
    f.render_widget(
        Table::new(rows, widths).header(header).block(theme::panel("Devices")),
        table_area,
    );

    f.render_widget(verdict_panel(&report.verdict), verdict_area);
    f.render_widget(fit_panel(m, fit_area), fit_area);
}

fn device_row(d: &DeviceRow, selected: bool) -> Row<'_> {
    // The selection marker is a column rather than a highlight style: the plain-text renderer
    // has no colour to fall back on, and the two outputs should read the same way.
    let marker = if selected { "▸" } else { " " };
    let name_style = if d.is_inference_target() {
        theme::text()
    } else {
        // Not dimmed to hide it — a CPU device is the diagnostic that distinguishes an
        // unsourced runtime from a missing driver, so it stays visible and merely recedes.
        theme::subtle()
    };
    Row::new(vec![
        Cell::from(marker).style(theme::accent()),
        Cell::from(d.name.as_str()).style(name_style),
        Cell::from(d.backend.label()).style(theme::subtle()),
        Cell::from(d.driver.as_str()).style(theme::subtle()),
        Cell::from(format!("{} / {}", format::bytes(d.free_bytes), format::bytes(d.total_bytes))),
    ])
}

/// Rows the verdict's prose needs at this width.
///
/// Estimated rather than measured: ratatui wraps at word boundaries and does not expose the
/// resulting line count without an unstable feature. The estimate is deliberately generous —
/// a blank row inside a panel costs nothing, whereas one row too few truncates the remedy
/// mid-sentence, which is the "symptom without a cause" failure docs/ux.md rules out.
fn verdict_height(v: &Verdict, width: u16) -> u16 {
    // Two border columns and two padding columns on each side.
    let inner = width.saturating_sub(6).max(1) as usize;
    let rows = |s: &str| (s.chars().count().div_ceil(inner) + 1) as u16;
    let mut h = rows(&v.headline());
    if let Some(remedy) = v.remedy() {
        h += 1 + rows(remedy);
    }
    h
}

fn verdict_panel(v: &Verdict) -> Paragraph<'_> {
    let (mark, style) = if v.is_ready() {
        ("✓", Style::new().fg(theme::GOOD))
    } else {
        ("✗", Style::new().fg(theme::BAD))
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(format!("{mark} "), style),
        Span::styled(v.headline(), theme::text()),
    ])];
    if let Some(remedy) = v.remedy() {
        lines.push(Line::raw(""));
        lines.push(Line::styled(remedy, theme::subtle()));
    }
    Paragraph::new(lines).wrap(Wrap { trim: true }).block(theme::panel("Verdict"))
}

/// "What will fit" — the second half of the zero-argument output.
///
/// Two plans, stacked, because there are two tiers and only one of them is the card. The gauge
/// on top is the *host* budget; the table under it re-classifies as that gauge moves. That
/// adjacency is the point: a 12 GiB card offering to run a 59 GiB model is the whole claim of
/// this project, and it is legible in one frame only if the cause sits directly above the
/// effect.
fn fit_panel(m: &Model, area: Rect) -> Paragraph<'_> {
    let title = match m.ctx_request {
        Some(ctx) => format!("What will fit at {} ctx", format::count(ctx as i64)),
        None => "What will fit".to_string(),
    };
    let cols = Columns::of(&m.models);
    let mut rows = Vec::new();
    for (i, (card, fit)) in m.models.iter().zip(&m.fits).enumerate() {
        rows.push(fit_line(card, fit, m.placements.get(i), cols));
    }

    // What has to survive, in priority order: the model rows, then the controls, then the
    // heading and the spacing, then the prose. A clipped table would lose the rows the dials
    // exist to change, which is the wrong half to lose — so the controls compact themselves
    // instead. Two border rows and two padding rows come off the panel first.
    const CHROME: u16 = 4;
    let inner = area.height.saturating_sub(CHROME) as usize;
    // The empty-catalogue message is three lines and is the content in that case.
    let content = if rows.is_empty() { 3 } else { rows.len() };
    let room_for_prose = inner >= 4 + 2 + content;
    let dial_rows = if room_for_prose { 4 } else { 1 };
    let room_for_spacing = inner >= dial_rows + 2 + content;

    let mut lines = Vec::new();
    if let Some(budget) = m.budget {
        lines.extend(dial_lines(budget, m.ctx_request, room_for_prose));
        if room_for_spacing {
            lines.push(Line::raw(""));
        }
    }
    if rows.is_empty() {
        // Where it looked, not just that it found nothing. `docs/ux.md` rules out an error
        // that reports a symptom without naming the cause, and the cause is nearly always
        // that the files are in a different directory.
        lines.push(Line::styled(
            m.catalog_location.as_ref().map_or_else(
                || "No models in the catalog yet.".to_string(),
                |dir| format!("No GGUF files in {dir}"),
            ),
            theme::subtle(),
        ));
        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "Point moearc at them with --models-dir <DIR>, or set MOEARC_MODELS.",
            Style::new().fg(theme::FAINT),
        ));
    } else {
        if room_for_spacing {
            lines.push(fit_header(cols));
        }
        lines.extend(rows);
        lines.push(Line::raw(""));
        // A refusal is the planner's sentence, verbatim, on the screen where the dial that
        // caused it lives. It already names the achievable number and, where there is one,
        // what would fix it — better copy than anything this module could write, and putting
        // it behind a keystroke would hide the honest state at exactly the wrong moment.
        for (card, fit) in m.models.iter().zip(&m.fits) {
            if let FitOutcome::DoesNotFit { reason, .. } = &fit.outcome {
                lines.push(Line::from(vec![
                    Span::styled(format!("{} — ", card.id), Style::new().fg(theme::WARN)),
                    Span::styled(reason.clone(), theme::subtle()),
                ]));
            }
        }
        // What the requested context cost, per model, in the planner's own words. This is the
        // coupling the second dial exists to show: the residency column moved, and this says
        // by how much and why.
        for (card, fit) in m.models.iter().zip(&m.fits) {
            if let Some(note) = fit.yield_note() {
                lines.push(Line::from(vec![
                    Span::styled(format!("{} — ", card.id), theme::subtle()),
                    Span::styled(note.to_string(), theme::subtle()),
                ]));
            }
        }
        // Said once, on the screen where it is visible, because the wording of the tier is the
        // thing most likely to be misread: a model past the budget is slower, not broken.
        if m.placements.iter().any(|p| p.tier == Tier::RunsPagesFromDisk) {
            lines.push(Line::styled(
                "Past the budget is not past the machine: the excess is paged in from the \
                 drive as it is needed, which is slower and is not a failure.",
                theme::subtle(),
            ));
        }
        lines.push(Line::styled(
            "Residency and context are computed from this card's free VRAM. The headroom \
             behind them is provisional, not measured on Arc.",
            Style::new().fg(theme::WARN),
        ));
        if m.fits.iter().any(Fit::context_at_floor) {
            lines.push(Line::styled(
                "A context at the minimum is not the card's limit — experts took everything \
                 above it. [ and ] trade expert slots back for context.",
                theme::subtle(),
            ));
            // 🔴 Measured, and cited on screen because it is the reason to move the dial rather
            // than an opinion about it. Source: `bench/traces/qwen3-30b-prose.decode.ndjson`,
            // a real decode trace — the hottest of that model's 6,144 expert slots is touched
            // on 97.9% of tokens, only 1.4% of slots exceed 50%, and 78% sit below 10%. A KV
            // byte is touched on every token, so per byte of VRAM a KV byte beats every expert
            // slot in the model. Nothing here is extrapolated to another model: the sentence
            // names the one it was measured on.
            lines.push(Line::styled(
                "Measured on a real Qwen3-30B routing trace: only 1.4% of its 6,144 expert \
                 slots are touched on more than half of tokens, and every KV byte is touched \
                 on all of them.",
                Style::new().fg(theme::FAINT),
            ));
        }
    }
    // 🔴 `trim: false`, and it is load-bearing rather than a preference. Ratatui trims leading
    // whitespace off every line when trimming is on, which silently deletes the two columns
    // that align the heading row with the rows under it and the indent that ties each dial to
    // its own explanation. A table whose heading is two columns left of its data reads as a
    // rendering bug, which on the screen this tool is judged by is expensive.
    Paragraph::new(lines).wrap(Wrap { trim: false }).block(
        theme::panel(title).title_bottom(
            Line::from(Span::styled(" moearc info <model> ", Style::new().fg(theme::FAINT)))
                .alignment(Alignment::Right),
        ),
    )
}

/// The two dials, and what each one is doing to the table under it.
///
/// 🔴 **They are one budget, split across two pools.** Host RAM bounds what the *machine* can
/// keep of the mapping, so it decides whether a cache miss is a copy or a drive read. Context
/// bounds what the *card* spends on KV, and the card has exactly two claimants — every page of
/// context is an expert slot sold. Neither dial computes anything here: the first is
/// `moearc_engine::host_budget`, the second is `moearc_engine::memory::plan`, and this function
/// draws what they return.
///
/// Lines rather than panels of their own. A panel each would cost eight rows of chrome on the
/// one screen that already has three, and the controls belong *with* the table they change.
/// `prose` drops the two explanatory lines when the terminal is short — losing an explanation
/// is better than losing the rows the explanation is about.
fn dial_lines(b: HostBudget, ctx: Option<u32>, prose: bool) -> Vec<Line<'static>> {
    // Wide enough to read a position off at video bitrate, narrow enough to leave the numbers
    // beside it on the same line at 100 columns.
    const BAR: usize = 26;
    let ctx_text = match ctx {
        Some(t) => format!("{} tokens", format::count(t as i64)),
        // Not "0" and not a default constant: it is a different request. See `fit::plan`.
        None => "largest that fits".to_string(),
    };

    if !prose {
        // One line for both, for a terminal with no rows to spare.
        return vec![Line::from(vec![
            Span::styled("Host RAM  ", theme::subtle()),
            Span::styled(
                format!("{} of {}", format::bytes(b.bytes()), format::bytes(b.max_bytes())),
                theme::text(),
            ),
            Span::styled("  -/+     ", Style::new().fg(theme::FAINT)),
            Span::styled("Context  ", theme::subtle()),
            Span::styled(ctx_text, theme::text()),
            Span::styled("  [/]", Style::new().fg(theme::FAINT)),
        ])];
    }

    let mut ram = vec![Span::styled("Host RAM  ", theme::subtle())];
    ram.extend(gauge_spans(b.fraction_of_ceiling(), BAR));
    ram.push(Span::styled(
        format!("  {} of {}", format::bytes(b.bytes()), format::bytes(b.max_bytes())),
        theme::text(),
    ));
    ram.push(Span::styled("   -/+", Style::new().fg(theme::FAINT)));

    let memory = b.memory();
    let ram_note = match b.source() {
        // A clamp is reported rather than applied quietly: the user typed a number and did not
        // get it, and that is theirs to know.
        BudgetSource::Clamped { asked } => format!(
            "{} was asked for — this machine has {} available, and {} of that is kept for it",
            format::bytes(asked),
            format::bytes(memory.available_bytes),
            format::bytes(b.reserved_bytes())
        ),
        _ => format!(
            "{} available of {} fitted · {} kept for the machine",
            format::bytes(memory.available_bytes),
            format::bytes(memory.total_bytes),
            format::bytes(b.reserved_bytes())
        ),
    };

    let rungs = (CONTEXT_LADDER.len() - 1).max(1) as f64;
    let mut context = vec![Span::styled("Context   ", theme::subtle())];
    context.extend(gauge_spans(ladder_index(ctx) as f64 / rungs, BAR));
    context.push(Span::styled(format!("  {ctx_text}"), theme::text()));
    context.push(Span::styled("   [/]", Style::new().fg(theme::FAINT)));

    vec![
        Line::from(ram),
        Line::styled(format!("          {ram_note}"), Style::new().fg(theme::FAINT)),
        Line::from(context),
        Line::styled(
            format!(
                "          {} KV · every page of it is an expert slot given back",
                KvPrecision::default().label()
            ),
            Style::new().fg(theme::FAINT),
        ),
    ]
}

/// A gauge drawn as text, so it can sit on a line beside its label inside a paragraph.
///
/// Heavy rule for the filled part, light for the rest — the same shape as the KV gauge on the
/// serving screen, which is a `LineGauge` and cannot be used here.
fn gauge_spans(fraction: f64, width: usize) -> Vec<Span<'static>> {
    let filled = ((fraction.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    vec![
        Span::styled("━".repeat(filled), theme::accent()),
        Span::styled("─".repeat(width - filled), Style::new().fg(theme::FAINT)),
    ]
}

/// The table's heading row.
///
/// Worth its two lines of code: it is what lets each cell hold a bare number instead of
/// repeating its own units, which is where the width for the host column came from.
fn fit_header(cols: Columns) -> Line<'static> {
    Line::styled(
        format!(
            "  {:<id$} {:<quant$} {:>size$}  {:<tier$}  {:<res$} {:>ctx$}",
            "model",
            "quant",
            "size",
            "host",
            "residency",
            "ctx",
            id = cols.id,
            quant = cols.quant,
            size = cols.size,
            tier = Columns::TIER,
            res = cols.residency,
            ctx = cols.ctx
        ),
        Style::new().fg(theme::FAINT),
    )
}

/// The colour for a host tier.
///
/// 🔴 Restrained on purpose, and the paging tier is *not* a warning colour. It is a supported
/// mode of this engine — the mode that lets a 59 GiB model run on an 11 GiB card — and
/// painting it amber would tell the viewer the opposite of what the tool does.
fn tier_style(tier: Tier) -> Style {
    match tier {
        Tier::RunsFromRam => Style::new().fg(theme::GOOD),
        Tier::RunsPagesFromDisk => theme::text(),
        Tier::WillNotFit => theme::subtle(),
    }
}

/// One row of "what will fit".
///
/// The widths come from [`Columns`], which the plain renderer uses too. They used to be
/// constants, and constants were fine for fixtures with thirteen-character handles: a real
/// directory produced rows that ran past the panel border, on the one screen the whole tool is
/// judged by.
///
/// The host column and the residency column are two different tiers and stay two cells: a model
/// can page from disk and still plan cleanly onto the card, and collapsing that into one verdict
/// would have to invent a precedence between them.
///
/// The leading mark is the row's own answer to "can I run this here", so it needs *both* — the
/// card's plan has to succeed and the machine has to be able to hold the file. A green tick
/// beside "won't fit" is a contradiction a viewer reads as a rendering bug.
fn fit_line<'a>(
    card: &'a ModelCard,
    fit: &Fit,
    placement: Option<&Placement>,
    cols: Columns,
) -> Line<'a> {
    let runs = fit.fits() && placement.is_none_or(|p| p.tier.runs());
    let (mark, style) = if runs {
        ("✓", Style::new().fg(theme::GOOD))
    } else {
        ("·", Style::new().fg(theme::SUBTLE))
    };
    let (tier_text, tier) = match placement {
        Some(p) => (p.tier.label(), tier_style(p.tier)),
        // Before the probe lands there is no verdict, and a blank cell says that better than a
        // guess would.
        None => ("", theme::subtle()),
    };
    Line::from(vec![
        Span::styled(format!("{mark} "), style),
        Span::styled(format!("{:<w$} ", card.id, w = cols.id), theme::text()),
        Span::styled(format!("{:<w$} ", card.quant, w = cols.quant), theme::subtle()),
        Span::styled(
            format!("{:>w$}  ", format::bytes(card.file_bytes), w = cols.size),
            theme::subtle(),
        ),
        Span::styled(format!("{tier_text:<w$}  ", w = Columns::TIER), tier),
        Span::styled(
            format!("{:<w$} ", fit.residency_cell(), w = cols.residency),
            if runs { theme::text() } else { theme::subtle() },
        ),
        Span::styled(format!("{:>w$}", fit.context_cell(), w = cols.ctx), theme::subtle()),
    ])
}

// ---------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------

fn models_screen(m: &Model, f: &mut Frame, area: Rect) {
    let [left, right] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .areas(area);
    let [list_area, input_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(5)])
        .areas(left);

    let cols = Columns::of(&m.models);
    let items: Vec<ListItem> = m
        .models
        .iter()
        .zip(m.fits.iter().map(Some).chain(std::iter::repeat(None)))
        .map(|(card, fit)| ListItem::new(model_list_line(card, fit, cols)))
        .collect();
    let mut state = ListState::default().with_selected(Some(m.model_row));
    f.render_stateful_widget(
        List::new(items)
            .block(theme::tight_panel("Models"))
            .highlight_symbol("▸ ")
            .highlight_style(Style::new().fg(theme::ACCENT)),
        list_area,
        &mut state,
    );

    f.render_widget(repo_input(m), input_area);
    f.render_widget(model_detail(m), right);
}

fn model_list_line<'a>(card: &'a ModelCard, fit: Option<&Fit>, cols: Columns) -> Line<'a> {
    // `docs/ux.md`: a model we have not run does not get a green checkmark. The three states
    // are distinguished by glyph as well as colour, so the distinction survives a pipe.
    let (mark, style) = match (card.measured, fit.is_some_and(Fit::fits)) {
        (true, true) => ("✓", Style::new().fg(theme::GOOD)),
        (false, true) => ("~", Style::new().fg(theme::WARN)),
        _ => ("·", Style::new().fg(theme::SUBTLE)),
    };
    Line::from(vec![
        Span::styled(format!("{mark} "), style),
        Span::styled(format!("{:<w$} ", card.id, w = cols.id), theme::text()),
        Span::styled(if card.local { "local" } else { "remote" }, theme::subtle()),
    ])
}

fn repo_input(m: &Model) -> Paragraph<'_> {
    let (value, style) = if m.editing {
        (format!("{}▏", m.repo_input.value()), theme::text())
    } else if m.repo_input.value().is_empty() {
        ("press / to paste a Hugging Face repo id".to_string(), Style::new().fg(theme::FAINT))
    } else {
        (m.repo_input.value().to_string(), theme::subtle())
    };
    let block = if m.editing {
        theme::panel("Pull from Hugging Face").border_style(theme::accent())
    } else {
        theme::panel("Pull from Hugging Face")
    };
    Paragraph::new(Line::styled(value, style)).block(block)
}

fn model_detail(m: &Model) -> Paragraph<'_> {
    let Some(card) = m.selected_model() else {
        return Paragraph::new(Line::styled("No model selected.", theme::subtle()))
            .block(theme::panel("Detail"));
    };
    let mut lines = Vec::new();
    // A repo id when we have one, the file when we do not — never a field labelled "repo"
    // holding something a user cannot paste into `moearc pull`.
    if let Some(repo) = &card.repo {
        lines.push(theme::field("repo", Span::styled(repo.as_str(), theme::text()), LABEL_W));
    }
    if let Some(file) = &card.file {
        lines.push(theme::field("file", Span::styled(file.as_str(), theme::subtle()), LABEL_W));
    }
    lines.extend([
        theme::field("quantisation", card.quant.clone(), LABEL_W),
        theme::field("parameters", card.params(), LABEL_W),
        theme::field("download", format::bytes(card.file_bytes), LABEL_W),
        theme::field("experts", card.experts(), LABEL_W),
        // The number residency is counted in, next to the number the model is described by,
        // because 4,608 slots and 128 experts are both true and only one of them is the plan's
        // denominator.
        theme::field("slots", card.slots(), LABEL_W),
        theme::field(
            "trained ctx",
            format!("{} tok", format::count(card.trained_context_tokens as i64)),
            LABEL_W,
        ),
        theme::field(
            "footprint",
            if card.measured {
                Span::styled("measured on Arc", Style::new().fg(theme::GOOD))
            } else {
                Span::styled("never run on Arc", Style::new().fg(theme::WARN))
            },
            LABEL_W,
        ),
        Line::raw(""),
    ]);

    // The host tier, before the card's plan. Deliberately first: it is the question that
    // decides whether a model is worth downloading at all, and the VRAM split underneath it is
    // the same either way.
    if let Some(p) = m.selected_placement() {
        lines.push(Line::styled("Host RAM", theme::heading()));
        lines.push(theme::field("tier", Span::styled(p.tier.label(), tier_style(p.tier)), LABEL_W));
        if p.tier == Tier::RunsPagesFromDisk {
            lines.push(theme::field(
                "in RAM",
                format!(
                    "{} of {}",
                    format::percent(p.ram_bytes as i64, (p.ram_bytes + p.disk_bytes) as i64),
                    format::bytes(p.ram_bytes + p.disk_bytes)
                ),
                LABEL_W,
            ));
        }
        lines.push(Line::styled(format!("· {}", p.reason), theme::subtle()));
        lines.push(Line::raw(""));
    }

    match m.selected_fit().map(|f| &f.outcome) {
        Some(FitOutcome::Fits { rationale, .. }) => {
            lines.push(Line::styled("Planned split", theme::heading()));
            lines.extend(split_fields(&m.selected_fit().unwrap().outcome));
            lines.push(Line::raw(""));
            // The planner's own reasoning, verbatim. `Reason` exists in the engine so the
            // interface can show its work; putting it behind -v would be the "numbers the
            // user cannot derive" failure from docs/ux.md.
            lines.push(Line::styled("Why", theme::heading()));
            for step in rationale {
                lines.push(Line::styled(format!("· {step}"), theme::subtle()));
            }
        }
        Some(FitOutcome::DoesNotFit { headline, reason }) => {
            // The glyph carries the emphasis the capital used to.  is one string
            // used in a table cell and here, and two spellings of it would drift.
            lines.push(Line::styled(format!("✗ {headline}"), Style::new().fg(theme::BAD)));
            lines.push(Line::raw(""));
            lines.push(Line::styled(reason.clone(), theme::subtle()));
        }
        None => lines.push(Line::styled("Not yet planned.", theme::subtle())),
    }

    Paragraph::new(lines).wrap(Wrap { trim: true }).block(theme::panel("Detail"))
}

/// The chosen split as label/value rows.
///
/// Shared by the model detail pane and the serving panel: `docs/ux.md` asks that serving
/// print what it decided, and "what it decided" must be the same words in both places or one
/// of them is wrong.
fn split_fields(outcome: &FitOutcome) -> Vec<Line<'static>> {
    let FitOutcome::Fits {
        resident_experts,
        total_experts,
        context_tokens,
        kv_pages,
        ceiling_tokens,
        ..
    } = outcome
    else {
        return Vec::new();
    };
    let mut lines = vec![
        theme::field(
            "residency",
            format!(
                // Separated to match the row in "What will fit". Four thousand six hundred
                // and eight is the sort of number a reader compares by eye, once.
                "{} / {}  ({})",
                format::count(*resident_experts as i64),
                format::count(*total_experts as i64),
                format::percent(*resident_experts as i64, *total_experts as i64)
            ),
            LABEL_W,
        ),
        theme::field(
            "context",
            format!(
                "{} tok · {} pages",
                format::count(*context_tokens as i64),
                format::count(*kv_pages as i64)
            ),
            LABEL_W,
        ),
    ];
    if let Some(ceiling) = ceiling_tokens {
        // The other end of the tradeoff. Without it the context figure reads as a hard limit
        // of the card rather than as the price of keeping experts resident.
        lines.push(theme::field(
            "ceiling",
            format!("{} tok min residency", format::count(*ceiling as i64)),
            LABEL_W,
        ));
    }
    lines
}

// ---------------------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------------------

fn download_screen(m: &Model, f: &mut Frame, area: Rect) {
    let Some(d) = &m.download else {
        f.render_widget(
            Paragraph::new(Line::styled("Nothing downloading.", theme::subtle()))
                .block(theme::panel("Download")),
            area,
        );
        return;
    };

    let block = theme::panel(format!("Pulling {}", d.plan.repo));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let [gauge_area, _, detail_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)])
        .areas(inner);

    let percent = (d.fraction() * 100.0).round() as u16;
    f.render_widget(
        Gauge::default()
            .ratio(d.fraction())
            .gauge_style(Style::new().fg(theme::ACCENT))
            .label(Span::styled(format!("{percent}%"), theme::text())),
        gauge_area,
    );

    let status = if d.cancelled {
        Span::styled("cancelled", Style::new().fg(theme::BAD))
    } else if d.finished() {
        Span::styled("complete — checksum verified", Style::new().fg(theme::GOOD))
    } else {
        return_spinner_status(m)
    };
    let mut lines = vec![
        theme::field(
            "transferred",
            format!("{} of {}", format::bytes(d.done_bytes), format::bytes(d.plan.total_bytes)),
            LABEL_W,
        ),
        theme::field("rate", format::rate(d.plan.bytes_per_sec), LABEL_W),
        theme::field(
            "eta",
            d.eta_secs().map_or_else(|| "—".to_string(), format::duration),
            LABEL_W,
        ),
        theme::field("status", status, LABEL_W),
    ];
    if d.plan.resume_from > 0 {
        lines.push(theme::field("resumed from", format::bytes(d.plan.resume_from), LABEL_W));
    }
    f.render_widget(Paragraph::new(lines), detail_area);
}

fn return_spinner_status(m: &Model) -> Span<'static> {
    let frame = spinner(m, "");
    let glyph = frame.spans.first().map_or(" ".to_string(), |s| s.content.to_string());
    Span::styled(format!("{glyph} downloading"), theme::subtle())
}

// ---------------------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------------------

fn serving_screen(m: &Model, f: &mut Frame, area: Rect) {
    let Some(s) = &m.serving else {
        f.render_widget(
            Paragraph::new(Line::styled("No server running.", theme::subtle()))
                .block(theme::panel("Serving")),
            area,
        );
        return;
    };

    let [top, bottom] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(8)])
        .areas(area);
    let [left, right] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .areas(top);

    let mut server = vec![
        theme::field(
            "endpoint",
            Span::styled(
                format!("http://{}:{}/v1", s.host, s.port),
                Style::new().fg(theme::ACCENT),
            ),
            LABEL_W,
        ),
        theme::field("model", Span::styled(s.model.as_str(), theme::text()), LABEL_W),
        theme::field("uptime", format::duration(s.ticks * super::model::TICK_MS / 1000), LABEL_W),
        Line::raw(""),
        Line::styled("What it decided", theme::heading()),
    ];
    match &s.fit.outcome {
        FitOutcome::Fits { .. } => server.extend(split_fields(&s.fit.outcome)),
        FitOutcome::DoesNotFit { reason, .. } => {
            server.push(Line::styled(reason.clone(), Style::new().fg(theme::BAD)))
        }
    }
    server.push(theme::field(
        "requested",
        match s.fit.requested_ctx {
            Some(ctx) => format!("{} tok", format::count(ctx as i64)),
            None => "largest that fits".to_string(),
        },
        LABEL_W,
    ));
    f.render_widget(
        Paragraph::new(server).wrap(Wrap { trim: true }).block(theme::panel("Server")),
        left,
    );

    let v = &s.vitals;
    // Capacity comes from the plan, occupancy from the server. Deriving the page count here
    // rather than taking one from each source is what keeps the two halves of this bar
    // consistent.
    let kv_total = match &s.fit.outcome {
        FitOutcome::Fits { kv_pages, .. } => (*kv_pages).max(1) as i64,
        FitOutcome::DoesNotFit { .. } => 1,
    };
    let kv_fraction = v.kv_utilisation.clamp(0.0, 1.0);
    let kv_used = (kv_fraction * kv_total as f64).round() as i64;
    let vitals_block = theme::panel("Vitals");
    let vitals_inner = vitals_block.inner(right);
    f.render_widget(vitals_block, right);
    let [numbers, _, kv_bar] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Length(1), Constraint::Length(2)])
        .areas(vitals_inner);
    f.render_widget(
        Paragraph::new(vec![
            theme::field("generation", format!("{:.1} tok/s", v.tokens_per_sec), LABEL_W),
            theme::field("prompt", format!("{:.0} tok/s", v.prompt_tokens_per_sec), LABEL_W),
            theme::field("in flight", format!("{} requests", v.active_requests), LABEL_W),
            theme::field(
                "expert hits",
                format::percent((v.expert_hit_rate * 100.0) as i64, 100),
                LABEL_W,
            ),
        ]),
        numbers,
    );
    f.render_widget(
        LineGauge::default()
            .ratio(kv_fraction)
            .filled_style(Style::new().fg(theme::ACCENT))
            .unfilled_style(Style::new().fg(theme::FAINT))
            .label(Span::styled(
                format!("kv {} / {} pages", format::count(kv_used), format::count(kv_total)),
                theme::subtle(),
            )),
        kv_bar,
    );

    throughput_panel(&s.history, f, bottom);
}

/// The generation-rate strip.
///
/// Rescaled to its own min/max before drawing. A sparkline anchors bar height at zero, and a
/// steady 61–65 tok/s then renders as a wall of full blocks that shows nothing. The range is
/// printed in the title, so the axis the bars are drawn against is stated rather than implied
/// — which is the difference between auto-scaling and a misleading chart.
fn throughput_panel(history: &[u64], f: &mut Frame, area: Rect) {
    let (lo, hi) = history.iter().fold((u64::MAX, 0), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let title = if history.is_empty() {
        "Generation, tokens/sec".to_string()
    } else {
        format!("Generation, tokens/sec  ({lo}–{hi})")
    };
    let block = theme::panel(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    // A flat series would otherwise divide by zero; give it a full-height flat line, which is
    // the honest picture of a rate that is not moving.
    let span = hi.saturating_sub(lo).max(1);
    let scaled: Vec<u64> = history.iter().map(|v| (v - lo) * 100 / span).collect();
    f.render_widget(Sparkline::default().data(scaled).style(theme::accent()), inner);
}

// ---------------------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------------------

/// The keybind reference — and, deliberately, the flag each key corresponds to.
///
/// `docs/ux.md` forbids anything being reachable only through the interface. Printing the
/// equivalent flag beside every key turns that rule into something a user can see and a
/// reviewer can check, instead of a promise made in a design document.
fn help_overlay(f: &mut Frame, area: Rect) {
    const ROWS: &[(&str, &str, &str)] = &[
        ("↑ ↓ / j k", "move the selection", ""),
        ("tab", "next screen", ""),
        ("- / +", "host RAM budget", "--host-budget <SIZE>"),
        ("0 / m", "no budget / all of it", "--host-budget 0"),
        ("[ / ]", "context length", "--ctx <tokens>"),
        ("r", "rescan devices", "moearc"),
        ("/", "paste a repo id", "moearc pull <repo-id>"),
        ("d", "download the selection", "moearc pull <model>"),
        ("s", "serve / stop", "moearc serve <model>"),
        ("⏎", "open the selection", "moearc info <model>"),
        ("x", "cancel a download", ""),
        ("q / ctrl-c", "quit", ""),
    ];

    let w = 68.min(area.width.saturating_sub(4));
    let h = (ROWS.len() as u16 + 7).min(area.height);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };

    let mut lines =
        vec![Line::styled("Every key here is also a flag.", theme::subtle()), Line::raw("")];
    for (key, what, flag) in ROWS {
        lines.push(Line::from(vec![
            Span::styled(format!("{key:<12}"), Style::new().fg(theme::ACCENT)),
            Span::styled(format!("{what:<26}"), theme::text()),
            Span::styled(*flag, Style::new().fg(theme::FAINT)),
        ]));
    }

    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines).block(theme::panel("Keys").border_style(theme::accent())),
        rect,
    );
}

// ---------------------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod snapshot {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    /// Render one frame and return it as text.
    ///
    /// This is the screenshot. A terminal interface has no other way to prove it drew what it
    /// claims, and because `view` is pure the result is stable enough to assert on.
    pub fn frame(m: &Model, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("test backend");
        terminal.draw(|f| view(m, f)).expect("draw");
        let buf = terminal.backend().buffer();
        (0..buf.area.height)
            .map(|y| {
                let mut line = String::new();
                for x in 0..buf.area.width {
                    line.push_str(buf[(x, y)].symbol());
                }
                line.trim_end().to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Print a dump under a banner, so `cargo test -- --nocapture` doubles as a screenshot
    /// session.
    pub fn show(name: &str, dump: &str) {
        println!("\n──────── {name} ────────\n{dump}\n");
    }
}

#[cfg(test)]
mod tests {
    use super::snapshot::{frame, show};
    use super::*;
    use crate::source::{
        CpuOnlyCause, DeviceReport, DeviceSource, HostSource, ModelCatalog, ServeStats,
        StubCatalog, StubDeviceSource, StubHost, StubServeStats, StubTransfers, TransferSource,
    };
    use crate::tui::model::{Msg, Serving, update};

    const W: u16 = 100;
    const H: u16 = 30;

    fn loaded() -> Model {
        crate::tui::model::tests::loaded()
    }

    /// The interface holding the four real GGUF files, planned against the reference card.
    ///
    /// The rows are wider than the fixtures in every column that has one — a 25-character
    /// handle, a 59 GiB file, a five-digit slot count — which is the point. See
    /// [`StubCatalog::as_measured`].
    fn measured() -> Model {
        let mut m = Model::new(
            None,
            None,
            Some(crate::source::Sources::real(std::path::PathBuf::new()).stub_parts),
        );
        update(&mut m, Msg::Detected(StubDeviceSource.detect().unwrap()));
        update(&mut m, Msg::Catalog(StubCatalog::as_measured()));
        update(&mut m, Msg::Host(StubHost.probe().unwrap()));
        m
    }

    /// The same, with the host budget wound down to `bytes`.
    ///
    /// The budget goes through the reducer rather than being poked into the model, so these
    /// frames are the ones a key press produces and not a state only a test can reach.
    fn measured_at_budget(bytes: u64) -> Model {
        let mut m = measured();
        m.budget = Some(m.budget.unwrap().set(bytes));
        m.recompute_placements();
        m
    }

    #[test]
    fn device_report_screen() {
        let m = loaded();
        let dump = frame(&m, W, H);
        show("moearc — device report", &dump);
        assert!(dump.contains("Intel Arc B580 Graphics"));
        assert!(dump.contains("level_zero"));
        assert!(dump.contains("11.4 GiB"), "free VRAM is the number every plan starts from");
        assert!(dump.contains("is ready"), "the verdict line, not just a table");
        assert!(dump.contains("What will fit"));
        assert!(dump.contains("╭") && dump.contains("╮"), "rounded borders, per docs/ux.md");
        assert!(
            dump.contains(crate::source::Sources::stub().stub_parts),
            "provenance is always on screen, and says which parts"
        );
    }

    #[test]
    fn detecting_screen_shows_a_spinner_not_a_blank_panel() {
        let mut m = loaded();
        update(
            &mut m,
            Msg::Key(ratatui::crossterm::event::KeyEvent::new(
                ratatui::crossterm::event::KeyCode::Char('r'),
                ratatui::crossterm::event::KeyModifiers::NONE,
            )),
        );
        let dump = frame(&m, W, H);
        show("moearc — detecting", &dump);
        assert!(dump.contains("Looking for a Level Zero device"));
    }

    #[test]
    fn a_cpu_only_machine_is_told_why_and_what_to_do() {
        let mut m = loaded();
        let mut report = m.report.clone().unwrap();
        report.devices.retain(|d| !d.is_inference_target());
        report.verdict = crate::source::Verdict::CpuOnly { cause: CpuOnlyCause::RuntimeNotSourced };
        update(&mut m, Msg::Detected(DeviceReport { ..report }));
        let dump = frame(&m, W, H);
        show("moearc — cpu-only verdict", &dump);
        assert!(dump.contains("oneAPI runtime is not on this shell"));
        assert!(dump.contains("unsourced"), "the remedy, not just the symptom");
    }

    #[test]
    fn model_picker_screen() {
        let mut m = loaded();
        m.screen = Screen::Models;
        m.model_row = 1;
        let dump = frame(&m, W, H);
        show("moearc ls — model picker", &dump);
        assert!(dump.contains("gpt-oss-20b"));
        assert!(dump.contains("▸ "), "the selection is visible without colour");
        assert!(dump.contains("press / to paste a Hugging Face repo id"));
        assert!(dump.contains("residency"), "the split is shown, never asked for");
    }

    #[test]
    fn model_picker_with_the_repo_field_focused() {
        let mut m = loaded();
        m.screen = Screen::Models;
        m.editing = true;
        for c in "unsloth/Qwen3-Coder-30B-A3B-GGUF".chars() {
            update(
                &mut m,
                Msg::Key(ratatui::crossterm::event::KeyEvent::new(
                    ratatui::crossterm::event::KeyCode::Char(c),
                    ratatui::crossterm::event::KeyModifiers::NONE,
                )),
            );
        }
        let dump = frame(&m, W, H);
        show("moearc ls — pasting a repo id", &dump);
        assert!(dump.contains("unsloth/Qwen3-Coder-30B-A3B-GGUF"));
    }

    #[test]
    fn a_model_that_does_not_fit_says_so_with_the_shortfall() {
        let mut m = loaded();
        m.screen = Screen::Models;
        m.model_row = 3; // qwen3-235b-a22b
        let dump = frame(&m, W, H);
        show("moearc info — a model that will not fit", &dump);
        assert!(dump.contains("will not fit"), "the headline names why there is no plan");
        // The engine names the shortfall and a way out; "out of memory" would not.
        assert!(dump.contains("quantisation"), "the miss should say what would fix it");
    }

    #[test]
    fn download_screen() {
        let mut m = loaded();
        update(&mut m, Msg::TransferReady(StubTransfers.plan("mixtral-8x7b").unwrap()));
        for _ in 0..80 {
            update(&mut m, Msg::Tick);
        }
        let dump = frame(&m, W, H);
        show("moearc pull — download", &dump);
        // The handle resolved to a repo id, which is what a user would need to copy.
        assert!(dump.contains("Pulling mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF"));
        assert!(dump.contains("transferred"));
        assert!(dump.contains("eta"));
        assert!(dump.contains('%'), "a progress bar with a number on it");
    }

    #[test]
    fn serving_screen() {
        let mut m = loaded();
        let card = StubCatalog.resolve("qwen3-30b-a3b").unwrap();
        let fit = m.fits[0].clone();
        update(
            &mut m,
            Msg::Serving(Box::new(Serving {
                model: card.id,
                host: "127.0.0.1".into(),
                port: 8080,
                fit,
                vitals: StubServeStats.sample(0),
                history: Vec::new(),
                ticks: 0,
            })),
        );
        for t in 0..48 {
            update(&mut m, Msg::Tick);
            update(&mut m, Msg::Vitals(StubServeStats.sample(t)));
        }
        let dump = frame(&m, W, H);
        show("moearc serve — live stats", &dump);
        assert!(dump.contains("http://127.0.0.1:8080/v1"), "the endpoint, ready to paste");
        assert!(dump.contains("What it decided"), "startup shows its reasoning");
        assert!(dump.contains("tok/s"));
        assert!(dump.contains("Generation, tokens/sec"));
        // The gauge's two halves must come out of the same geometry. An earlier version drew
        // occupancy from the sample and capacity from the plan, and rendered "187 / 12".
        // Side-by-side panels share a line, so read only the gauge's own text.
        let kv = dump
            .lines()
            .find_map(|l| l.split_once("kv "))
            .map(|(_, rest)| rest)
            .expect("a kv gauge");
        let nums: Vec<i64> = kv.split_whitespace().take(3).filter_map(|t| t.parse().ok()).collect();
        assert_eq!(nums.len(), 2, "expected `kv <used> / <total> pages`, got {kv:?}");
        assert!(nums[0] <= nums[1], "kv used {} exceeds capacity {}", nums[0], nums[1]);
    }

    #[test]
    fn help_overlay_lists_the_flag_for_every_key() {
        let mut m = loaded();
        m.help = true;
        let dump = frame(&m, W, H);
        show("moearc — help overlay", &dump);
        assert!(dump.contains("Every key here is also a flag."));
        for flag in ["moearc pull <model>", "moearc serve <model>", "moearc info <model>"] {
            assert!(dump.contains(flag), "missing the flag equivalent for {flag}");
        }
    }

    #[test]
    fn a_failure_reaches_the_footer_instead_of_the_terminal() {
        let mut m = loaded();
        update(&mut m, Msg::Failed("unknown model `nope`".into()));
        let dump = frame(&m, W, H);
        show("moearc — a failure in the footer", &dump);
        assert!(dump.contains("unknown model `nope`"));
    }

    #[test]
    fn every_enumerated_device_reaches_the_screen() {
        // Regression: the table was sized by hand and swallowed its last row. The row it
        // swallowed was the CPU device, which is the evidence that distinguishes an unsourced
        // runtime from a dead card — so the bug removed a diagnosis, not a cosmetic line.
        let m = loaded();
        let dump = frame(&m, W, H);
        for d in &m.report.as_ref().unwrap().devices {
            assert!(dump.contains(d.name.as_str()), "{} was clipped", d.name);
        }
    }

    #[test]
    fn a_remedy_is_never_truncated_mid_sentence() {
        let mut m = loaded();
        let mut report = m.report.clone().unwrap();
        report.devices.retain(|d| !d.is_inference_target());
        report.verdict = crate::source::Verdict::for_devices(&report.devices);
        update(&mut m, Msg::Detected(report));
        for width in [60, 80, 100, 140] {
            let dump = frame(&m, width, 34);
            // Compare word by word: the panel re-breaks the text, so the sentence is present
            // without any single line of it matching.
            for word in CpuOnlyCause::RuntimeNotSourced.remedy().split_whitespace() {
                assert!(dump.contains(word), "`{word}` missing at width {width}");
            }
        }
    }

    #[test]
    fn every_real_model_keeps_its_whole_row_on_one_line() {
        // The regression this file already caught four of: a column sized for a fixture and
        // overrun by real data. `olmoe-1b-7b-0924-instruct` is twice the handle
        // `gpt-oss-20b` is, `qwen3.6-35b-a3b-ud` has 10,240 residency slots where the fixture
        // has 128, and this is the screen the tool is judged by.
        //
        // Asserted per line, not per dump: a wrapped row still puts every substring somewhere
        // in the frame, and wrapping is the failure being tested for.
        for width in [100, 120, 160] {
            let m = measured();
            let dump = frame(&m, width, 36);
            show(&format!("moearc — real models at {width} columns"), &dump);
            assert!(
                !dump.contains("stub data"),
                "the device table and the model list are read from the machine; a blanket \
                 fixture marker over them under-claims, which is its own kind of wrong"
            );
            for (card, fit) in m.models.iter().zip(&m.fits) {
                let row = dump
                    .lines()
                    .find(|l| l.contains(card.id.as_str()))
                    .unwrap_or_else(|| panic!("no row for {} at width {width}", card.id));
                let tier = m.placements[m.models.iter().position(|c| c.id == card.id).unwrap()]
                    .tier
                    .label()
                    .to_string();
                for part in [
                    &card.quant,
                    &format::bytes(card.file_bytes),
                    &tier,
                    &fit.residency_cell(),
                    &fit.context_cell(),
                ] {
                    assert!(
                        row.contains(part.as_str()),
                        "`{part}` is not on {}'s row at width {width}: {row:?}",
                        card.id
                    );
                }
            }
        }
    }

    #[test]
    fn a_model_five_times_the_size_of_the_card_reports_its_residency_rather_than_refusing() {
        // The whole thesis on one line: a 59 GiB model on an 11 GiB card, with the fraction
        // actually resident stated. Hiding the fraction would make the claim prettier and
        // unverifiable; it is the number that makes it credible.
        let m = measured();
        let card = &m.models[0];
        assert_eq!(card.id, "gpt-oss-120b", "the largest model sorts first");
        let fit = &m.fits[0];
        let FitOutcome::Fits { resident_experts, total_experts, .. } = fit.outcome else {
            panic!("a 59 GiB model must plan, not bail: {:?}", fit.outcome)
        };
        assert_eq!(total_experts, 4_608, "36 MoE blocks x 128 experts, not 128");
        assert!(
            resident_experts > card.expert_slots_active && resident_experts < total_experts,
            "a partial residency is the interesting case: {resident_experts} of {total_experts}"
        );
        let dump = frame(&m, 100, 36);
        assert!(dump.contains("59.0 GiB"));
        assert!(dump.contains(&fit.residency_cell()), "the residency figure reaches the screen");
    }

    // -----------------------------------------------------------------------------------
    // The two dials
    // -----------------------------------------------------------------------------------

    #[test]
    fn the_host_budget_is_on_screen_with_the_machine_it_came_from() {
        let m = measured();
        let dump = frame(&m, 110, 40);
        show("moearc — host RAM at the ceiling", &dump);
        assert!(dump.contains("Host RAM"), "the control is visible without a keystroke");
        let b = m.budget.unwrap();
        assert!(dump.contains(&format::bytes(b.bytes())));
        // The ceiling is explained rather than asserted: a number the user cannot go past has
        // to say what is holding it there.
        assert!(dump.contains("kept for the machine"));
        assert!(dump.contains(&format::bytes(b.memory().total_bytes)));
    }

    #[test]
    fn winding_the_budget_down_moves_models_from_ram_to_disk_and_none_of_them_out() {
        // The demo, as a snapshot pair. The same four real models, two budgets, and the only
        // thing that changes is which tier each is in — nothing becomes unrunnable.
        let high = frame(&measured(), 110, 40);
        show("moearc — host RAM at the ceiling", &high);
        let low = frame(&measured_at_budget(8 << 30), 110, 40);
        show("moearc — host RAM wound down to 8 GiB", &low);

        assert!(high.contains("runs from RAM"));
        assert!(low.contains("runs, pages from disk"));
        assert!(!low.contains("won't fit"), "a smaller budget is slower, never fatal");
        // olmoe is 3.9 GiB and stays inside an 8 GiB budget, so the two tiers are on screen
        // together — which is what makes the distinction readable.
        assert!(low.contains("runs from RAM"));
    }

    #[test]
    fn a_zero_budget_still_runs_every_model() {
        let dump = frame(&measured_at_budget(0), 110, 40);
        show("moearc — host RAM at zero", &dump);
        assert!(!dump.contains("won't fit"));
        assert!(dump.contains("runs, pages from disk"));
        assert!(dump.contains("is not a failure"), "the wording carries the claim, not a colour");
    }

    #[test]
    fn the_context_dial_is_on_screen_and_its_ends_are_reachable() {
        let mut m = measured();
        let auto = frame(&m, 110, 40);
        show("moearc — context at `largest that fits`", &auto);
        assert!(auto.contains("Context"));
        assert!(auto.contains("largest that fits"));

        m.ctx_request = Some(65_536);
        m.recompute_fits();
        let long = frame(&m, 110, 40);
        show("moearc — context wound to 64K", &long);
        assert!(long.contains("65,536 tokens"), "the dial says what was asked for");
        assert!(long.contains("What will fit at 65,536 ctx"));
    }

    #[test]
    fn a_context_this_card_cannot_serve_shows_the_planner_s_own_refusal() {
        // 🔴 The honest state at f16, and it must be legible rather than smoothed over: 256K of
        // KV is past what an 11.33 GiB card can hold for these models. The sentence on screen
        // is `moearc_engine::memory`'s, which names the achievable number.
        let mut m = measured();
        m.ctx_request = Some(262_144);
        m.recompute_fits();
        let dump = frame(&m, 110, 44);
        show("moearc — a context the card cannot serve", &dump);
        let refused: Vec<&crate::source::ModelCard> =
            m.models.iter().zip(&m.fits).filter(|(_, f)| !f.fits()).map(|(c, _)| c).collect();
        assert!(!refused.is_empty(), "256K must be refused somewhere on a 12 GiB card");
        for card in refused {
            assert!(dump.contains(card.id.as_str()));
        }
        assert!(
            dump.contains("does not fit") || dump.contains("trained"),
            "the refusal reaches the screen: {dump}"
        );
    }

    #[test]
    fn a_model_that_is_not_here_and_will_not_fit_the_drive_is_the_only_refusal() {
        // 🔴 The third tier, and the only one that is a refusal. It is reachable **only** for a
        // model that is not on this machine: a file already on the drive has spent its bytes
        // and always runs, however far past the budget it is. That asymmetry is the whole
        // point — see `moearc_engine::host_budget::place`.
        //
        // The catalogue on a real machine is all-local today, so this state cannot be produced
        // from a directory scan; the fixture's remote entries are what exercise it, and the
        // frame says so in its own footer.
        let mut m = loaded();
        update(
            &mut m,
            Msg::Host(crate::source::HostReport {
                total_bytes: 96 << 30,
                available_bytes: 74_088_284_160,
                // A drive with 20 GiB left on it.
                models_free_bytes: 20 << 30,
            }),
        );
        let dump = frame(&m, 110, 40);
        show("moearc — a model with nowhere to land", &dump);

        for (card, p) in m.models.iter().zip(&m.placements) {
            if card.local {
                assert!(p.tier.runs(), "{} is here, so it runs", card.id);
            }
        }
        // mixtral is 24.6 GiB and not here: it needs room the drive does not have.
        assert!(dump.contains("won't fit"));
        assert!(dump.contains("runs from RAM"), "and the other tiers are still on screen");
    }

    #[test]
    fn the_dials_never_push_a_model_row_off_a_short_terminal() {
        // The controls compact themselves rather than costing rows. Losing the explanation is
        // survivable; losing the table the explanation is about is not.
        let m = measured();
        for height in [28, 30, 34, 40] {
            let dump = frame(&m, 110, height);
            for card in &m.models {
                assert!(
                    dump.contains(card.id.as_str()),
                    "{} was pushed off screen at height {height}",
                    card.id
                );
            }
            assert!(dump.contains("Host RAM"), "and the control survives too, at {height}");
        }
    }

    #[test]
    fn the_detail_pane_names_the_slot_count_the_plan_is_denominated_in() {
        // 128 experts and 4,608 slots are both true, and only one of them is the denominator
        // of the residency figure two lines below it.
        let mut m = measured();
        m.screen = Screen::Models;
        m.model_row = 0;
        let dump = frame(&m, 120, 36);
        show("moearc ls — a real model's detail", &dump);
        assert!(dump.contains("128 per block, 4 routed"), "the model's own geometry");
        assert!(dump.contains("4,608 across 36 blocks"), "and what residency counts");
        assert!(dump.contains("gpt-oss-120b-MXFP4.gguf"), "the file it was read from");
    }

    #[test]
    fn a_context_the_model_cannot_use_is_never_offered() {
        // olmoe is a 4,096-token model and this card has room for eleven times that. The
        // number on screen is a claim about the model, so it stops where the model does.
        let m = measured();
        let (card, fit) = m
            .models
            .iter()
            .zip(&m.fits)
            .find(|(c, _)| c.id.starts_with("olmoe"))
            .expect("olmoe is in the measured set");
        assert_eq!(card.trained_context_tokens, 4_096);
        let FitOutcome::Fits { context_tokens, ceiling_tokens, .. } = fit.outcome else {
            panic!("olmoe fits comfortably")
        };
        assert!(context_tokens <= 4_096, "{context_tokens} tokens is more than it was trained for");
        assert!(
            ceiling_tokens.is_none_or(|c| c <= 4_096),
            "the ceiling is held to the same limit, or it re-tells the same lie"
        );
        let dump = frame(&m, 120, 36);
        let row = dump.lines().find(|l| l.contains("olmoe")).expect("a row for olmoe");
        assert_eq!(fit.context_cell(), "4,096");
        assert!(row.contains(&fit.context_cell()), "the capped context is on its own row: {row:?}");
    }

    #[test]
    fn an_empty_model_directory_says_where_it_looked() {
        let mut m = Model::new(None, None, None);
        m.catalog_location = Some("/srv/models".to_string());
        update(&mut m, Msg::Detected(StubDeviceSource.detect().unwrap()));
        update(&mut m, Msg::Catalog(Vec::new()));
        update(&mut m, Msg::Host(StubHost.probe().unwrap()));
        let dump = frame(&m, 100, 30);
        show("moearc — nothing in the model directory", &dump);
        assert!(dump.contains("/srv/models"), "the directory is the cause, not a detail");
        assert!(dump.contains("--models-dir"), "and there is one thing to do about it");
    }

    #[test]
    fn every_screen_survives_a_terminal_too_small_to_hold_it() {
        // Ratatui panics on out-of-bounds writes, so a cramped terminal is a crash risk in
        // exactly the situation where a crash is least excusable: a split pane.
        for mut m in [loaded(), measured()] {
            for screen in [Screen::Devices, Screen::Models, Screen::Download, Screen::Serving] {
                m.screen = screen;
                for (w, h) in [(20, 6), (40, 10), (200, 60)] {
                    let _ = frame(&m, w, h);
                }
            }
        }
    }
}
