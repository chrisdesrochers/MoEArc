//! Rendering. A pure function of [`Model`] — it reads no state of its own and mutates
//! nothing, which is what lets the tests at the bottom render every screen into a
//! `TestBackend` and assert on the text that comes out.
//!
//! Those snapshots are the closest thing a terminal interface has to a screenshot, and they
//! are the reason this file has no hidden animation state: anything that varied with wall
//! time would make them flap.

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
use crate::fit::{Fit, FitOutcome};
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
        Screen::Devices => {
            &[("↑↓", "select"), ("r", "rescan"), ("⏎", "models"), ("?", "help"), ("q", "quit")]
        }
        Screen::Models if m.editing => &[("⏎", "pull"), ("esc", "cancel")],
        Screen::Models => &[
            ("↑↓", "select"),
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
    } else if m.stubbed {
        // Provenance, permanently visible. Everything on screen is fixture data until the
        // device and model backends are wired, and a number that looks measured and is not
        // is worse than a blank panel.
        spans.push(Span::styled("   stub data", Style::new().fg(theme::FAINT)));
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
    f.render_widget(fit_panel(m), fit_area);
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
fn fit_panel(m: &Model) -> Paragraph<'_> {
    let title = match m.ctx_request {
        Some(ctx) => format!("What will fit at {} ctx", format::count(ctx as i64)),
        None => "What will fit".to_string(),
    };
    let mut lines = Vec::new();
    for (card, fit) in m.models.iter().zip(&m.fits) {
        lines.push(fit_line(card, fit));
    }
    if lines.is_empty() {
        lines.push(Line::styled("No models in the catalog yet.", theme::subtle()));
    } else {
        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "Residency and context are computed from this card's free VRAM. The headroom \
             behind them is provisional, not measured on Arc.",
            Style::new().fg(theme::WARN),
        ));
    }
    Paragraph::new(lines).wrap(Wrap { trim: true }).block(
        theme::panel(title).title_bottom(
            Line::from(Span::styled(" moearc info <model> ", Style::new().fg(theme::FAINT)))
                .alignment(Alignment::Right),
        ),
    )
}

fn fit_line<'a>(card: &'a ModelCard, fit: &Fit) -> Line<'a> {
    let (mark, style) = if fit.fits() {
        ("✓", Style::new().fg(theme::GOOD))
    } else {
        ("·", Style::new().fg(theme::SUBTLE))
    };
    Line::from(vec![
        Span::styled(format!("{mark} "), style),
        Span::styled(format!("{:<20}", card.id), theme::text()),
        Span::styled(format!("{:<10}", card.quant), theme::subtle()),
        Span::styled(format!("{:>10}  ", format::bytes(card.file_bytes)), theme::subtle()),
        Span::styled(fit.summary(), if fit.fits() { theme::text() } else { theme::subtle() }),
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

    let items: Vec<ListItem> = m
        .models
        .iter()
        .zip(m.fits.iter().map(Some).chain(std::iter::repeat(None)))
        .map(|(card, fit)| ListItem::new(model_list_line(card, fit)))
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

fn model_list_line<'a>(card: &'a ModelCard, fit: Option<&Fit>) -> Line<'a> {
    // `docs/ux.md`: a model we have not run does not get a green checkmark. The three states
    // are distinguished by glyph as well as colour, so the distinction survives a pipe.
    let (mark, style) = match (card.measured, fit.is_some_and(Fit::fits)) {
        (true, true) => ("✓", Style::new().fg(theme::GOOD)),
        (false, true) => ("~", Style::new().fg(theme::WARN)),
        _ => ("·", Style::new().fg(theme::SUBTLE)),
    };
    Line::from(vec![
        Span::styled(format!("{mark} "), style),
        Span::styled(format!("{:<20}", card.id), theme::text()),
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
    let mut lines = vec![
        theme::field("repo", Span::styled(card.repo.as_str(), theme::text()), LABEL_W),
        theme::field("quantisation", card.quant.clone(), LABEL_W),
        theme::field("parameters", format!("{} active", card.params()), LABEL_W),
        theme::field("download", format::bytes(card.file_bytes), LABEL_W),
        theme::field(
            "experts",
            format!("{} of {} per token", card.experts_active, card.experts_total),
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
    ];

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
        Some(FitOutcome::DoesNotFit { reason }) => {
            lines.push(Line::styled("Will not fit on this card", Style::new().fg(theme::BAD)));
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
                "{resident_experts} / {total_experts}  ({})",
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
        FitOutcome::DoesNotFit { reason } => {
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
        CpuOnlyCause, DeviceReport, ModelCatalog, ServeStats, StubCatalog, StubServeStats,
        StubTransfers, TransferSource,
    };
    use crate::tui::model::{Msg, Serving, update};

    const W: u16 = 100;
    const H: u16 = 30;

    fn loaded() -> Model {
        crate::tui::model::tests::loaded()
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
        assert!(dump.contains("stub data"), "provenance is always on screen");
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
        assert!(dump.contains("Will not fit on this card"));
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
    fn every_screen_survives_a_terminal_too_small_to_hold_it() {
        // Ratatui panics on out-of-bounds writes, so a cramped terminal is a crash risk in
        // exactly the situation where a crash is least excusable: a split pane.
        let mut m = loaded();
        for screen in [Screen::Devices, Screen::Models, Screen::Download, Screen::Serving] {
            m.screen = screen;
            for (w, h) in [(20, 6), (40, 10), (200, 60)] {
                let _ = frame(&m, w, h);
            }
        }
    }
}
