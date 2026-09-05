//! The interface runtime: terminal setup, the event pump, and effects.
//!
//! Everything here is the impure half. State transitions live in [`model::update`]; this
//! module only reads events, performs the [`model::Action`]s that reducer asks for, and feeds
//! the results back in as messages.

pub mod model;
pub mod view;

use std::io::{Stdout, stdout};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::cli::{Cli, Command};
use crate::fit;
use crate::source::Sources;
use model::{Action, Model, Msg, Screen, Serving, TICK_MS, update};

/// Puts the terminal back the way it was, including on a panic.
///
/// Without this a panic mid-frame leaves the user in raw mode inside an alternate screen: no
/// echo, no line editing, and no visible prompt. They have to `reset` blind. A guard costs
/// four lines and removes the worst failure this crate can inflict.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen);
    }
}

pub fn run(cli: &Cli, sources: &Sources) -> Result<ExitCode> {
    let mut m = Model::new(cli.ctx, sources.stubbed);
    let mut pending = vec![Action::Detect];

    // Each subcommand opens on its own screen. That is the other half of the mapping in
    // `cli.rs`: a flag reaches a screen, and a screen has a flag.
    match &cli.command {
        None => {}
        Some(Command::Ls(_)) | Some(Command::Info(_)) => m.screen = Screen::Models,
        Some(Command::Pull(a)) => pending.push(Action::Download(a.model.clone())),
        Some(Command::Serve(a)) => {
            m.ctx_request = a.ctx;
            pending.push(Action::Serve(a.model.clone()));
        }
    }

    enable_raw_mode().context("could not put the terminal into raw mode")?;
    let _guard = TerminalGuard;
    execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal: Terminal<CrosstermBackend<Stdout>> =
        Terminal::new(CrosstermBackend::new(stdout()))?;
    terminal.clear()?;

    let tick = Duration::from_millis(TICK_MS);
    let mut last_tick = Instant::now();
    let mut inbox: Vec<Msg> = Vec::new();

    loop {
        for action in pending.drain(..) {
            inbox.extend(perform(action, sources, &mut m));
        }
        if m.quit {
            break;
        }

        terminal.draw(|f| view::view(&m, f))?;

        // Block on input for whatever is left of the frame rather than spinning: an idle TUI
        // that burns a core is a bug users feel through the fan before they see it.
        let timeout = tick.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
        {
            inbox.push(Msg::Key(key));
        }
        if last_tick.elapsed() >= tick {
            inbox.push(Msg::Tick);
            if m.serving.is_some() {
                inbox.push(Msg::Vitals(sources.serve.sample(m.tick)));
            }
            last_tick = Instant::now();
        }

        for msg in inbox.drain(..) {
            match update(&mut m, msg) {
                Action::None => {}
                Action::Quit => m.quit = true,
                other => pending.push(other),
            }
        }
        if m.quit {
            break;
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// Carry out one effect and return what came back.
///
/// Failures become [`Msg::Failed`] rather than propagating: an unknown model handle is a
/// typo, and tearing the interface down over a typo would be a worse answer than a line in
/// the footer.
fn perform(action: Action, sources: &Sources, m: &mut Model) -> Vec<Msg> {
    match action {
        Action::None => Vec::new(),
        Action::Quit => {
            m.quit = true;
            Vec::new()
        }
        Action::Detect => {
            let mut out = Vec::new();
            match sources.devices.detect() {
                Ok(r) => out.push(Msg::Detected(r)),
                Err(e) => out.push(Msg::Failed(format!("{e:#}"))),
            }
            match sources.models.curated() {
                Ok(c) => out.push(Msg::Catalog(c)),
                Err(e) => out.push(Msg::Failed(format!("{e:#}"))),
            }
            out
        }
        Action::Download(repo) => match sources.transfers.plan(&repo) {
            Ok(plan) => vec![Msg::TransferReady(plan)],
            Err(e) => vec![Msg::Failed(format!("{e:#}"))],
        },
        Action::Serve(id) => match sources.models.resolve(&id) {
            Err(e) => vec![Msg::Failed(format!("{e:#}"))],
            Ok(card) => {
                let Some(device) = m.report.as_ref().and_then(|r| r.primary()) else {
                    return vec![Msg::Failed(
                        "no Level Zero device to serve on — see the Devices screen".to_string(),
                    )];
                };
                let plan = fit::plan(device, &card, m.ctx_request);
                vec![Msg::Serving(Box::new(Serving {
                    model: card.id,
                    host: "127.0.0.1".to_string(),
                    port: 8080,
                    fit: plan,
                    vitals: sources.serve.sample(0),
                    history: Vec::new(),
                    ticks: 0,
                }))]
            }
        },
    }
}
