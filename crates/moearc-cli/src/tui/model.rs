//! The Elm loop: state, messages, and the reducer.
//!
//! `docs/ux.md` points at Bubbletea, and the half of Bubbletea that ports is the
//! architecture: state is one value, every input is a message, and the only thing that
//! changes state is [`update`]. Ratatui imposes no architecture, so this is a hand-rolled
//! pattern rather than a framework — the reducer below is the whole of it.
//!
//! The rule that keeps it worth having: [`update`] performs no I/O and reads no clock.
//! Effects are requested by returning an [`Action`] and are carried out by the runtime in
//! `super::run`, which feeds the result back as another [`Msg`]. That is what makes the tests
//! at the bottom of this file possible — key event in, asserted state out, no terminal
//! anywhere.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use throbber_widgets_tui::ThrobberState;
use tui_input::Input;
use tui_input::backend::crossterm::to_input_request;

use crate::fit::{self, Fit};
use crate::source::{DeviceReport, ModelCard, ServeSample, TransferPlan};

/// The frame interval. Also the unit progress is integrated over, so the two cannot drift
/// apart the way a separate animation clock would.
pub const TICK_MS: u64 = 100;

/// How many samples the serving sparkline keeps. Sized to the panel, not to a duration:
/// history nobody can see is a memory leak with a nice name.
pub const VITALS_HISTORY: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Devices,
    Models,
    Download,
    Serving,
}

impl Screen {
    pub fn title(self) -> &'static str {
        match self {
            Self::Devices => "Devices",
            Self::Models => "Models",
            Self::Download => "Download",
            Self::Serving => "Serving",
        }
    }
}

/// A download in flight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Download {
    pub plan: TransferPlan,
    pub done_bytes: u64,
    pub ticks: u64,
    pub cancelled: bool,
}

impl Download {
    pub fn fraction(&self) -> f64 {
        if self.plan.total_bytes == 0 {
            return 1.0;
        }
        (self.done_bytes as f64 / self.plan.total_bytes as f64).clamp(0.0, 1.0)
    }

    pub fn finished(&self) -> bool {
        self.done_bytes >= self.plan.total_bytes
    }

    /// Seconds left at the current rate, or `None` once there is nothing left to wait for.
    pub fn eta_secs(&self) -> Option<u64> {
        if self.finished() || self.cancelled || self.plan.bytes_per_sec == 0 {
            return None;
        }
        Some((self.plan.total_bytes - self.done_bytes) / self.plan.bytes_per_sec)
    }
}

/// A running server.
#[derive(Debug, Clone, PartialEq)]
pub struct Serving {
    pub model: String,
    pub host: String,
    pub port: u16,
    /// The split that was chosen, kept so the panel can show the reasoning rather than only
    /// the outcome. `docs/ux.md`: startup prints what it decided.
    pub fit: Fit,
    pub vitals: ServeSample,
    /// Recent tokens/sec, oldest first.
    pub history: Vec<u64>,
    pub ticks: u64,
}

/// Everything on screen.
pub struct Model {
    pub screen: Screen,
    /// `None` while detection is in flight — which is what puts the spinner up.
    pub report: Option<DeviceReport>,
    pub device_row: usize,
    pub models: Vec<ModelCard>,
    /// Parallel to `models`. Recomputed whenever either input changes, never cached across a
    /// device change: a stale fit is a wrong answer that looks authoritative.
    pub fits: Vec<Fit>,
    pub model_row: usize,
    pub repo_input: Input,
    pub editing: bool,
    pub download: Option<Download>,
    pub serving: Option<Serving>,
    /// The context the user asked for, or `None` for "whatever fits". Not defaulted to a
    /// constant: see `fit::plan`.
    pub ctx_request: Option<u32>,
    pub tick: u64,
    pub throbber: ThrobberState,
    pub help: bool,
    pub status: Option<String>,
    /// Which parts of what is on screen are fixtures, or `None` when none are.
    ///
    /// One field rather than a flag plus a string. The flag and the string were briefly both
    /// here, and two fields describing one fact is how a footer ends up reading "stub data"
    /// over measured numbers: nothing makes the second one move when the first does. See
    /// `Sources::stub_parts`.
    pub provenance: Option<&'static str>,
    /// Where the catalogue looked for models. Shown only when it found none — "no models" is
    /// a symptom, and the directory it searched is the cause.
    pub catalog_location: Option<String>,
    pub quit: bool,
}

impl Model {
    pub fn new(ctx_request: Option<u32>, provenance: Option<&'static str>) -> Self {
        Self {
            screen: Screen::Devices,
            report: None,
            device_row: 0,
            models: Vec::new(),
            fits: Vec::new(),
            model_row: 0,
            repo_input: Input::default(),
            editing: false,
            download: None,
            serving: None,
            ctx_request,
            tick: 0,
            throbber: ThrobberState::default(),
            help: false,
            status: None,
            provenance,
            catalog_location: None,
            quit: false,
        }
    }

    /// The tabs, in order. Download and Serving appear only once they exist: a tab that leads
    /// to an empty screen teaches the user the interface is decorative.
    pub fn screens(&self) -> Vec<Screen> {
        let mut v = vec![Screen::Devices, Screen::Models];
        if self.download.is_some() {
            v.push(Screen::Download);
        }
        if self.serving.is_some() {
            v.push(Screen::Serving);
        }
        v
    }

    pub fn selected_model(&self) -> Option<&ModelCard> {
        self.models.get(self.model_row)
    }

    pub fn selected_fit(&self) -> Option<&Fit> {
        self.fits.get(self.model_row)
    }

    /// Re-plan every model against the primary device.
    ///
    /// Cheap enough to redo wholesale on any change, and doing so removes the class of bug
    /// where one row is planned against the old free-VRAM figure.
    pub fn recompute_fits(&mut self) {
        let Some(device) = self.report.as_ref().and_then(|r| r.primary()) else {
            self.fits.clear();
            return;
        };
        self.fits = self.models.iter().map(|m| fit::plan(device, m, self.ctx_request)).collect();
    }

    fn move_selection(&mut self, delta: isize) {
        let (cursor, len) = match self.screen {
            Screen::Devices => {
                (&mut self.device_row, self.report.as_ref().map_or(0, |r| r.devices.len()))
            }
            Screen::Models => (&mut self.model_row, self.models.len()),
            _ => return,
        };
        if len == 0 {
            return;
        }
        // Wrap rather than clamp: a four-row list is faster to traverse by wrapping, and
        // there is no scroll position to lose.
        let next = (*cursor as isize + delta).rem_euclid(len as isize) as usize;
        *cursor = next;
    }

    fn cycle_screen(&mut self, delta: isize) {
        let screens = self.screens();
        let at = screens.iter().position(|s| *s == self.screen).unwrap_or(0);
        let next = (at as isize + delta).rem_euclid(screens.len() as isize) as usize;
        self.screen = screens[next];
    }
}

/// Everything that can happen.
#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    Key(KeyEvent),
    /// One frame interval elapsed.
    Tick,
    /// Device detection finished.
    Detected(DeviceReport),
    /// The catalog arrived.
    Catalog(Vec<ModelCard>),
    /// A download was sized and may begin.
    TransferReady(TransferPlan),
    /// A server came up.
    Serving(Box<Serving>),
    /// One sample of a running server's vitals.
    Vitals(ServeSample),
    /// Something failed, with a cause the user can act on.
    Failed(String),
}

/// An effect the runtime should perform. Returned rather than executed, so [`update`] stays
/// pure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    None,
    Quit,
    /// Enumerate devices and load the catalog.
    Detect,
    /// Size and start a download of this handle or repo id.
    Download(String),
    /// Start a server for this handle.
    Serve(String),
}

/// The reducer.
pub fn update(m: &mut Model, msg: Msg) -> Action {
    match msg {
        Msg::Tick => {
            m.tick += 1;
            m.throbber.calc_next();
            advance_download(m);
            if let Some(s) = m.serving.as_mut() {
                s.ticks += 1;
            }
            Action::None
        }
        Msg::Detected(report) => {
            m.device_row = 0;
            m.report = Some(report);
            m.recompute_fits();
            Action::None
        }
        Msg::Catalog(models) => {
            m.model_row = 0;
            m.models = models;
            m.recompute_fits();
            Action::None
        }
        Msg::TransferReady(plan) => {
            m.status = None;
            m.download =
                Some(Download { done_bytes: plan.resume_from, plan, ticks: 0, cancelled: false });
            m.screen = Screen::Download;
            Action::None
        }
        Msg::Serving(serving) => {
            m.status = None;
            m.serving = Some(*serving);
            m.screen = Screen::Serving;
            Action::None
        }
        Msg::Vitals(sample) => {
            if let Some(s) = m.serving.as_mut() {
                s.vitals = sample;
                s.history.push(sample.tokens_per_sec.round().max(0.0) as u64);
                if s.history.len() > VITALS_HISTORY {
                    s.history.remove(0);
                }
            }
            Action::None
        }
        Msg::Failed(why) => {
            m.status = Some(why);
            Action::None
        }
        Msg::Key(key) => on_key(m, key),
    }
}

/// Integrate the download over one frame.
///
/// Deliberately derived from tick count and the plan's rate rather than from wall time: a
/// paused process should not "catch up" by jumping the bar, and a test should be able to
/// assert exactly where the bar is after N frames.
fn advance_download(m: &mut Model) {
    let Some(d) = m.download.as_mut() else { return };
    if d.cancelled || d.finished() {
        return;
    }
    d.ticks += 1;
    let per_tick = d.plan.bytes_per_sec * TICK_MS / 1000;
    d.done_bytes = (d.done_bytes + per_tick).min(d.plan.total_bytes);
}

fn on_key(m: &mut Model, key: KeyEvent) -> Action {
    // Terminals that report key releases would otherwise fire every binding twice.
    if key.kind != KeyEventKind::Press {
        return Action::None;
    }

    // Ctrl-C outranks everything, including a focused text field. A text field that traps the
    // universal quit key is the single most infuriating thing a TUI can do.
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return Action::Quit;
    }

    if m.editing {
        return on_key_editing(m, key);
    }

    match key.code {
        KeyCode::Char('q') => return Action::Quit,
        KeyCode::Char('?') => {
            m.help = !m.help;
            return Action::None;
        }
        KeyCode::Tab => {
            m.cycle_screen(1);
            return Action::None;
        }
        KeyCode::BackTab => {
            m.cycle_screen(-1);
            return Action::None;
        }
        KeyCode::Char(c @ '1'..='4') => {
            let want = c as usize - '1' as usize;
            let screens = m.screens();
            if let Some(s) = screens.get(want) {
                m.screen = *s;
            }
            return Action::None;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            m.move_selection(-1);
            return Action::None;
        }
        KeyCode::Down | KeyCode::Char('j') => {
            m.move_selection(1);
            return Action::None;
        }
        _ => {}
    }

    match m.screen {
        Screen::Devices => match key.code {
            KeyCode::Char('r') => {
                // Drop the old report first: the spinner is the honest state while a fresh
                // enumeration is running, and leaving the stale table up implies otherwise.
                m.report = None;
                m.fits.clear();
                Action::Detect
            }
            KeyCode::Enter => {
                m.screen = Screen::Models;
                Action::None
            }
            _ => Action::None,
        },
        Screen::Models => match key.code {
            KeyCode::Char('/') => {
                m.editing = true;
                Action::None
            }
            KeyCode::Char('d') => {
                m.selected_model().map_or(Action::None, |c| Action::Download(c.id.clone()))
            }
            KeyCode::Char('s') => {
                m.selected_model().map_or(Action::None, |c| Action::Serve(c.id.clone()))
            }
            KeyCode::Enter => match m.selected_model() {
                // Enter does the obvious next thing for the row under the cursor: serve what
                // is already here, fetch what is not. Nothing is hidden behind it that is not
                // also on its own key.
                Some(c) if c.local => Action::Serve(c.id.clone()),
                Some(c) => Action::Download(c.id.clone()),
                None => Action::None,
            },
            KeyCode::Esc => {
                m.screen = Screen::Devices;
                Action::None
            }
            _ => Action::None,
        },
        Screen::Download => match key.code {
            KeyCode::Char('x') => {
                if let Some(d) = m.download.as_mut() {
                    d.cancelled = true;
                }
                Action::None
            }
            KeyCode::Esc | KeyCode::Enter => {
                m.screen = Screen::Models;
                Action::None
            }
            _ => Action::None,
        },
        Screen::Serving => match key.code {
            KeyCode::Char('s') => {
                m.serving = None;
                m.screen = Screen::Models;
                Action::None
            }
            KeyCode::Esc => {
                m.screen = Screen::Models;
                Action::None
            }
            _ => Action::None,
        },
    }
}

/// Keys while the repo-id field has focus.
///
/// Only Enter and Esc are intercepted; everything else goes to `tui-input`, which owns
/// cursor motion, word deletion and paste. Re-implementing those is how a text field ends up
/// subtly wrong.
fn on_key_editing(m: &mut Model, key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Esc => {
            m.editing = false;
            m.repo_input.reset();
            Action::None
        }
        KeyCode::Enter => {
            let value = m.repo_input.value_and_reset();
            m.editing = false;
            if value.trim().is_empty() {
                Action::None
            } else {
                Action::Download(value.trim().to_string())
            }
        }
        _ => {
            if let Some(req) = to_input_request(&crossterm::event::Event::Key(key)) {
                m.repo_input.handle(req);
            }
            Action::None
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::source::{
        DeviceSource, ModelCatalog, ServeStats, StubCatalog, StubDeviceSource, StubServeStats,
        StubTransfers, TransferSource,
    };

    fn key(code: KeyCode) -> Msg {
        Msg::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    /// A model in the state the runtime hands to `view` once startup has settled.
    pub(crate) fn loaded() -> Model {
        let mut m = Model::new(None, Some(crate::source::Sources::stub().stub_parts));
        update(&mut m, Msg::Detected(StubDeviceSource.detect().unwrap()));
        update(&mut m, Msg::Catalog(StubCatalog.curated().unwrap()));
        m
    }

    #[test]
    fn detection_populates_the_table_and_plans_every_model() {
        let m = loaded();
        assert_eq!(m.report.as_ref().unwrap().devices.len(), 3);
        assert_eq!(m.fits.len(), m.models.len());
        assert!(m.fits.iter().any(|f| f.fits()), "at least one model should fit");
    }

    #[test]
    fn the_spinner_state_is_re_entered_on_a_rescan() {
        let mut m = loaded();
        assert_eq!(update(&mut m, key(KeyCode::Char('r'))), Action::Detect);
        assert!(m.report.is_none(), "the stale table must come down with the request");
        assert!(m.fits.is_empty());
    }

    #[test]
    fn selection_wraps_in_both_directions() {
        let mut m = loaded();
        update(&mut m, key(KeyCode::Up));
        assert_eq!(m.device_row, 2, "up from the first row wraps to the last");
        update(&mut m, key(KeyCode::Down));
        assert_eq!(m.device_row, 0);
        update(&mut m, key(KeyCode::Char('j')));
        assert_eq!(m.device_row, 1, "vim keys are the same binding");
    }

    #[test]
    fn tab_only_cycles_screens_that_exist() {
        let mut m = loaded();
        update(&mut m, key(KeyCode::Tab));
        assert_eq!(m.screen, Screen::Models);
        update(&mut m, key(KeyCode::Tab));
        assert_eq!(m.screen, Screen::Devices, "no download or server yet, so it wraps at two");
    }

    #[test]
    fn enter_serves_a_local_model_and_downloads_a_remote_one() {
        let mut m = loaded();
        m.screen = Screen::Models;
        m.model_row = 0; // qwen3-30b-a3b, local
        assert_eq!(update(&mut m, key(KeyCode::Enter)), Action::Serve("qwen3-30b-a3b".into()));
        m.model_row = 2; // mixtral-8x7b, not local
        assert_eq!(update(&mut m, key(KeyCode::Enter)), Action::Download("mixtral-8x7b".into()));
    }

    #[test]
    fn every_enter_shortcut_is_also_its_own_key() {
        // The docs/ux.md rule that nothing is TUI-only has an in-TUI corollary: nothing is
        // Enter-only either, or the flag mapping in cli.rs has no counterpart to point at.
        let mut m = loaded();
        m.screen = Screen::Models;
        m.model_row = 0;
        assert_eq!(
            update(&mut m, key(KeyCode::Char('d'))),
            Action::Download("qwen3-30b-a3b".into())
        );
        assert_eq!(update(&mut m, key(KeyCode::Char('s'))), Action::Serve("qwen3-30b-a3b".into()));
    }

    #[test]
    fn slash_focuses_the_repo_field_and_enter_submits_what_was_typed() {
        let mut m = loaded();
        m.screen = Screen::Models;
        update(&mut m, key(KeyCode::Char('/')));
        assert!(m.editing);
        for c in "org/model-GGUF".chars() {
            update(&mut m, key(KeyCode::Char(c)));
        }
        assert_eq!(m.repo_input.value(), "org/model-GGUF");
        assert_eq!(update(&mut m, key(KeyCode::Enter)), Action::Download("org/model-GGUF".into()));
        assert!(!m.editing);
        assert_eq!(m.repo_input.value(), "", "the field clears so the next paste starts empty");
    }

    #[test]
    fn a_focused_field_swallows_q_but_never_ctrl_c() {
        let mut m = loaded();
        m.screen = Screen::Models;
        update(&mut m, key(KeyCode::Char('/')));
        assert_eq!(update(&mut m, key(KeyCode::Char('q'))), Action::None);
        assert_eq!(m.repo_input.value(), "q", "q is text here, not a quit");
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(update(&mut m, Msg::Key(ctrl_c)), Action::Quit);
    }

    #[test]
    fn escape_abandons_the_field_without_starting_a_download() {
        let mut m = loaded();
        m.screen = Screen::Models;
        update(&mut m, key(KeyCode::Char('/')));
        update(&mut m, key(KeyCode::Char('x')));
        assert_eq!(update(&mut m, key(KeyCode::Esc)), Action::None);
        assert!(!m.editing);
        assert_eq!(m.repo_input.value(), "");
    }

    #[test]
    fn a_download_advances_by_exactly_one_frame_of_bytes_per_tick() {
        let mut m = loaded();
        let plan = StubTransfers.plan("mixtral-8x7b").unwrap();
        let per_tick = plan.bytes_per_sec * TICK_MS / 1000;
        update(&mut m, Msg::TransferReady(plan));
        assert_eq!(m.screen, Screen::Download);
        update(&mut m, Msg::Tick);
        update(&mut m, Msg::Tick);
        assert_eq!(m.download.as_ref().unwrap().done_bytes, per_tick * 2);
    }

    #[test]
    fn a_cancelled_download_stops_moving_and_reports_no_eta() {
        let mut m = loaded();
        update(&mut m, Msg::TransferReady(StubTransfers.plan("mixtral-8x7b").unwrap()));
        update(&mut m, Msg::Tick);
        let at_cancel = m.download.as_ref().unwrap().done_bytes;
        update(&mut m, key(KeyCode::Char('x')));
        update(&mut m, Msg::Tick);
        let d = m.download.as_ref().unwrap();
        assert_eq!(d.done_bytes, at_cancel);
        assert_eq!(d.eta_secs(), None);
    }

    #[test]
    fn the_vitals_history_is_bounded_by_what_the_panel_can_show() {
        let mut m = loaded();
        let fit = m.fits[0].clone();
        update(
            &mut m,
            Msg::Serving(Box::new(Serving {
                model: "qwen3-30b-a3b".into(),
                host: "127.0.0.1".into(),
                port: 8080,
                fit,
                vitals: StubServeStats.sample(0),
                history: Vec::new(),
                ticks: 0,
            })),
        );
        for t in 0..(VITALS_HISTORY as u64 * 2) {
            update(&mut m, Msg::Vitals(StubServeStats.sample(t)));
        }
        assert_eq!(m.serving.as_ref().unwrap().history.len(), VITALS_HISTORY);
    }

    #[test]
    fn stopping_a_server_removes_its_tab() {
        let mut m = loaded();
        let fit = m.fits[0].clone();
        update(
            &mut m,
            Msg::Serving(Box::new(Serving {
                model: "qwen3-30b-a3b".into(),
                host: "127.0.0.1".into(),
                port: 8080,
                fit,
                vitals: StubServeStats.sample(0),
                history: Vec::new(),
                ticks: 0,
            })),
        );
        assert_eq!(m.screens().len(), 3);
        update(&mut m, key(KeyCode::Char('s')));
        assert!(m.serving.is_none());
        assert_eq!(m.screen, Screen::Models);
        assert_eq!(m.screens().len(), 2);
    }

    #[test]
    fn a_failure_becomes_a_status_line_rather_than_a_panic() {
        let mut m = loaded();
        update(&mut m, Msg::Failed("unknown model `nope`".into()));
        assert_eq!(m.status.as_deref(), Some("unknown model `nope`"));
        assert!(!m.quit);
    }

    #[test]
    fn q_and_ctrl_c_both_quit_from_the_top_level() {
        let mut m = loaded();
        assert_eq!(update(&mut m, key(KeyCode::Char('q'))), Action::Quit);
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(update(&mut m, Msg::Key(ctrl_c)), Action::Quit);
    }

    #[test]
    fn key_releases_are_ignored_so_bindings_do_not_fire_twice() {
        let mut m = loaded();
        let mut release = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        assert_eq!(update(&mut m, Msg::Key(release)), Action::None);
        assert_eq!(m.device_row, 0);
    }
}
