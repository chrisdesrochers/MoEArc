//! The look.
//!
//! `docs/ux.md` names Charm's `lipgloss` as the reference — rounded borders, generous
//! padding, a restrained palette, aligned columns, nothing shouting. That is a handful of
//! decisions, and they live here so they are made once instead of re-argued in every widget.
//!
//! Colours are 256-colour indices rather than truecolour on purpose: indexed colours are
//! remapped by the user's terminal theme, so the palette stays legible on a light background
//! without us detecting anything.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding};

/// The accent. One hue, spent only on focus and the active tab. A second accent is how an
/// interface starts shouting.
pub const ACCENT: Color = Color::Indexed(105);
/// Present, but not the point: units, hints, inactive tabs.
pub const SUBTLE: Color = Color::Indexed(244);
/// Quieter still: borders and rules, which should be felt rather than read.
pub const FAINT: Color = Color::Indexed(238);
pub const TEXT: Color = Color::Indexed(252);
pub const GOOD: Color = Color::Indexed(78);
pub const WARN: Color = Color::Indexed(179);
pub const BAD: Color = Color::Indexed(174);

pub fn subtle() -> Style {
    Style::new().fg(SUBTLE)
}

pub fn text() -> Style {
    Style::new().fg(TEXT)
}

pub fn accent() -> Style {
    Style::new().fg(ACCENT)
}

pub fn heading() -> Style {
    Style::new().fg(TEXT).add_modifier(Modifier::BOLD)
}

/// A bordered panel: rounded, faint-bordered, padded on all four sides.
///
/// The padding is the whole trick. Terminal UIs default to text jammed against a border, and
/// two columns of horizontal breathing room is most of the distance between "dashboard" and
/// "science experiment".
pub fn panel(title: impl Into<String>) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(FAINT))
        .padding(Padding::symmetric(2, 1))
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled(title.into(), heading()),
            Span::raw(" "),
        ]))
}

/// A panel with no inner padding, for widgets that manage their own margins (gauges, lists
/// with a selection gutter) and would otherwise be indented twice.
pub fn tight_panel(title: impl Into<String>) -> Block<'static> {
    panel(title).padding(Padding::new(2, 2, 1, 0))
}

/// A `label  value` pair, aligned to `width` so stacked pairs form a column.
pub fn field<'a>(label: &'a str, value: impl Into<Span<'a>>, width: usize) -> Line<'a> {
    // A label exactly as wide as the column would otherwise butt straight into its value
    // ("resident experts29 / 32"). Widening here keeps that a local misalignment rather than
    // an unreadable run-together, and means no call site can reintroduce it by renaming.
    let width = width.max(label.chars().count() + 2);
    Line::from(vec![Span::styled(format!("{label:<width$}"), subtle()), value.into()])
}
