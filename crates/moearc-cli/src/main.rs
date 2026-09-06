//! The `moearc` binary.
//!
//! `docs/ux.md` is the spec this crate answers to, and its headline constraint is that the
//! tool must not feel like a science experiment: four commands, one binary, the hardware
//! found for you, and the cache split computed rather than asked for.
//!
//! Two structural rules follow from that and shape everything here:
//!
//! * **The interface is the default face, and nothing lives only inside it.** Every screen
//!   has a subcommand and every in-interface action has a flag, so the same binary works in a
//!   shell script and in CI where there is no terminal at all. The mapping is tabulated in
//!   [`cli`] and re-stated in the interface's own help overlay, so it is visible from both
//!   sides.
//! * **The interface owns no facts.** Device enumeration and model metadata arrive through
//!   the traits in [`source`], and are answered by [`detect`] and [`catalog`] against the real
//!   machine. The seam is not scaffolding left over from before those crates existed: a UI
//!   wired straight to hardware cannot be snapshot-tested, because its frames would change
//!   with the card, the driver and the free VRAM at the moment the test ran.

mod bench;
mod catalog;
mod cli;
mod detect;
mod fit;
mod format;
mod host;
mod plain;
mod source;
mod theme;
mod tui;

use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    let cli = cli::Cli::parse();
    // Real detection, with the fixture available behind MOEARC_STUB for interface work
    // on a machine with no Arc card.
    let sources = if std::env::var_os("MOEARC_STUB").is_some() {
        source::Sources::stub()
    } else {
        source::Sources::real(catalog::models_dir(cli.global.models_dir.as_deref()))
    };

    // 🔴 `bench` never opens the interface, whichever way it was invoked. A measurement
    // taken underneath a renderer that is redrawing at 20 Hz on the same box is measuring the
    // renderer as much as the engine, and the artefact it writes is a file rather than a
    // screen. This is the one exception to "every screen has a subcommand": the subcommand
    // has no screen, on purpose, and `cli.rs`'s table records that.
    let plain = cli.plain_output()
        || matches!(cli.command, Some(cli::Command::Bench(_) | cli::Command::BenchRun(_)));
    let result = if plain { plain::run(&cli, &sources) } else { tui::run(&cli, &sources) };

    match result {
        Ok(code) => code,
        Err(err) => {
            // `{err:#}` flattens the whole anyhow chain onto one line. The cause is the
            // point: docs/ux.md rules out errors that report a symptom without naming it.
            eprintln!("moearc: {err:#}");
            ExitCode::FAILURE
        }
    }
}
