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
//!   the traits in [`source`], which is what lets this crate compile and its screens be
//!   snapshot-tested before `moearc-device` and `moearc-model` exist.

mod cli;
mod fit;
mod format;
mod plain;
mod source;
mod theme;
mod tui;

use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    let cli = cli::Cli::parse();
    let sources = source::Sources::stub();

    let result =
        if cli.plain_output() { plain::run(&cli, &sources) } else { tui::run(&cli, &sources) };

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
