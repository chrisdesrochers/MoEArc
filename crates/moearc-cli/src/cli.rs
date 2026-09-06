//! The command surface.
//!
//! `docs/ux.md` fixes the shape: four commands, and the zero-argument case is the one a new
//! user reaches first. It also fixes a rule that constrains this file specifically —
//! **nothing may be reachable only through the TUI** — so every screen has a subcommand and
//! every in-TUI choice has a flag. The mapping is exact and worth keeping that way:
//!
//! | TUI                         | flag                                        |
//! | --------------------------- | ------------------------------------------- |
//! | device report (opening view)| `moearc` (or `moearc --no-tui`)             |
//! | model picker                | `moearc ls` / `moearc ls --all`             |
//! | picker → paste a repo id    | `moearc pull <repo-id>`                     |
//! | picker → open a model       | `moearc info <model>`                       |
//! | download screen             | `moearc pull <model>`                       |
//! | serving screen              | `moearc serve <model>`                      |
//! | the split it chose          | `moearc serve <model> --dry-run`            |
//! | context slider              | `--ctx <tokens>`                            |
//! | expert-slot override        | `--moe-cache <slots>`                       |
//! | host RAM budget             | `--host-budget <SIZE>` / `$MOEARC_HOST_BUDGET` |

use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "moearc",
    version,
    about = "Run large mixture-of-experts models on Intel Arc.",
    long_about = "Run large mixture-of-experts models on Intel Arc.\n\n\
                  With no arguments, moearc reports the devices it found and what will fit \
                  on them. Everything the interface can do is also a flag, so the same \
                  binary works in a script.",
    disable_help_subcommand = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Check this context length instead of reporting the largest that fits.
    #[arg(long, value_name = "TOKENS")]
    pub ctx: Option<u32>,

    #[command(flatten)]
    pub global: GlobalArgs,
}

#[derive(Debug, Clone, Args)]
pub struct GlobalArgs {
    /// Where the GGUF files are. Defaults to $MOEARC_MODELS, then a cache directory.
    ///
    /// 🔴 There is no compiled-in path. Models live where the user put them, which is nowhere
    /// this program can guess — see [`crate::catalog::models_dir`] for the order it resolves.
    #[arg(long, global = true, value_name = "DIR")]
    pub models_dir: Option<PathBuf>,

    /// Plain text instead of the interface. Implied by --json, and by a non-terminal stdout.
    #[arg(long, global = true)]
    pub no_tui: bool,

    /// Machine-readable JSON on stdout. Implies --no-tui.
    #[arg(long, global = true)]
    pub json: bool,

    /// Show more: -v adds the numbers behind each decision, -vv adds diagnostics.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Host RAM to keep model weights in, e.g. `24G`. Also `$MOEARC_HOST_BUDGET`.
    ///
    /// Weights past this are paged from the drive on demand — slower, and it still runs. The
    /// value is clamped to what this machine has available, so it cannot starve the OS.
    #[arg(long, global = true, value_name = "SIZE", value_parser = crate::host::parse_size)]
    pub host_budget: Option<u64>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List models on this machine.
    Ls(LsArgs),
    /// Download a model from Hugging Face.
    Pull(PullArgs),
    /// Start the OpenAI-compatible server.
    Serve(ServeArgs),
    /// Show what we know about a model, and how it would run here.
    Info(InfoArgs),
}

#[derive(Debug, Args)]
pub struct LsArgs {
    /// Include curated models that are not downloaded yet.
    #[arg(long)]
    pub all: bool,

    /// Only models whose footprint was measured on an Arc card, never derived from a header.
    #[arg(long)]
    pub measured: bool,
}

#[derive(Debug, Args)]
pub struct PullArgs {
    /// A curated model handle, or any Hugging Face repo id.
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Do not ask before starting a multi-gigabyte download.
    #[arg(short, long)]
    pub yes: bool,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// A model handle from `moearc ls`.
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Context length in tokens. The unit you actually think in; pages are our problem.
    #[arg(long, value_name = "TOKENS")]
    pub ctx: Option<u32>,

    /// Port for the OpenAI-compatible API.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Address to bind. Loopback by default: a local inference server that listens on every
    /// interface the moment it starts is a surprise, not a convenience.
    #[arg(long, default_value = "127.0.0.1", value_name = "ADDR")]
    pub host: String,

    /// Escape hatch: pin the number of resident expert slots instead of computing it.
    #[arg(long, value_name = "SLOTS")]
    pub moe_cache: Option<u32>,

    /// Print the plan and exit without starting the server.
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct InfoArgs {
    /// A curated model handle, or any Hugging Face repo id.
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Plan for this context length instead of the largest that fits.
    #[arg(long, value_name = "TOKENS")]
    pub ctx: Option<u32>,
}

impl Cli {
    /// Whether to take the plain-text path.
    ///
    /// The `IsTerminal` check is the load-bearing part. `--no-tui` covers the user who knows
    /// to ask; the automatic case covers `moearc | grep`, a CI job and an ssh command with no
    /// pty — where a TUI does not degrade into plain text, it emits escape codes into a pipe
    /// and hangs waiting for a keypress that is never coming.
    pub fn plain_output(&self) -> bool {
        self.global.json || self.global.no_tui || !std::io::stdout().is_terminal()
    }

    /// The host RAM budget the user asked for, if any: the flag, then the environment.
    ///
    /// `None` is not zero. It means "no preference", and the engine answers it with a
    /// defensible default rather than with a constant chosen here — the same rule `--ctx`
    /// follows, and for the same reason.
    ///
    /// A malformed environment variable is ignored rather than fatal. A stale export in a shell
    /// profile should not stop the tool from starting; the flag is the channel for a value the
    /// user is asserting right now, and clap rejects a bad one there.
    pub fn host_budget(&self) -> Option<u64> {
        self.global.host_budget.or_else(|| {
            std::env::var(crate::host::HOST_BUDGET_ENV)
                .ok()
                .and_then(|v| crate::host::parse_size(&v).ok())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_surface_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn no_arguments_is_the_device_report() {
        let cli = Cli::parse_from(["moearc"]);
        assert!(cli.command.is_none());
    }

    #[test]
    fn the_model_directory_is_a_flag_and_defaults_to_unset() {
        // Unset here, not defaulted here: the resolution order lives in one place, and a
        // clap default would quietly win over $MOEARC_MODELS.
        assert!(Cli::parse_from(["moearc"]).global.models_dir.is_none());
        let cli = Cli::parse_from(["moearc", "ls", "--models-dir", "/srv/models"]);
        assert_eq!(cli.global.models_dir.as_deref(), Some(std::path::Path::new("/srv/models")));
    }

    #[test]
    fn json_forces_the_plain_path_even_on_a_terminal() {
        let cli = Cli::parse_from(["moearc", "--json"]);
        assert!(cli.plain_output());
    }

    #[test]
    fn global_flags_are_accepted_after_a_subcommand() {
        // The ordering users actually type. Without `global = true` this parse fails, and it
        // fails only at runtime, so it is worth an explicit test.
        let cli = Cli::parse_from(["moearc", "serve", "qwen3-30b-a3b", "--json", "-vv"]);
        assert!(cli.global.json);
        assert_eq!(cli.global.verbose, 2);
    }

    #[test]
    fn serve_defaults_to_loopback_on_a_predictable_port() {
        let Some(Command::Serve(a)) = Cli::parse_from(["moearc", "serve", "m"]).command else {
            panic!("expected serve");
        };
        assert_eq!(a.port, 8080);
        assert_eq!(a.host, "127.0.0.1");
        assert!(a.ctx.is_none());
        assert!(a.moe_cache.is_none());
    }

    #[test]
    fn the_host_budget_is_a_size_and_defaults_to_no_preference() {
        assert!(Cli::parse_from(["moearc"]).global.host_budget.is_none());
        let cli = Cli::parse_from(["moearc", "ls", "--host-budget", "24G"]);
        assert_eq!(cli.global.host_budget, Some(24 << 30));
        // Zero is a setting, not an absence: "keep nothing in RAM".
        assert_eq!(Cli::parse_from(["moearc", "--host-budget", "0"]).global.host_budget, Some(0));
    }

    #[test]
    fn a_malformed_host_budget_is_rejected_at_parse_time() {
        assert!(Cli::try_parse_from(["moearc", "--host-budget", "lots"]).is_err());
    }

    #[test]
    fn context_is_expressed_in_tokens_everywhere_it_appears() {
        let Some(Command::Serve(a)) =
            Cli::parse_from(["moearc", "serve", "m", "--ctx", "32768"]).command
        else {
            panic!("expected serve");
        };
        assert_eq!(a.ctx, Some(32768));
        let cli = Cli::parse_from(["moearc", "--ctx", "4096"]);
        assert_eq!(cli.ctx, Some(4096));
    }
}
