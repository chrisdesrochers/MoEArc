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
//! | *(none — deliberately)*     | `moearc bench`                              |
//!
//! 🔴 The last row is the single exception, and it runs the other way: `bench` has a
//! subcommand and **no screen**. A timed measurement taken underneath a renderer redrawing on
//! the same box measures the renderer too, and what `bench` produces is an artefact file meant
//! to be pasted into an issue rather than a view meant to be looked at. `bench-run` is hidden
//! for a different reason — it is one child invocation, spawned by `bench` itself.

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
// `BenchArgs` is several times the size of the other variants, and that is fine here: this
// enum is built once from `argv` at start-up and never stored in a collection or moved in a
// loop, so boxing it would buy nothing and cost clap's derive, which cannot see through a
// `Box` to an `Args` implementation.
#[allow(clippy::large_enum_variant)]
pub enum Command {
    /// List models on this machine.
    Ls(LsArgs),
    /// Download a model from Hugging Face.
    Pull(PullArgs),
    /// Start the OpenAI-compatible server.
    Serve(ServeArgs),
    /// Show what we know about a model, and how it would run here.
    Info(InfoArgs),
    /// Measure this machine, and refuse to print a number it cannot stand behind.
    Bench(BenchArgs),
    /// One timed invocation. Internal: `moearc bench` re-executes itself with this, because
    /// PROTOCOL §5 wants independent processes rather than iterations inside one.
    #[command(hide = true)]
    BenchRun(BenchRunArgs),
}

/// `moearc bench`.
///
/// The flags fall into four groups: **what to run**, **the shape replay**, **the timed run**,
/// and **the incumbent**. Everything that a check compares against is also a flag, so a user
/// who disagrees with a threshold can move it — and the artefact records the value that was
/// actually in force, so moving it is visible rather than silent.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Replay the committed routing traces. Deterministic, needs no GPU, and is the result
    /// that should reproduce on any machine. This is the default when nothing else is asked.
    #[arg(long)]
    pub shape: bool,

    /// Time this machine's engine. Needs `--model`, `--prompt-ids`, a GPU build and a quiet
    /// box, and produces a number that is an artefact of this machine rather than a portable
    /// result.
    #[arg(long)]
    pub absolutes: bool,

    /// Both of the above.
    #[arg(long)]
    pub all: bool,

    /// Run the checks and print the verdict without measuring anything.
    #[arg(long)]
    pub check: bool,

    /// Proceed past a refusal. Every figure produced is then stamped untrusted, and the
    /// artefact says so in the headline position where the result would have been.
    #[arg(long)]
    pub force: bool,

    /// The model to time: a handle from `moearc ls`, or a path to a GGUF.
    #[arg(long, value_name = "MODEL")]
    pub model: Option<String>,

    /// Directory of captured routing traces.
    #[arg(long, default_value = "bench/traces", value_name = "DIR")]
    pub traces: PathBuf,

    /// One capture file. Repeatable, and overrides `--traces`.
    #[arg(long, value_name = "FILE")]
    pub trace: Vec<PathBuf>,

    /// Replay every capture in the directory, not only the ones taken from `--model`.
    ///
    /// 🔴 Off by default, for a correctness reason before a speed one: a hit rate is a
    /// property of one model's routing, and PROTOCOL §9's last rule was learned by carrying a
    /// coverage curve from Qwen3-30B (8 of 128 experts active) onto gpt-oss (4 of 128) and
    /// calling it conservative when it was optimistic. With `--model` given, its own captures
    /// are the ones that describe it. Every skipped file is named in the artefact.
    #[arg(long)]
    pub all_traces: bool,

    /// The dynamic policy under test: `lru`, `lfu`, `lru-k:<k>`, `slru:<pct>`,
    /// `2q:<kin>:<kout>`, `w-tinylfu:<window>:<protected>`, `phase-lru`, `optimal`.
    #[arg(long, default_value = "lru", value_name = "SPEC")]
    pub policy: String,

    /// Capacity ladder for the curve. Defaults to multiples of the trace's own peak
    /// single-step demand, so the same ladder means the same thing on every model.
    #[arg(long, value_delimiter = ',', value_name = "N,...")]
    pub slots: Option<Vec<u32>>,

    /// Also simulate Belady's optimal, to bound what any online policy could reach.
    #[arg(long)]
    pub optimal: bool,

    /// Bytes one resident expert slot occupies, if it cannot be read from the model.
    ///
    /// Without it the staged-byte columns are omitted rather than estimated: hit rate
    /// predicts staged bytes only once the slot size is known, and PROTOCOL §9 forbids
    /// inventing the conversion.
    #[arg(long, value_name = "SIZE", value_parser = crate::host::parse_size)]
    pub slot_bytes: Option<u64>,

    /// Context depths to measure decode throughput at, in tokens.
    #[arg(long, value_delimiter = ',', default_value = "128,512", value_name = "N,...")]
    pub depths: Vec<u32>,

    /// Tokens generated per timed pass. Held constant across depths on purpose, so the router
    /// churn generation itself causes is constant while depth varies.
    #[arg(long, default_value_t = 64, value_name = "N")]
    pub tokens: u32,

    /// Independent invocations of this binary per point. Three is the minimum a stddev is
    /// quoted from.
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub repeats: usize,

    /// Host expert threads. Pinned for every engine, and read back from each.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Expert residency: `all`, `<slots>`, `static:<blocks>`, `plan:<bytes>`.
    #[arg(long, default_value = "all", value_name = "SPEC")]
    pub residency: String,

    /// Which of a block's misses go to the CPU: `off`, `frac:<0..1>`, `over:<n>`, `all`.
    #[arg(long = "host-policy", default_value = "off", value_name = "SPEC")]
    pub host_policy: String,

    /// Context length for the timed session.
    #[arg(long, value_name = "TOKENS")]
    pub ctx: Option<u32>,

    /// File of committed prompt token ids. Required for `--absolutes`.
    ///
    /// 🔴 There is no generator and no default. PROTOCOL §8: a tiled prompt revisits the same
    /// experts and flatters the hit rate, and the exact ids have to be in the repository for
    /// the run to be reproducible from it. `bench/references/*.ids` holds them.
    #[arg(long, value_name = "FILE")]
    pub prompt_ids: Option<PathBuf>,

    /// Also collect a synchronous per-phase profile, for attributing cost between staging and
    /// attention. Destroys overlap, so its absolute values are inflated and only the growth
    /// ratio between depths is a finding.
    #[arg(long)]
    pub attribution: bool,

    /// Path to `llama-bench`. Never searched for: PROTOCOL §2 records a glob-ordered pick that
    /// silently selected a Vulkan build 4.8x slower than SYCL.
    #[arg(long, value_name = "PATH")]
    pub llama_bench: Option<PathBuf>,

    /// Thread counts to sweep the incumbent over. §1 requires it be quoted at its best
    /// configuration, not its first.
    #[arg(long, value_delimiter = ',', value_name = "N,...")]
    pub llama_bench_threads: Vec<usize>,

    /// Extra argument for `llama-bench`, e.g. `--llama-bench-arg "-ncmoe 31"`. Repeatable, and
    /// recorded in the artefact.
    #[arg(long = "llama-bench-arg", value_name = "ARG")]
    pub llama_bench_arg: Vec<String>,

    /// `-r` inside each `llama-bench` process. Independent invocations are `--repeats`.
    #[arg(long, default_value_t = 5, value_name = "N")]
    pub llama_bench_inner_repeats: u32,

    /// The backend every timed run must be on. A mismatch is a refusal, not a note.
    #[arg(long, default_value = "level_zero", value_name = "NAME")]
    pub expect_backend: String,

    /// Override the 1-minute load average above which a timed run is refused.
    ///
    /// The default is one eighth of the machine's logical CPUs with a floor of 2.0; `moearc
    /// bench --check` prints the value in force. Raising it is recorded in the artefact.
    #[arg(long, value_name = "LOAD")]
    pub max_load: Option<f64>,

    /// Block device to attribute `/proc/diskstats` reads to. Resolved from the model's
    /// filesystem when it can be; a device that cannot be attributed is reported as unknown
    /// rather than summed across the machine.
    #[arg(long, value_name = "DEV")]
    pub disk_dev: Option<String>,

    /// Write the artefact here instead of to stdout.
    #[arg(long, value_name = "FILE")]
    pub out: Option<PathBuf>,
}

/// One child of `moearc bench --absolutes`. Not part of the public surface.
#[derive(Debug, Args)]
pub struct BenchRunArgs {
    #[arg(long, value_name = "PATH")]
    pub model: PathBuf,
    #[arg(long, default_value_t = 1, value_name = "TOKENS")]
    pub depth: u32,
    #[arg(long, default_value_t = 64, value_name = "N")]
    pub tokens: u32,
    #[arg(long, default_value = "all", value_name = "SPEC")]
    pub residency: String,
    #[arg(long, default_value = "off", value_name = "SPEC")]
    pub host: String,
    #[arg(long, value_name = "TOKENS")]
    pub ctx: Option<u32>,
    #[arg(long, value_name = "FILE")]
    pub prompt_ids: PathBuf,
    #[arg(long, value_name = "DEV")]
    pub disk_dev: Option<String>,
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
    fn bench_defaults_to_the_result_that_reproduces() {
        // Nothing asked for means the deterministic replay, not a timed run: the safe default
        // is the one that cannot produce an untrustworthy number.
        let Some(Command::Bench(a)) = Cli::parse_from(["moearc", "bench"]).command else {
            panic!("expected bench");
        };
        assert!(!a.absolutes && !a.all && !a.shape);
        assert!(!a.force, "--force must never be on by default");
        assert_eq!(a.repeats, 3);
        assert_eq!(a.policy, "lru");
        assert_eq!(a.expect_backend, "level_zero");
        assert!(a.prompt_ids.is_none());
        assert!(a.max_load.is_none());
    }

    #[test]
    fn bench_takes_its_lists_comma_separated() {
        let Some(Command::Bench(a)) = Cli::parse_from([
            "moearc",
            "bench",
            "--slots",
            "144,300,600",
            "--depths",
            "0,512,2048",
            "--llama-bench-threads",
            "4,8,16",
        ])
        .command
        else {
            panic!("expected bench");
        };
        assert_eq!(a.slots.unwrap(), vec![144, 300, 600]);
        assert_eq!(a.depths, vec![0, 512, 2048]);
        assert_eq!(a.llama_bench_threads, vec![4, 8, 16]);
    }

    #[test]
    fn the_bench_worker_is_hidden_but_reachable() {
        // Hidden from --help because a user never types it; still parseable, because `bench`
        // re-executes this binary with it.
        let Some(Command::BenchRun(a)) = Cli::parse_from([
            "moearc",
            "bench-run",
            "--model",
            "/m.gguf",
            "--depth",
            "512",
            "--prompt-ids",
            "/p.ids",
        ])
        .command
        else {
            panic!("expected bench-run");
        };
        assert_eq!(a.depth, 512);
        assert_eq!(a.tokens, 64);
        assert_eq!(a.model, std::path::PathBuf::from("/m.gguf"));
    }

    #[test]
    fn bench_never_opens_the_interface() {
        // main.rs routes it to the plain path whatever the terminal is; this asserts the
        // subcommand exists to be matched on there.
        assert!(matches!(Cli::parse_from(["moearc", "bench"]).command, Some(Command::Bench(_))));
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
