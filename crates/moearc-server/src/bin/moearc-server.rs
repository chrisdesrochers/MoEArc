//! `moearc-server` — run the OpenAI-compatible API.
//!
//! Standalone so the serving layer can be started, curled and profiled without the TUI in the
//! way. `moearc serve` in `moearc-cli` is expected to call [`moearc_server::serve`] directly
//! rather than shell out to this.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use moearc_server::chat::ChatTemplate;
use moearc_server::generate::{EchoGenerator, SharedGenerator};
use moearc_server::sampling::SamplingParams;
use moearc_server::state::ServerState;
use moearc_server::tokenize::{Tokenizer, load_for_model};

#[derive(Parser, Debug)]
#[command(name = "moearc-server", about = "OpenAI-compatible inference server", version)]
struct Args {
    /// Address to bind.
    ///
    /// 🔴 Defaults to loopback and is not widened by any other flag. An inference endpoint is
    /// unauthenticated arbitrary compute; exposing it is a deliberate act that has to be typed
    /// out in full.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// A GGUF file, or a directory holding `tokenizer.json`.
    ///
    /// Optional *only* while the generator is a stub: without it the server runs on an
    /// in-memory fixture vocabulary and says so on every `/health`.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Override the tokeniser source (a `tokenizer.json`, or a GGUF).
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Override the chat template with a Jinja file.
    #[arg(long)]
    chat_template: Option<PathBuf>,

    /// The id reported by `/v1/models`. Defaults to the model file's stem.
    #[arg(long)]
    model_id: Option<String>,

    /// Default sampling, used when a request specifies nothing.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// Fix the sampling seed, making every completion reproducible.
    #[arg(long)]
    seed: Option<u64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "moearc_server=info,tower_http=warn".into()),
        )
        .init();

    let args = Args::parse();

    let tokenizer_path = args.tokenizer.clone().or_else(|| args.model.clone());
    let tokenizer = match &tokenizer_path {
        Some(p) => {
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
                Tokenizer::from_tokenizer_json(p)?
            } else {
                load_for_model(p)?
            }
        }
        None => moearc_server::testing::tiny_tokenizer(),
    };

    let template = match &args.chat_template {
        Some(p) => {
            let src = std::fs::read_to_string(p)
                .with_context(|| format!("reading chat template {}", p.display()))?;
            ChatTemplate::from_source(&src, moearc_server::chat::TemplateSource::JinjaFile)?
        }
        None => match &args.model {
            Some(p) => ChatTemplate::discover(p)?,
            None => ChatTemplate::chatml(),
        },
    };

    let model_id = args.model_id.clone().unwrap_or_else(|| {
        // A directory keeps its whole name; only a file loses its extension. `file_stem` on
        // both turns the directory `Qwen3.6-35B-A3B-NVFP4` into the model id `Qwen3`, because
        // `.6-35B-A3B-NVFP4` looks like an extension.
        args.model
            .as_deref()
            .and_then(|p| if p.is_dir() { p.file_name() } else { p.file_stem() })
            .map_or_else(|| "moearc-echo".to_string(), |s| s.to_string_lossy().into_owned())
    });

    // ─────────────────────────────────────────────────────────────────────────────────────
    // 🔴 THE INTEGRATION POINT. This is the one line that changes when the engine lands:
    //
    //     let generator: SharedGenerator = Arc::new(moearc_engine::Session::load(&path)?);
    //
    // Nothing else in this crate refers to a concrete generator type. See `generate.rs` for
    // the contract the implementation has to meet.
    // ─────────────────────────────────────────────────────────────────────────────────────
    let generator: SharedGenerator = Arc::new(EchoGenerator::new(tokenizer.vocab_size()));

    let defaults = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        max_tokens: args.max_tokens,
        stop_tokens: Vec::new(),
        stop_strings: Vec::new(),
    };

    let state = Arc::new(ServerState::new(
        Arc::new(tokenizer),
        Arc::new(template),
        generator,
        model_id,
        defaults,
    ));

    let addr = SocketAddr::new(args.host, args.port);
    let (listener, local) =
        moearc_server::bind(addr).await.with_context(|| format!("binding {addr}"))?;

    // docs/ux.md: "Startup prints what it decided ... so a user can see the reasoning without
    // enabling debug logging."
    println!("{}", state.banner());
    println!("\nlistening   http://{local}");
    if !local.ip().is_loopback() {
        println!(
            "\n\u{26a0}  Bound to a non-loopback address. This endpoint has no authentication."
        );
    }

    moearc_server::serve(listener, state, shutdown_signal()).await?;
    Ok(())
}

/// Ctrl-C, or SIGTERM under a supervisor.
async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok() };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok()?.recv().await
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<Option<()>>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}
