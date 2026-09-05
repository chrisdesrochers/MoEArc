//! The HTTP surface: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.

use std::convert::Infallible;

use axum::Router;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::{ApiError, stream_error_payload};
use crate::generate::{GenerationStats, Generator, StopReason};
use crate::openai::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, CompletionChoice, CompletionRequest, CompletionResponse,
    Model, ModelList, PromptField, ResponseMessage, StreamOptions, Usage, response_id, unix_now,
};
use crate::sampling::SamplingParams;
use crate::state::AppState;
use crate::tokenize::{IncrementalDecoder, Tokenizer};

/// Channel depth between the blocking generator thread and the SSE writer.
///
/// Bounded on purpose. An unbounded channel lets a fast generator queue an entire completion in
/// memory for a client that has stopped reading; a bounded one applies backpressure, which is
/// also how a disconnect is noticed promptly rather than after the whole generation is spent.
const STREAM_BUFFER: usize = 64;

/// Build the router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model}", get(retrieve_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .fallback(not_found)
        .with_state(state)
}

async fn not_found(uri: axum::http::Uri) -> ApiError {
    ApiError::not_found(format!(
        "no route for `{uri}` — MoEArc serves /v1/chat/completions, /v1/completions, \
         /v1/models and /health"
    ))
}

/// Liveness plus the facts an operator needs before trusting an answer.
///
/// `stubbed` is here for the same reason `moearc-cli` surfaces it: a health check that returns
/// `ok` while the generator is a fixture is a health check that lies.
async fn health(State(state): State<AppState>) -> Response {
    axum::Json(json!({
        "status": "ok",
        "model": state.model_id,
        "tokenizer": state.tokenizer.source().to_string(),
        "vocab_size": state.tokenizer.vocab_size(),
        "chat_template": state.template.source().to_string(),
        "generator": state.generator.name(),
        "stubbed": state.generator.is_stub(),
        "warnings": state.vocab_mismatch().map(|w| vec![w]).unwrap_or_default(),
    }))
    .into_response()
}

async fn list_models(State(state): State<AppState>) -> Response {
    axum::Json(ModelList { object: "list", data: vec![model_entry(&state.model_id)] })
        .into_response()
}

async fn retrieve_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Response, ApiError> {
    if model == state.model_id {
        Ok(axum::Json(model_entry(&state.model_id)).into_response())
    } else {
        Err(ApiError::not_found(format!(
            "this server serves `{}` only; it was asked for `{model}`",
            state.model_id
        )))
    }
}

fn model_entry(id: &str) -> Model {
    Model { id: id.to_string(), object: "model", created: unix_now(), owned_by: "moearc" }
}

/// Parse a JSON body into `T`, reporting failures in OpenAI's envelope.
///
/// Deliberately not `axum::Json`: its rejection body is axum's own plain-text shape, which an
/// OpenAI client cannot parse and will surface as an empty error. serde's message already names
/// the offending field and position, so it is passed through.
fn parse_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))
}

// ---------------------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let req: ChatCompletionRequest = parse_body(&body)?;
    req.sampling.validate().map_err(|m| ApiError::invalid_param("n", m))?;
    if req.messages.is_empty() {
        return Err(ApiError::invalid_param(
            "messages",
            "`messages` must contain at least one message",
        ));
    }

    let messages = serde_json::to_value(&req.messages)
        .map_err(|e| ApiError::internal(&anyhow::anyhow!("re-serialising messages: {e}")))?;
    let prompt = state
        .template
        .render(&messages, req.tools.as_ref(), true, &state.template_context())
        // A template that calls `raise_exception` is rejecting the *client's* input, so this is
        // a 400 naming the model's own reason, not a 500.
        .map_err(|e| ApiError::invalid_param("messages", format!("{e:#}")))?;

    let params = req.sampling.to_params(&state.defaults);
    let prompt_tokens =
        state.tokenizer.encode(&prompt, false).map_err(|e| ApiError::internal(&e))?;
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    if req.stream {
        Ok(chat_stream(state, prompt_tokens, params, model, req.stream_options))
    } else {
        let out = run_blocking(state.clone(), prompt_tokens, params).await?;
        Ok(axum::Json(ChatCompletionResponse {
            id: response_id("chatcmpl"),
            object: "chat.completion",
            created: unix_now(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage { role: "assistant".into(), content: out.text },
                logprobs: None,
                finish_reason: out.stats.stop_reason.as_openai().to_string(),
            }],
            usage: Usage::new(out.stats.prompt_tokens, out.stats.completion_tokens),
        })
        .into_response())
    }
}

fn chat_stream(
    state: AppState,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    model: String,
    opts: Option<StreamOptions>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(STREAM_BUFFER);
    let id = response_id("chatcmpl");
    let created = unix_now();
    let include_usage = opts.is_some_and(|o| o.include_usage);

    let chunk = move |id: &str, model: &str, delta: ChatDelta, finish: Option<String>| {
        ChatCompletionChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta,
                logprobs: None,
                finish_reason: finish,
            }],
            usage: None,
        }
    };

    // `spawn_blocking`, not `spawn`: `Generator::generate` is synchronous and will hold a
    // device fence for the length of a generation. On a worker thread that stalls every other
    // request on the same runtime thread.
    tokio::task::spawn_blocking(move || {
        // The role-only opening chunk is what OpenAI sends, and several clients use it to
        // decide the message exists before any text arrives.
        let first =
            chunk(&id, &model, ChatDelta { role: Some("assistant".into()), content: None }, None);
        if send_json(&tx, &first).is_err() {
            return;
        }

        let result = generate_text(
            state.tokenizer.as_ref(),
            state.generator.as_ref(),
            &prompt_tokens,
            &params,
            |text| {
                let c = chunk(
                    &id,
                    &model,
                    ChatDelta { role: None, content: Some(text.to_string()) },
                    None,
                );
                send_json(&tx, &c).is_ok()
            },
        );

        match result {
            Ok(out) => {
                let last = chunk(
                    &id,
                    &model,
                    ChatDelta { role: None, content: None },
                    Some(out.stats.stop_reason.as_openai().to_string()),
                );
                let _ = send_json(&tx, &last);
                if include_usage {
                    // OpenAI's usage chunk carries an empty `choices` array.
                    let mut u = chunk(&id, &model, ChatDelta { role: None, content: None }, None);
                    u.choices.clear();
                    u.usage =
                        Some(Usage::new(out.stats.prompt_tokens, out.stats.completion_tokens));
                    let _ = send_json(&tx, &u);
                }
            }
            Err(e) => {
                let _ = tx.blocking_send(Ok(Event::default()
                    .event("error")
                    .data(stream_error_payload(&format!("{e:#}")))));
            }
        }
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------------------------------
// Legacy completions
// ---------------------------------------------------------------------------------------

async fn completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let req: CompletionRequest = parse_body(&body)?;
    req.sampling.validate().map_err(|m| ApiError::invalid_param("n", m))?;

    // The prompt reaches the model verbatim: no chat template. That is the point of this
    // endpoint, and applying one would silently change what a base-model user asked for.
    let (prompt_text, prompt_tokens) = match &req.prompt {
        PromptField::Text(t) => {
            (t.clone(), state.tokenizer.encode(t, false).map_err(|e| ApiError::internal(&e))?)
        }
        PromptField::Texts(v) => {
            let [t] = v.as_slice() else {
                return Err(ApiError::invalid_param(
                    "prompt",
                    format!(
                        "`prompt` held {} prompts — MoEArc completes one prompt per request. \
                         Send them as separate requests.",
                        v.len()
                    ),
                ));
            };
            (t.clone(), state.tokenizer.encode(t, false).map_err(|e| ApiError::internal(&e))?)
        }
        PromptField::Tokens(ids) => {
            let text = state.tokenizer.decode(ids, false).map_err(|e| ApiError::internal(&e))?;
            (text, ids.clone())
        }
    };

    let params = req.sampling.to_params(&state.defaults);
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let echo = req.echo.then_some(prompt_text);

    if req.stream {
        Ok(completion_stream(state, prompt_tokens, params, model, req.stream_options, echo))
    } else {
        let out = run_blocking(state.clone(), prompt_tokens, params).await?;
        let text = match echo {
            Some(p) => p + &out.text,
            None => out.text,
        };
        Ok(axum::Json(CompletionResponse {
            id: response_id("cmpl"),
            object: "text_completion",
            created: unix_now(),
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                logprobs: None,
                finish_reason: Some(out.stats.stop_reason.as_openai().to_string()),
            }],
            usage: Some(Usage::new(out.stats.prompt_tokens, out.stats.completion_tokens)),
        })
        .into_response())
    }
}

fn completion_stream(
    state: AppState,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    model: String,
    opts: Option<StreamOptions>,
    echo: Option<String>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(STREAM_BUFFER);
    let id = response_id("cmpl");
    let created = unix_now();
    let include_usage = opts.is_some_and(|o| o.include_usage);

    let chunk =
        move |id: &str, model: &str, text: String, finish: Option<String>| CompletionResponse {
            id: id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            choices: vec![CompletionChoice {
                index: 0,
                text,
                logprobs: None,
                finish_reason: finish,
            }],
            usage: None,
        };

    tokio::task::spawn_blocking(move || {
        if let Some(p) = echo
            && send_json(&tx, &chunk(&id, &model, p, None)).is_err()
        {
            return;
        }
        let result = generate_text(
            state.tokenizer.as_ref(),
            state.generator.as_ref(),
            &prompt_tokens,
            &params,
            |text| send_json(&tx, &chunk(&id, &model, text.to_string(), None)).is_ok(),
        );
        match result {
            Ok(out) => {
                let mut last = chunk(
                    &id,
                    &model,
                    String::new(),
                    Some(out.stats.stop_reason.as_openai().to_string()),
                );
                if include_usage {
                    last.usage =
                        Some(Usage::new(out.stats.prompt_tokens, out.stats.completion_tokens));
                }
                let _ = send_json(&tx, &last);
            }
            Err(e) => {
                let _ = tx.blocking_send(Ok(Event::default()
                    .event("error")
                    .data(stream_error_payload(&format!("{e:#}")))));
            }
        }
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()).into_response()
}

// ---------------------------------------------------------------------------------------
// Generation plumbing
// ---------------------------------------------------------------------------------------

/// A finished generation.
#[derive(Debug)]
pub struct GenOutput {
    pub text: String,
    pub stats: GenerationStats,
}

/// Run a generation on a blocking thread and collect the whole thing.
async fn run_blocking(
    state: AppState,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
) -> Result<GenOutput, ApiError> {
    tokio::task::spawn_blocking(move || {
        generate_text(
            state.tokenizer.as_ref(),
            state.generator.as_ref(),
            &prompt_tokens,
            &params,
            |_| true,
        )
    })
    .await
    .map_err(|e| ApiError::internal(&anyhow::anyhow!("generation task failed: {e}")))?
    .map_err(|e| ApiError::internal(&e))
}

/// Drive a [`Generator`], detokenising incrementally and enforcing stop strings.
///
/// `sink` receives each newly-available piece of text and returns *continue*; returning `false`
/// stops generation. Stop **strings** are enforced here rather than in the engine because they
/// are a property of decoded text and the engine has no detokeniser — see [`crate::generate`].
///
/// The stop string itself is never emitted, matching OpenAI: the text is truncated at the match,
/// including when the match lands in the middle of the token that produced it.
pub fn generate_text(
    tokenizer: &Tokenizer,
    generator: &dyn Generator,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    mut sink: impl FnMut(&str) -> bool,
) -> anyhow::Result<GenOutput> {
    let mut decoder = IncrementalDecoder::new();
    let mut text = String::new();
    // Bytes of `text` already handed to `sink`.
    let mut emitted = 0usize;
    let mut decode_error: Option<anyhow::Error> = None;
    let mut hit_stop_string = false;
    let mut sink_alive = true;

    let mut stats = {
        let mut on_token = |id: u32| -> bool {
            let delta = match decoder.push(tokenizer, id) {
                Ok(d) => d,
                Err(e) => {
                    decode_error = Some(e);
                    return false;
                }
            };
            if delta.is_empty() {
                // A partial UTF-8 sequence. Nothing to emit yet; keep going.
                return true;
            }
            text.push_str(&delta);

            if let Some(at) = first_stop_match(&text, &params.stop_strings) {
                hit_stop_string = true;
                text.truncate(at);
                if at > emitted {
                    sink_alive = sink(&text[emitted..at]);
                    emitted = at;
                }
                return false;
            }

            // 🔴 Hold back any tail that could still turn into a stop string.
            //
            // Without this, a stop string spanning two tokens leaks its own prefix to the
            // client: the non-streaming body is truncated correctly while the SSE stream has
            // already sent `alpha STO`. Found by the byte-per-token fixture, which splits every
            // stop string; a tokeniser that happened to emit `STOP` as one token would have
            // hidden the bug indefinitely.
            let safe = text.len() - held_back(&text, &params.stop_strings);
            if safe > emitted {
                sink_alive = sink(&text[emitted..safe]);
                emitted = safe;
            }
            sink_alive
        };
        generator.generate(prompt_tokens, params, &mut on_token)?
    };

    if let Some(e) = decode_error {
        return Err(e);
    }
    // Generation ended with text still held back against a stop string that never completed.
    if !hit_stop_string && sink_alive && emitted < text.len() {
        sink(&text[emitted..]);
    }
    if hit_stop_string {
        // A stop string is a normal end of turn, not a cancellation.
        stats.stop_reason = StopReason::EndOfTurn;
    }
    Ok(GenOutput { text, stats })
}

/// Length of the longest suffix of `text` that is a **proper prefix** of some stop string.
///
/// That suffix cannot safely be sent yet: the next token might complete the stop string, and
/// text already on the wire cannot be recalled.
fn held_back(text: &str, stops: &[String]) -> usize {
    let mut hold = 0usize;
    for stop in stops.iter().filter(|s| !s.is_empty()) {
        // Proper prefixes only — a full match is `first_stop_match`'s job, and would otherwise
        // be held back forever.
        for n in (1..stop.len().min(text.len() + 1)).rev() {
            let at = text.len() - n;
            // Requiring a char boundary keeps the later slice safe, and a byte-level match that
            // straddles one cannot be a real prefix match anyway: stop strings are valid UTF-8.
            if text.is_char_boundary(at) && text.as_bytes()[at..] == stop.as_bytes()[..n] {
                hold = hold.max(n);
                break;
            }
        }
    }
    hold
}

/// Byte offset of the earliest stop-string match, if any.
fn first_stop_match(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min()
}

/// Serialise and enqueue one SSE data frame. `Err` means the client is gone.
fn send_json<T: serde::Serialize>(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    value: &T,
) -> Result<(), ()> {
    let body = serde_json::to_string(value).map_err(|_| ())?;
    tx.blocking_send(Ok(Event::default().data(body))).map_err(|_| ())
}

/// Bind and serve until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: AppState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state).into_make_service()).with_graceful_shutdown(shutdown).await
}

/// Convenience for tests and for `moearc serve`: bind, and hand back the real address.
///
/// Returns the bound address because binding to port 0 is how a test gets a port that is
/// definitely free — asking for a fixed one makes the suite fail when a developer happens to be
/// running the server.
pub async fn bind(
    addr: std::net::SocketAddr,
) -> std::io::Result<(tokio::net::TcpListener, std::net::SocketAddr)> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    Ok((listener, local))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::chat::ChatTemplate;
    use crate::generate::EchoGenerator;
    use crate::state::ServerState;
    use crate::testing::tiny_tokenizer;

    fn state() -> AppState {
        let tk = Arc::new(tiny_tokenizer());
        let n = tk.vocab_size();
        Arc::new(ServerState::new(
            tk,
            Arc::new(ChatTemplate::chatml()),
            Arc::new(EchoGenerator::new(n)),
            "moearc-echo",
            SamplingParams { temperature: 0.0, max_tokens: 64, ..Default::default() },
        ))
    }

    fn greedy(max_tokens: usize, stops: &[&str]) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            max_tokens,
            stop_strings: stops.iter().map(|s| (*s).to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn generation_reassembles_the_prompt_text_exactly() {
        let s = state();
        let ids = s.tokenizer.encode("Hello, world!", false).unwrap();
        let out = generate_text(
            &s.tokenizer,
            s.generator.as_ref(),
            &ids,
            &greedy(ids.len(), &[]),
            |_| true,
        )
        .unwrap();
        assert_eq!(out.text, "Hello, world!");
        assert_eq!(out.stats.completion_tokens, ids.len());
    }

    #[test]
    fn streaming_pieces_concatenate_to_the_same_text() {
        let s = state();
        let ids = s.tokenizer.encode("Ünïcödé 🚀 works", false).unwrap();
        let mut pieces = Vec::new();
        let out =
            generate_text(&s.tokenizer, s.generator.as_ref(), &ids, &greedy(ids.len(), &[]), |t| {
                pieces.push(t.to_string());
                true
            })
            .unwrap();
        assert_eq!(pieces.concat(), out.text);
        assert_eq!(out.text, "Ünïcödé 🚀 works");
        // Multi-byte characters must not be split across chunks into replacement characters.
        assert!(!pieces.iter().any(|p| p.contains('\u{FFFD}')), "{pieces:?}");
    }

    #[test]
    fn a_stop_string_truncates_and_is_not_emitted() {
        let s = state();
        let ids = s.tokenizer.encode("alpha STOP beta", false).unwrap();
        let mut streamed = String::new();
        let out = generate_text(
            &s.tokenizer,
            s.generator.as_ref(),
            &ids,
            &greedy(ids.len(), &["STOP"]),
            |t| {
                streamed.push_str(t);
                true
            },
        )
        .unwrap();
        assert_eq!(out.text, "alpha ");
        assert_eq!(streamed, "alpha ", "the stop string must not reach the client");
        assert_eq!(out.stats.stop_reason, StopReason::EndOfTurn);
    }

    #[test]
    fn a_sink_that_gives_up_stops_generation() {
        let s = state();
        let ids = s.tokenizer.encode("a long prompt that keeps going", false).unwrap();
        let mut n = 0;
        let out = generate_text(
            &s.tokenizer,
            s.generator.as_ref(),
            &ids,
            &greedy(ids.len(), &[]),
            |_| {
                n += 1;
                n < 3
            },
        )
        .unwrap();
        assert_eq!(n, 3);
        assert_eq!(out.stats.stop_reason, StopReason::Cancelled);
    }

    #[test]
    fn max_tokens_reports_length() {
        let s = state();
        let ids = s.tokenizer.encode("plenty of tokens here", false).unwrap();
        let out =
            generate_text(&s.tokenizer, s.generator.as_ref(), &ids, &greedy(3, &[]), |_| true)
                .unwrap();
        assert_eq!(out.stats.stop_reason, StopReason::Length);
        assert_eq!(out.stats.completion_tokens, 3);
    }

    #[test]
    fn a_stop_string_split_across_tokens_never_leaks_its_prefix() {
        // The fixture emits one token per byte, so "STOP" arrives as S, T, O, P. This is the
        // regression test for text going out on the wire that a later token retracts.
        let s = state();
        let ids = s.tokenizer.encode("alpha STOP beta", false).unwrap();
        let mut pieces: Vec<String> = Vec::new();
        let out = generate_text(
            &s.tokenizer,
            s.generator.as_ref(),
            &ids,
            &greedy(ids.len(), &["STOP"]),
            |t| {
                pieces.push(t.to_string());
                true
            },
        )
        .unwrap();
        assert_eq!(pieces.concat(), "alpha ");
        assert_eq!(out.text, "alpha ");
    }

    #[test]
    fn held_back_text_is_flushed_when_the_stop_never_completes() {
        // "STO" is a prefix of "STOP" but generation ends first; it must still be delivered.
        let s = state();
        let ids = s.tokenizer.encode("alpha STO", false).unwrap();
        let mut streamed = String::new();
        let out = generate_text(
            &s.tokenizer,
            s.generator.as_ref(),
            &ids,
            &greedy(ids.len(), &["STOP"]),
            |t| {
                streamed.push_str(t);
                true
            },
        )
        .unwrap();
        assert_eq!(out.text, "alpha STO");
        assert_eq!(streamed, "alpha STO", "held-back text must not be dropped");
    }

    #[test]
    fn held_back_measures_the_longest_pending_prefix() {
        let stops = vec!["STOP".to_string(), "END".to_string()];
        assert_eq!(held_back("alpha ST", &stops), 2);
        assert_eq!(held_back("alpha E", &stops), 1);
        assert_eq!(held_back("alpha x", &stops), 0);
        // A complete match is not held back — `first_stop_match` handles it.
        assert_eq!(held_back("alpha STOP", &stops), 0);
        // Multi-byte characters must not be sliced mid-character.
        assert_eq!(held_back("é", &["\u{e9}x".to_string()]), 2);
    }

    #[test]
    fn earliest_stop_string_wins() {
        assert_eq!(first_stop_match("abcXdefY", &["Y".into(), "X".into()]), Some(3));
        assert_eq!(first_stop_match("abc", &["Z".into()]), None);
        assert_eq!(
            first_stop_match("abc", &[String::new()]),
            None,
            "an empty stop must be ignored"
        );
    }
}
