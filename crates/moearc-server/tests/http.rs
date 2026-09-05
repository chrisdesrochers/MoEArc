//! End-to-end tests over a real TCP socket with a real HTTP client.
//!
//! Calling the handlers directly would not prove the thing that matters: that an *unmodified*
//! OpenAI client works. So each test starts the actual server on an ephemeral loopback port
//! and drives it with `reqwest` — real sockets, real chunked transfer, real SSE framing.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use futures_util::StreamExt;
use moearc_server::chat::ChatTemplate;
use moearc_server::generate::EchoGenerator;
use moearc_server::sampling::SamplingParams;
use moearc_server::state::ServerState;
use moearc_server::testing::tiny_tokenizer;
use serde_json::{Value, json};

/// Start a server on a free loopback port; returns its base URL and a shutdown handle.
async fn spawn() -> (String, tokio::sync::oneshot::Sender<()>) {
    let tk = Arc::new(tiny_tokenizer());
    let vocab = tk.vocab_size();
    let state = Arc::new(ServerState::new(
        tk,
        Arc::new(ChatTemplate::chatml()),
        Arc::new(EchoGenerator::new(vocab)),
        "moearc-echo",
        // Greedy by default so every assertion below is on an exact string, not a distribution.
        SamplingParams { temperature: 0.0, max_tokens: 128, ..Default::default() },
    ));

    // Port 0: the OS picks a free port. A fixed port makes the suite fail whenever a developer
    // happens to have the server running.
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
    let (listener, local) = moearc_server::bind(addr).await.unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        moearc_server::serve(listener, state, async {
            rx.await.ok();
        })
        .await
        .unwrap();
    });
    (format!("http://{local}"), tx)
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

#[tokio::test]
async fn health_reports_the_stub_out_loud() {
    let (base, _s) = spawn().await;
    let v: Value =
        client().get(format!("{base}/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["model"], "moearc-echo");
    assert_eq!(
        v["stubbed"], true,
        "a health check that hides the stub is a health check that lies"
    );
    assert!(v["tokenizer"].as_str().unwrap().contains("fixture"));
}

#[tokio::test]
async fn models_list_matches_the_openai_shape() {
    let (base, _s) = spawn().await;
    let v: Value =
        client().get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "moearc-echo");
    assert_eq!(v["data"][0]["object"], "model");

    let one: Value = client()
        .get(format!("{base}/v1/models/moearc-echo"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(one["id"], "moearc-echo");

    let missing = client().get(format!("{base}/v1/models/nope")).send().await.unwrap();
    assert_eq!(missing.status(), 404);
}

#[tokio::test]
async fn chat_completion_returns_the_openai_envelope() {
    let (base, _s) = spawn().await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "moearc-echo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 24
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["index"], 0);
    assert!(v["choices"][0]["logprobs"].is_null());
    assert!(v["choices"][0]["finish_reason"].is_string());

    // The stub echoes the rendered ChatML prompt at temperature 0, so the content is
    // predictable — which is exactly what makes the whole path assertable without a model.
    // Note the missing `<|im_start|>`: detokenisation skips special tokens, which is what
    // stops a model's turn markers from leaking into an assistant message.
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with("user\nHello"), "{content:?}");

    let usage = &v["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );
}

/// The important one: a stream must arrive as many frames, not one blob.
#[tokio::test]
async fn chat_completion_streams_as_multiple_sse_chunks() {
    let (base, _s) = spawn().await;
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "Streaming works"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": 0,
            "max_tokens": 40
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"),
        "{:?}",
        resp.headers()["content-type"]
    );

    // Count network reads as well as frames: one big read holding everything would still parse
    // as "multiple chunks" while not actually having streamed.
    let mut reads = 0usize;
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(b) = stream.next().await {
        reads += 1;
        buf.push_str(std::str::from_utf8(&b.unwrap()).unwrap());
    }

    let frames: Vec<&str> =
        buf.split("\n\n").filter_map(|f| f.trim().strip_prefix("data: ")).collect();
    assert!(frames.len() > 3, "expected many frames, got {}: {buf:?}", frames.len());
    assert_eq!(*frames.last().unwrap(), "[DONE]", "the stream must terminate with [DONE]");
    assert!(reads > 1, "the whole body arrived in one read — that is not streaming");

    let parsed: Vec<Value> =
        frames[..frames.len() - 1].iter().map(|f| serde_json::from_str(f).unwrap()).collect();

    assert_eq!(parsed[0]["object"], "chat.completion.chunk");
    assert_eq!(parsed[0]["choices"][0]["delta"]["role"], "assistant");

    // Exactly one chunk carries a finish_reason, and it is not the usage chunk.
    let finished: Vec<&Value> =
        parsed.iter().filter(|c| c["choices"][0]["finish_reason"].is_string()).collect();
    assert_eq!(finished.len(), 1, "one and only one chunk may finish the choice");

    let usage_chunk = parsed.last().unwrap();
    assert!(usage_chunk["usage"]["total_tokens"].as_u64().unwrap() > 0);
    assert_eq!(usage_chunk["choices"].as_array().unwrap().len(), 0);

    // Reassembling the deltas must give the same text a non-streaming call returns.
    let streamed: String =
        parsed.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();

    let whole: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "Streaming works"}],
            "temperature": 0,
            "max_tokens": 40
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(streamed, whole["choices"][0]["message"]["content"].as_str().unwrap());
}

#[tokio::test]
async fn legacy_completions_work_including_echo_and_streaming() {
    let (base, _s) = spawn().await;
    let v: Value = client()
        .post(format!("{base}/v1/completions"))
        .json(&json!({"prompt": "abc", "echo": true, "temperature": 0, "max_tokens": 3}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["object"], "text_completion");
    // echo=true prepends the prompt; the stub then echoes it again.
    assert_eq!(v["choices"][0]["text"], "abcabc");

    let body = client()
        .post(format!("{base}/v1/completions"))
        .json(&json!({"prompt": "abc", "stream": true, "temperature": 0, "max_tokens": 3}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("data: [DONE]"), "{body}");
    assert!(body.matches("data: ").count() > 2, "{body}");
}

#[tokio::test]
async fn a_seed_makes_a_sampled_completion_reproducible() {
    let (base, _s) = spawn().await;
    let ask = |seed: u64| {
        let base = base.clone();
        async move {
            let v: Value = client()
                .post(format!("{base}/v1/chat/completions"))
                .json(&json!({
                    "messages": [{"role": "user", "content": "vary me"}],
                    "temperature": 1.3, "seed": seed, "max_tokens": 16
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            v["choices"][0]["message"]["content"].as_str().unwrap().to_string()
        }
    };
    assert_eq!(ask(7).await, ask(7).await);
    assert_ne!(ask(7).await, ask(8).await);
}

#[tokio::test]
async fn stop_strings_truncate_the_response() {
    let (base, _s) = spawn().await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0, "max_tokens": 128, "stop": "\n"
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    // The rendered prompt is `<|im_start|>user\nx<|im_end|>...`; with special tokens skipped
    // the echo is `user\nx...`, and the stop string cuts it at the first newline.
    assert_eq!(content, "user");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn errors_come_back_in_the_openai_envelope() {
    let (base, _s) = spawn().await;

    let bad = client()
        .post(format!("{base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
    let v: Value = bad.json().await.unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");

    let n = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "hi"}], "n": 3}))
        .send()
        .await
        .unwrap();
    assert_eq!(n.status(), 400);
    let v: Value = n.json().await.unwrap();
    assert_eq!(v["error"]["param"], "n");

    let empty = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(empty.status(), 400);

    let nowhere = client().get(format!("{base}/v1/nope")).send().await.unwrap();
    assert_eq!(nowhere.status(), 404);
    let v: Value = nowhere.json().await.unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("/v1/chat/completions"));
}

#[tokio::test]
async fn unknown_request_fields_are_tolerated() {
    let (base, _s) = spawn().await;
    let r = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "moearc-echo",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": false, "top_logprobs": null, "user": "someone",
            "presence_penalty": 0.2, "response_format": {"type": "text"},
            "parallel_tool_calls": true, "service_tier": "auto", "store": false,
            "max_tokens": 8
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "an unmodified OpenAI client must not get a 400");
}

#[tokio::test]
async fn concurrent_requests_do_not_interfere() {
    let (base, _s) = spawn().await;
    let mut set = tokio::task::JoinSet::new();
    for i in 0..12u32 {
        let base = base.clone();
        set.spawn(async move {
            let v: Value = client()
                .post(format!("{base}/v1/chat/completions"))
                .json(&json!({
                    "messages": [{"role": "user", "content": format!("request-{i}")}],
                    "temperature": 0, "max_tokens": 64
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            (i, v["choices"][0]["message"]["content"].as_str().unwrap().to_string())
        });
    }
    while let Some(r) = set.join_next().await {
        let (i, content) = r.unwrap();
        assert!(content.contains(&format!("request-{i}")), "response {i} carried another's prompt");
    }
}
