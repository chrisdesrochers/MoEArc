//! The OpenAI wire format.
//!
//! These types exist so that clients written against `api.openai.com` work against MoEArc
//! unmodified — which is `docs/ux.md`'s whole ask for the serving layer ("serving is the boring
//! part"). That imposes two rules on everything below:
//!
//! - **Unknown request fields are accepted and ignored, never rejected.** Clients send
//!   `logprobs`, `user`, `metadata`, `service_tier` and whatever OpenAI added last month. A
//!   400 on an unrecognised field would break working clients for no benefit.
//! - **Fields we cannot honour are *refused by name*,** not silently dropped. `n = 4` returning
//!   one choice is a wrong answer wearing the shape of a right one; see [`ChatCompletionRequest::validate`].
//!
//! Extensions beyond OpenAI's schema (`top_k`, `repetition_penalty`, `min_tokens`) are additive
//! and optional, matching the names llama.cpp and vLLM already use so that clients which know
//! about them need no third spelling.

use serde::{Deserialize, Serialize};

use crate::sampling::SamplingParams;

/// `stop` is a string or an array of strings in the OpenAI schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/// `prompt` in the legacy completions API is a string, an array of strings, or token ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
}

/// One chat message.
///
/// `content` stays as raw JSON: OpenAI allows both a plain string and an array of typed parts,
/// and the chat template is written to handle whichever the client sent. Collapsing to a
/// `String` here would quietly discard image parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `tool_calls`, `tool_call_id`, and anything else a template may branch on, carried
    /// through untouched.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    /// When true, the stream ends with an extra chunk carrying `usage`.
    #[serde(default)]
    pub include_usage: bool,
}

/// Sampling knobs shared by both completion endpoints.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SamplingFields {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Extension (llama.cpp / vLLM spelling). Not in OpenAI's schema.
    pub top_k: Option<usize>,
    /// Extension. Not in OpenAI's schema.
    pub repetition_penalty: Option<f32>,
    /// OpenAI's `frequency_penalty`, in `-2.0..=2.0`. Mapped onto `repetition_penalty` when
    /// the latter is absent, so an unmodified OpenAI client still gets the behaviour it asked
    /// for rather than none.
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    /// OpenAI's replacement for `max_tokens`. Takes precedence when both are present, matching
    /// their deprecation.
    pub max_completion_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub stop: Option<StopField>,
    pub n: Option<u32>,
}

impl SamplingFields {
    /// Fold the request onto defaults, clamping every knob into its valid range.
    ///
    /// Clamping rather than rejecting: a client sending `top_p = 1.2` means "no nucleus
    /// filtering", and failing the request teaches them nothing. Values that would change the
    /// *meaning* of the request (`n`) are rejected in `validate` instead.
    pub fn to_params(&self, defaults: &SamplingParams) -> SamplingParams {
        let repetition_penalty = self
            .repetition_penalty
            .or_else(|| {
                // frequency_penalty ∈ [-2, 2] with 0 = off; repetition_penalty ∈ (0, ∞) with
                // 1 = off. `1 + f/2` maps one onto the other monotonically and keeps both
                // "off" points aligned.
                self.frequency_penalty.map(|f| 1.0 + f.clamp(-2.0, 2.0) / 2.0)
            })
            .unwrap_or(defaults.repetition_penalty);

        SamplingParams {
            temperature: self.temperature.unwrap_or(defaults.temperature).max(0.0),
            top_p: self.top_p.unwrap_or(defaults.top_p).clamp(f32::MIN_POSITIVE, 1.0),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            repetition_penalty: repetition_penalty.max(f32::MIN_POSITIVE),
            seed: self.seed.or(defaults.seed),
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(defaults.max_tokens)
                .clamp(1, 32_768),
            stop_tokens: defaults.stop_tokens.clone(),
            stop_strings: self.stop.clone().map(StopField::into_vec).unwrap_or_default(),
        }
    }

    /// The one field whose unsupported values must not be silently accepted.
    pub fn validate(&self) -> Result<(), String> {
        match self.n {
            Some(n) if n != 1 => Err(format!(
                "`n` must be 1 — MoEArc returns a single choice per request, and answering \
                 `n = {n}` with one choice would be a wrong answer in the right shape. Send {n} \
                 requests, or omit `n`."
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: PromptField,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub echo: bool,
    #[serde(flatten)]
    pub sampling: SamplingFields,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ResponseMessage,
    /// Always present, always `null` unless `logprobs` was requested — which MoEArc does not
    /// yet support. Clients type it as nullable and omitting the key breaks the strict ones.
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<Model>,
}

/// Seconds since the Unix epoch, for the `created` field.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A response id, e.g. `chatcmpl-000000018f3a2c41`.
///
/// A monotonic counter mixed with the start time: unique within a process and across restarts,
/// with no dependency on a UUID crate. Clients only require opacity and stability, not
/// randomness.
pub fn response_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:016x}", (unix_now() << 20) ^ n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn a_minimal_openai_request_parses() {
        let r = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(r.messages.len(), 1);
        assert!(!r.stream);
        assert_eq!(r.model.as_deref(), Some("m"));
    }

    #[test]
    fn unknown_fields_are_ignored_not_rejected() {
        // The exact shape a current openai-python client sends. Every one of these is a field
        // MoEArc has no use for, and every one of them must not break the request.
        let r = parse(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "logprobs":false,"user":"u","service_tier":"auto","metadata":{"a":"b"},
                "parallel_tool_calls":true,"response_format":{"type":"text"},"store":false}"#,
        );
        assert_eq!(r.messages[0].role, "user");
    }

    #[test]
    fn structured_content_and_tool_fields_survive_a_round_trip() {
        let r = parse(
            r#"{"messages":[
                 {"role":"user","content":[{"type":"text","text":"hi"}]},
                 {"role":"assistant","tool_calls":[{"id":"c1","type":"function"}]},
                 {"role":"tool","tool_call_id":"c1","content":"42"}]}"#,
        );
        let back = serde_json::to_value(&r.messages).unwrap();
        assert_eq!(back[0]["content"][0]["type"], "text");
        assert_eq!(back[1]["tool_calls"][0]["id"], "c1");
        assert_eq!(back[2]["tool_call_id"], "c1");
    }

    #[test]
    fn stop_accepts_a_string_or_an_array() {
        let one = parse(r#"{"messages":[],"stop":"END"}"#);
        assert_eq!(one.sampling.stop.unwrap().into_vec(), vec!["END"]);
        let many = parse(r#"{"messages":[],"stop":["A","B"]}"#);
        assert_eq!(many.sampling.stop.unwrap().into_vec(), vec!["A", "B"]);
    }

    #[test]
    fn max_completion_tokens_beats_max_tokens() {
        let r = parse(r#"{"messages":[],"max_tokens":10,"max_completion_tokens":20}"#);
        assert_eq!(r.sampling.to_params(&SamplingParams::default()).max_tokens, 20);
    }

    #[test]
    fn out_of_range_knobs_are_clamped_not_rejected() {
        let r = parse(r#"{"messages":[],"top_p":1.7,"temperature":-3,"max_tokens":9999999}"#);
        let p = r.sampling.to_params(&SamplingParams::default());
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.max_tokens, 32_768);
    }

    #[test]
    fn frequency_penalty_maps_onto_repetition_penalty() {
        let off = parse(r#"{"messages":[],"frequency_penalty":0}"#);
        assert_eq!(off.sampling.to_params(&SamplingParams::default()).repetition_penalty, 1.0);
        let on = parse(r#"{"messages":[],"frequency_penalty":1.0}"#);
        assert_eq!(on.sampling.to_params(&SamplingParams::default()).repetition_penalty, 1.5);
        // An explicit repetition_penalty wins over the mapped one.
        let both = parse(r#"{"messages":[],"frequency_penalty":2.0,"repetition_penalty":1.1}"#);
        assert_eq!(both.sampling.to_params(&SamplingParams::default()).repetition_penalty, 1.1);
    }

    #[test]
    fn n_greater_than_one_is_refused_by_name() {
        assert!(parse(r#"{"messages":[],"n":1}"#).sampling.validate().is_ok());
        assert!(parse(r#"{"messages":[]}"#).sampling.validate().is_ok());
        let err = parse(r#"{"messages":[],"n":4}"#).sampling.validate().unwrap_err();
        assert!(err.contains('4'), "{err}");
    }

    #[test]
    fn completion_prompt_accepts_all_three_shapes() {
        let t: CompletionRequest = serde_json::from_str(r#"{"prompt":"hi"}"#).unwrap();
        assert!(matches!(t.prompt, PromptField::Text(_)));
        let a: CompletionRequest = serde_json::from_str(r#"{"prompt":["a","b"]}"#).unwrap();
        assert!(matches!(a.prompt, PromptField::Texts(_)));
        let k: CompletionRequest = serde_json::from_str(r#"{"prompt":[1,2,3]}"#).unwrap();
        assert!(matches!(k.prompt, PromptField::Tokens(_)));
    }

    #[test]
    fn response_ids_are_unique() {
        let ids: std::collections::HashSet<String> =
            (0..1000).map(|_| response_id("chatcmpl")).collect();
        assert_eq!(ids.len(), 1000);
    }

    #[test]
    fn a_chunk_omits_absent_delta_fields() {
        // OpenAI's final chunk is `{"delta":{},"finish_reason":"stop"}`. A `"content":null`
        // there makes some clients append the string "null".
        let chunk = ChatCompletionChunk {
            id: "x".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: None, content: None },
                logprobs: None,
                finish_reason: Some("stop".into()),
            }],
            usage: None,
        };
        let v = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"], serde_json::json!({}));
        assert!(v.get("usage").is_none());
    }
}
