//! Errors, in OpenAI's envelope.
//!
//! Clients parse `{"error": {"message", "type", "param", "code"}}`; the good ones surface
//! `message` to the user verbatim. So the message is the whole product here — it is the only
//! part of a failure most people will ever read, which is why `docs/ux.md` rules out errors
//! that report a symptom without naming a cause.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;

/// A failed request.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    /// OpenAI's error `type` discriminator.
    pub kind: &'static str,
    pub message: String,
    pub param: Option<String>,
}

#[derive(Serialize)]
struct Envelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    param: Option<&'a str>,
    code: Option<&'a str>,
}

impl ApiError {
    /// The client sent something we cannot act on.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
            param: None,
        }
    }

    /// As [`Self::invalid_request`], naming the offending field so a client can point at it.
    pub fn invalid_param(param: &str, message: impl Into<String>) -> Self {
        Self { param: Some(param.to_string()), ..Self::invalid_request(message) }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "not_found_error",
            message: message.into(),
            param: None,
        }
    }

    /// Something on our side broke.
    ///
    /// Takes the full `{:#}` chain rather than the outermost message: "tokenising failed" alone
    /// is unactionable, and the operator reading the response is usually the same person who
    /// would otherwise have to go and find the log line.
    pub fn internal(err: &anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: format!("{err:#}"),
            param: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::error!(status = %self.status, message = %self.message, "request failed");
        } else {
            tracing::debug!(status = %self.status, message = %self.message, "request rejected");
        }
        let body = Envelope {
            error: ErrorBody {
                message: &self.message,
                kind: self.kind,
                param: self.param.as_deref(),
                code: None,
            },
        };
        (self.status, axum::Json(body)).into_response()
    }
}

/// Render an error into the JSON an SSE stream carries when it fails *after* headers are sent.
///
/// 🔴 Once a 200 and the SSE content type are on the wire, the status code is spent — a failure
/// mid-stream cannot become a 500. The only honest option left is an `error` event in the
/// stream, which the OpenAI clients do check for. Silently closing the connection instead is
/// what makes a client hang or report a truncated answer as a complete one.
pub fn stream_error_payload(message: &str) -> String {
    json!({
        "error": {
            "message": message,
            "type": "server_error",
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null,
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(e: ApiError) -> (StatusCode, serde_json::Value) {
        let r = e.into_response();
        let status = r.status();
        // Small bodies only; these are error envelopes.
        let bytes = futures_executor_block_on(r.into_body());
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// axum's body is a stream; the test needs it as bytes without pulling in a runtime.
    fn futures_executor_block_on(body: axum::body::Body) -> Vec<u8> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async { axum::body::to_bytes(body, 64 * 1024).await.unwrap().to_vec() })
    }

    #[test]
    fn invalid_request_matches_the_openai_envelope() {
        let (status, v) = body_of(ApiError::invalid_param("n", "`n` must be 1"));
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "`n` must be 1");
        assert_eq!(v["error"]["param"], "n");
        assert!(v["error"]["code"].is_null());
    }

    #[test]
    fn internal_errors_carry_the_whole_cause_chain() {
        let err = anyhow::anyhow!("no such file").context("reading tokenizer.json");
        let (status, v) = body_of(ApiError::internal(&err));
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("reading tokenizer.json"), "{msg}");
        assert!(msg.contains("no such file"), "outermost message alone is unactionable: {msg}");
    }

    #[test]
    fn stream_errors_are_parseable_by_the_same_client_code() {
        let v: serde_json::Value = serde_json::from_str(&stream_error_payload("boom")).unwrap();
        assert_eq!(v["error"]["message"], "boom");
        assert_eq!(v["error"]["type"], "server_error");
    }
}
