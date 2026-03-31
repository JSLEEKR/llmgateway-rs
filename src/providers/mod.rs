//! Provider module — request/response translation for different LLM providers.
//!
//! Translates between the OpenAI-compatible API format (used as the gateway's
//! lingua franca) and provider-native formats (Anthropic, Google, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    /// OpenAI-compatible generic provider (DeepSeek, Together, etc.)
    Generic,
}

impl Provider {
    /// Detect provider from a provider name string.
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "openai" | "azure" | "azure-openai" => Provider::OpenAI,
            "anthropic" | "claude" => Provider::Anthropic,
            "google" | "gemini" | "vertex" => Provider::Google,
            _ => Provider::Generic,
        }
    }

    /// Get the chat completions endpoint path for this provider.
    pub fn chat_endpoint(&self) -> &str {
        match self {
            Provider::OpenAI | Provider::Generic => "/v1/chat/completions",
            Provider::Anthropic => "/v1/messages",
            Provider::Google => "/v1/models/{model}:generateContent",
        }
    }

    /// Get the auth header name for this provider.
    pub fn auth_header(&self) -> &str {
        match self {
            Provider::OpenAI | Provider::Generic => "Authorization",
            Provider::Anthropic => "x-api-key",
            Provider::Google => "Authorization",
        }
    }

    /// Format the auth header value.
    pub fn auth_value(&self, api_key: &str) -> String {
        match self {
            Provider::OpenAI | Provider::Generic | Provider::Google => {
                format!("Bearer {}", api_key)
            }
            Provider::Anthropic => api_key.to_string(),
        }
    }

    /// Additional required headers for this provider.
    pub fn extra_headers(&self) -> Vec<(&str, &str)> {
        match self {
            Provider::Anthropic => vec![
                ("anthropic-version", "2023-06-01"),
            ],
            _ => vec![],
        }
    }
}

/// OpenAI-compatible chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Extra fields we pass through.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Transform an OpenAI-format request into the provider's native format.
pub fn transform_request(provider: &Provider, request: &ChatRequest) -> serde_json::Value {
    match provider {
        Provider::OpenAI | Provider::Generic => {
            // Already in OpenAI format — pass through
            serde_json::to_value(request).unwrap_or_default()
        }
        Provider::Anthropic => transform_to_anthropic(request),
        Provider::Google => transform_to_google(request),
    }
}

/// Transform OpenAI format → Anthropic Messages API format.
fn transform_to_anthropic(request: &ChatRequest) -> serde_json::Value {
    let mut messages = Vec::new();
    let mut system_text = String::new();

    for msg in &request.messages {
        if msg.role == "system" {
            // Anthropic uses a separate `system` field
            if let Some(text) = msg.content.as_str() {
                system_text = text.to_string();
            }
        } else {
            let content = if let Some(text) = msg.content.as_str() {
                serde_json::json!([{"type": "text", "text": text}])
            } else {
                msg.content.clone()
            };
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": content,
            }));
        }
    }

    let mut body = serde_json::json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens.unwrap_or(4096),
    });

    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if let Some(stream) = request.stream {
        body["stream"] = serde_json::json!(stream);
    }

    body
}

/// Transform OpenAI format → Google Gemini format.
fn transform_to_google(request: &ChatRequest) -> serde_json::Value {
    let mut contents = Vec::new();
    let mut system_instruction = None;

    for msg in &request.messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            "system" => {
                if let Some(text) = msg.content.as_str() {
                    system_instruction = Some(serde_json::json!({
                        "parts": [{"text": text}]
                    }));
                }
                continue;
            }
            r => r,
        };

        let parts = if let Some(text) = msg.content.as_str() {
            vec![serde_json::json!({"text": text})]
        } else {
            vec![serde_json::json!({"text": msg.content.to_string()})]
        };

        contents.push(serde_json::json!({
            "role": role,
            "parts": parts,
        }));
    }

    let mut body = serde_json::json!({
        "contents": contents,
    });

    if let Some(si) = system_instruction {
        body["systemInstruction"] = si;
    }

    let mut generation_config = serde_json::Map::new();
    if let Some(temp) = request.temperature {
        generation_config.insert("temperature".to_string(), serde_json::json!(temp));
    }
    if let Some(max_tokens) = request.max_tokens {
        generation_config.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
    }
    if let Some(top_p) = request.top_p {
        generation_config.insert("topP".to_string(), serde_json::json!(top_p));
    }
    if !generation_config.is_empty() {
        body["generationConfig"] = serde_json::Value::Object(generation_config);
    }

    body
}

/// Transform a provider's native response back into OpenAI-compatible format.
pub fn transform_response(
    provider: &Provider,
    response: &serde_json::Value,
    model: &str,
) -> serde_json::Value {
    match provider {
        Provider::OpenAI | Provider::Generic => response.clone(),
        Provider::Anthropic => transform_from_anthropic(response, model),
        Provider::Google => transform_from_google(response, model),
    }
}

/// Transform Anthropic Messages API response → OpenAI format.
fn transform_from_anthropic(response: &serde_json::Value, model: &str) -> serde_json::Value {
    let content = response
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|block| block.get("type").and_then(|t| t.as_str()) == Some("text"))
                .and_then(|block| block.get("text").and_then(|t| t.as_str()))
        })
        .unwrap_or("");

    let usage = response.get("usage");
    let prompt_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("msg-unknown");

    serde_json::json!({
        "id": format!("chatcmpl-{}", id),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": response.get("stop_reason")
                .and_then(|v| v.as_str())
                .map(|r| match r {
                    "end_turn" => "stop",
                    "max_tokens" => "length",
                    _ => r,
                })
                .unwrap_or("stop"),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

/// Transform Google Gemini response → OpenAI format.
fn transform_from_google(response: &serde_json::Value, model: &str) -> serde_json::Value {
    let content = response
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.get(0))
        .and_then(|p| p.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    let usage_metadata = response.get("usageMetadata");
    let prompt_tokens = usage_metadata
        .and_then(|u| u.get("promptTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = usage_metadata
        .and_then(|u| u.get("candidatesTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let finish_reason = response
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finishReason"))
        .and_then(|v| v.as_str())
        .map(|r| match r {
            "STOP" => "stop",
            "MAX_TOKENS" => "length",
            "SAFETY" => "content_filter",
            _ => "stop",
        })
        .unwrap_or("stop");

    serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

/// Build the full URL for a provider request.
pub fn build_url(base_url: &str, provider: &Provider, model: &str) -> String {
    let endpoint = provider.chat_endpoint();
    let url = format!(
        "{}{}",
        base_url.trim_end_matches('/'),
        endpoint
    );
    url.replace("{model}", model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_openai_request() -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: serde_json::json!("You are a helpful assistant."),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: serde_json::json!("Hello!"),
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(1000),
            stream: Some(false),
            top_p: None,
            seed: None,
            extra: HashMap::new(),
        }
    }

    // --- Provider Detection Tests ---

    #[test]
    fn test_provider_from_name_openai() {
        assert_eq!(Provider::from_name("openai"), Provider::OpenAI);
        assert_eq!(Provider::from_name("OpenAI"), Provider::OpenAI);
        assert_eq!(Provider::from_name("azure"), Provider::OpenAI);
    }

    #[test]
    fn test_provider_from_name_anthropic() {
        assert_eq!(Provider::from_name("anthropic"), Provider::Anthropic);
        assert_eq!(Provider::from_name("claude"), Provider::Anthropic);
    }

    #[test]
    fn test_provider_from_name_google() {
        assert_eq!(Provider::from_name("google"), Provider::Google);
        assert_eq!(Provider::from_name("gemini"), Provider::Google);
        assert_eq!(Provider::from_name("vertex"), Provider::Google);
    }

    #[test]
    fn test_provider_from_name_generic() {
        assert_eq!(Provider::from_name("deepseek"), Provider::Generic);
        assert_eq!(Provider::from_name("together"), Provider::Generic);
    }

    // --- Auth Tests ---

    #[test]
    fn test_provider_auth_header() {
        assert_eq!(Provider::OpenAI.auth_header(), "Authorization");
        assert_eq!(Provider::Anthropic.auth_header(), "x-api-key");
    }

    #[test]
    fn test_provider_auth_value() {
        assert_eq!(Provider::OpenAI.auth_value("sk-123"), "Bearer sk-123");
        assert_eq!(Provider::Anthropic.auth_value("sk-ant-123"), "sk-ant-123");
    }

    #[test]
    fn test_provider_extra_headers() {
        let anthropic_headers = Provider::Anthropic.extra_headers();
        assert!(anthropic_headers.iter().any(|(k, _)| *k == "anthropic-version"));
    }

    // --- Request Transform Tests ---

    #[test]
    fn test_transform_openai_passthrough() {
        let req = make_openai_request();
        let body = transform_request(&Provider::OpenAI, &req);
        assert_eq!(body["model"], "gpt-4o");
        assert!(body["messages"].is_array());
    }

    #[test]
    fn test_transform_to_anthropic() {
        let req = make_openai_request();
        let body = transform_request(&Provider::Anthropic, &req);

        // System message should be in `system` field, not in messages
        assert_eq!(body["system"], "You are a helpful assistant.");

        // Messages should only have user message
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");

        assert_eq!(body["max_tokens"], 1000);
        assert_eq!(body["temperature"], 0.7);
    }

    #[test]
    fn test_transform_to_anthropic_no_system() {
        let req = ChatRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: serde_json::json!("Hi"),
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            top_p: None,
            seed: None,
            extra: HashMap::new(),
        };
        let body = transform_request(&Provider::Anthropic, &req);
        assert!(body.get("system").is_none());
        assert_eq!(body["max_tokens"], 4096); // default
    }

    #[test]
    fn test_transform_to_google() {
        let req = make_openai_request();
        let body = transform_request(&Provider::Google, &req);

        assert!(body["contents"].is_array());
        let contents = body["contents"].as_array().unwrap();
        // System message should NOT be in contents
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");

        // System instruction
        assert!(body["systemInstruction"].is_object());
    }

    #[test]
    fn test_transform_to_google_generation_config() {
        let req = make_openai_request();
        let body = transform_request(&Provider::Google, &req);
        let gen_config = body.get("generationConfig").unwrap();
        assert_eq!(gen_config["temperature"], 0.7);
        assert_eq!(gen_config["maxOutputTokens"], 1000);
    }

    // --- Response Transform Tests ---

    #[test]
    fn test_transform_anthropic_response() {
        let anthropic_resp = serde_json::json!({
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello! How can I help you?"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 10
            }
        });

        let openai = transform_response(&Provider::Anthropic, &anthropic_resp, "claude-3-5-sonnet");
        assert_eq!(openai["object"], "chat.completion");
        assert_eq!(
            openai["choices"][0]["message"]["content"],
            "Hello! How can I help you?"
        );
        assert_eq!(openai["choices"][0]["finish_reason"], "stop");
        assert_eq!(openai["usage"]["prompt_tokens"], 25);
        assert_eq!(openai["usage"]["completion_tokens"], 10);
        assert_eq!(openai["usage"]["total_tokens"], 35);
    }

    #[test]
    fn test_transform_google_response() {
        let google_resp = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 8,
                "totalTokenCount": 23
            }
        });

        let openai = transform_response(&Provider::Google, &google_resp, "gemini-2.0-flash");
        assert_eq!(openai["object"], "chat.completion");
        assert_eq!(
            openai["choices"][0]["message"]["content"],
            "Hello from Gemini!"
        );
        assert_eq!(openai["choices"][0]["finish_reason"], "stop");
        assert_eq!(openai["usage"]["prompt_tokens"], 15);
        assert_eq!(openai["usage"]["completion_tokens"], 8);
    }

    #[test]
    fn test_transform_openai_response_passthrough() {
        let resp = serde_json::json!({"id": "chatcmpl-123", "model": "gpt-4o"});
        let result = transform_response(&Provider::OpenAI, &resp, "gpt-4o");
        assert_eq!(result, resp);
    }

    // --- URL Building Tests ---

    #[test]
    fn test_build_url_openai() {
        let url = build_url("https://api.openai.com", &Provider::OpenAI, "gpt-4o");
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_build_url_anthropic() {
        let url = build_url("https://api.anthropic.com", &Provider::Anthropic, "claude-3-5-sonnet");
        assert_eq!(url, "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn test_build_url_google() {
        let url = build_url("https://generativelanguage.googleapis.com", &Provider::Google, "gemini-2.0-flash");
        assert_eq!(url, "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent");
    }

    #[test]
    fn test_build_url_trailing_slash() {
        let url = build_url("https://api.openai.com/", &Provider::OpenAI, "gpt-4o");
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    // --- Chat Endpoint Tests ---

    #[test]
    fn test_chat_endpoint() {
        assert_eq!(Provider::OpenAI.chat_endpoint(), "/v1/chat/completions");
        assert_eq!(Provider::Anthropic.chat_endpoint(), "/v1/messages");
        assert_eq!(Provider::Generic.chat_endpoint(), "/v1/chat/completions");
    }

    // --- Anthropic stop reason mapping ---

    #[test]
    fn test_anthropic_max_tokens_stop_reason() {
        let resp = serde_json::json!({
            "id": "msg_123",
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100}
        });
        let openai = transform_response(&Provider::Anthropic, &resp, "claude-3-5-sonnet");
        assert_eq!(openai["choices"][0]["finish_reason"], "length");
    }

    // --- Google finish reason mapping ---

    #[test]
    fn test_google_safety_finish_reason() {
        let resp = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": ""}], "role": "model"},
                "finishReason": "SAFETY"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0}
        });
        let openai = transform_response(&Provider::Google, &resp, "gemini-2.0-flash");
        assert_eq!(openai["choices"][0]["finish_reason"], "content_filter");
    }
}
