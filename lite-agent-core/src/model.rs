use crate::error::{AgentError, Result};
use crate::projection::ChatMessage;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRequest {
    pub messages: Vec<ChatMessage>,
    pub functions: Vec<FunctionSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelResponse {
    AssistantMessage { text: String },
    FunctionCalls { calls: Vec<ModelFunctionCall> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStreamEvent {
    AssistantDelta { text: String },
}

pub type ModelStreamHandler<'a> = dyn FnMut(ModelStreamEvent) + Send + 'a;

pub trait ModelClient: Send + Sync {
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>>;

    fn stream_complete<'a>(
        &'a self,
        request: ModelRequest,
        on_event: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let response = self.complete(request).await?;
            if let ModelResponse::AssistantMessage { text } = &response {
                on_event(ModelStreamEvent::AssistantDelta { text: text.clone() });
            }
            Ok(response)
        })
    }
}

#[derive(Debug, Clone)]
pub struct ChatCompletionsClient {
    http: reqwest::Client,
    config: ModelConfig,
}

impl ChatCompletionsClient {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            config,
        }
    }
}

impl ModelClient for ChatCompletionsClient {
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            );
            let body = OpenAiChatRequest::from_model_request(&self.config.model, request, false);
            let response = self
                .http
                .post(url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await?;
            let status = response.status();
            let raw = response.text().await?;
            if !status.is_success() {
                return Err(AgentError::Http(format!(
                    "HTTP status {status} for chat/completions: {raw}"
                )));
            }
            let response: OpenAiChatResponse = serde_json::from_str(&raw)?;

            let choice = response.choices.into_iter().next().ok_or_else(|| {
                AgentError::Model("chat/completions returned no choices".to_string())
            })?;

            if let Some(tool_calls) = choice.message.tool_calls {
                let mut calls = Vec::with_capacity(tool_calls.len());
                for call in tool_calls {
                    let arguments =
                        serde_json::from_str(&call.function.arguments).map_err(|error| {
                            AgentError::Model(format!(
                                "invalid JSON arguments for tool call {}: {error}",
                                call.id
                            ))
                        })?;
                    calls.push(ModelFunctionCall {
                        call_id: call.id,
                        name: call.function.name,
                        arguments,
                    });
                }
                Ok(ModelResponse::FunctionCalls { calls })
            } else if let Some(content) = choice.message.content {
                Ok(ModelResponse::AssistantMessage { text: content })
            } else {
                Err(AgentError::Model(
                    "chat/completions response had no content or tool calls".to_string(),
                ))
            }
        })
    }

    fn stream_complete<'a>(
        &'a self,
        request: ModelRequest,
        on_event: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            );
            let body = OpenAiChatRequest::from_model_request(&self.config.model, request, true);
            let response = self
                .http
                .post(url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await?;
            let status = response.status();
            if !status.is_success() {
                let raw = response.text().await?;
                return Err(AgentError::Http(format!(
                    "HTTP status {status} for streamed chat/completions: {raw}"
                )));
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut assistant_text = String::new();
            let mut tool_calls = BTreeMap::<usize, PartialToolCall>::new();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(frame_end) = buffer.find("\n\n") {
                    let frame = buffer[..frame_end].to_string();
                    buffer.drain(..frame_end + 2);
                    handle_sse_frame(&frame, &mut assistant_text, &mut tool_calls, on_event)
                        .await?;
                }
            }
            if !buffer.trim().is_empty() {
                handle_sse_frame(&buffer, &mut assistant_text, &mut tool_calls, on_event).await?;
            }

            if !tool_calls.is_empty() {
                let calls = tool_calls
                    .into_values()
                    .map(PartialToolCall::finish)
                    .collect::<Result<Vec<_>>>()?;
                Ok(ModelResponse::FunctionCalls { calls })
            } else {
                Ok(ModelResponse::AssistantMessage {
                    text: assistant_text,
                })
            }
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    tools: Vec<OpenAiTool>,
    tool_choice: &'static str,
    stream: bool,
}

impl OpenAiChatRequest {
    fn from_model_request(model: &str, request: ModelRequest, stream: bool) -> Self {
        Self {
            model: model.to_string(),
            messages: request
                .messages
                .into_iter()
                .map(OpenAiMessage::from)
                .collect(),
            tools: request
                .functions
                .into_iter()
                .map(|function| OpenAiTool {
                    kind: "function",
                    function,
                })
                .collect(),
            tool_choice: "auto",
            stream,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiRequestToolCall>,
}

impl From<ChatMessage> for OpenAiMessage {
    fn from(message: ChatMessage) -> Self {
        match message {
            ChatMessage::System { content } => Self {
                role: "system",
                content: Some(content),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            ChatMessage::User { content } => Self {
                role: "user",
                content: Some(content),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => Self {
                role: "assistant",
                content,
                name: None,
                tool_call_id: None,
                tool_calls: tool_calls
                    .into_iter()
                    .map(OpenAiRequestToolCall::from)
                    .collect(),
            },
            ChatMessage::Tool {
                tool_call_id,
                name: _,
                content,
            } => Self {
                role: "tool",
                content: Some(content.to_string()),
                name: None,
                tool_call_id: Some(tool_call_id),
                tool_calls: Vec::new(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequestToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiRequestFunctionCall,
}

impl From<ModelFunctionCall> for OpenAiRequestToolCall {
    fn from(call: ModelFunctionCall) -> Self {
        Self {
            id: call.call_id,
            kind: "function",
            function: OpenAiRequestFunctionCall {
                name: call.name,
                arguments: call.arguments.to_string(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequestFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionSpec,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl PartialToolCall {
    fn apply_delta(&mut self, delta: OpenAiStreamToolCall) {
        if let Some(id) = delta.id {
            self.call_id = Some(id);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                self.name = Some(name);
            }
            if let Some(arguments) = function.arguments {
                self.arguments.push_str(&arguments);
            }
        }
    }

    fn finish(self) -> Result<ModelFunctionCall> {
        let call_id = self
            .call_id
            .ok_or_else(|| AgentError::Model("streamed tool call missing id".to_string()))?;
        let name = self.name.ok_or_else(|| {
            AgentError::Model(format!("streamed tool call {call_id} missing name"))
        })?;
        let arguments = serde_json::from_str(&self.arguments).map_err(|error| {
            AgentError::Model(format!(
                "invalid JSON arguments for streamed tool call {call_id}: {error}"
            ))
        })?;
        Ok(ModelFunctionCall {
            call_id,
            name,
            arguments,
        })
    }
}

async fn handle_sse_frame(
    frame: &str,
    assistant_text: &mut String,
    tool_calls: &mut BTreeMap<usize, PartialToolCall>,
    on_event: &mut ModelStreamHandler<'_>,
) -> Result<()> {
    for line in frame.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let event: OpenAiChatStreamResponse = serde_json::from_str(data)?;
        for choice in event.choices {
            if let Some(content) = choice.delta.content {
                assistant_text.push_str(&content);
                on_event(ModelStreamEvent::AssistantDelta { text: content });
            }
            if let Some(deltas) = choice.delta.tool_calls {
                for delta in deltas {
                    tool_calls
                        .entry(delta.index)
                        .or_default()
                        .apply_delta(delta);
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct OpenAiChatStreamResponse {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamToolCall {
    index: usize,
    id: Option<String>,
    function: Option<OpenAiStreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{handle_sse_frame, PartialToolCall};
    use super::{FunctionSpec, ModelFunctionCall, ModelRequest, OpenAiChatRequest};
    use crate::projection::ChatMessage;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    #[test]
    fn serializes_structured_tool_history() {
        let request = ModelRequest {
            messages: vec![
                ChatMessage::Assistant {
                    content: None,
                    tool_calls: vec![ModelFunctionCall {
                        call_id: "call_1".to_string(),
                        name: "exec_command".to_string(),
                        arguments: json!({ "cmd": "ls" }),
                    }],
                },
                ChatMessage::Tool {
                    tool_call_id: "call_1".to_string(),
                    name: "exec_command".to_string(),
                    content: json!({ "stdout": "src\n" }),
                },
            ],
            functions: vec![FunctionSpec {
                name: "exec_command".to_string(),
                description: "run command".to_string(),
                parameters: json!({ "type": "object" }),
            }],
        };

        let body = OpenAiChatRequest::from_model_request("model", request, false);
        let value = serde_json::to_value(body).expect("json");
        let messages = value["messages"].as_array().expect("messages");

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "exec_command"
        );
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_1");
        assert_eq!(messages[1]["name"], Value::Null);
        assert_ne!(messages[1]["content"], Value::Null);
    }

    #[tokio::test]
    async fn parses_streaming_content_deltas() {
        let mut text = String::new();
        let mut calls = BTreeMap::<usize, PartialToolCall>::new();
        let mut deltas = Vec::new();
        let frame = r#"data: {"choices":[{"delta":{"content":"hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

data: [DONE]

"#;

        handle_sse_frame(frame, &mut text, &mut calls, &mut |event| {
            deltas.push(event);
        })
        .await
        .expect("frame");

        assert_eq!(text, "hello");
        assert_eq!(deltas.len(), 2);
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn parses_streaming_tool_call_deltas() {
        let mut text = String::new();
        let mut calls = BTreeMap::<usize, PartialToolCall>::new();
        let frame = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"exec_command","arguments":"{\"cmd\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"ls\"}"}}]}}]}

data: [DONE]

"#;

        handle_sse_frame(frame, &mut text, &mut calls, &mut |_event| {})
            .await
            .expect("frame");

        let call = calls.remove(&0).expect("call").finish().expect("finish");
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "exec_command");
        assert_eq!(call.arguments, json!({ "cmd": "ls" }));
    }
}
