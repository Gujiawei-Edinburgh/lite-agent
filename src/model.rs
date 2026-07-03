use crate::error::{AgentError, Result};
use crate::projection::{ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

pub trait ModelClient: Send + Sync {
    fn complete<'a>(
        &'a self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>>;
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
            let body = OpenAiChatRequest::from_model_request(&self.config.model, request);
            let response: OpenAiChatResponse = self
                .http
                .post(url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;

            let choice = response.choices.into_iter().next().ok_or_else(|| {
                AgentError::Model("chat/completions returned no choices".to_string())
            })?;

            if let Some(tool_calls) = choice.message.tool_calls {
                let calls = tool_calls
                    .into_iter()
                    .map(|call| {
                        let arguments = serde_json::from_str(&call.function.arguments)
                            .unwrap_or(Value::String(call.function.arguments));
                        ModelFunctionCall {
                            call_id: call.id,
                            name: call.function.name,
                            arguments,
                        }
                    })
                    .collect();
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
    fn from_model_request(model: &str, request: ModelRequest) -> Self {
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
            stream: false,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<ChatMessage> for OpenAiMessage {
    fn from(message: ChatMessage) -> Self {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        Self {
            role,
            content: message.content,
            name: message.name,
            tool_call_id: message.tool_call_id,
        }
    }
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
