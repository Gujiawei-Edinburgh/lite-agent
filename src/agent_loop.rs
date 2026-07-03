use crate::error::{AgentError, Result};
use crate::events::{ThreadEvent, ThreadEventKind};
use crate::functions::{FunctionContext, FunctionExecution, FunctionRegistry};
use crate::model::{ModelClient, ModelRequest, ModelResponse};
use crate::projection::{ChatMessage, ChatRole, ThreadProjection};
use crate::store::ThreadStore;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_model_iterations: usize,
    pub system_prompt: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_model_iterations: 8,
            system_prompt: concat!(
                "You are a lite Q&A agent. Use functions only when they are useful. ",
                "Events and state are factual; do not invent completed goals. ",
                "Ask the user when required information is missing."
            )
            .to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    AssistantMessage { text: String },
    WaitingForUser { request_id: String, prompt: String },
    Failed { error: String },
}

#[derive(Clone)]
pub struct Agent {
    config: AgentConfig,
    store: Arc<dyn ThreadStore>,
    model_client: Arc<dyn ModelClient>,
    function_registry: FunctionRegistry,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        store: Arc<dyn ThreadStore>,
        model_client: Arc<dyn ModelClient>,
        function_registry: FunctionRegistry,
    ) -> Self {
        Self {
            config,
            store,
            model_client,
            function_registry,
        }
    }

    pub async fn run_turn(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
    ) -> Result<TurnOutcome> {
        let response_to = match self.store.load(thread_id).await {
            Ok(thread) => ThreadProjection::from_thread(&thread)
                .pending_user_input_request
                .map(|request| request.request_id),
            Err(AgentError::ThreadNotFound(_)) => None,
            Err(error) => return Err(error),
        };

        let mut thread = self
            .store
            .append(
                thread_id,
                vec![ThreadEvent::new(ThreadEventKind::UserInputReceived {
                    text: user_text.into(),
                    response_to,
                })],
            )
            .await?;

        for _ in 0..self.config.max_model_iterations {
            let projection = ThreadProjection::from_thread(&thread);
            let request = self.model_request_from_projection(projection);
            let response = self.model_client.complete(request).await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    self.store
                        .append(
                            thread_id,
                            vec![ThreadEvent::new(ThreadEventKind::TurnFailed {
                                error: error.to_string(),
                            })],
                        )
                        .await?;
                    return Ok(TurnOutcome::Failed {
                        error: error.to_string(),
                    });
                }
            };

            match response {
                ModelResponse::AssistantMessage { text } => {
                    self.store
                        .append(
                            thread_id,
                            vec![ThreadEvent::new(ThreadEventKind::AssistantMessageEmitted {
                                text: text.clone(),
                            })],
                        )
                        .await?;
                    return Ok(TurnOutcome::AssistantMessage { text });
                }
                ModelResponse::FunctionCalls { calls } => {
                    if calls.is_empty() {
                        let error = "model returned an empty function call list".to_string();
                        self.store
                            .append(
                                thread_id,
                                vec![ThreadEvent::new(ThreadEventKind::TurnFailed {
                                    error: error.clone(),
                                })],
                            )
                            .await?;
                        return Ok(TurnOutcome::Failed { error });
                    }

                    for call in calls {
                        thread = self
                            .store
                            .append(
                                thread_id,
                                vec![ThreadEvent::new(ThreadEventKind::FunctionCallRequested {
                                    call_id: call.call_id.clone(),
                                    name: call.name.clone(),
                                    arguments: call.arguments.clone(),
                                })],
                            )
                            .await?;

                        let context = FunctionContext {
                            projection: ThreadProjection::from_thread(&thread),
                        };
                        let execution = self
                            .function_registry
                            .call(&call.name, call.arguments.clone(), context)
                            .await;

                        match execution {
                            Ok(FunctionExecution::Completed {
                                output,
                                mut extra_events,
                            }) => {
                                extra_events.push(ThreadEvent::new(
                                    ThreadEventKind::FunctionCallCompleted {
                                        call_id: call.call_id,
                                        name: call.name,
                                        output,
                                    },
                                ));
                                thread = self.store.append(thread_id, extra_events).await?;
                            }
                            Ok(FunctionExecution::WaitingForUser {
                                request_id,
                                prompt,
                                output,
                                mut extra_events,
                            }) => {
                                extra_events.push(ThreadEvent::new(
                                    ThreadEventKind::FunctionCallCompleted {
                                        call_id: call.call_id,
                                        name: call.name,
                                        output,
                                    },
                                ));
                                self.store.append(thread_id, extra_events).await?;
                                return Ok(TurnOutcome::WaitingForUser { request_id, prompt });
                            }
                            Err(error) => {
                                thread = self
                                    .store
                                    .append(
                                        thread_id,
                                        vec![ThreadEvent::new(
                                            ThreadEventKind::FunctionCallFailed {
                                                call_id: call.call_id,
                                                name: call.name,
                                                error: error.to_string(),
                                            },
                                        )],
                                    )
                                    .await?;
                            }
                        }
                    }
                }
            }
        }

        let error = AgentError::MaxIterations(self.config.max_model_iterations).to_string();
        self.store
            .append(
                thread_id,
                vec![ThreadEvent::new(ThreadEventKind::TurnFailed {
                    error: error.clone(),
                })],
            )
            .await?;
        Ok(TurnOutcome::Failed { error })
    }

    fn model_request_from_projection(&self, projection: ThreadProjection) -> ModelRequest {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            content: self.config.system_prompt.clone(),
            name: None,
            tool_call_id: None,
        }];
        messages.extend(projection.messages_for_model);

        ModelRequest {
            messages,
            functions: self.function_registry.specs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::events::ThreadEventKind;
    use crate::functions::builtin_registry;
    use crate::model::{ModelClient, ModelFunctionCall, ModelRequest, ModelResponse};
    use crate::store::{JsonFileThreadStore, ThreadStore};
    use crate::{Agent, AgentConfig, Result, TurnOutcome};
    use serde_json::json;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    struct MockModel {
        responses: Mutex<VecDeque<ModelResponse>>,
    }

    impl MockModel {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }
    }

    impl ModelClient for MockModel {
        fn complete<'a>(
            &'a self,
            _request: ModelRequest,
        ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
            Box::pin(async move {
                self.responses
                    .lock()
                    .expect("lock")
                    .pop_front()
                    .ok_or_else(|| crate::AgentError::Model("no mock response".to_string()))
            })
        }
    }

    fn agent_with(store: Arc<dyn ThreadStore>, responses: Vec<ModelResponse>) -> Agent {
        Agent::new(
            AgentConfig::default(),
            store,
            Arc::new(MockModel::new(responses)),
            builtin_registry(),
        )
    }

    #[tokio::test]
    async fn simple_assistant_message_ends_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::AssistantMessage {
                text: "hello".to_string(),
            }],
        );

        let outcome = agent.run_turn("t", "hi").await.expect("turn");
        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "hello".to_string()
            }
        );
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.events.len(), 2);
    }

    #[tokio::test]
    async fn update_goal_then_message() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "update_goal".to_string(),
                        arguments: json!({ "objective": "ship", "status": "active" }),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "goal set".to_string(),
                },
            ],
        );

        let outcome = agent.run_turn("t", "set a goal").await.expect("turn");
        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "goal set".to_string()
            }
        );
        let thread = store.load("t").await.expect("thread");
        assert!(thread
            .events
            .iter()
            .any(|event| matches!(event.kind, ThreadEventKind::GoalUpdated { .. })));
    }

    #[tokio::test]
    async fn ask_user_stops_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::FunctionCalls {
                calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "ask_user".to_string(),
                    arguments: json!({ "prompt": "Which one?" }),
                }],
            }],
        );

        let outcome = agent.run_turn("t", "compare").await.expect("turn");
        assert!(matches!(outcome, TurnOutcome::WaitingForUser { .. }));
        let thread = store.load("t").await.expect("thread");
        assert!(thread
            .events
            .iter()
            .any(|event| matches!(event.kind, ThreadEventKind::UserInputRequested { .. })));
    }

    #[tokio::test]
    async fn max_iterations_fails_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = Agent::new(
            AgentConfig {
                max_model_iterations: 1,
                ..AgentConfig::default()
            },
            store,
            Arc::new(MockModel::new(vec![ModelResponse::FunctionCalls {
                calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "get_goal".to_string(),
                    arguments: json!({}),
                }],
            }])),
            builtin_registry(),
        );

        let outcome = agent.run_turn("t", "goal?").await.expect("turn");
        assert!(matches!(outcome, TurnOutcome::Failed { .. }));
    }
}
