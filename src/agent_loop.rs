use crate::error::{AgentError, Result};
use crate::events::{
    Thread, ToolResult, Turn, TurnId, TurnItem, TurnItemKind, TurnItemSource, TurnStatus,
};
use crate::functions::{FunctionContext, FunctionExecution, FunctionRegistry, ThreadUpdate};
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
                "Thread goal is explicit durable state. Turn items are factual append-only records. ",
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

#[derive(Debug, Clone)]
struct Session {
    _thread_id: String,
    active_turn_id: TurnId,
    projection: ThreadProjection,
}

impl Session {
    fn from_thread(thread: &Thread, active_turn_id: TurnId) -> Self {
        Self {
            _thread_id: thread.id.clone(),
            active_turn_id,
            projection: ThreadProjection::from_thread(thread),
        }
    }
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
        let mut thread = match self.store.load(thread_id).await {
            Ok(thread) => thread,
            Err(AgentError::ThreadNotFound(_)) => Thread::new(thread_id),
            Err(error) => return Err(error),
        };
        let response_to = ThreadProjection::from_thread(&thread)
            .pending_user_input_request
            .map(|request| request.request_id);

        let mut turn = Turn::new();
        let turn_id = turn.id.clone();
        turn.push_item(TurnItem::new(
            TurnItemSource::User,
            TurnItemKind::UserInput {
                text: user_text.into(),
                response_to,
            },
        ));
        thread.turns.push(turn);
        thread = self.store.save(thread).await?;

        for _ in 0..self.config.max_model_iterations {
            let session = Session::from_thread(&thread, turn_id.clone());
            let request = self.model_request_from_projection(session.projection.clone());
            let response = self.model_client.complete(request).await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    let error = error.to_string();
                    self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                    self.store.save(thread).await?;
                    return Ok(TurnOutcome::Failed { error });
                }
            };

            match response {
                ModelResponse::AssistantMessage { text } => {
                    Self::push_turn_items(
                        &mut thread,
                        &session.active_turn_id,
                        vec![TurnItem::new(
                            TurnItemSource::Model,
                            TurnItemKind::ModelMessage { text: text.clone() },
                        )],
                    )?;
                    Self::set_turn_status(
                        &mut thread,
                        &session.active_turn_id,
                        TurnStatus::Completed,
                    )?;
                    self.store.save(thread).await?;
                    return Ok(TurnOutcome::AssistantMessage { text });
                }
                ModelResponse::FunctionCalls { calls } => {
                    if calls.is_empty() {
                        let error = "model returned an empty function call list".to_string();
                        self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                        self.store.save(thread).await?;
                        return Ok(TurnOutcome::Failed { error });
                    }

                    for call in calls {
                        Self::push_turn_items(
                            &mut thread,
                            &session.active_turn_id,
                            vec![TurnItem::new(
                                TurnItemSource::Model,
                                TurnItemKind::ModelFunctionCall {
                                    call_id: call.call_id.clone(),
                                    name: call.name.clone(),
                                    arguments: call.arguments.clone(),
                                },
                            )],
                        )?;
                        thread = self.store.save(thread).await?;

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
                                thread_update,
                                mut extra_items,
                            }) => {
                                Self::apply_thread_update(&mut thread, thread_update);
                                extra_items.push(TurnItem::new(
                                    TurnItemSource::Tool,
                                    TurnItemKind::ToolOutput {
                                        call_id: call.call_id,
                                        name: call.name,
                                        result: ToolResult::Success { output },
                                    },
                                ));
                                Self::push_turn_items(&mut thread, &turn_id, extra_items)?;
                                thread = self.store.save(thread).await?;
                            }
                            Ok(FunctionExecution::WaitingForUser {
                                request_id,
                                prompt,
                                output,
                                thread_update,
                                mut extra_items,
                            }) => {
                                Self::apply_thread_update(&mut thread, thread_update);
                                extra_items.push(TurnItem::new(
                                    TurnItemSource::Tool,
                                    TurnItemKind::ToolOutput {
                                        call_id: call.call_id,
                                        name: call.name,
                                        result: ToolResult::Success { output },
                                    },
                                ));
                                Self::push_turn_items(&mut thread, &turn_id, extra_items)?;
                                Self::set_turn_status(
                                    &mut thread,
                                    &turn_id,
                                    TurnStatus::WaitingForUser,
                                )?;
                                self.store.save(thread).await?;
                                return Ok(TurnOutcome::WaitingForUser { request_id, prompt });
                            }
                            Err(error) => {
                                Self::push_turn_items(
                                    &mut thread,
                                    &turn_id,
                                    vec![TurnItem::new(
                                        TurnItemSource::Tool,
                                        TurnItemKind::ToolOutput {
                                            call_id: call.call_id,
                                            name: call.name,
                                            result: ToolResult::Error {
                                                error: error.to_string(),
                                            },
                                        },
                                    )],
                                )?;
                                thread = self.store.save(thread).await?;
                            }
                        }
                    }
                }
            }
        }

        let error = AgentError::MaxIterations(self.config.max_model_iterations).to_string();
        self.fail_turn(&mut thread, &turn_id, error.clone())?;
        self.store.save(thread).await?;
        Ok(TurnOutcome::Failed { error })
    }

    fn model_request_from_projection(&self, projection: ThreadProjection) -> ModelRequest {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            content: self.config.system_prompt.clone(),
            name: None,
            tool_call_id: None,
        }];
        if let Some(goal) = &projection.goal {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: format!(
                    "Current thread goal: objective={}, status={:?}, notes={}",
                    goal.objective,
                    goal.status,
                    goal.notes.as_deref().unwrap_or("")
                ),
                name: None,
                tool_call_id: None,
            });
        }
        messages.extend(projection.messages_for_model);

        ModelRequest {
            messages,
            functions: self.function_registry.specs(),
        }
    }

    fn apply_thread_update(thread: &mut Thread, update: Option<ThreadUpdate>) {
        if let Some(ThreadUpdate::Goal(goal)) = update {
            thread.goal = Some(goal);
        }
    }

    fn push_turn_items(thread: &mut Thread, turn_id: &str, items: Vec<TurnItem>) -> Result<()> {
        let turn = thread
            .turn_mut(turn_id)
            .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
        for item in items {
            turn.push_item(item);
        }
        Ok(())
    }

    fn set_turn_status(thread: &mut Thread, turn_id: &str, status: TurnStatus) -> Result<()> {
        let turn = thread
            .turn_mut(turn_id)
            .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
        turn.set_status(status);
        Ok(())
    }

    fn fail_turn(&self, thread: &mut Thread, turn_id: &str, error: String) -> Result<()> {
        Self::push_turn_items(
            thread,
            turn_id,
            vec![TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::TurnFailed { error },
            )],
        )?;
        Self::set_turn_status(thread, turn_id, TurnStatus::Failed)
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{ToolResult, TurnItemKind, TurnStatus};
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
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].status, TurnStatus::Completed);
        assert_eq!(thread.turns[0].items.len(), 2);
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
        assert_eq!(
            thread.goal.as_ref().map(|goal| goal.objective.as_str()),
            Some("ship")
        );
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::GoalUpdated { .. })));
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
        assert_eq!(thread.turns[0].status, TurnStatus::WaitingForUser);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::UserInputRequested { .. })));
    }

    #[tokio::test]
    async fn function_failure_becomes_tool_error() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "missing".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "could not call it".to_string(),
                },
            ],
        );

        let outcome = agent.run_turn("t", "call missing").await.expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        let thread = store.load("t").await.expect("thread");
        assert!(thread.turns[0].items.iter().any(|item| matches!(
            &item.kind,
            TurnItemKind::ToolOutput {
                result: ToolResult::Error { .. },
                ..
            }
        )));
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
            store.clone(),
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
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Failed);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::TurnFailed { .. })));
    }
}
