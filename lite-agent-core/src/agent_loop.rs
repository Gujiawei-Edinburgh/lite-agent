use crate::error::{AgentError, Result};
use crate::events::{
    Thread, TokenUsage, ToolResult, Turn, TurnId, TurnItem, TurnItemKind, TurnItemSource,
    TurnStatus,
};
use crate::functions::{FunctionContext, FunctionExecution, FunctionRegistry, ThreadUpdate};
use crate::model::{ModelClient, ModelRequest, ModelResponse, ModelStreamEvent};
use crate::projection::{ChatMessage, ThreadProjection};
use crate::store::ThreadStore;
use chrono::{Local, Utc};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

pub trait RuntimeContextProvider: Send + Sync {
    fn context_for_turn(&self, input: RuntimeContextInput<'_>) -> Option<String>;
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeContextInput<'a> {
    pub thread_id: &'a str,
    pub user_text: &'a str,
}

#[derive(Clone)]
pub struct AgentConfig {
    pub max_model_iterations: usize,
    pub system_prompt: String,
    pub runtime_context_provider: Option<Arc<dyn RuntimeContextProvider>>,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("max_model_iterations", &self.max_model_iterations)
            .field("system_prompt", &self.system_prompt)
            .field(
                "runtime_context_provider",
                &self.runtime_context_provider.is_some(),
            )
            .finish()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_model_iterations: 128,
            system_prompt: concat!(
                "You are a lite Q&A agent. Use functions only when they are useful. ",
                "Thread goal is explicit durable state. Turn items are factual append-only records. ",
                "Ask the user when required information is missing."
            )
            .to_string(),
            runtime_context_provider: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    AssistantMessage { text: String },
    WaitingForUser { request_id: String, prompt: String },
    Failed { error: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnStreamEvent {
    TurnStarted {
        thread_id: String,
        turn_id: TurnId,
    },
    ModelRequestStarted {
        iteration: usize,
    },
    ModelMessage {
        text: String,
    },
    ModelMessageDelta {
        text: String,
    },
    FunctionCallsRequested {
        calls: Vec<crate::model::ModelFunctionCall>,
    },
    FunctionStarted {
        call_id: String,
        name: String,
    },
    FunctionCompleted {
        call_id: String,
        name: String,
    },
    FunctionFailed {
        call_id: String,
        name: String,
        error: String,
    },
    WaitingForUser {
        request_id: String,
        prompt: String,
    },
    TurnCompleted {
        outcome: TurnOutcome,
    },
    TurnFailed {
        error: String,
    },
    TurnTokenUsage {
        usage: TokenUsage,
    },
}

pub type TurnEventHandler<'a> = dyn FnMut(TurnStreamEvent) + Send + 'a;

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
    session_locks: Arc<AsyncMutex<BTreeMap<String, Arc<AsyncMutex<()>>>>>,
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
            session_locks: Arc::new(AsyncMutex::new(BTreeMap::new())),
        }
    }

    pub async fn run_turn(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
    ) -> Result<TurnOutcome> {
        self.run_turn_stream(thread_id, user_text, |_event| {})
            .await
    }

    pub async fn run_turn_stream<'a, F>(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
        mut on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let _session_lock = self.acquire_session_lock(thread_id).await;
        tracing::debug!(thread_id, "session lock acquired");

        let user_text = user_text.into();
        let runtime_context = self
            .config
            .runtime_context_provider
            .as_ref()
            .and_then(|provider| {
                provider.context_for_turn(RuntimeContextInput {
                    thread_id,
                    user_text: &user_text,
                })
            });

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
                text: user_text,
                response_to,
            },
        ));
        thread.turns.push(turn);
        thread = self.store.save(thread).await?;
        tracing::info!(thread_id, turn_id, "turn started");
        on_event(TurnStreamEvent::TurnStarted {
            thread_id: thread_id.to_string(),
            turn_id: turn_id.clone(),
        });
        let mut turn_token_usage = TokenUsage::default();

        let outcome = 'turn_loop: loop {
            for iteration in 0..self.config.max_model_iterations {
                let session = Session::from_thread(&thread, turn_id.clone());
                let request = self.model_request_from_projection(
                    session.projection.clone(),
                    runtime_context.as_deref(),
                );
                on_event(TurnStreamEvent::ModelRequestStarted { iteration });
                let mut model_event_handler = |event| match event {
                    ModelStreamEvent::AssistantDelta { text } => {
                        on_event(TurnStreamEvent::ModelMessageDelta { text });
                    }
                    ModelStreamEvent::TokenUsage { usage } => {
                        turn_token_usage.add_assign(usage);
                    }
                };
                let response = self
                    .model_client
                    .stream_complete(request, &mut model_event_handler)
                    .await;
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        let error = error.to_string();
                        tracing::error!(error, "turn failed during model request");
                        self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                        break 'turn_loop TurnOutcome::Failed { error };
                    }
                };

                match response {
                    ModelResponse::AssistantMessage { text } => {
                        on_event(TurnStreamEvent::ModelMessage { text: text.clone() });
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
                        break 'turn_loop TurnOutcome::AssistantMessage { text };
                    }
                    ModelResponse::FunctionCalls { calls } => {
                        if calls.is_empty() {
                            let error = "model returned an empty function call list".to_string();
                            tracing::warn!(error, "empty function call list");
                            self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                            break 'turn_loop TurnOutcome::Failed { error };
                        }

                        on_event(TurnStreamEvent::FunctionCallsRequested {
                            calls: calls.clone(),
                        });
                        let call_items = calls
                            .iter()
                            .map(|call| {
                                TurnItem::new(
                                    TurnItemSource::Model,
                                    TurnItemKind::ModelFunctionCall {
                                        call_id: call.call_id.clone(),
                                        name: call.name.clone(),
                                        arguments: call.arguments.clone(),
                                    },
                                )
                            })
                            .collect();
                        Self::push_turn_items(&mut thread, &session.active_turn_id, call_items)?;
                        thread = self.store.save(thread).await?;

                        for call in calls {
                            let call_id = call.call_id.clone();
                            let name = call.name.clone();
                            on_event(TurnStreamEvent::FunctionStarted {
                                call_id: call_id.clone(),
                                name: name.clone(),
                            });
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
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    Self::push_turn_items(&mut thread, &turn_id, extra_items)?;
                                    thread = self.store.save(thread).await?;
                                    on_event(TurnStreamEvent::FunctionCompleted { call_id, name });
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
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    Self::push_turn_items(&mut thread, &turn_id, extra_items)?;
                                    Self::set_turn_status(
                                        &mut thread,
                                        &turn_id,
                                        TurnStatus::WaitingForUser,
                                    )?;
                                    on_event(TurnStreamEvent::FunctionCompleted { call_id, name });
                                    on_event(TurnStreamEvent::WaitingForUser {
                                        request_id: request_id.clone(),
                                        prompt: prompt.clone(),
                                    });
                                    break 'turn_loop TurnOutcome::WaitingForUser {
                                        request_id,
                                        prompt,
                                    };
                                }
                                Err(error) => {
                                    let error_text = error.to_string();
                                    Self::push_turn_items(
                                        &mut thread,
                                        &turn_id,
                                        vec![TurnItem::new(
                                            TurnItemSource::Tool,
                                            TurnItemKind::ToolOutput {
                                                call_id: call_id.clone(),
                                                name: name.clone(),
                                                result: ToolResult::Error {
                                                    error: error_text.clone(),
                                                },
                                            },
                                        )],
                                    )?;
                                    thread = self.store.save(thread).await?;
                                    on_event(TurnStreamEvent::FunctionFailed {
                                        call_id,
                                        name,
                                        error: error_text,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            let error = AgentError::MaxIterations(self.config.max_model_iterations).to_string();
            tracing::warn!(error, "turn exceeded max iterations");
            self.fail_turn(&mut thread, &turn_id, error.clone())?;
            break 'turn_loop TurnOutcome::Failed { error };
        };

        Self::apply_turn_token_usage(&mut thread, turn_token_usage);
        self.store.save(thread).await?;
        if let TurnOutcome::Failed { error } = &outcome {
            on_event(TurnStreamEvent::TurnFailed {
                error: error.clone(),
            });
        }
        Self::emit_turn_token_usage(&mut on_event, turn_token_usage);
        on_event(TurnStreamEvent::TurnCompleted {
            outcome: outcome.clone(),
        });
        Ok(outcome)
    }

    fn apply_turn_token_usage(thread: &mut Thread, usage: TokenUsage) {
        if !usage.is_zero() {
            thread.token_usage.add_assign(usage);
        }
    }

    fn emit_turn_token_usage(on_event: &mut TurnEventHandler<'_>, usage: TokenUsage) {
        if !usage.is_zero() {
            on_event(TurnStreamEvent::TurnTokenUsage { usage });
        }
    }

    async fn acquire_session_lock(&self, thread_id: &str) -> OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.session_locks.lock().await;
            locks
                .entry(thread_id.to_string())
                .or_insert_with(|| Arc::new(AsyncMutex::new(())))
                .clone()
        };
        lock.lock_owned().await
    }

    fn model_request_from_projection(
        &self,
        projection: ThreadProjection,
        runtime_context: Option<&str>,
    ) -> ModelRequest {
        let mut system_content = self.config.system_prompt.clone();
        if let Some(runtime_context) = runtime_context {
            let runtime_context = runtime_context.trim();
            if !runtime_context.is_empty() {
                system_content.push_str(
                    "\nCurrent turn runtime context. This host-supplied context applies only to this model request and is not durable thread state:\n",
                );
                system_content.push_str(runtime_context);
            }
        }
        system_content.push_str(&format!(
            "\nCurrent time context: local={}, utc={}. Use this as the current date/time for time-sensitive answers.",
            Local::now().format("%Y-%m-%d %H:%M:%S %:z"),
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        if let Some(goal) = &projection.goal {
            system_content.push_str(&format!(
                "\nCurrent thread goal: objective={}, status={:?}, notes={}",
                goal.objective,
                goal.status,
                goal.notes.as_deref().unwrap_or("")
            ));
        }
        let mut messages = vec![ChatMessage::System {
            content: system_content,
        }];
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
    use crate::events::{TokenUsage, ToolResult, TurnItemKind, TurnStatus};
    use crate::functions::builtin_registry;
    use crate::model::{
        ModelClient, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent,
        ModelStreamHandler,
    };
    use crate::store::{JsonFileThreadStore, ThreadStore};
    use crate::{
        Agent, AgentConfig, Result, RuntimeContextInput, RuntimeContextProvider, TurnOutcome,
        TurnStreamEvent,
    };
    use serde_json::json;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::time::{sleep, Duration};

    struct MockModel {
        responses: Mutex<VecDeque<ModelResponse>>,
        requests: Mutex<Vec<ModelRequest>>,
    }

    impl MockModel {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl ModelClient for MockModel {
        fn stream_complete<'a>(
            &'a self,
            request: ModelRequest,
            _on_event: &'a mut ModelStreamHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
            Box::pin(async move {
                self.requests.lock().expect("requests").push(request);
                self.responses
                    .lock()
                    .expect("lock")
                    .pop_front()
                    .ok_or_else(|| crate::AgentError::Model("no mock response".to_string()))
            })
        }
    }

    struct SlowCountingModel {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl SlowCountingModel {
        fn new(active: Arc<AtomicUsize>, max_active: Arc<AtomicUsize>) -> Self {
            Self { active, max_active }
        }
    }

    impl ModelClient for SlowCountingModel {
        fn stream_complete<'a>(
            &'a self,
            _request: ModelRequest,
            _on_event: &'a mut ModelStreamHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
            Box::pin(async move {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_active.fetch_max(active, Ordering::SeqCst);
                sleep(Duration::from_millis(50)).await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(ModelResponse::AssistantMessage {
                    text: "done".to_string(),
                })
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
    async fn concurrent_turns_for_same_thread_are_serialized() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let agent = Arc::new(Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(SlowCountingModel::new(active, max_active.clone())),
            builtin_registry(),
        ));

        let first_agent = agent.clone();
        let first = tokio::spawn(async move { first_agent.run_turn("t", "first").await });
        let second_agent = agent.clone();
        let second = tokio::spawn(async move { second_agent.run_turn("t", "second").await });

        first.await.expect("join").expect("first turn");
        second.await.expect("join").expect("second turn");

        assert_eq!(max_active.load(Ordering::SeqCst), 1);
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns.len(), 2);
        assert!(thread
            .turns
            .iter()
            .all(|turn| turn.status == TurnStatus::Completed));
    }

    #[test]
    fn model_request_includes_current_time_context() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = agent_with(
            store,
            vec![ModelResponse::AssistantMessage {
                text: "hello".to_string(),
            }],
        );

        let request = agent.model_request_from_projection(Default::default(), None);
        let Some(crate::projection::ChatMessage::System { content }) = request.messages.first()
        else {
            panic!("missing system message");
        };

        assert!(content.contains("Current time context: local="));
        assert!(content.contains("utc="));
        assert!(content.contains("time-sensitive answers"));
    }

    #[tokio::test]
    async fn runtime_context_is_added_to_system_prompt_not_user_input() {
        struct DummyRuntimeContextProvider;

        impl RuntimeContextProvider for DummyRuntimeContextProvider {
            fn context_for_turn(&self, input: RuntimeContextInput<'_>) -> Option<String> {
                Some(format!(
                    "<runtime thread=\"{}\">query={}</runtime>",
                    input.thread_id, input.user_text
                ))
            }
        }

        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let model = Arc::new(MockModel::new(vec![ModelResponse::AssistantMessage {
            text: "hello".to_string(),
        }]));
        let agent = Agent::new(
            AgentConfig {
                runtime_context_provider: Some(Arc::new(DummyRuntimeContextProvider)),
                ..AgentConfig::default()
            },
            store.clone(),
            model.clone(),
            builtin_registry(),
        );

        let outcome = agent.run_turn("thread-a", "coffee shop lights").await;
        assert!(matches!(outcome, Ok(TurnOutcome::AssistantMessage { .. })));

        let requests = model.requests.lock().expect("requests");
        let system = requests[0]
            .messages
            .iter()
            .find_map(|message| match message {
                crate::projection::ChatMessage::System { content } => Some(content),
                _ => None,
            })
            .expect("system message");
        assert!(system.contains("<runtime thread=\"thread-a\">query=coffee shop lights</runtime>"));

        let thread = store.load("thread-a").await.expect("thread");
        let TurnItemKind::UserInput { text, .. } = &thread.turns[0].items[0].kind else {
            panic!("missing user input");
        };
        assert_eq!(text, "coffee shop lights");
    }

    #[tokio::test]
    async fn stream_emits_intermediate_and_final_events() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = agent_with(
            store,
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "get_goal".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "done".to_string(),
                },
            ],
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();

        let outcome = agent
            .run_turn_stream("t", "goal?", move |event| {
                captured.lock().expect("lock").push(event);
            })
            .await
            .expect("turn");

        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "done".to_string()
            }
        );
        let events = events.lock().expect("lock");
        assert!(events
            .iter()
            .any(|event| matches!(event, TurnStreamEvent::FunctionStarted { .. })));
        assert!(events
            .iter()
            .any(|event| matches!(event, TurnStreamEvent::TurnCompleted { .. })));
    }

    #[tokio::test]
    async fn turn_token_usage_is_emitted_and_persisted_on_thread() {
        struct UsageModel;

        impl ModelClient for UsageModel {
            fn stream_complete<'a>(
                &'a self,
                _request: ModelRequest,
                on_event: &'a mut ModelStreamHandler<'a>,
            ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
                Box::pin(async move {
                    on_event(ModelStreamEvent::TokenUsage {
                        usage: TokenUsage {
                            input_tokens: 10,
                            cached_input_tokens: 2,
                            output_tokens: 4,
                            total_tokens: 14,
                        },
                    });
                    Ok(ModelResponse::AssistantMessage {
                        text: "done".to_string(),
                    })
                })
            }
        }

        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::new(temp.path()));
        let agent = Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(UsageModel),
            builtin_registry(),
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();

        let outcome = agent
            .run_turn_stream("t", "usage?", move |event| {
                captured.lock().expect("lock").push(event);
            })
            .await
            .expect("turn");

        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        let events = events.lock().expect("lock");
        let usage = events
            .iter()
            .find_map(|event| match event {
                TurnStreamEvent::TurnTokenUsage { usage } => Some(*usage),
                _ => None,
            })
            .expect("usage event");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cached_input_tokens, 2);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.total_tokens, 14);

        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.token_usage, usage);
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
