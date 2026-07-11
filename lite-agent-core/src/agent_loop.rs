use crate::context::{CompactingContextBuilder, ContextBuildInput, ContextBuilder};
use crate::error::{AgentError, Result};
use crate::events::{
    Suspension, Thread, TokenUsage, ToolResult, Turn, TurnId, TurnItem, TurnItemKind,
    TurnItemSource, TurnStatus,
};
use crate::functions::{FunctionContext, FunctionExecution, FunctionRegistry, ThreadUpdate};
use crate::model::{ModelClient, ModelRequest, ModelResponse, ModelStreamEvent};
use crate::projection::ThreadProjection;
use crate::store::ThreadStore;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::watch;

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
                "You are an agent runtime assistant. Use functions only when they are useful. ",
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
    Suspended { suspension: Suspension },
    Failed { error: String },
    Aborted { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnStreamEvent {
    State(TurnStateEvent),
    Model(TurnModelEvent),
    Runtime(RuntimeEvent),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnStateEvent {
    TurnStarted {
        thread_id: String,
        turn_id: TurnId,
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
    Suspended {
        suspension: Suspension,
    },
    TurnFinished {
        outcome: TurnOutcome,
    },
    TurnFailed {
        error: String,
    },
    TurnAborted {
        reason: String,
    },
    TurnTokenUsage {
        usage: TokenUsage,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnModelEvent {
    RequestStarted { iteration: usize },
    AssistantMessage { text: String },
    AssistantDelta { text: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeEvent {
    pub source: String,
    pub message: String,
    pub metadata: Value,
}

pub type TurnEventHandler<'a> = dyn FnMut(TurnStreamEvent) + Send + 'a;

#[derive(Debug, Clone)]
pub struct TurnAbortHandle {
    sender: watch::Sender<bool>,
}

#[derive(Debug, Clone)]
pub struct TurnAbortSignal {
    receiver: watch::Receiver<bool>,
}

pub fn turn_abort_pair() -> (TurnAbortHandle, TurnAbortSignal) {
    let (sender, receiver) = watch::channel(false);
    (TurnAbortHandle { sender }, TurnAbortSignal { receiver })
}

impl TurnAbortHandle {
    pub fn abort(&self) {
        let _ = self.sender.send(true);
    }
}

impl TurnAbortSignal {
    async fn cancelled(&mut self) {
        if *self.receiver.borrow() {
            return;
        }
        while self.receiver.changed().await.is_ok() {
            if *self.receiver.borrow() {
                return;
            }
        }
        std::future::pending::<()>().await;
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCallHookContext {
    pub thread_id: String,
    pub turn_id: TurnId,
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
    pub projection: ThreadProjection,
    /// Set for after-hooks after the resulting thread state has been reconstructed.
    pub projection_after: Option<ThreadProjection>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionCallHookResult {
    Completed {
        output: Value,
    },
    Suspended {
        suspension: Suspension,
        output: Value,
    },
    Failed {
        error: String,
    },
}

pub trait FunctionCallHook: Send + Sync {
    fn before_call<'a>(
        &'a self,
        _context: FunctionCallHookContext,
        _emit: &'a mut TurnEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn after_call<'a>(
        &'a self,
        _context: FunctionCallHookContext,
        _result: FunctionCallHookResult,
        _emit: &'a mut TurnEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
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
    function_call_hooks: Vec<Arc<dyn FunctionCallHook>>,
    context_builder: Arc<dyn ContextBuilder>,
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
            function_call_hooks: Vec::new(),
            context_builder: Arc::new(CompactingContextBuilder::default()),
        }
    }

    pub fn with_function_call_hook<H>(mut self, hook: H) -> Self
    where
        H: FunctionCallHook + 'static,
    {
        self.function_call_hooks.push(Arc::new(hook));
        self
    }

    pub fn with_function_call_hooks(mut self, hooks: Vec<Arc<dyn FunctionCallHook>>) -> Self {
        self.function_call_hooks.extend(hooks);
        self
    }

    pub fn with_context_builder<C>(mut self, context_builder: C) -> Self
    where
        C: ContextBuilder + 'static,
    {
        self.context_builder = Arc::new(context_builder);
        self
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
        on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let (_abort_handle, abort_signal) = turn_abort_pair();
        self.run_turn_stream_abortable(thread_id, user_text, abort_signal, on_event)
            .await
    }

    pub async fn run_turn_stream_abortable<'a, F>(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
        mut abort_signal: TurnAbortSignal,
        mut on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let session_lock = self.store.session_lock(thread_id);
        let _session_lock = session_lock.lock().await;
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
            .pending_suspension
            .map(|suspension| suspension.suspension.id);

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
        thread = self.commit_thread(thread).await?;
        tracing::info!(thread_id, turn_id, "turn started");
        on_event(TurnStreamEvent::State(TurnStateEvent::TurnStarted {
            thread_id: thread_id.to_string(),
            turn_id: turn_id.clone(),
        }));
        let mut turn_token_usage = TokenUsage::default();

        let outcome = 'turn_loop: {
            for iteration in 0..self.config.max_model_iterations {
                let session = Session::from_thread(&thread, turn_id.clone());
                let request = self.model_request_from_projection(
                    session.projection.clone(),
                    runtime_context.as_deref(),
                )?;
                on_event(TurnStreamEvent::Model(TurnModelEvent::RequestStarted {
                    iteration,
                }));
                let mut model_event_handler = |event| match event {
                    ModelStreamEvent::AssistantDelta { text } => {
                        on_event(TurnStreamEvent::Model(TurnModelEvent::AssistantDelta {
                            text,
                        }));
                    }
                    ModelStreamEvent::TokenUsage { usage } => {
                        turn_token_usage.add_assign(usage);
                    }
                };
                let model_call = self
                    .model_client
                    .stream_complete(request, &mut model_event_handler);
                let response = self
                    .await_step_or_abort(
                        model_call,
                        &mut abort_signal,
                        &mut thread,
                        thread_id,
                        &session.active_turn_id,
                        "model request",
                    )
                    .await?;
                let Some(response) = response else {
                    break 'turn_loop TurnOutcome::Aborted {
                        reason: "turn aborted by caller".to_string(),
                    };
                };
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        let error = error.to_string();
                        tracing::error!(error, "turn failed during model request");
                        self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                        break 'turn_loop TurnOutcome::Failed { error };
                    }
                };

                let response = match response {
                    ModelResponse::AssistantMessage { text } => ModelResponse::Assistant {
                        text: Some(text),
                        function_calls: Vec::new(),
                    },
                    ModelResponse::FunctionCalls { calls } => ModelResponse::Assistant {
                        text: None,
                        function_calls: calls,
                    },
                    response => response,
                };
                match response {
                    ModelResponse::Assistant {
                        text,
                        function_calls,
                    } => {
                        if let Some(text) = &text {
                            on_event(TurnStreamEvent::Model(TurnModelEvent::AssistantMessage {
                                text: text.clone(),
                            }));
                            Self::push_turn_items(
                                &mut thread,
                                &session.active_turn_id,
                                vec![TurnItem::new(
                                    TurnItemSource::Model,
                                    TurnItemKind::ModelMessage { text: text.clone() },
                                )],
                            )?;
                            thread = self.commit_thread(thread).await?;
                        }
                        if function_calls.is_empty() {
                            let text = text.unwrap_or_default();
                            Self::set_turn_status(
                                &mut thread,
                                &session.active_turn_id,
                                TurnStatus::Completed,
                            )?;
                            break 'turn_loop TurnOutcome::AssistantMessage { text };
                        }
                        let calls = function_calls;
                        if calls.is_empty() {
                            let error = "model returned an empty function call list".to_string();
                            tracing::warn!(error, "empty function call list");
                            self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                            break 'turn_loop TurnOutcome::Failed { error };
                        }

                        on_event(TurnStreamEvent::State(
                            TurnStateEvent::FunctionCallsRequested {
                                calls: calls.clone(),
                            },
                        ));
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
                        thread = self.commit_thread(thread).await?;

                        for (call_index, call) in calls.iter().enumerate() {
                            let call_id = call.call_id.clone();
                            let name = call.name.clone();
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                                call_id: call_id.clone(),
                                name: name.clone(),
                            }));
                            let projection = ThreadProjection::from_thread(&thread);
                            let hook_context = FunctionCallHookContext {
                                thread_id: thread.id.clone(),
                                turn_id: turn_id.clone(),
                                call_id: call_id.clone(),
                                name: name.clone(),
                                arguments: call.arguments.clone(),
                                projection: projection.clone(),
                                projection_after: None,
                            };
                            let pre_hook_result = self
                                .run_before_function_call_hooks(hook_context.clone(), &mut on_event)
                                .await;
                            let execution = match pre_hook_result {
                                Ok(()) => {
                                    let context = FunctionContext {
                                        thread_id: thread.id.clone(),
                                        turn_id: turn_id.clone(),
                                        call_id: call_id.clone(),
                                        projection,
                                        abort_signal: abort_signal.clone(),
                                    };
                                    let function_call = self.function_registry.call(
                                        &call.name,
                                        call.arguments.clone(),
                                        context,
                                    );
                                    self.await_step_or_abort(
                                        function_call,
                                        &mut abort_signal,
                                        &mut thread,
                                        thread_id,
                                        &turn_id,
                                        "function call",
                                    )
                                    .await?
                                }
                                Err(error) => Some(Err(error)),
                            };

                            let Some(execution) = execution else {
                                break 'turn_loop TurnOutcome::Aborted {
                                    reason: "turn aborted by caller".to_string(),
                                };
                            };
                            match execution {
                                Ok(FunctionExecution::Completed {
                                    output,
                                    thread_update,
                                }) => {
                                    let update_item =
                                        Self::apply_thread_update(&mut thread, thread_update);
                                    let hook_result = FunctionCallHookResult::Completed {
                                        output: output.clone(),
                                    };
                                    let mut func_items =
                                        update_item.into_iter().collect::<Vec<_>>();
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Tool,
                                        TurnItemKind::ToolOutput {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    thread = self.commit_thread(thread).await?;
                                    let mut after_context = hook_context.clone();
                                    after_context.projection_after =
                                        Some(ThreadProjection::from_thread(&thread));
                                    self.run_after_function_call_hooks(
                                        after_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionCompleted { call_id, name },
                                    ));
                                }
                                Ok(FunctionExecution::Suspended {
                                    suspension,
                                    output,
                                    thread_update,
                                }) => {
                                    let update_item =
                                        Self::apply_thread_update(&mut thread, thread_update);
                                    let hook_result = FunctionCallHookResult::Suspended {
                                        suspension: suspension.clone(),
                                        output: output.clone(),
                                    };
                                    let mut func_items =
                                        update_item.into_iter().collect::<Vec<_>>();
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Tool,
                                        TurnItemKind::ToolOutput {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Runtime,
                                        TurnItemKind::SuspensionCreated {
                                            suspension: suspension.clone(),
                                        },
                                    ));
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    Self::set_turn_status(
                                        &mut thread,
                                        &turn_id,
                                        TurnStatus::Suspended,
                                    )?;
                                    thread = self.commit_thread(thread).await?;
                                    let mut after_context = hook_context.clone();
                                    after_context.projection_after =
                                        Some(ThreadProjection::from_thread(&thread));
                                    self.run_after_function_call_hooks(
                                        after_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionCompleted { call_id, name },
                                    ));
                                    on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                                        suspension: suspension.clone(),
                                    }));
                                    // A suspended call ends this model iteration. Record explicit
                                    // tool errors for later calls so the next request has no
                                    // unmatched assistant tool-call records.
                                    let skipped_items = calls
                                        .iter()
                                        .skip(call_index + 1)
                                        .map(|skipped_call| {
                                            let skipped_error =
                                                "function not executed because a previous function suspended the turn";
                                            on_event(TurnStreamEvent::State(
                                                TurnStateEvent::FunctionFailed {
                                                    call_id: skipped_call.call_id.clone(),
                                                    name: skipped_call.name.clone(),
                                                    error: skipped_error.to_string(),
                                                },
                                            ));
                                            TurnItem::new(
                                                TurnItemSource::Tool,
                                                TurnItemKind::ToolOutput {
                                                    call_id: skipped_call.call_id.clone(),
                                                    name: skipped_call.name.clone(),
                                                    result: ToolResult::Error {
                                                        error: skipped_error.to_string(),
                                                    },
                                                },
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    Self::push_turn_items(&mut thread, &turn_id, skipped_items)?;
                                    thread = self.commit_thread(thread).await?;
                                    break 'turn_loop TurnOutcome::Suspended { suspension };
                                }
                                Err(error) => {
                                    let error_text = error.to_string();
                                    let hook_result = FunctionCallHookResult::Failed {
                                        error: error_text.clone(),
                                    };
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
                                    thread = self.commit_thread(thread).await?;
                                    let mut after_context = hook_context.clone();
                                    after_context.projection_after =
                                        Some(ThreadProjection::from_thread(&thread));
                                    self.run_after_function_call_hooks(
                                        after_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionFailed {
                                            call_id,
                                            name,
                                            error: error_text,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                    ModelResponse::AssistantMessage { .. }
                    | ModelResponse::FunctionCalls { .. } => {
                        unreachable!("legacy model response variants are normalized above")
                    }
                }
            }

            let error = AgentError::MaxIterations(self.config.max_model_iterations).to_string();
            tracing::warn!(error, "turn exceeded max iterations");
            self.fail_turn(&mut thread, &turn_id, error.clone())?;
            break 'turn_loop TurnOutcome::Failed { error };
        };

        Self::apply_turn_token_usage(&mut thread, turn_token_usage);
        self.commit_thread(thread).await?;
        if let TurnOutcome::Failed { error } = &outcome {
            on_event(TurnStreamEvent::State(TurnStateEvent::TurnFailed {
                error: error.clone(),
            }));
        }
        if let TurnOutcome::Aborted { reason } = &outcome {
            on_event(TurnStreamEvent::State(TurnStateEvent::TurnAborted {
                reason: reason.clone(),
            }));
        }
        Self::emit_turn_token_usage(&mut on_event, turn_token_usage);
        on_event(TurnStreamEvent::State(TurnStateEvent::TurnFinished {
            outcome: outcome.clone(),
        }));
        Ok(outcome)
    }

    fn apply_turn_token_usage(thread: &mut Thread, usage: TokenUsage) {
        if !usage.is_zero() {
            thread.token_usage.add_assign(usage);
        }
    }

    async fn await_step_or_abort<T, F>(
        &self,
        future: F,
        abort_signal: &mut TurnAbortSignal,
        thread: &mut Thread,
        thread_id: &str,
        turn_id: &str,
        step: &str,
    ) -> Result<Option<Result<T>>>
    where
        F: Future<Output = Result<T>> + Send,
    {
        tokio::select! {
            result = future => Ok(Some(result)),
            () = abort_signal.cancelled() => {
                let reason = "turn aborted by caller".to_string();
                tracing::info!(thread_id, turn_id, step, "turn aborted");
                self.abort_turn(thread, turn_id, reason)?;
                Ok(None)
            }
        }
    }

    fn emit_turn_token_usage(on_event: &mut TurnEventHandler<'_>, usage: TokenUsage) {
        if !usage.is_zero() {
            on_event(TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage {
                usage,
            }));
        }
    }

    async fn run_before_function_call_hooks(
        &self,
        context: FunctionCallHookContext,
        on_event: &mut TurnEventHandler<'_>,
    ) -> Result<()> {
        for hook in &self.function_call_hooks {
            hook.before_call(context.clone(), on_event).await?;
        }
        Ok(())
    }

    async fn run_after_function_call_hooks(
        &self,
        context: FunctionCallHookContext,
        result: FunctionCallHookResult,
        on_event: &mut TurnEventHandler<'_>,
    ) {
        for hook in &self.function_call_hooks {
            if let Err(error) = hook
                .after_call(context.clone(), result.clone(), on_event)
                .await
            {
                tracing::warn!(error = %error, "function call post-hook failed");
                on_event(TurnStreamEvent::Runtime(RuntimeEvent {
                    source: "function_call_hook".to_string(),
                    message: "post-hook failed".to_string(),
                    metadata: serde_json::json!({
                        "call_id": context.call_id,
                        "name": context.name,
                        "error": error.to_string(),
                    }),
                }));
            }
        }
    }

    async fn commit_thread(&self, thread: Thread) -> Result<Thread> {
        let expected_version = thread.version;
        self.store.commit(thread, expected_version).await
    }

    fn model_request_from_projection(
        &self,
        projection: ThreadProjection,
        runtime_context: Option<&str>,
    ) -> Result<ModelRequest> {
        Ok(ModelRequest {
            messages: self.context_builder.build(ContextBuildInput {
                projection: &projection,
                system_prompt: &self.config.system_prompt,
                runtime_context,
            })?,
            functions: self.function_registry.specs(),
        })
    }

    fn apply_thread_update(thread: &mut Thread, update: Option<ThreadUpdate>) -> Option<TurnItem> {
        match update {
            Some(ThreadUpdate::Goal(goal)) => {
                let previous = thread.goal.replace(goal.clone());
                Some(TurnItem::new(
                    TurnItemSource::Runtime,
                    TurnItemKind::GoalUpdated {
                        previous,
                        current: goal,
                    },
                ))
            }
            None => None,
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

    fn abort_turn(&self, thread: &mut Thread, turn_id: &str, reason: String) -> Result<()> {
        Self::push_turn_items(
            thread,
            turn_id,
            vec![TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::TurnAborted { reason },
            )],
        )?;
        Self::set_turn_status(thread, turn_id, TurnStatus::Aborted)
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
        turn_abort_pair, Agent, AgentConfig, AgentError, FunctionCallHook, FunctionCallHookContext,
        FunctionCallHookResult, Result, RuntimeContextInput, RuntimeContextProvider, RuntimeEvent,
        TurnOutcome, TurnStateEvent, TurnStreamEvent,
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

    struct PendingModel;

    impl ModelClient for PendingModel {
        fn stream_complete<'a>(
            &'a self,
            _request: ModelRequest,
            _on_event: &'a mut ModelStreamHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
            Box::pin(async move { std::future::pending::<Result<ModelResponse>>().await })
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

    struct RecordingHook {
        label: &'static str,
        events: Arc<Mutex<Vec<String>>>,
        fail_before: bool,
        fail_after: bool,
    }

    impl RecordingHook {
        fn new(label: &'static str, events: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                label,
                events,
                fail_before: false,
                fail_after: false,
            }
        }

        fn fail_before(mut self) -> Self {
            self.fail_before = true;
            self
        }

        fn fail_after(mut self) -> Self {
            self.fail_after = true;
            self
        }
    }

    impl FunctionCallHook for RecordingHook {
        fn before_call<'a>(
            &'a self,
            context: FunctionCallHookContext,
            emit: &'a mut super::TurnEventHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                self.events
                    .lock()
                    .expect("events")
                    .push(format!("before:{}:{}", self.label, context.name));
                emit(TurnStreamEvent::Runtime(RuntimeEvent {
                    source: self.label.to_string(),
                    message: "before".to_string(),
                    metadata: json!({ "name": context.name }),
                }));
                if self.fail_before {
                    return Err(AgentError::Function {
                        name: context.name,
                        message: format!("{} blocked call", self.label),
                    });
                }
                Ok(())
            })
        }

        fn after_call<'a>(
            &'a self,
            context: FunctionCallHookContext,
            result: FunctionCallHookResult,
            _emit: &'a mut super::TurnEventHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                let result_label = match result {
                    FunctionCallHookResult::Completed { .. } => "completed",
                    FunctionCallHookResult::Suspended { .. } => "waiting",
                    FunctionCallHookResult::Failed { .. } => "failed",
                };
                self.events.lock().expect("events").push(format!(
                    "after:{}:{}:{result_label}",
                    self.label, context.name
                ));
                if self.fail_after {
                    return Err(AgentError::Function {
                        name: context.name,
                        message: format!("{} post-hook failed", self.label),
                    });
                }
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn simple_assistant_message_ends_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
    async fn persists_assistant_text_when_response_also_requests_tools() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::Assistant {
                    text: Some("I will check the goal first.".to_string()),
                    function_calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "get_goal".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::Assistant {
                    text: Some("The goal is not set.".to_string()),
                    function_calls: Vec::new(),
                },
            ],
        );

        agent.run_turn("t", "check").await.expect("turn");
        let thread = store.load("t").await.expect("thread");
        let messages = thread.turns[0]
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                TurnItemKind::ModelMessage { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            messages,
            vec!["I will check the goal first.", "The goal is not set."]
        );
    }

    #[tokio::test]
    async fn concurrent_turns_for_same_thread_are_serialized() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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

    #[tokio::test]
    async fn abort_during_model_request_persists_aborted_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
        let agent = Arc::new(Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(PendingModel),
            builtin_registry(),
        ));
        let (abort_handle, abort_signal) = turn_abort_pair();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let started_tx = Arc::new(Mutex::new(Some(started_tx)));
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = events.clone();
        let captured_started = started_tx.clone();
        let turn_agent = agent.clone();

        let turn = tokio::spawn(async move {
            turn_agent
                .run_turn_stream_abortable("t", "slow", abort_signal, move |event| {
                    if matches!(
                        event,
                        TurnStreamEvent::State(TurnStateEvent::TurnStarted { .. })
                    ) {
                        if let Some(sender) = captured_started.lock().expect("started").take() {
                            let _ = sender.send(());
                        }
                    }
                    captured_events.lock().expect("events").push(event);
                })
                .await
        });

        started_rx.await.expect("turn started");
        abort_handle.abort();
        let outcome = turn.await.expect("join").expect("turn");

        assert!(matches!(outcome, TurnOutcome::Aborted { .. }));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].status, TurnStatus::Aborted);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::TurnAborted { .. })));
        assert!(events.lock().expect("events").iter().any(|event| matches!(
            event,
            TurnStreamEvent::State(TurnStateEvent::TurnAborted { .. })
        )));
    }

    #[test]
    fn model_request_includes_current_time_context() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
        let agent = agent_with(
            store,
            vec![ModelResponse::AssistantMessage {
                text: "hello".to_string(),
            }],
        );

        let request = agent
            .model_request_from_projection(Default::default(), None)
            .expect("context");
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
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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

        let system = {
            let requests = model.requests.lock().expect("requests");
            requests[0]
                .messages
                .iter()
                .find_map(|message| match message {
                    crate::projection::ChatMessage::System { content } => Some(content),
                    _ => None,
                })
                .expect("system message")
                .clone()
        };
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
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
        assert!(events.iter().any(|event| matches!(
            event,
            TurnStreamEvent::State(TurnStateEvent::FunctionStarted { .. })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            TurnStreamEvent::State(TurnStateEvent::TurnFinished { .. })
        )));
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
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
        let usage = {
            let events = events.lock().expect("lock");
            events
                .iter()
                .find_map(|event| match event {
                    TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage { usage }) => {
                        Some(*usage)
                    }
                    _ => None,
                })
                .expect("usage event")
        };
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
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
        assert!(matches!(outcome, TurnOutcome::Suspended { .. }));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Suspended);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::SuspensionCreated { .. })));
    }

    #[tokio::test]
    async fn function_failure_becomes_tool_error() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
    async fn function_call_hooks_are_stacked_in_order() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
        let hook_events = Arc::new(Mutex::new(Vec::new()));
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
        )
        .with_function_call_hook(RecordingHook::new("a", hook_events.clone()))
        .with_function_call_hook(RecordingHook::new("b", hook_events.clone()));

        let outcome = agent.run_turn("t", "goal?").await.expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));

        assert_eq!(
            *hook_events.lock().expect("events"),
            vec![
                "before:a:get_goal",
                "before:b:get_goal",
                "after:a:get_goal:completed",
                "after:b:get_goal:completed",
            ]
        );
    }

    #[tokio::test]
    async fn pre_hook_failure_blocks_function_and_records_tool_error() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
        let hook_events = Arc::new(Mutex::new(Vec::new()));
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "get_goal".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "blocked".to_string(),
                },
            ],
        )
        .with_function_call_hook(RecordingHook::new("policy", hook_events.clone()).fail_before());

        let outcome = agent.run_turn("t", "goal?").await.expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        assert_eq!(
            *hook_events.lock().expect("events"),
            vec!["before:policy:get_goal", "after:policy:get_goal:failed"]
        );

        let thread = store.load("t").await.expect("thread");
        let tool_outputs = thread.turns[0]
            .items
            .iter()
            .filter(|item| matches!(item.kind, TurnItemKind::ToolOutput { .. }))
            .count();
        assert_eq!(tool_outputs, 1);
        assert!(thread.turns[0].items.iter().any(|item| matches!(
            &item.kind,
            TurnItemKind::ToolOutput {
                call_id,
                name,
                result: ToolResult::Error { error },
            } if call_id == "c1"
                && name == "get_goal"
                && error.contains("policy blocked call")
        )));
    }

    #[tokio::test]
    async fn post_hook_failure_is_non_blocking_runtime_event() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
        let hook_events = Arc::new(Mutex::new(Vec::new()));
        let agent = agent_with(
            store.clone(),
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
        )
        .with_function_call_hook(RecordingHook::new("audit", hook_events.clone()).fail_after());
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();

        let outcome = agent
            .run_turn_stream("t", "goal?", move |event| {
                captured.lock().expect("events").push(event);
            })
            .await
            .expect("turn");

        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "done".to_string()
            }
        );
        assert_eq!(
            *hook_events.lock().expect("events"),
            vec!["before:audit:get_goal", "after:audit:get_goal:completed"]
        );
        assert!(events.lock().expect("events").iter().any(|event| matches!(
            event,
            TurnStreamEvent::Runtime(RuntimeEvent { source, message, metadata })
                if source == "function_call_hook"
                    && message == "post-hook failed"
                    && metadata["name"] == "get_goal"
        )));

        let thread = store.load("t").await.expect("thread");
        assert!(thread.turns[0].items.iter().any(|item| matches!(
            &item.kind,
            TurnItemKind::ToolOutput {
                result: ToolResult::Success { .. },
                ..
            }
        )));
    }

    #[tokio::test]
    async fn max_iterations_fails_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(JsonFileThreadStore::open(temp.path()).expect("store"));
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
