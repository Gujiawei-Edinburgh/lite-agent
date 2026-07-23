use crate::context::{CompactingContextBuilder, ContextBuildInput, ContextBuilder};
use crate::error::{AgentError, Result};
use crate::functions::{
    FunctionCallExecution, FunctionContext, FunctionRegistry, RuntimeEffect, SuspensionResolution,
};
use crate::model::{ModelClient, ModelRequest, ModelResponse, ModelStreamEvent};
use crate::session::{SessionCoordinator, SessionLease};
use crate::store::{ThreadContextCache, ThreadStore};
use crate::trace::{
    NoopTraceCollector, TraceCollector, TraceEvent, TraceEventKind, TraceTurnStatus,
};
use lite_agent_kernel::events::{
    Suspension, Thread, TokenUsage, ToolResult, Turn, TurnId, TurnItem, TurnItemKind,
    TurnItemSource, TurnStatus,
};
use lite_agent_kernel::projection::ThreadProjection;
use lite_agent_kernel::RevisionToken;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone)]
pub struct AgentConfig {
    pub max_model_iterations: usize,
    pub system_prompt: String,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("max_model_iterations", &self.max_model_iterations)
            .field("system_prompt", &self.system_prompt)
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
    pub async fn wait_cancelled(&mut self) {
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
    trace_collector: Arc<dyn TraceCollector>,
    session_coordinator: Arc<dyn SessionCoordinator>,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        store: Arc<dyn ThreadStore>,
        model_client: Arc<dyn ModelClient>,
        function_registry: FunctionRegistry,
        session_coordinator: Arc<dyn SessionCoordinator>,
    ) -> Self {
        Self {
            config,
            store,
            model_client,
            function_registry,
            function_call_hooks: Vec::new(),
            context_builder: Arc::new(CompactingContextBuilder::default()),
            trace_collector: Arc::new(NoopTraceCollector),
            session_coordinator,
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

    pub fn with_trace_collector<C>(mut self, trace_collector: C) -> Self
    where
        C: TraceCollector + 'static,
    {
        self.trace_collector = Arc::new(trace_collector);
        self
    }

    pub fn with_shared_trace_collector(mut self, trace_collector: Arc<dyn TraceCollector>) -> Self {
        self.trace_collector = trace_collector;
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
        abort_signal: TurnAbortSignal,
        on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let lease = self.session_coordinator.acquire(thread_id).await?;
        self.run_turn_stream_abortable_internal(
            thread_id,
            Some(user_text.into()),
            abort_signal,
            on_event,
            0,
            &lease,
        )
        .await
    }

    /// Resume the existing turn associated with a suspension.
    ///
    /// The caller must update any external authorization state before calling
    /// this method. The original function call is executed in the suspended
    /// turn; no user-input item or new turn is created.
    pub async fn resume_suspended_turn<'a, F>(
        &self,
        thread_id: &str,
        suspension_id: &str,
        resolution: SuspensionResolution,
        on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let (_abort_handle, mut abort_signal) = turn_abort_pair();
        let lease = self.session_coordinator.acquire(thread_id).await?;
        let mut on_event = on_event;
        let mut suspended_outcome = None;
        let should_continue = {
            let mut thread = self.store.load(thread_id).await?;
            let projection = ThreadProjection::from_thread(&thread);
            let pending = projection
                .pending_suspension
                .clone()
                .ok_or_else(|| AgentError::TurnNotFound(suspension_id.to_string()))?;
            if pending.suspension.id != suspension_id {
                return Err(AgentError::TurnNotFound(suspension_id.to_string()));
            }
            let deferred = pending
                .suspension
                .payload
                .get("deferred")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let SuspensionResolution::UserInput { text } = resolution.clone() {
                if deferred {
                    return Err(AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "a pre-execution suspension requires approval or denial"
                            .to_string(),
                    });
                }
                Self::push_turn_items(
                    &mut thread,
                    &pending.turn_id,
                    vec![TurnItem::new(
                        TurnItemSource::User,
                        TurnItemKind::UserInput {
                            text,
                            response_to: Some(suspension_id.to_string()),
                        },
                    )],
                )?;
                Self::set_turn_status(&mut thread, &pending.turn_id, TurnStatus::Running)?;
                self.commit_thread(thread, lease.fence()).await?;
                true
            } else {
                let call_id = pending
                    .suspension
                    .payload
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "suspension does not identify a resumable call".to_string(),
                    })?;
                let call = thread
                    .turns
                    .iter()
                    .find(|turn| turn.id == pending.turn_id)
                    .and_then(|turn| {
                        turn.items.iter().rev().find_map(|item| match &item.kind {
                            TurnItemKind::ModelResponse { function_calls, .. } => function_calls
                                .iter()
                                .find(|call| call.call_id == call_id)
                                .cloned(),
                            _ => None,
                        })
                    })
                    .ok_or_else(|| AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "suspended function call was not found".to_string(),
                    })?;
                if !deferred {
                    return Err(AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "approval or denial can only resume a pre-execution suspension"
                            .to_string(),
                    });
                }
                if let SuspensionResolution::Deny { reason } = resolution.clone() {
                    let error_text = format!("execution denied by user: {reason}");
                    Self::push_turn_items(
                        &mut thread,
                        &pending.turn_id,
                        vec![TurnItem::new(
                            TurnItemSource::Tool,
                            TurnItemKind::ToolOutput {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                result: ToolResult::Error {
                                    error: error_text.clone(),
                                },
                            },
                        )],
                    )?;
                    Self::set_turn_status(&mut thread, &pending.turn_id, TurnStatus::Running)?;
                    self.commit_thread(thread, lease.fence()).await?;
                    on_event(TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                        call_id: call.call_id,
                        name: call.name,
                        error: error_text,
                    }));
                    true
                } else if matches!(resolution, SuspensionResolution::Approve) {
                    on_event(TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                    }));
                    let context = FunctionContext {
                        thread_id: thread.id.clone(),
                        turn_id: pending.turn_id.clone(),
                        call_id: call.call_id.clone(),
                        projection: projection.clone(),
                        abort_signal: abort_signal.clone(),
                    };
                    let execution =
                        self.function_registry
                            .call(&call.name, call.arguments.clone(), context);
                    let execution = self
                        .await_step_or_abort(
                            execution,
                            &mut abort_signal,
                            &mut thread,
                            thread_id,
                            &pending.turn_id,
                            "resumed function call",
                        )
                        .await?;
                    let Some(execution) = execution else {
                        return Ok(TurnOutcome::Aborted {
                            reason: "turn aborted by caller".to_string(),
                        });
                    };
                    match execution {
                        Ok(FunctionCallExecution::Completed { output, effects }) => {
                            let mut items = Self::apply_runtime_effects(&thread, effects);
                            items.push(TurnItem::new(
                                TurnItemSource::Tool,
                                TurnItemKind::ToolOutput {
                                    call_id: call.call_id.clone(),
                                    name: call.name.clone(),
                                    result: ToolResult::Success {
                                        output: output.clone(),
                                    },
                                },
                            ));
                            Self::push_turn_items(&mut thread, &pending.turn_id, items)?;
                            Self::set_turn_status(
                                &mut thread,
                                &pending.turn_id,
                                TurnStatus::Running,
                            )?;
                            self.commit_thread(thread, lease.fence()).await?;
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionCompleted {
                                call_id: call.call_id,
                                name: call.name,
                            }));
                            true
                        }
                        Ok(FunctionCallExecution::SuspendedBeforeExecution {
                            suspension, ..
                        })
                        | Ok(FunctionCallExecution::SuspendedAfterExecution {
                            suspension, ..
                        }) => {
                            suspended_outcome = Some(TurnOutcome::Suspended { suspension });
                            false
                        }
                        Err(error) => {
                            let error_text = error.to_string();
                            Self::push_turn_items(
                                &mut thread,
                                &pending.turn_id,
                                vec![TurnItem::new(
                                    TurnItemSource::Tool,
                                    TurnItemKind::ToolOutput {
                                        call_id: call.call_id.clone(),
                                        name: call.name.clone(),
                                        result: ToolResult::Error {
                                            error: error_text.clone(),
                                        },
                                    },
                                )],
                            )?;
                            Self::set_turn_status(
                                &mut thread,
                                &pending.turn_id,
                                TurnStatus::Running,
                            )?;
                            self.commit_thread(thread, lease.fence()).await?;
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                                call_id: call.call_id,
                                name: call.name,
                                error: error_text,
                            }));
                            true
                        }
                    }
                } else {
                    return Err(AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "unsupported suspension resolution".to_string(),
                    });
                }
            }
        };
        if let Some(outcome) = suspended_outcome {
            return Ok(outcome);
        }
        if should_continue {
            return self
                .run_turn_stream_abortable_internal(
                    thread_id,
                    None,
                    abort_signal,
                    on_event,
                    1,
                    &lease,
                )
                .await;
        }
        Ok(TurnOutcome::Failed {
            error: "suspended turn could not be resumed".to_string(),
        })
    }

    async fn run_turn_stream_abortable_internal<'a, F>(
        &self,
        thread_id: &str,
        user_text: Option<String>,
        mut abort_signal: TurnAbortSignal,
        mut on_event: F,
        mut trace_sequence: u64,
        lease: &SessionLease,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        tracing::debug!(thread_id, "turn started with session lease");

        let mut thread = match self.store.load(thread_id).await {
            Ok(thread) => thread,
            Err(AgentError::ThreadNotFound(_)) => Thread::new(thread_id),
            Err(error) => return Err(error),
        };
        let (turn_id, trace_user_text) = if let Some(user_text) = user_text {
            if let Some(pending) = ThreadProjection::from_thread(&thread).pending_suspension {
                return Err(AgentError::SuspendedTurn {
                    thread_id: thread_id.to_string(),
                    suspension_id: pending.suspension.id,
                });
            }
            let mut turn = Turn::new();
            let turn_id = turn.id.clone();
            turn.push_item(TurnItem::new(
                TurnItemSource::User,
                TurnItemKind::UserInput {
                    text: user_text.clone(),
                    response_to: None,
                },
            ));
            thread.turns.push(turn);
            thread = self.commit_thread(thread, lease.fence()).await?;
            (turn_id, Some(user_text))
        } else {
            let projection = ThreadProjection::from_thread(&thread);
            if let Some(pending) = projection.pending_suspension {
                return Err(AgentError::SuspendedTurn {
                    thread_id: thread_id.to_string(),
                    suspension_id: pending.suspension.id,
                });
            }
            let turn_id = projection
                .active_turn_id
                .ok_or_else(|| AgentError::TurnNotFound(thread_id.to_string()))?;
            (turn_id, None)
        };
        tracing::info!(thread_id, turn_id, "turn started");
        if let Some(trace_user_text) = trace_user_text {
            self.record_trace(
                &mut trace_sequence,
                thread_id,
                &turn_id,
                TraceEventKind::UserInput {
                    text: trace_user_text,
                    response_to: None,
                },
            );
        }
        on_event(TurnStreamEvent::State(TurnStateEvent::TurnStarted {
            thread_id: thread_id.to_string(),
            turn_id: turn_id.clone(),
        }));
        let mut turn_token_usage = TokenUsage::default();

        let outcome = 'turn_loop: {
            for iteration in 0..self.config.max_model_iterations {
                let session = Session::from_thread(&thread, turn_id.clone());
                let cached_context = self.store.load_context_cache(thread_id).await?;
                let request = self
                    .model_request_from_projection(
                        thread_id,
                        &thread.revision,
                        session.projection.clone(),
                        cached_context.as_ref(),
                    )
                    .await?;
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
                        if text.is_none() && function_calls.is_empty() {
                            let error = "model returned neither assistant text nor function calls"
                                .to_string();
                            tracing::error!(error, "empty model response");
                            self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                            break 'turn_loop TurnOutcome::Failed { error };
                        }
                        if let Some(text) = &text {
                            on_event(TurnStreamEvent::Model(TurnModelEvent::AssistantMessage {
                                text: text.clone(),
                            }));
                        }
                        let trace_model_response = TraceEventKind::ModelResponse {
                            text: text.clone(),
                            function_calls: function_calls.clone(),
                        };
                        Self::push_turn_items(
                            &mut thread,
                            &session.active_turn_id,
                            vec![TurnItem::new(
                                TurnItemSource::Model,
                                TurnItemKind::ModelResponse {
                                    text: text.clone(),
                                    function_calls: function_calls.clone(),
                                },
                            )],
                        )?;
                        if function_calls.is_empty() {
                            let text = text.unwrap_or_default();
                            Self::set_turn_status(
                                &mut thread,
                                &session.active_turn_id,
                                TurnStatus::Completed,
                            )?;
                            thread = self.commit_thread(thread, lease.fence()).await?;
                            self.record_trace(
                                &mut trace_sequence,
                                thread_id,
                                &session.active_turn_id,
                                trace_model_response,
                            );
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
                        thread = self.commit_thread(thread, lease.fence()).await?;
                        self.record_trace(
                            &mut trace_sequence,
                            thread_id,
                            &session.active_turn_id,
                            trace_model_response,
                        );

                        for (call_index, call) in calls.iter().enumerate() {
                            let call_id = call.call_id.clone();
                            let name = call.name.clone();
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                                call_id: call_id.clone(),
                                name: name.clone(),
                            }));
                            self.record_trace(
                                &mut trace_sequence,
                                thread_id,
                                &turn_id,
                                TraceEventKind::FunctionCall {
                                    call_id: call_id.clone(),
                                    name: name.clone(),
                                    arguments: call.arguments.clone(),
                                },
                            );
                            let mut hook_context = FunctionCallHookContext {
                                thread_id: thread.id.clone(),
                                turn_id: turn_id.clone(),
                                call_id: call_id.clone(),
                                name: name.clone(),
                                arguments: call.arguments.clone(),
                                projection: ThreadProjection::from_thread(&thread),
                            };
                            let (entered_hooks, pre_hook_result) = self
                                .run_before_function_call_hooks(&hook_context, &mut on_event)
                                .await;
                            let execution = match pre_hook_result {
                                Ok(()) => {
                                    let context = FunctionContext {
                                        thread_id: thread.id.clone(),
                                        turn_id: turn_id.clone(),
                                        call_id: call_id.clone(),
                                        projection: hook_context.projection.clone(),
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
                                Ok(FunctionCallExecution::Completed { output, effects }) => {
                                    let update_items =
                                        Self::apply_runtime_effects(&thread, effects);
                                    let hook_result = FunctionCallHookResult::Completed {
                                        output: output.clone(),
                                    };
                                    let trace_tool_output = TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Success {
                                            output: output.clone(),
                                        },
                                    };
                                    let mut func_items = update_items;
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Tool,
                                        TurnItemKind::ToolOutput {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        trace_tool_output,
                                    );
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    self.run_after_function_call_hooks(
                                        &entered_hooks,
                                        &hook_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionCompleted { call_id, name },
                                    ));
                                }
                                Ok(FunctionCallExecution::SuspendedBeforeExecution {
                                    suspension,
                                    effects,
                                }) => {
                                    let mut func_items =
                                        Self::apply_runtime_effects(&thread, effects);
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Runtime,
                                        TurnItemKind::SuspensionCreated {
                                            suspension: suspension.clone(),
                                        },
                                    ));
                                    let skipped_calls = calls
                                        .iter()
                                        .skip(call_index + 1)
                                        .map(|skipped_call| {
                                            let error = "function not executed because a previous function suspended the turn";
                                            func_items.push(TurnItem::new(
                                                TurnItemSource::Tool,
                                                TurnItemKind::ToolOutput {
                                                    call_id: skipped_call.call_id.clone(),
                                                    name: skipped_call.name.clone(),
                                                    result: ToolResult::Error {
                                                        error: error.to_string(),
                                                    },
                                                },
                                            ));
                                            (
                                                skipped_call.call_id.clone(),
                                                skipped_call.name.clone(),
                                                error.to_string(),
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    Self::set_turn_status(
                                        &mut thread,
                                        &turn_id,
                                        TurnStatus::Suspended,
                                    )?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    for (skipped_call_id, skipped_name, error) in &skipped_calls {
                                        self.record_trace(
                                            &mut trace_sequence,
                                            thread_id,
                                            &turn_id,
                                            TraceEventKind::ToolOutput {
                                                call_id: skipped_call_id.clone(),
                                                name: skipped_name.clone(),
                                                result: ToolResult::Error {
                                                    error: error.clone(),
                                                },
                                            },
                                        );
                                    }
                                    on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                                        suspension: suspension.clone(),
                                    }));
                                    for (skipped_call_id, skipped_name, error) in skipped_calls {
                                        on_event(TurnStreamEvent::State(
                                            TurnStateEvent::FunctionFailed {
                                                call_id: skipped_call_id,
                                                name: skipped_name,
                                                error,
                                            },
                                        ));
                                    }
                                    break 'turn_loop TurnOutcome::Suspended { suspension };
                                }
                                Ok(FunctionCallExecution::SuspendedAfterExecution {
                                    suspension,
                                    output,
                                    effects,
                                }) => {
                                    let update_items =
                                        Self::apply_runtime_effects(&thread, effects);
                                    let hook_result = FunctionCallHookResult::Suspended {
                                        suspension: suspension.clone(),
                                        output: output.clone(),
                                    };
                                    let trace_tool_output = TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Success {
                                            output: output.clone(),
                                        },
                                    };
                                    let mut func_items = update_items;
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
                                    let skipped_calls = calls
                                        .iter()
                                        .skip(call_index + 1)
                                        .map(|skipped_call| {
                                            let error = "function not executed because a previous function suspended the turn";
                                            func_items.push(TurnItem::new(
                                                TurnItemSource::Tool,
                                                TurnItemKind::ToolOutput {
                                                    call_id: skipped_call.call_id.clone(),
                                                    name: skipped_call.name.clone(),
                                                    result: ToolResult::Error {
                                                        error: error.to_string(),
                                                    },
                                                },
                                            ));
                                            (
                                                skipped_call.call_id.clone(),
                                                skipped_call.name.clone(),
                                                error.to_string(),
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    Self::set_turn_status(
                                        &mut thread,
                                        &turn_id,
                                        TurnStatus::Suspended,
                                    )?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        trace_tool_output,
                                    );
                                    for (skipped_call_id, skipped_name, error) in &skipped_calls {
                                        self.record_trace(
                                            &mut trace_sequence,
                                            thread_id,
                                            &turn_id,
                                            TraceEventKind::ToolOutput {
                                                call_id: skipped_call_id.clone(),
                                                name: skipped_name.clone(),
                                                result: ToolResult::Error {
                                                    error: error.clone(),
                                                },
                                            },
                                        );
                                    }
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    self.run_after_function_call_hooks(
                                        &entered_hooks,
                                        &hook_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionCompleted {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                        },
                                    ));
                                    on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                                        suspension: suspension.clone(),
                                    }));
                                    for (skipped_call_id, skipped_name, error) in skipped_calls {
                                        on_event(TurnStreamEvent::State(
                                            TurnStateEvent::FunctionFailed {
                                                call_id: skipped_call_id,
                                                name: skipped_name,
                                                error,
                                            },
                                        ));
                                    }
                                    break 'turn_loop TurnOutcome::Suspended { suspension };
                                }
                                Err(error) => {
                                    let error_text = error.to_string();
                                    let hook_result = FunctionCallHookResult::Failed {
                                        error: error_text.clone(),
                                    };
                                    let trace_tool_output = TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Error {
                                            error: error_text.clone(),
                                        },
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
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        trace_tool_output,
                                    );
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    self.run_after_function_call_hooks(
                                        &entered_hooks,
                                        &hook_context,
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
        self.commit_thread(thread, lease.fence()).await?;
        let trace_status = match &outcome {
            TurnOutcome::AssistantMessage { .. } => TraceTurnStatus::Completed,
            TurnOutcome::Suspended { .. } => TraceTurnStatus::Suspended,
            TurnOutcome::Failed { .. } => TraceTurnStatus::Failed,
            TurnOutcome::Aborted { .. } => TraceTurnStatus::Aborted,
        };
        self.record_trace(
            &mut trace_sequence,
            thread_id,
            &turn_id,
            TraceEventKind::TurnFinished {
                status: trace_status,
            },
        );
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

    fn record_trace(
        &self,
        sequence: &mut u64,
        thread_id: &str,
        turn_id: &str,
        kind: TraceEventKind,
    ) {
        *sequence = sequence.saturating_add(1);
        self.trace_collector.record(TraceEvent {
            thread_id: thread_id.to_string(),
            turn_id: turn_id.to_string(),
            sequence: *sequence,
            occurred_at: lite_agent_kernel::now_timestamp(),
            kind,
        });
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
            () = abort_signal.wait_cancelled() => {
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
        context: &FunctionCallHookContext,
        on_event: &mut TurnEventHandler<'_>,
    ) -> (Vec<Arc<dyn FunctionCallHook>>, Result<()>) {
        let mut entered_hooks = Vec::new();
        for hook in &self.function_call_hooks {
            if let Err(error) = hook.before_call((*context).clone(), on_event).await {
                return (entered_hooks, Err(error));
            }
            entered_hooks.push(hook.clone());
        }
        (entered_hooks, Ok(()))
    }

    async fn run_after_function_call_hooks(
        &self,
        hooks: &[Arc<dyn FunctionCallHook>],
        context: &FunctionCallHookContext,
        result: FunctionCallHookResult,
        on_event: &mut TurnEventHandler<'_>,
    ) {
        for hook in hooks.iter().rev() {
            if let Err(error) = hook
                .after_call((*context).clone(), result.clone(), on_event)
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

    async fn commit_thread(
        &self,
        thread: Thread,
        lease_fence: &crate::session::LeaseFence,
    ) -> Result<Thread> {
        self.store.compare_and_commit(thread, lease_fence).await
    }

    async fn model_request_from_projection(
        &self,
        thread_id: &str,
        thread_revision: &RevisionToken,
        projection: ThreadProjection,
        cached_context: Option<&ThreadContextCache>,
    ) -> Result<ModelRequest> {
        let context = self
            .context_builder
            .build(ContextBuildInput {
                thread_id,
                thread_revision,
                projection: &projection,
                system_prompt: &self.config.system_prompt,
                cached_context,
            })
            .await?;
        if let Some(cache) = context.cache {
            self.store.save_context_cache(cache).await?;
        }
        Ok(ModelRequest {
            messages: context.messages,
            functions: self.function_registry.specs(),
        })
    }

    fn apply_runtime_effects(thread: &Thread, effects: Vec<RuntimeEffect>) -> Vec<TurnItem> {
        effects
            .into_iter()
            .map(|effect| match effect {
                RuntimeEffect::SetGoal(goal) => {
                    let previous = ThreadProjection::from_thread(thread).goal;
                    TurnItem::new(
                        TurnItemSource::Runtime,
                        TurnItemKind::GoalUpdated {
                            previous,
                            current: goal,
                        },
                    )
                }
            })
            .collect()
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
    use crate::functions::builtin_registry;
    use crate::model::{
        ModelClient, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent,
        ModelStreamHandler,
    };
    use crate::store::{ThreadContextCache, ThreadStore};
    use crate::{
        turn_abort_pair, Agent, AgentConfig, AgentError, FunctionCallHook, FunctionCallHookContext,
        FunctionCallHookResult, Result, RuntimeEvent, TurnOutcome, TurnStateEvent, TurnStreamEvent,
    };
    use lite_agent_kernel::events::{TokenUsage, ToolResult, TurnItemKind, TurnStatus};
    use lite_agent_kernel::projection::ThreadProjection;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct TestStore {
        threads: Mutex<std::collections::BTreeMap<String, lite_agent_kernel::events::Thread>>,
        caches: Mutex<std::collections::BTreeMap<String, ThreadContextCache>>,
    }

    impl ThreadStore for TestStore {
        fn load<'a>(
            &'a self,
            thread_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<lite_agent_kernel::events::Thread>> + Send + 'a>>
        {
            Box::pin(async move {
                self.threads
                    .lock()
                    .expect("threads")
                    .get(thread_id)
                    .cloned()
                    .ok_or_else(|| AgentError::ThreadNotFound(thread_id.to_string()))
            })
        }

        fn compare_and_commit<'a>(
            &'a self,
            mut thread: lite_agent_kernel::events::Thread,
            _lease_fence: &'a crate::session::LeaseFence,
        ) -> Pin<Box<dyn Future<Output = Result<lite_agent_kernel::events::Thread>> + Send + 'a>>
        {
            Box::pin(async move {
                let mut threads = self.threads.lock().expect("threads");
                let current_revision = threads
                    .get(&thread.id)
                    .map_or_else(lite_agent_kernel::RevisionToken::initial, |current| {
                        current.revision.clone()
                    });
                if current_revision != thread.revision {
                    return Err(AgentError::RevisionConflict {
                        thread_id: thread.id.clone(),
                        expected: thread.revision.clone(),
                        actual: current_revision,
                    });
                }
                let next = thread
                    .revision
                    .as_bytes()
                    .try_into()
                    .map(u64::from_be_bytes)
                    .unwrap_or(0)
                    .saturating_add(1);
                thread.revision = lite_agent_kernel::RevisionToken::from_u64(next);
                thread.touch();
                threads.insert(thread.id.clone(), thread.clone());
                Ok(thread)
            })
        }

        fn load_context_cache<'a>(
            &'a self,
            thread_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<Option<ThreadContextCache>>> + Send + 'a>> {
            Box::pin(async move { Ok(self.caches.lock().expect("caches").get(thread_id).cloned()) })
        }

        fn save_context_cache<'a>(
            &'a self,
            cache: ThreadContextCache,
        ) -> Pin<Box<dyn Future<Output = Result<ThreadContextCache>> + Send + 'a>> {
            Box::pin(async move {
                self.caches
                    .lock()
                    .expect("caches")
                    .insert(cache.thread_id.clone(), cache.clone());
                Ok(cache)
            })
        }
    }

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
            Arc::new(crate::session::LocalSessionCoordinator::default()),
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
        let store = Arc::new(TestStore::default());
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
    async fn empty_model_response_fails_turn() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::Assistant {
                text: None,
                function_calls: Vec::new(),
            }],
        );

        let outcome = agent.run_turn("t", "hello").await.expect("turn");
        assert!(
            matches!(outcome, TurnOutcome::Failed { error } if error.contains("neither assistant text nor function calls"))
        );
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Failed);
    }

    #[tokio::test]
    async fn persists_assistant_text_when_response_also_requests_tools() {
        let store = Arc::new(TestStore::default());
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
                TurnItemKind::ModelResponse { text, .. } => text.as_deref(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            messages,
            vec!["I will check the goal first.", "The goal is not set."]
        );
    }

    #[tokio::test]
    async fn abort_during_model_request_persists_aborted_turn() {
        let store = Arc::new(TestStore::default());
        let agent = Arc::new(Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(PendingModel),
            builtin_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
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

    #[tokio::test]
    async fn stream_emits_intermediate_and_final_events() {
        let store = Arc::new(TestStore::default());
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

        let store = Arc::new(TestStore::default());
        let agent = Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(UsageModel),
            builtin_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
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
        let store = Arc::new(TestStore::default());
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
        let projection = ThreadProjection::from_thread(&thread);
        assert_eq!(
            projection.goal.as_ref().map(|goal| goal.objective.as_str()),
            Some("ship")
        );
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::GoalUpdated { .. })));
    }

    #[tokio::test]
    async fn ask_user_stops_turn() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::FunctionCalls {
                calls: vec![
                    ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "ask_user".to_string(),
                        arguments: json!({ "prompt": "Which one?" }),
                    },
                    ModelFunctionCall {
                        call_id: "c2".to_string(),
                        name: "get_goal".to_string(),
                        arguments: json!({}),
                    },
                ],
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
        assert_eq!(
            thread.turns[0]
                .items
                .iter()
                .filter(|item| matches!(item.kind, TurnItemKind::ToolOutput { .. }))
                .count(),
            2
        );
        let error = agent.run_turn("t", "another prompt").await.unwrap_err();
        assert!(matches!(error, AgentError::SuspendedTurn { .. }));
    }

    #[tokio::test]
    async fn function_failure_becomes_tool_error() {
        let store = Arc::new(TestStore::default());
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
        let store = Arc::new(TestStore::default());
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
                "after:b:get_goal:completed",
                "after:a:get_goal:completed",
            ]
        );
    }

    #[tokio::test]
    async fn pre_hook_failure_blocks_function_and_records_tool_error() {
        let store = Arc::new(TestStore::default());
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
        .with_function_call_hook(RecordingHook::new("audit", hook_events.clone()))
        .with_function_call_hook(RecordingHook::new("policy", hook_events.clone()).fail_before());

        let outcome = agent.run_turn("t", "goal?").await.expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        assert_eq!(
            *hook_events.lock().expect("events"),
            vec![
                "before:audit:get_goal",
                "before:policy:get_goal",
                "after:audit:get_goal:failed",
            ]
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
        let store = Arc::new(TestStore::default());
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
        let store = Arc::new(TestStore::default());
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
            Arc::new(crate::session::LocalSessionCoordinator::default()),
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
