use crate::agent_loop::TurnAbortSignal;
use crate::error::{AgentError, Result};
use crate::model::FunctionSpec;
use lite_agent_kernel::events::{new_id, GoalState, GoalStatus, Suspension, SuspensionKind};
use lite_agent_kernel::projection::ThreadProjection;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FunctionContext {
    pub thread_id: String,
    pub turn_id: String,
    pub call_id: String,
    pub projection: ThreadProjection,
    pub abort_signal: TurnAbortSignal,
}

#[derive(Debug, Clone)]
pub enum ThreadUpdate {
    Goal(GoalState),
}

#[derive(Debug, Clone)]
pub enum FunctionExecution {
    Completed {
        output: Value,
        thread_update: Option<ThreadUpdate>,
    },
    Suspended {
        suspension: Suspension,
        output: Value,
        thread_update: Option<ThreadUpdate>,
    },
}

pub trait AgentFunction: Send + Sync {
    fn spec(&self) -> FunctionSpec;
    fn call<'a>(
        &'a self,
        args: Value,
        context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>>;
}

pub struct SimpleFunction<F> {
    spec: FunctionSpec,
    handler: F,
}

impl<F> SimpleFunction<F> {
    pub fn new(spec: FunctionSpec, handler: F) -> Self {
        Self { spec, handler }
    }
}

impl<F, Fut> AgentFunction for SimpleFunction<F>
where
    F: Fn(Value, FunctionContext) -> Fut + Send + Sync,
    Fut: Future<Output = Result<FunctionExecution>> + Send + 'static,
{
    fn spec(&self) -> FunctionSpec {
        self.spec.clone()
    }

    fn call<'a>(
        &'a self,
        args: Value,
        context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin((self.handler)(args, context))
    }
}

#[derive(Clone, Default)]
pub struct FunctionRegistry {
    functions: BTreeMap<String, Arc<dyn AgentFunction>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F>(&mut self, function: F)
    where
        F: AgentFunction + 'static,
    {
        self.functions
            .insert(function.spec().name.clone(), Arc::new(function));
    }

    pub fn specs(&self) -> Vec<FunctionSpec> {
        self.functions
            .values()
            .map(|function| function.spec())
            .collect()
    }

    pub async fn call(
        &self,
        name: &str,
        args: Value,
        context: FunctionContext,
    ) -> Result<FunctionExecution> {
        let function = self
            .functions
            .get(name)
            .ok_or_else(|| AgentError::FunctionNotFound(name.to_string()))?;
        function.call(args, context).await
    }
}

pub fn builtin_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    registry.register(GetGoal);
    registry.register(UpdateGoal);
    registry.register(AskUser);
    registry
}

struct GetGoal;

impl AgentFunction for GetGoal {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "get_goal".to_string(),
            description: "Return the current thread goal, if one has been set.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        }
    }

    fn call<'a>(
        &'a self,
        _args: Value,
        context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            Ok(FunctionExecution::Completed {
                output: json!({ "goal": context.projection.goal }),
                thread_update: None,
            })
        })
    }
}

struct UpdateGoal;

#[derive(Debug, Deserialize)]
struct UpdateGoalArgs {
    objective: String,
    status: GoalStatus,
    notes: Option<String>,
}

impl AgentFunction for UpdateGoal {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "update_goal".to_string(),
            description: "Set or update the explicit goal state for this thread.".to_string(),
            parameters: json!({
                "type": "object",
                "required": ["objective", "status"],
                "properties": {
                    "objective": { "type": "string" },
                    "status": {
                        "type": "string",
                        "enum": ["active", "complete", "blocked"]
                    },
                    "notes": { "type": "string" }
                },
                "additionalProperties": false
            }),
        }
    }

    fn call<'a>(
        &'a self,
        args: Value,
        _context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: UpdateGoalArgs = serde_json::from_value(args).map_err(|error| {
                AgentError::InvalidFunctionArguments {
                    name: "update_goal".to_string(),
                    message: error.to_string(),
                }
            })?;
            let current = GoalState {
                objective: parsed.objective,
                status: parsed.status,
                notes: parsed.notes,
            };
            Ok(FunctionExecution::Completed {
                output: json!({ "goal": current }),
                thread_update: Some(ThreadUpdate::Goal(current)),
            })
        })
    }
}

struct AskUser;

#[derive(Debug, Deserialize)]
struct AskUserArgs {
    prompt: String,
}

impl AgentFunction for AskUser {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "ask_user".to_string(),
            description: "Ask the user a follow-up question and stop the current turn.".to_string(),
            parameters: json!({
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": { "type": "string" }
                },
                "additionalProperties": false
            }),
        }
    }

    fn call<'a>(
        &'a self,
        args: Value,
        _context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            let parsed: AskUserArgs = serde_json::from_value(args).map_err(|error| {
                AgentError::InvalidFunctionArguments {
                    name: "ask_user".to_string(),
                    message: error.to_string(),
                }
            })?;
            let request_id = new_id("req");
            Ok(FunctionExecution::Suspended {
                suspension: Suspension {
                    id: request_id.clone(),
                    kind: SuspensionKind::UserInput,
                    payload: json!({ "prompt": parsed.prompt.clone() }),
                },
                output: json!({
                    "request_id": request_id,
                    "prompt": parsed.prompt,
                    "status": "suspended"
                }),
                thread_update: None,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use lite_agent_kernel::events::{GoalStatus, SuspensionKind, Thread};
    use lite_agent_kernel::projection::ThreadProjection;

    use super::{builtin_registry, FunctionContext, FunctionExecution, ThreadUpdate};
    use serde_json::json;

    #[tokio::test]
    async fn update_goal_returns_thread_update_and_item() {
        let registry = builtin_registry();
        let execution = registry
            .call(
                "update_goal",
                json!({ "objective": "ship v1", "status": "active" }),
                FunctionContext {
                    thread_id: "t".to_string(),
                    turn_id: "turn".to_string(),
                    call_id: "call".to_string(),
                    projection: ThreadProjection::from_thread(&Thread::new("t")),
                    abort_signal: crate::agent_loop::turn_abort_pair().1,
                },
            )
            .await
            .expect("call");

        let FunctionExecution::Completed { thread_update, .. } = execution else {
            panic!("expected completion");
        };
        assert!(matches!(
            thread_update,
            Some(ThreadUpdate::Goal(goal)) if goal.status == GoalStatus::Active
        ));
    }

    #[tokio::test]
    async fn ask_user_returns_suspended_state() {
        let registry = builtin_registry();
        let execution = registry
            .call(
                "ask_user",
                json!({ "prompt": "Which thread?" }),
                FunctionContext {
                    thread_id: "t".to_string(),
                    turn_id: "turn".to_string(),
                    call_id: "call".to_string(),
                    projection: ThreadProjection::from_thread(&Thread::new("t")),
                    abort_signal: crate::agent_loop::turn_abort_pair().1,
                },
            )
            .await
            .expect("call");

        let FunctionExecution::Suspended { suspension, .. } = execution else {
            panic!("expected waiting");
        };
        assert_eq!(suspension.kind, SuspensionKind::UserInput);
        assert_eq!(suspension.payload["prompt"], "Which thread?");
    }
}
