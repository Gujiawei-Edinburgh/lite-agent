use crate::error::{AgentError, Result};
use crate::events::{new_id, GoalState, GoalStatus, TurnItem, TurnItemKind, TurnItemSource};
use crate::model::FunctionSpec;
use crate::projection::ThreadProjection;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FunctionContext {
    pub projection: ThreadProjection,
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
        extra_items: Vec<TurnItem>,
    },
    WaitingForUser {
        request_id: String,
        prompt: String,
        output: Value,
        thread_update: Option<ThreadUpdate>,
        extra_items: Vec<TurnItem>,
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
                extra_items: Vec::new(),
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
        context: FunctionContext,
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
            let item = TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::GoalUpdated {
                    previous: context.projection.goal,
                    current: current.clone(),
                },
            );

            Ok(FunctionExecution::Completed {
                output: json!({ "goal": current }),
                thread_update: Some(ThreadUpdate::Goal(current)),
                extra_items: vec![item],
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
            let item = TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::UserInputRequested {
                    request_id: request_id.clone(),
                    prompt: parsed.prompt.clone(),
                },
            );

            Ok(FunctionExecution::WaitingForUser {
                request_id: request_id.clone(),
                prompt: parsed.prompt.clone(),
                output: json!({
                    "request_id": request_id,
                    "prompt": parsed.prompt,
                    "status": "waiting_for_user"
                }),
                thread_update: None,
                extra_items: vec![item],
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{GoalStatus, Thread, TurnItemKind};
    use crate::projection::ThreadProjection;

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
                    projection: ThreadProjection::from_thread(&Thread::new("t")),
                },
            )
            .await
            .expect("call");

        let FunctionExecution::Completed {
            thread_update,
            extra_items,
            ..
        } = execution
        else {
            panic!("expected completion");
        };
        assert!(matches!(
            thread_update,
            Some(ThreadUpdate::Goal(goal)) if goal.status == GoalStatus::Active
        ));
        assert!(matches!(
            extra_items[0].kind,
            TurnItemKind::GoalUpdated { .. }
        ));
    }

    #[tokio::test]
    async fn ask_user_returns_waiting_state() {
        let registry = builtin_registry();
        let execution = registry
            .call(
                "ask_user",
                json!({ "prompt": "Which thread?" }),
                FunctionContext {
                    projection: ThreadProjection::from_thread(&Thread::new("t")),
                },
            )
            .await
            .expect("call");

        let FunctionExecution::WaitingForUser {
            prompt,
            extra_items,
            ..
        } = execution
        else {
            panic!("expected waiting");
        };
        assert_eq!(prompt, "Which thread?");
        assert!(matches!(
            extra_items[0].kind,
            TurnItemKind::UserInputRequested { .. }
        ));
    }
}
