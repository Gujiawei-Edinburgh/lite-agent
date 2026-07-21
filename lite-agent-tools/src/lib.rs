//! Opt-in tools maintained alongside the lite-agent runtime.
//!
//! These tools observe or operate on the host environment and return ordinary
//! tool outputs. They do not mutate thread or runtime state.

use chrono::{Local, Utc};
use lite_agent_runtime::{
    AgentFunction, FunctionContext, FunctionExecution, FunctionRegistry, FunctionSpec, Result,
};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

pub mod sandbox;

#[derive(Debug, Clone, Copy, Default)]
pub struct GetCurrentTime;

impl AgentFunction for GetCurrentTime {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "get_current_time".to_string(),
            description: "Return the current UTC time and local time with its numeric offset. Use this for time-sensitive questions.".to_string(),
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
        _context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            let utc = Utc::now();
            let local = Local::now();
            Ok(FunctionExecution::Completed {
                output: json!({
                    "utc": utc.to_rfc3339(),
                    "local": local.to_rfc3339(),
                    "unix_seconds": utc.timestamp(),
                }),
            })
        })
    }
}

pub fn register_time_tools(registry: &mut FunctionRegistry) {
    registry.register(GetCurrentTime);
}

#[cfg(test)]
mod tests {
    use super::{register_time_tools, GetCurrentTime};
    use lite_agent_kernel::projection::ThreadProjection;
    use lite_agent_runtime::{
        turn_abort_pair, AgentFunction, FunctionCallExecution, FunctionContext, FunctionRegistry,
    };
    use serde_json::json;

    #[test]
    fn exposes_current_time_as_an_ordinary_tool() {
        let spec = GetCurrentTime.spec();
        assert_eq!(spec.name, "get_current_time");
        assert_eq!(spec.parameters["additionalProperties"], false);
    }

    #[tokio::test]
    async fn returns_a_factual_tool_output_without_runtime_effects() {
        let mut registry = FunctionRegistry::new();
        register_time_tools(&mut registry);
        let (_abort_handle, abort_signal) = turn_abort_pair();
        let execution = registry
            .call(
                "get_current_time",
                json!({}),
                FunctionContext {
                    thread_id: "thread".to_string(),
                    turn_id: "turn".to_string(),
                    call_id: "call".to_string(),
                    projection: ThreadProjection::default(),
                    abort_signal,
                },
            )
            .await
            .expect("tool call");

        let FunctionCallExecution::Completed { output, effects } = execution else {
            panic!("expected completed tool call");
        };
        assert!(output["utc"].is_string());
        assert!(output["local"].is_string());
        assert!(output["unix_seconds"].is_number());
        assert!(effects.is_empty());
    }
}
