use lite_agent_kernel::{ModelFunctionCall, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceEvent {
    pub thread_id: String,
    pub turn_id: String,
    pub sequence: u64,
    pub occurred_at: String,
    pub kind: TraceEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceEventKind {
    UserInput {
        text: String,
        response_to: Option<String>,
    },
    ModelResponse {
        text: Option<String>,
        function_calls: Vec<ModelFunctionCall>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: Value,
    },
    ToolOutput {
        call_id: String,
        name: String,
        result: ToolResult,
    },
    TurnFinished {
        status: TraceTurnStatus,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TraceTurnStatus {
    Completed,
    Suspended,
    Failed,
    Aborted,
}

pub trait TraceCollector: Send + Sync {
    fn record(&self, event: TraceEvent);

    fn flush<'a>(&'a self) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async {})
    }
}

#[derive(Debug, Default)]
pub struct NoopTraceCollector;

impl TraceCollector for NoopTraceCollector {
    fn record(&self, _event: TraceEvent) {}
}
