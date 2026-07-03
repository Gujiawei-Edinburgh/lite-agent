use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub type ThreadId = String;
pub type EventId = String;

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Thread {
    pub id: ThreadId,
    pub events: Vec<ThreadEvent>,
}

impl Thread {
    pub fn new(id: impl Into<ThreadId>) -> Self {
        Self {
            id: id.into(),
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThreadEvent {
    pub id: EventId,
    pub created_at: String,
    #[serde(flatten)]
    pub kind: ThreadEventKind,
}

impl ThreadEvent {
    pub fn new(kind: ThreadEventKind) -> Self {
        Self {
            id: new_id("evt"),
            created_at: now_timestamp(),
            kind,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThreadEventKind {
    UserInputReceived {
        text: String,
        response_to: Option<String>,
    },
    AssistantMessageEmitted {
        text: String,
    },
    FunctionCallRequested {
        call_id: String,
        name: String,
        arguments: Value,
    },
    FunctionCallCompleted {
        call_id: String,
        name: String,
        output: Value,
    },
    FunctionCallFailed {
        call_id: String,
        name: String,
        error: String,
    },
    GoalUpdated {
        objective: String,
        status: GoalStatus,
        notes: Option<String>,
    },
    UserInputRequested {
        request_id: String,
        prompt: String,
    },
    TurnFailed {
        error: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    Active,
    Complete,
    Blocked,
}

pub fn new_id(prefix: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{millis}_{counter}")
}

fn now_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("{seconds}")
}
