use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub type ThreadId = String;
pub type TurnId = String;
pub type TurnItemId = String;

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Thread {
    #[serde(default = "current_schema_version")]
    pub schema_version: u32,
    pub id: ThreadId,
    pub goal: Option<GoalState>,
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub token_usage: TokenUsage,
    pub created_at: String,
    pub updated_at: String,
}

impl Thread {
    pub fn new(id: impl Into<ThreadId>) -> Self {
        let now = now_timestamp();
        Self {
            schema_version: current_schema_version(),
            id: id.into(),
            goal: None,
            turns: Vec::new(),
            token_usage: TokenUsage::default(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = now_timestamp();
    }

    pub fn turn_mut(&mut self, turn_id: &str) -> Option<&mut Turn> {
        self.turns.iter_mut().find(|turn| turn.id == turn_id)
    }
}

fn current_schema_version() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsage {
    pub fn is_zero(&self) -> bool {
        self.input_tokens == 0
            && self.cached_input_tokens == 0
            && self.output_tokens == 0
            && self.total_tokens == 0
    }

    pub fn add_assign(&mut self, usage: TokenUsage) {
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(usage.cached_input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
    }
}

impl fmt::Display for TokenUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "input={}, cached_input={}, output={}, total={}",
            self.input_tokens, self.cached_input_tokens, self.output_tokens, self.total_tokens
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    pub id: TurnId,
    pub status: TurnStatus,
    pub items: Vec<TurnItem>,
    pub created_at: String,
    pub updated_at: String,
}

impl Turn {
    pub fn new() -> Self {
        let now = now_timestamp();
        Self {
            id: new_id("turn"),
            status: TurnStatus::Running,
            items: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn push_item(&mut self, item: TurnItem) {
        self.items.push(item);
        self.updated_at = now_timestamp();
    }

    pub fn set_status(&mut self, status: TurnStatus) {
        self.status = status;
        self.updated_at = now_timestamp();
    }
}

impl Default for Turn {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Running,
    WaitingForUser,
    Completed,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TurnItem {
    pub id: TurnItemId,
    pub created_at: String,
    pub source: TurnItemSource,
    #[serde(flatten)]
    pub kind: TurnItemKind,
}

impl TurnItem {
    pub fn new(source: TurnItemSource, kind: TurnItemKind) -> Self {
        Self {
            id: new_id("item"),
            created_at: now_timestamp(),
            source,
            kind,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnItemSource {
    User,
    Model,
    Tool,
    Runtime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnItemKind {
    UserInput {
        text: String,
        response_to: Option<String>,
    },
    ModelMessage {
        text: String,
    },
    ModelFunctionCall {
        call_id: String,
        name: String,
        arguments: Value,
    },
    ToolOutput {
        call_id: String,
        name: String,
        result: ToolResult,
    },
    UserInputRequested {
        request_id: String,
        prompt: String,
    },
    GoalUpdated {
        previous: Option<GoalState>,
        current: GoalState,
    },
    TurnFailed {
        error: String,
    },
    TurnAborted {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ToolResult {
    Success { output: Value },
    Error { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoalState {
    pub objective: String,
    pub status: GoalStatus,
    pub notes: Option<String>,
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

pub fn now_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("{seconds}")
}
