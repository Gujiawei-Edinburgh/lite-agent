use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("thread not found: {0}")]
    ThreadNotFound(String),

    #[error("invalid thread id: {0}")]
    InvalidThreadId(String),

    #[error("thread store directory is already owned: {0}")]
    StoreLocked(String),

    #[error("thread version conflict for {thread_id}: expected {expected}, actual {actual}")]
    VersionConflict {
        thread_id: String,
        expected: u64,
        actual: u64,
    },

    #[error("context window exceeded: estimated {estimated} tokens, limit {limit}")]
    ContextWindowExceeded { estimated: usize, limit: usize },

    #[error("context compactor exceeded its budget: estimated {estimated} tokens, limit {limit}")]
    ContextCompactorContractViolation { estimated: usize, limit: usize },

    #[error("turn not found: {0}")]
    TurnNotFound(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("http error: {0}")]
    Http(String),

    #[error("http client error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("model error: {0}")]
    Model(String),

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("function error in {name}: {message}")]
    Function { name: String, message: String },

    #[error("invalid function arguments for {name}: {message}")]
    InvalidFunctionArguments { name: String, message: String },

    #[error("turn exceeded max model iterations: {0}")]
    MaxIterations(usize),

    #[error("logging error: {0}")]
    Logging(String),
}
