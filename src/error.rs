use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("thread not found: {0}")]
    ThreadNotFound(String),

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
}
