use crate::error::{AgentError, Result};
use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

pub struct LoggingGuard {
    _guard: WorkerGuard,
}

pub fn init_file_logging(state_dir: impl Into<PathBuf>) -> Result<LoggingGuard> {
    let file_appender = tracing_appender::rolling::never(state_dir.into(), "lite-agent.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::TRACE.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .json()
        .with_ansi(false)
        .with_env_filter(filter)
        .with_writer(non_blocking)
        .try_init()
        .map_err(|error| AgentError::Logging(error.to_string()))?;

    Ok(LoggingGuard { _guard: guard })
}
