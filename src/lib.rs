pub mod agent_loop;
pub mod error;
pub mod events;
pub mod functions;
pub mod model;
pub mod projection;
pub mod store;

pub use agent_loop::{Agent, AgentConfig, TurnOutcome};
pub use error::{AgentError, Result};
pub use events::{EventId, Thread, ThreadEvent, ThreadId};
pub use functions::{builtin_registry, FunctionRegistry};
pub use model::{ChatCompletionsClient, ModelClient, ModelConfig};
pub use projection::{GoalState, ThreadProjection};
pub use store::{JsonFileThreadStore, ThreadStore};
