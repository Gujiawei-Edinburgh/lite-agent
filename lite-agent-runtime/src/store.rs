use lite_agent_kernel::Thread;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

pub type SessionLock = Arc<AsyncMutex<()>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ThreadContextCache {
    pub thread_id: String,
    pub source_version: u64,
    pub policy_version: String,
    pub covered_message_count: usize,
    pub summary: String,
}

pub trait ThreadStore: Send + Sync {
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::Result<Thread>> + Send + 'a>>;

    fn commit<'a>(
        &'a self,
        thread: Thread,
        expected_version: u64,
    ) -> Pin<Box<dyn Future<Output = crate::Result<Thread>> + Send + 'a>>;

    fn session_lock(&self, thread_id: &str) -> SessionLock;

    fn load_context_cache<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::Result<Option<ThreadContextCache>>> + Send + 'a>>;

    fn save_context_cache<'a>(
        &'a self,
        cache: ThreadContextCache,
    ) -> Pin<Box<dyn Future<Output = crate::Result<ThreadContextCache>> + Send + 'a>>;
}
