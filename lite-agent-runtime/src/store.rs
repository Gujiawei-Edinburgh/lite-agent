use crate::session::LeaseFence;
use lite_agent_kernel::{RevisionToken, Thread};
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ThreadContextCache {
    pub thread_id: String,
    pub source_revision: RevisionToken,
    pub policy_version: String,
    pub covered_message_count: usize,
    pub summary: String,
}

pub trait ThreadStore: Send + Sync {
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::Result<Thread>> + Send + 'a>>;

    /// Commits `thread` only when its embedded revision still matches storage.
    /// The lease fence must also be validated by stores that persist lease ownership.
    fn compare_and_commit<'a>(
        &'a self,
        thread: Thread,
        lease_fence: &'a LeaseFence,
    ) -> Pin<Box<dyn Future<Output = crate::Result<Thread>> + Send + 'a>>;

    fn load_context_cache<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::Result<Option<ThreadContextCache>>> + Send + 'a>>;

    fn save_context_cache<'a>(
        &'a self,
        cache: ThreadContextCache,
    ) -> Pin<Box<dyn Future<Output = crate::Result<ThreadContextCache>> + Send + 'a>>;
}
