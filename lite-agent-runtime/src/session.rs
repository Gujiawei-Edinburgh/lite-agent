use crate::{AgentError, Result};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// An opaque fencing token issued for one leased session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LeaseFence(Vec<u8>);

impl LeaseFence {
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// The ownership handle for one active thread session.
pub struct SessionLease {
    fence: LeaseFence,
    _guard: Box<dyn Send + Sync>,
}

impl SessionLease {
    pub fn new(fence: LeaseFence, guard: impl Send + Sync + 'static) -> Self {
        Self {
            fence,
            _guard: Box::new(guard),
        }
    }

    pub fn fence(&self) -> &LeaseFence {
        &self.fence
    }
}

pub trait SessionCoordinator: Send + Sync {
    fn acquire<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SessionLease>> + Send + 'a>>;
}

/// In-process coordinator intended for tests and single-process local adapters.
#[derive(Default)]
pub struct LocalSessionCoordinator {
    slots: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
    next_fence: AtomicU64,
}

impl LocalSessionCoordinator {
    fn slot(&self, thread_id: &str) -> Result<Arc<AsyncMutex<()>>> {
        self.slots
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("local coordinator poisoned".to_string()))
            .map(|mut slots| {
                slots
                    .entry(thread_id.to_string())
                    .or_insert_with(|| Arc::new(AsyncMutex::new(())))
                    .clone()
            })
    }
}

impl SessionCoordinator for LocalSessionCoordinator {
    fn acquire<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SessionLease>> + Send + 'a>> {
        Box::pin(async move {
            let slot = self.slot(thread_id)?;
            let guard: OwnedMutexGuard<()> = slot.lock_owned().await;
            let fence = LeaseFence::from_bytes(
                self.next_fence
                    .fetch_add(1, Ordering::Relaxed)
                    .saturating_add(1)
                    .to_be_bytes(),
            );
            Ok(SessionLease::new(fence, guard))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{LocalSessionCoordinator, SessionCoordinator};
    use std::sync::Arc;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn local_coordinator_serializes_same_thread() {
        let coordinator = Arc::new(LocalSessionCoordinator::default());
        let first = coordinator.acquire("thread").await.expect("first lease");
        let (started, wait_started) = oneshot::channel();
        let coordinator_for_task = coordinator.clone();
        let second = tokio::spawn(async move {
            let _ = coordinator_for_task
                .acquire("thread")
                .await
                .expect("second lease");
            let _ = started.send(());
        });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), wait_started)
                .await
                .is_err()
        );
        drop(first);
        tokio::time::timeout(std::time::Duration::from_secs(1), second)
            .await
            .expect("second lease acquired")
            .expect("task completed");
    }

    #[tokio::test]
    async fn leases_receive_distinct_fences() {
        let coordinator = LocalSessionCoordinator::default();
        let first = coordinator.acquire("one").await.expect("first lease");
        drop(first);
        let second = coordinator.acquire("one").await.expect("second lease");
        assert_ne!(second.fence().as_bytes(), &[0; 8]);
    }
}
