use crate::error::{AgentError, Result};
use crate::events::Thread;
use fs2::FileExt;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex as AsyncMutex;

pub type SessionLock = Arc<AsyncMutex<()>>;

pub trait ThreadStore: Send + Sync {
    /// Load one durable thread snapshot.
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn commit<'a>(
        &'a self,
        thread: Thread,
        expected_version: u64,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn session_lock(&self, thread_id: &str) -> SessionLock;
}

/// Single-process JSON persistence intended for local development and examples.
/// Production deployments should provide a repository with atomic/versioned commits.
#[derive(Debug, Clone)]
pub struct JsonFileThreadStore {
    state_dir: PathBuf,
    _lock: Arc<StoreLock>,
    session_locks: Arc<std::sync::Mutex<BTreeMap<String, SessionLock>>>,
    commit_lock: Arc<AsyncMutex<()>>,
}

#[derive(Debug)]
struct StoreLock {
    file: File,
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

impl JsonFileThreadStore {
    /// Opens and exclusively owns a state directory until the returned store is dropped.
    ///
    /// Share the returned store with multiple agents using `Arc`. A second process, or a
    /// second independently opened store, cannot use the same directory concurrently.
    pub fn open(state_dir: impl Into<PathBuf>) -> Result<Self> {
        let state_dir = state_dir.into();
        std::fs::create_dir_all(&state_dir)?;
        let lock_path = state_dir.join(".store.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        if let Err(error) = file.try_lock_exclusive() {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                return Err(AgentError::StoreLocked(state_dir.display().to_string()));
            }
            return Err(error.into());
        }
        Ok(Self {
            state_dir,
            _lock: Arc::new(StoreLock { file }),
            session_locks: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            commit_lock: Arc::new(AsyncMutex::new(())),
        })
    }

    fn thread_path(&self, thread_id: &str) -> PathBuf {
        self.state_dir
            .join("threads")
            .join(format!("{thread_id}.json"))
    }

    fn validate_thread_id(thread_id: &str) -> Result<()> {
        if thread_id.is_empty()
            || thread_id == "."
            || thread_id == ".."
            || thread_id.contains('/')
            || thread_id.contains('\\')
        {
            return Err(AgentError::InvalidThreadId(thread_id.to_string()));
        }
        Ok(())
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    async fn write_thread(&self, thread: &Thread) -> Result<()> {
        let path = self.thread_path(&thread.id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let raw = serde_json::to_string_pretty(thread)?;
        let temporary_path =
            path.with_extension(format!("json.{}.tmp", crate::events::new_id("write")));
        fs::write(&temporary_path, raw).await?;
        fs::rename(temporary_path, path).await?;
        Ok(())
    }

    async fn commit_thread(&self, mut thread: Thread, expected_version: u64) -> Result<Thread> {
        let _commit_guard = self.commit_lock.lock().await;
        Self::validate_thread_id(&thread.id)?;
        let current = match self.load(&thread.id).await {
            Ok(current) => current,
            Err(AgentError::ThreadNotFound(_)) => Thread::new(thread.id.clone()),
            Err(error) => return Err(error),
        };
        if current.version != expected_version {
            return Err(AgentError::VersionConflict {
                thread_id: thread.id.clone(),
                expected: expected_version,
                actual: current.version,
            });
        }
        thread.version = expected_version.saturating_add(1);
        thread.touch();
        self.write_thread(&thread).await?;
        Ok(thread)
    }
}

impl ThreadStore for JsonFileThreadStore {
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            Self::validate_thread_id(thread_id)?;
            let path = self.thread_path(thread_id);
            match fs::read_to_string(path).await {
                Ok(raw) => Ok(serde_json::from_str(&raw)?),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    Err(AgentError::ThreadNotFound(thread_id.to_string()))
                }
                Err(error) => Err(error.into()),
            }
        })
    }

    fn commit<'a>(
        &'a self,
        thread: Thread,
        expected_version: u64,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move { self.commit_thread(thread, expected_version).await })
    }

    fn session_lock(&self, thread_id: &str) -> SessionLock {
        let mut locks = self.session_locks.lock().expect("session lock registry");
        locks
            .entry(thread_id.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{Thread, Turn, TurnItem, TurnItemKind, TurnItemSource, TurnStatus};

    use super::{JsonFileThreadStore, ThreadStore};

    #[tokio::test]
    async fn round_trips_thread_with_turn_items() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = JsonFileThreadStore::open(temp.path()).expect("store");
        let mut turn = Turn::new();
        let turn_id = turn.id.clone();
        turn.push_item(TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelMessage {
                text: "hello".to_string(),
            },
        ));

        store
            .commit(
                Thread {
                    turns: vec![turn],
                    ..Thread::new("t1")
                },
                0,
            )
            .await
            .expect("commit");

        let thread = store.load("t1").await.expect("load");
        assert_eq!(thread.id, "t1");
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].id, turn_id);
        assert_eq!(thread.turns[0].items.len(), 1);
    }

    #[tokio::test]
    async fn commits_thread_snapshot() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = JsonFileThreadStore::open(temp.path()).expect("store");
        let mut turn = Turn::new();
        turn.push_item(TurnItem::new(
            TurnItemSource::User,
            TurnItemKind::UserInput {
                text: "hi".to_string(),
                response_to: None,
            },
        ));
        turn.set_status(TurnStatus::Completed);
        let thread = store
            .commit(
                Thread {
                    turns: vec![turn],
                    ..Thread::new("t1")
                },
                0,
            )
            .await
            .expect("commit");

        assert_eq!(thread.turns[0].status, TurnStatus::Completed);
        assert_eq!(thread.turns[0].items.len(), 1);
    }

    #[tokio::test]
    async fn rejects_path_traversal_thread_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = JsonFileThreadStore::open(temp.path()).expect("store");
        let error = store.load("../outside").await.expect_err("invalid id");
        assert!(matches!(
            error,
            crate::AgentError::InvalidThreadId(value) if value == "../outside"
        ));
    }

    #[test]
    fn only_one_store_can_own_a_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let first = JsonFileThreadStore::open(temp.path()).expect("first store");
        let error = JsonFileThreadStore::open(temp.path()).expect_err("second store");
        assert!(matches!(error, crate::AgentError::StoreLocked(_)));
        drop(first);
        JsonFileThreadStore::open(temp.path()).expect("lock released after drop");
    }

    #[tokio::test]
    async fn rejects_stale_thread_version() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = JsonFileThreadStore::open(temp.path()).expect("store");
        let created = store.commit(Thread::new("t1"), 0).await.expect("create");
        assert_eq!(created.version, 1);

        let stale = store.load("t1").await.expect("load");
        let mut current = stale.clone();
        current.goal = Some(crate::events::GoalState {
            objective: "first".to_string(),
            status: crate::events::GoalStatus::Active,
            notes: None,
        });
        let committed = store.commit(current, stale.version).await.expect("commit");
        assert_eq!(committed.version, 2);

        let error = store.commit(stale, 1).await.expect_err("stale commit");
        assert!(matches!(
            error,
            crate::AgentError::VersionConflict {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }
}
