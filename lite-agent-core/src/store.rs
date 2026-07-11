use crate::error::{AgentError, Result};
use crate::events::{Thread, Turn, TurnItem, TurnStatus};
use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio::fs;

pub trait ThreadStore: Send + Sync {
    /// Load one durable thread snapshot.
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn save<'a>(
        &'a self,
        thread: Thread,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn append_turn<'a>(
        &'a self,
        thread_id: &'a str,
        turn: Turn,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn append_turn_items<'a>(
        &'a self,
        thread_id: &'a str,
        turn_id: &'a str,
        items: Vec<TurnItem>,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn update_turn_status<'a>(
        &'a self,
        thread_id: &'a str,
        turn_id: &'a str,
        status: TurnStatus,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;
}

/// Single-process JSON persistence intended for local development and examples.
/// Production deployments should provide a repository with atomic/versioned commits.
#[derive(Debug, Clone)]
pub struct JsonFileThreadStore {
    state_dir: PathBuf,
    _lock: Arc<StoreLock>,
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

    fn save<'a>(
        &'a self,
        mut thread: Thread,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            Self::validate_thread_id(&thread.id)?;
            thread.touch();
            self.write_thread(&thread).await?;
            Ok(thread)
        })
    }

    fn append_turn<'a>(
        &'a self,
        thread_id: &'a str,
        turn: Turn,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            Self::validate_thread_id(thread_id)?;
            let mut thread = match self.load(thread_id).await {
                Ok(thread) => thread,
                Err(AgentError::ThreadNotFound(_)) => Thread::new(thread_id),
                Err(error) => return Err(error),
            };
            thread.turns.push(turn);
            thread.touch();
            self.write_thread(&thread).await?;
            Ok(thread)
        })
    }

    fn append_turn_items<'a>(
        &'a self,
        thread_id: &'a str,
        turn_id: &'a str,
        items: Vec<TurnItem>,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            Self::validate_thread_id(thread_id)?;
            let mut thread = self.load(thread_id).await?;
            let turn = thread
                .turn_mut(turn_id)
                .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
            for item in items {
                turn.push_item(item);
            }
            thread.touch();
            self.write_thread(&thread).await?;
            Ok(thread)
        })
    }

    fn update_turn_status<'a>(
        &'a self,
        thread_id: &'a str,
        turn_id: &'a str,
        status: TurnStatus,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            Self::validate_thread_id(thread_id)?;
            let mut thread = self.load(thread_id).await?;
            let turn = thread
                .turn_mut(turn_id)
                .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
            turn.set_status(status);
            thread.touch();
            self.write_thread(&thread).await?;
            Ok(thread)
        })
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
            .save(Thread {
                turns: vec![turn],
                ..Thread::new("t1")
            })
            .await
            .expect("save");

        let thread = store.load("t1").await.expect("load");
        assert_eq!(thread.id, "t1");
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].id, turn_id);
        assert_eq!(thread.turns[0].items.len(), 1);
    }

    #[tokio::test]
    async fn appends_items_and_updates_status() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = JsonFileThreadStore::open(temp.path()).expect("store");
        let turn = Turn::new();
        let turn_id = turn.id.clone();
        store.append_turn("t1", turn).await.expect("append turn");

        store
            .append_turn_items(
                "t1",
                &turn_id,
                vec![TurnItem::new(
                    TurnItemSource::User,
                    TurnItemKind::UserInput {
                        text: "hi".to_string(),
                        response_to: None,
                    },
                )],
            )
            .await
            .expect("append items");
        let thread = store
            .update_turn_status("t1", &turn_id, TurnStatus::Completed)
            .await
            .expect("status");

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
}
