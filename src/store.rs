use crate::error::{AgentError, Result};
use crate::events::{Thread, ThreadEvent, ThreadId};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio::fs;

pub trait ThreadStore: Send + Sync {
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;

    fn append<'a>(
        &'a self,
        thread_id: &'a str,
        events: Vec<ThreadEvent>,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>>;
}

#[derive(Debug, Clone)]
pub struct JsonFileThreadStore {
    state_dir: PathBuf,
}

impl JsonFileThreadStore {
    pub fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
        }
    }

    fn thread_path(&self, thread_id: &str) -> PathBuf {
        self.state_dir
            .join("threads")
            .join(format!("{thread_id}.json"))
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }
}

impl ThreadStore for JsonFileThreadStore {
    fn load<'a>(
        &'a self,
        thread_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
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

    fn append<'a>(
        &'a self,
        thread_id: &'a str,
        events: Vec<ThreadEvent>,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            let mut thread = match self.load(thread_id).await {
                Ok(thread) => thread,
                Err(AgentError::ThreadNotFound(_)) => Thread {
                    id: ThreadId::from(thread_id),
                    events: Vec::new(),
                },
                Err(error) => return Err(error),
            };

            thread.events.extend(events);
            let path = self.thread_path(thread_id);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
            let raw = serde_json::to_string_pretty(&thread)?;
            fs::write(path, raw).await?;
            Ok(thread)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{ThreadEvent, ThreadEventKind};

    use super::{JsonFileThreadStore, ThreadStore};

    #[tokio::test]
    async fn round_trips_json_events() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = JsonFileThreadStore::new(temp.path());
        store
            .append(
                "t1",
                vec![ThreadEvent::new(ThreadEventKind::AssistantMessageEmitted {
                    text: "hello".to_string(),
                })],
            )
            .await
            .expect("append");

        let thread = store.load("t1").await.expect("load");
        assert_eq!(thread.id, "t1");
        assert_eq!(thread.events.len(), 1);
    }
}
