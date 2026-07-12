use lite_agent_runtime::{TraceCollector, TraceEvent};
use std::collections::BTreeMap;
use std::fs::{create_dir_all, File, OpenOptions};
use std::future::Future;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Mutex;

#[derive(Debug)]
pub struct JsonlTraceCollector {
    trace_dir: PathBuf,
    files: Mutex<BTreeMap<String, BufWriter<File>>>,
}

impl JsonlTraceCollector {
    pub fn new(state_dir: impl Into<PathBuf>) -> io::Result<Self> {
        let trace_dir = state_dir.into().join("traces");
        create_dir_all(&trace_dir)?;
        Ok(Self {
            trace_dir,
            files: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn trace_dir(&self) -> &Path {
        &self.trace_dir
    }

    fn validate_thread_id(thread_id: &str) -> bool {
        !thread_id.is_empty()
            && thread_id != "."
            && thread_id != ".."
            && !thread_id.contains('/')
            && !thread_id.contains('\\')
    }

    fn open_file(&self, thread_id: &str) -> io::Result<BufWriter<File>> {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.trace_dir.join(format!("{thread_id}.jsonl")))
            .map(BufWriter::new)
    }
}

impl TraceCollector for JsonlTraceCollector {
    fn record(&self, event: TraceEvent) {
        if !Self::validate_thread_id(&event.thread_id) {
            tracing::error!(thread_id = %event.thread_id, "refusing to write trace for invalid thread id");
            return;
        }

        let thread_id = event.thread_id.clone();
        let raw = match serde_json::to_vec(&event) {
            Ok(raw) => raw,
            Err(error) => {
                tracing::error!(thread_id, error = %error, "failed to serialize trace event");
                return;
            }
        };

        let mut files = match self.files.lock() {
            Ok(files) => files,
            Err(error) => {
                tracing::error!(error = %error, "trace file registry is poisoned");
                return;
            }
        };
        let writer = match files.entry(thread_id.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => match self.open_file(&thread_id) {
                Ok(writer) => entry.insert(writer),
                Err(error) => {
                    tracing::error!(thread_id, error = %error, "failed to open trace file");
                    return;
                }
            },
        };

        if let Err(error) = writer.write_all(&raw).and_then(|_| writer.write_all(b"\n")) {
            tracing::error!(thread_id, error = %error, "failed to write trace event");
        }
    }

    fn flush<'a>(&'a self) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let mut files = match self.files.lock() {
                Ok(files) => files,
                Err(error) => {
                    tracing::error!(error = %error, "trace file registry is poisoned");
                    return;
                }
            };
            for (thread_id, writer) in files.iter_mut() {
                if let Err(error) = writer.flush().and_then(|_| writer.get_ref().sync_data()) {
                    tracing::error!(thread_id, error = %error, "failed to flush trace file");
                }
            }
        })
    }
}
