//! Semantic contracts for native command sandbox backends.
//!
//! This crate does not implement an operating-system sandbox. It defines the
//! isolation guarantees requested by `exec_command` and the execution boundary
//! used by native, container, or user-provided backends.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use thiserror::Error;

pub type SandboxResult<T> = Result<T, SandboxError>;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("invalid sandbox request: {0}")]
    InvalidRequest(String),

    #[error("sandbox policy is unsupported: {0}")]
    UnsupportedPolicy(String),

    #[error("sandbox launch failed: {0}")]
    Launch(String),

    #[error("sandbox I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A backend that executes a native command under a semantic isolation policy.
pub trait SandboxBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Check whether this backend can honor the requested guarantees.
    ///
    /// Backends must reject unsupported restrictions instead of silently
    /// running the command with weaker isolation.
    fn check_policy(&self, policy: &SandboxPolicy) -> SandboxResult<()>;

    fn execute<'a>(
        &'a self,
        request: SandboxRequest,
    ) -> Pin<Box<dyn Future<Output = SandboxResult<SandboxOutput>> + Send + 'a>>;
}

/// A cancellation signal shared by the agent runtime and a sandbox backend.
///
/// Backends should use this signal to terminate the complete command process
/// tree, not only the direct child process.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct SandboxRequest {
    /// The requested executable path on the host.
    pub program: PathBuf,
    pub args: Vec<String>,
    /// The requested working directory on the host.
    pub cwd: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub cancellation: CancellationToken,
    pub policy: SandboxPolicy,
}

impl SandboxRequest {
    pub fn validate(&self) -> SandboxResult<()> {
        if self.program.as_os_str().is_empty() {
            return Err(SandboxError::InvalidRequest(
                "program must not be empty".to_string(),
            ));
        }
        if !self.cwd.is_absolute() {
            return Err(SandboxError::InvalidRequest(
                "working directory must be an absolute host path".to_string(),
            ));
        }
        self.policy.validate()
    }
}

/// Semantic isolation requirements for one command execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicy {
    pub filesystem: FilesystemPolicy,
    pub network: NetworkAccess,
    pub process: ProcessPolicy,
    pub identity: IdentityIsolation,
}

impl SandboxPolicy {
    pub fn workspace_read_only(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: FilesystemPolicy::workspace(host_path, FilesystemAccess::ReadOnly),
            ..Self::default()
        }
    }

    pub fn workspace_read_write(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: FilesystemPolicy::workspace(host_path, FilesystemAccess::ReadWrite),
            ..Self::default()
        }
    }

    pub fn workspace_read_write_with_host_network(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: FilesystemPolicy::workspace(host_path, FilesystemAccess::ReadWrite),
            network: NetworkAccess::Host,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> SandboxResult<()> {
        self.filesystem.validate()
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            filesystem: FilesystemPolicy::default(),
            network: NetworkAccess::Isolated,
            process: ProcessPolicy::default(),
            identity: IdentityIsolation::Unprivileged,
        }
    }
}

/// Filesystem visibility and write access requested by the command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FilesystemPolicy {
    /// No host filesystem access other than what the backend needs to launch
    /// the command.
    #[default]
    Isolated,
    /// The host filesystem is visible with the requested workspace roots
    /// writable or read-only according to each entry.
    Workspace { roots: Vec<WorkspaceRoot> },
    /// The command can access the host filesystem without filesystem
    /// isolation. This must be selected explicitly.
    Host,
}

impl FilesystemPolicy {
    pub fn workspace(host_path: impl Into<PathBuf>, access: FilesystemAccess) -> Self {
        Self::Workspace {
            roots: vec![WorkspaceRoot {
                path: host_path.into(),
                access,
            }],
        }
    }

    fn validate(&self) -> SandboxResult<()> {
        let roots = match self {
            Self::Workspace { roots } => roots,
            Self::Isolated | Self::Host => return Ok(()),
        };

        for root in roots {
            if root.path.as_os_str().is_empty() {
                return Err(SandboxError::InvalidRequest(
                    "workspace path must not be empty".to_string(),
                ));
            }
            if !root.path.is_absolute() {
                return Err(SandboxError::InvalidRequest(
                    "workspace path must be absolute".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceRoot {
    pub path: PathBuf,
    pub access: FilesystemAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilesystemAccess {
    ReadOnly,
    ReadWrite,
}

/// Network visibility requested by the command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NetworkAccess {
    /// The command cannot use networking.
    Denied,
    /// The command runs in a separate network environment. It cannot access
    /// the host network, but the backend may provide isolated loopback or
    /// other explicitly configured connectivity.
    #[default]
    Isolated,
    /// The command shares the host network, including access to services such
    /// as a developer-configured loopback proxy.
    Host,
}

/// Process visibility and descendant cleanup are independent guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessPolicy {
    pub visibility: ProcessVisibility,
    pub terminate_descendants: bool,
}

impl Default for ProcessPolicy {
    fn default() -> Self {
        Self {
            visibility: ProcessVisibility::Isolated,
            terminate_descendants: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProcessVisibility {
    #[default]
    Isolated,
    Host,
}

/// Whether the command inherits the caller's host identity or runs with a
/// backend-selected unprivileged identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IdentityIsolation {
    #[default]
    Unprivileged,
    Host,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxOutput {
    pub status: SandboxStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxStatus {
    Exited { code: i32 },
    Signaled { signal: i32 },
    TimedOut,
    Cancelled,
}
