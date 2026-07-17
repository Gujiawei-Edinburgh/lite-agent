//! Contracts for native process sandbox backends.
//!
//! This crate intentionally does not implement an operating-system sandbox.
//! It defines the policy and execution boundary used by `exec_command` and by
//! backend implementations such as Seatbelt, Linux namespaces, or a user
//! supplied container runner.

use std::collections::{BTreeMap, BTreeSet};
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
    #[error("invalid sandbox policy: {0}")]
    InvalidPolicy(String),

    #[error("sandbox capability is unavailable: {0}")]
    UnsupportedCapability(String),

    #[error("sandbox launch failed: {0}")]
    Launch(String),

    #[error("sandbox I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("sandbox execution was cancelled")]
    Cancelled,
}

/// A backend that executes a native command under an isolation policy.
pub trait SandboxBackend: Send + Sync {
    fn name(&self) -> &str;

    fn capabilities(&self) -> SandboxCapabilities;

    fn execute<'a>(
        &'a self,
        request: SandboxRequest,
    ) -> Pin<Box<dyn Future<Output = SandboxResult<SandboxOutput>> + Send + 'a>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxCapabilities {
    pub user_namespace: bool,
    pub mount_namespace: bool,
    pub pid_namespace: bool,
    pub network_namespace: bool,
    pub filesystem_policy: bool,
    pub network_policy: bool,
    pub identity_restriction: bool,
    pub syscall_filtering: bool,
    pub resource_limits: bool,
    pub descriptor_isolation: bool,
    pub cancellation: bool,
}

/// A cancellation signal shared by the agent runtime and a sandbox backend.
/// Backends should terminate the complete child process tree when this becomes
/// cancelled.
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
    /// The program path visible inside the sandbox.
    pub program: PathBuf,
    pub args: Vec<String>,
    /// The working directory visible inside the sandbox.
    pub cwd: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub cancellation: CancellationToken,
    pub policy: SandboxPolicy,
}

impl SandboxRequest {
    pub fn validate(&self) -> SandboxResult<()> {
        if self.program.as_os_str().is_empty() {
            return Err(SandboxError::InvalidPolicy(
                "program must not be empty".to_string(),
            ));
        }
        if !self.cwd.is_absolute() {
            return Err(SandboxError::InvalidPolicy(
                "sandbox cwd must be an absolute guest path".to_string(),
            ));
        }
        self.policy.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicy {
    pub identity: IdentityPolicy,
    pub namespaces: NamespacePolicy,
    pub filesystem: FilesystemPolicy,
    pub network: NetworkPolicy,
    pub syscalls: SyscallPolicy,
    pub environment: EnvironmentPolicy,
    pub processes: ProcessPolicy,
    pub resources: ResourceLimits,
}

impl SandboxPolicy {
    pub fn workspace(host_path: impl Into<PathBuf>, access: FilesystemAccess) -> Self {
        Self {
            identity: IdentityPolicy::default(),
            namespaces: NamespacePolicy::default(),
            filesystem: FilesystemPolicy::workspace(host_path, access),
            syscalls: SyscallPolicy::default(),
            environment: EnvironmentPolicy::default(),
            processes: ProcessPolicy::default(),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> SandboxResult<()> {
        self.filesystem.validate()?;
        self.syscalls.validate()?;
        self.environment.validate()?;
        self.processes.validate()?;
        if self.resources.max_output_bytes == 0 {
            return Err(SandboxError::InvalidPolicy(
                "max_output_bytes must be greater than zero".to_string(),
            ));
        }
        if let NetworkPolicy::Proxy { url } = &self.network {
            if url.trim().is_empty() {
                return Err(SandboxError::InvalidPolicy(
                    "proxy URL must not be empty".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            identity: IdentityPolicy::default(),
            namespaces: NamespacePolicy::default(),
            filesystem: FilesystemPolicy::default(),
            network: NetworkPolicy::Denied,
            syscalls: SyscallPolicy::default(),
            environment: EnvironmentPolicy::default(),
            processes: ProcessPolicy::default(),
            resources: ResourceLimits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityPolicy {
    pub user: UserIdentity,
    pub no_new_privileges: bool,
    pub clear_supplementary_groups: bool,
}

impl Default for IdentityPolicy {
    fn default() -> Self {
        Self {
            user: UserIdentity::DropPrivileges,
            no_new_privileges: true,
            clear_supplementary_groups: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserIdentity {
    Inherit,
    DropPrivileges,
    Fixed { uid: u32, gid: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamespacePolicy {
    pub user: bool,
    pub mount: bool,
    pub pid: bool,
    pub network: bool,
    pub ipc: bool,
    pub uts: bool,
}

impl Default for NamespacePolicy {
    fn default() -> Self {
        Self {
            user: true,
            mount: true,
            pid: true,
            network: true,
            ipc: true,
            uts: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemPolicy {
    pub mounts: Vec<FilesystemMount>,
    pub hide_home: bool,
    pub isolated_tmp: bool,
}

impl FilesystemPolicy {
    pub fn workspace(host_path: impl Into<PathBuf>, access: FilesystemAccess) -> Self {
        Self {
            mounts: vec![FilesystemMount {
                host_path: host_path.into(),
                guest_path: PathBuf::from("/workspace"),
                access,
            }],
            ..Self::default()
        }
    }

    pub fn validate(&self) -> SandboxResult<()> {
        let mut guest_paths = BTreeSet::new();
        for mount in &self.mounts {
            if mount.host_path.as_os_str().is_empty() {
                return Err(SandboxError::InvalidPolicy(
                    "mount host path must not be empty".to_string(),
                ));
            }
            if !mount.guest_path.is_absolute() {
                return Err(SandboxError::InvalidPolicy(
                    "mount guest path must be absolute".to_string(),
                ));
            }
            if !guest_paths.insert(mount.guest_path.clone()) {
                return Err(SandboxError::InvalidPolicy(format!(
                    "duplicate mount guest path: {}",
                    mount.guest_path.display()
                )));
            }
        }
        Ok(())
    }
}

impl Default for FilesystemPolicy {
    fn default() -> Self {
        Self {
            mounts: Vec::new(),
            hide_home: true,
            isolated_tmp: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemMount {
    pub host_path: PathBuf,
    pub guest_path: PathBuf,
    pub access: FilesystemAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilesystemAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SyscallPolicy {
    #[default]
    BackendDefault,
    Disabled,
    Allowlist(BTreeSet<String>),
    Denylist(BTreeSet<String>),
}

impl SyscallPolicy {
    fn validate(&self) -> SandboxResult<()> {
        match self {
            Self::Allowlist(names) | Self::Denylist(names) if names.is_empty() => Err(
                SandboxError::InvalidPolicy("syscall policy list must not be empty".to_string()),
            ),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnvironmentPolicy {
    pub inherit_host: bool,
    pub allowed_variables: Option<BTreeSet<String>>,
    pub blocked_variables: BTreeSet<String>,
}

impl EnvironmentPolicy {
    fn validate(&self) -> SandboxResult<()> {
        if let Some(allowed) = &self.allowed_variables {
            if allowed.iter().any(|name| name.is_empty()) {
                return Err(SandboxError::InvalidPolicy(
                    "allowed environment variable names must not be empty".to_string(),
                ));
            }
        }
        if self.blocked_variables.iter().any(|name| name.is_empty()) {
            return Err(SandboxError::InvalidPolicy(
                "blocked environment variable names must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessPolicy {
    pub allow_child_processes: bool,
    pub kill_process_group_on_exit: bool,
    pub allow_ptrace: bool,
    pub close_inherited_descriptors: bool,
}

impl ProcessPolicy {
    fn validate(&self) -> SandboxResult<()> {
        if self.allow_ptrace && !self.allow_child_processes {
            return Err(SandboxError::InvalidPolicy(
                "ptrace requires child-process access".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for ProcessPolicy {
    fn default() -> Self {
        Self {
            allow_child_processes: true,
            kill_process_group_on_exit: true,
            allow_ptrace: false,
            close_inherited_descriptors: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum NetworkPolicy {
    #[default]
    Denied,
    InheritHost,
    Proxy {
        url: String,
    },
    Allowlist(Vec<NetworkRule>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkRule {
    pub host: String,
    pub ports: BTreeSet<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    pub wall_time: Option<Duration>,
    pub cpu_time: Option<Duration>,
    pub memory_bytes: Option<u64>,
    pub max_processes: Option<u32>,
    pub max_open_files: Option<u64>,
    pub max_file_size_bytes: Option<u64>,
    pub max_output_bytes: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            wall_time: Some(Duration::from_secs(30)),
            cpu_time: None,
            memory_bytes: None,
            max_processes: Some(64),
            max_open_files: Some(1024),
            max_file_size_bytes: None,
            max_output_bytes: 20 * 1024 * 1024,
        }
    }
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
