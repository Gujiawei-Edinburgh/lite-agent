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
use tokio::sync::Notify;

mod seatbelt;

pub use seatbelt::MacOsSeatbeltBackend;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
pub use linux::LinuxNativeBackend;

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

    /// Validate whether a request can be launched under its requested policy.
    ///
    /// This is the only phase that may produce an approval suspension. Once
    /// execution begins, a policy violation is a failed, single-attempt
    /// execution and must not be retried automatically.
    fn preflight(&self, request: &SandboxRequest) -> SandboxResult<SandboxPreflight> {
        request.validate()?;
        self.resolve_policy(&request.policy)?;
        Ok(SandboxPreflight::Allowed)
    }

    /// Resolve the requested guarantees into the policy this backend can
    /// actually enforce.
    ///
    /// The returned warnings must describe every requested guarantee that was
    /// weakened or changed by fallback.
    fn resolve_policy(&self, policy: &SandboxPolicy) -> SandboxResult<SandboxPolicyResolution>;

    fn execute<'a>(
        &'a self,
        request: SandboxRequest,
    ) -> Pin<Box<dyn Future<Output = SandboxResult<SandboxOutput>> + Send + 'a>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxPreflight {
    Allowed,
    PolicyViolation { reason: String },
}

/// A cancellation signal shared by the agent runtime and a sandbox backend.
///
/// Backends should use this signal to terminate the complete command process
/// tree, not only the direct child process.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        let notified = self.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !self.is_cancelled() {
            notified.await;
        }
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
    pub filesystem: PolicySetting<FilesystemPolicy>,
    pub network: PolicySetting<NetworkAccess>,
    pub process: PolicySetting<ProcessPolicy>,
    pub identity: PolicySetting<IdentityIsolation>,
}

impl SandboxPolicy {
    pub fn workspace_read_only(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace(
                host_path,
                FilesystemAccess::ReadOnly,
            )),
            ..Self::default()
        }
    }

    pub fn workspace_read_write(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace(
                host_path,
                FilesystemAccess::ReadWrite,
            )),
            ..Self::default()
        }
    }

    pub fn workspace_read_write_with_host_network(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace(
                host_path,
                FilesystemAccess::ReadWrite,
            )),
            network: PolicySetting::strict(NetworkAccess::Host),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> SandboxResult<()> {
        self.filesystem.requested.validate()
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::default()),
            network: PolicySetting::strict(NetworkAccess::Isolated),
            process: PolicySetting::fallback(ProcessPolicy::default()),
            identity: PolicySetting::fallback(IdentityIsolation::Unprivileged),
        }
    }
}

/// Controls what happens when a backend cannot provide one requested
/// isolation guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnsupportedPolicyBehavior {
    #[default]
    Error,
    WarnAndFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySetting<T> {
    pub requested: T,
    pub unsupported: UnsupportedPolicyBehavior,
}

impl<T> PolicySetting<T> {
    pub fn strict(requested: T) -> Self {
        Self {
            requested,
            unsupported: UnsupportedPolicyBehavior::Error,
        }
    }

    pub fn fallback(requested: T) -> Self {
        Self {
            requested,
            unsupported: UnsupportedPolicyBehavior::WarnAndFallback,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicyResolution {
    pub requested: SandboxPolicy,
    pub effective: EffectiveSandboxPolicy,
    pub warnings: Vec<SandboxWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveSandboxPolicy {
    pub filesystem: FilesystemPolicy,
    pub network: NetworkAccess,
    pub process: ProcessPolicy,
    pub identity: IdentityIsolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxPolicyDimension {
    Filesystem,
    Network,
    Process,
    Identity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxWarning {
    pub dimension: SandboxPolicyDimension,
    pub message: String,
}

/// Filesystem visibility and write access requested by the command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FilesystemPolicy {
    /// No host filesystem access other than what the backend needs to launch
    /// the command.
    #[default]
    Isolated,
    /// The host filesystem is visible with hierarchical access rules. The
    /// longest matching rule wins; paths without a matching rule inherit the
    /// default access.
    Workspace {
        default_access: FilesystemAccess,
        rules: Vec<FilesystemRule>,
    },
    /// The command can access the host filesystem without filesystem
    /// isolation. This must be selected explicitly.
    Host,
}

impl FilesystemPolicy {
    pub fn workspace(host_path: impl Into<PathBuf>, access: FilesystemAccess) -> Self {
        Self::Workspace {
            default_access: FilesystemAccess::ReadOnly,
            rules: vec![FilesystemRule {
                path: host_path.into(),
                access,
            }],
        }
    }

    /// Resolve the effective access for an absolute path using longest-prefix
    /// matching on complete path components.
    pub fn access_for(&self, path: impl AsRef<std::path::Path>) -> SandboxResult<FilesystemAccess> {
        let path = normalize_absolute_path(path.as_ref())?;
        match self {
            Self::Isolated => Ok(FilesystemAccess::Denied),
            Self::Host => Ok(FilesystemAccess::ReadWrite),
            Self::Workspace {
                default_access,
                rules,
            } => {
                self.validate()?;
                Ok(rules
                    .iter()
                    .filter_map(|rule| {
                        normalize_absolute_path(&rule.path)
                            .ok()
                            .filter(|rule_path| path.starts_with(rule_path))
                            .map(|rule_path| (rule_path.components().count(), rule.access))
                    })
                    .max_by_key(|(depth, _)| *depth)
                    .map(|(_, access)| access)
                    .unwrap_or(*default_access))
            }
        }
    }

    fn validate(&self) -> SandboxResult<()> {
        let rules = match self {
            Self::Workspace { rules, .. } => rules,
            Self::Isolated | Self::Host => return Ok(()),
        };

        let mut normalized_paths = std::collections::BTreeSet::new();
        for rule in rules {
            let normalized = normalize_absolute_path(&rule.path)?;
            if !normalized_paths.insert(normalized.clone()) {
                return Err(SandboxError::InvalidRequest(format!(
                    "duplicate filesystem rule: {}",
                    normalized.display()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemRule {
    pub path: PathBuf,
    pub access: FilesystemAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilesystemAccess {
    Denied,
    ReadOnly,
    ReadWrite,
}

fn normalize_absolute_path(path: &std::path::Path) -> SandboxResult<PathBuf> {
    if !path.is_absolute() {
        return Err(SandboxError::InvalidRequest(
            "filesystem paths must be absolute".to_string(),
        ));
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(SandboxError::InvalidRequest(format!(
                        "filesystem path escapes its root: {}",
                        path.display()
                    )));
                }
            }
            std::path::Component::Normal(value) => normalized.push(value),
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::{FilesystemAccess, FilesystemPolicy, FilesystemRule};

    #[test]
    fn filesystem_rules_use_longest_matching_path() {
        let policy = FilesystemPolicy::Workspace {
            default_access: FilesystemAccess::Denied,
            rules: vec![
                FilesystemRule {
                    path: "/project".into(),
                    access: FilesystemAccess::ReadWrite,
                },
                FilesystemRule {
                    path: "/project/src".into(),
                    access: FilesystemAccess::ReadOnly,
                },
                FilesystemRule {
                    path: "/project/src/generated".into(),
                    access: FilesystemAccess::ReadWrite,
                },
            ],
        };

        assert_eq!(
            policy.access_for("/project/main.rs").unwrap(),
            FilesystemAccess::ReadWrite
        );
        assert_eq!(
            policy.access_for("/project/src/main.rs").unwrap(),
            FilesystemAccess::ReadOnly
        );
        assert_eq!(
            policy.access_for("/project/src/generated/file.rs").unwrap(),
            FilesystemAccess::ReadWrite
        );
        assert_eq!(
            policy.access_for("/project-other/file").unwrap(),
            FilesystemAccess::Denied
        );
    }

    #[test]
    fn filesystem_rules_match_path_components_not_string_prefixes() {
        let policy = FilesystemPolicy::Workspace {
            default_access: FilesystemAccess::Denied,
            rules: vec![FilesystemRule {
                path: "/project/src".into(),
                access: FilesystemAccess::ReadWrite,
            }],
        };

        assert_eq!(
            policy.access_for("/project/src-old/file").unwrap(),
            FilesystemAccess::Denied
        );
    }

    #[test]
    fn filesystem_rules_reject_duplicate_normalized_paths() {
        let policy = FilesystemPolicy::Workspace {
            default_access: FilesystemAccess::Denied,
            rules: vec![
                FilesystemRule {
                    path: "/project/src".into(),
                    access: FilesystemAccess::ReadOnly,
                },
                FilesystemRule {
                    path: "/project/./src".into(),
                    access: FilesystemAccess::ReadWrite,
                },
            ],
        };

        assert!(policy.validate().is_err());
    }
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
    pub warnings: Vec<SandboxWarning>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxStatus {
    Exited { code: i32 },
    Signaled { signal: i32 },
    TimedOut,
    Cancelled,
    PolicyViolation { reason: String },
}

pub(crate) fn classify_policy_violation(status: SandboxStatus, stderr: &[u8]) -> SandboxStatus {
    if !matches!(status, SandboxStatus::Exited { .. }) {
        return status;
    }
    let message = String::from_utf8_lossy(stderr);
    let lower = message.to_ascii_lowercase();
    if lower.contains("operation not permitted") || lower.contains("permission denied") {
        return SandboxStatus::PolicyViolation {
            reason: message.trim().to_string(),
        };
    }
    status
}
