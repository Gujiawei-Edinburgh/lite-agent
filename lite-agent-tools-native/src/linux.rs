//! Minimal native Linux sandbox backend.
//!
//! This backend intentionally exposes no Linux mechanisms through the public
//! policy. It translates the semantic policy into user, mount, and network
//! namespace setup performed before `execve`.

use crate::{
    EffectiveSandboxPolicy, FilesystemAccess, FilesystemPolicy, IdentityIsolation, NetworkAccess,
    PolicySetting, ProcessVisibility, SandboxBackend, SandboxError, SandboxOutput, SandboxPolicy,
    SandboxPolicyDimension, SandboxPolicyResolution, SandboxRequest, SandboxResult, SandboxStatus,
    SandboxWarning, UnsupportedPolicyBehavior,
};
use std::ffi::CString;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::{sleep, Duration};

#[derive(Debug, Clone, Default)]
pub struct LinuxNativeBackend;

impl SandboxBackend for LinuxNativeBackend {
    fn name(&self) -> &str {
        "linux-native"
    }

    fn resolve_policy(&self, policy: &SandboxPolicy) -> SandboxResult<SandboxPolicyResolution> {
        policy.validate()?;
        let mut effective = EffectiveSandboxPolicy {
            filesystem: policy.filesystem.requested.clone(),
            network: policy.network.requested,
            process: policy.process.requested,
            identity: policy.identity.requested,
        };
        let mut warnings = Vec::new();

        if effective.process.visibility == ProcessVisibility::Isolated {
            unsupported(
                SandboxPolicyDimension::Process,
                &policy.process,
                "the first Linux launcher does not yet provide PID namespace isolation",
                &mut effective,
                &mut warnings,
                |effective| effective.process.visibility = ProcessVisibility::Host,
            )?;
        }

        if matches!(effective.filesystem, FilesystemPolicy::Isolated) {
            unsupported(
                SandboxPolicyDimension::Filesystem,
                &policy.filesystem,
                "Linux workspace sandboxing requires an explicit workspace policy",
                &mut effective,
                &mut warnings,
                |effective| effective.filesystem = FilesystemPolicy::Host,
            )?;
        }

        if has_denied_filesystem_rule(&effective.filesystem) {
            unsupported(
                SandboxPolicyDimension::Filesystem,
                &policy.filesystem,
                "Landlock path-deny rules are not enabled in the first Linux launcher",
                &mut effective,
                &mut warnings,
                |effective| soften_denied_filesystem(&mut effective.filesystem),
            )?;
        }

        Ok(SandboxPolicyResolution {
            requested: policy.clone(),
            effective,
            warnings,
        })
    }

    fn execute<'a>(
        &'a self,
        request: SandboxRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = SandboxResult<SandboxOutput>> + Send + 'a>,
    > {
        Box::pin(async move { self.execute_request(request).await })
    }
}

impl LinuxNativeBackend {
    async fn execute_request(&self, request: SandboxRequest) -> SandboxResult<SandboxOutput> {
        request.validate()?;
        let resolution = self.resolve_policy(&request.policy)?;
        let started = Instant::now();

        if request.cancellation.is_cancelled() {
            return Ok(SandboxOutput {
                status: SandboxStatus::Cancelled,
                stdout: Vec::new(),
                stderr: Vec::new(),
                warnings: resolution.warnings,
                stdout_truncated: false,
                stderr_truncated: false,
                duration: started.elapsed(),
            });
        }

        let effective = resolution.effective.clone();
        let mut command = Command::new(&request.program);
        command
            .args(&request.args)
            .current_dir(&request.cwd)
            .env_clear()
            .envs(&request.environment)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        unsafe {
            command.as_std_mut().pre_exec(move || {
                set_process_group()?;
                setup_linux_sandbox(&effective)
            });
        }

        let mut child = command.spawn().map_err(|error| {
            SandboxError::Launch(format!("{}: {error}", request.program.display()))
        })?;
        let child_pid = child.id().map(|pid| pid as i32);
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SandboxError::Launch("stdout pipe was not created".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| SandboxError::Launch("stderr pipe was not created".to_string()))?;
        let stdout_task = tokio::spawn(async move {
            let mut output = Vec::new();
            let result = stdout.take(20 * 1024 * 1024).read_to_end(&mut output).await;
            (result, output)
        });
        let stderr_task = tokio::spawn(async move {
            let mut output = Vec::new();
            let result = stderr.take(20 * 1024 * 1024).read_to_end(&mut output).await;
            (result, output)
        });

        let mut cancelled = false;
        let status = loop {
            tokio::select! {
                result = child.wait() => break result?,
                _ = sleep(Duration::from_millis(20)) => {
                    if request.cancellation.is_cancelled() {
                        cancelled = true;
                        break terminate_child(&mut child, child_pid).await?;
                    }
                }
            }
        };

        let (stdout_result, stdout) = stdout_task
            .await
            .map_err(|error| SandboxError::Launch(format!("stdout reader failed: {error}")))?;
        let (stderr_result, stderr) = stderr_task
            .await
            .map_err(|error| SandboxError::Launch(format!("stderr reader failed: {error}")))?;
        stdout_result?;
        stderr_result?;

        Ok(SandboxOutput {
            status: if cancelled {
                SandboxStatus::Cancelled
            } else if let Some(signal) = exit_signal(&status) {
                SandboxStatus::Signaled { signal }
            } else {
                SandboxStatus::Exited {
                    code: status.code().unwrap_or(-1),
                }
            },
            stdout,
            stderr,
            warnings: resolution.warnings,
            stdout_truncated: false,
            stderr_truncated: false,
            duration: started.elapsed(),
        })
    }
}

fn unsupported<T>(
    dimension: SandboxPolicyDimension,
    setting: &PolicySetting<T>,
    message: &str,
    effective: &mut EffectiveSandboxPolicy,
    warnings: &mut Vec<SandboxWarning>,
    fallback: impl FnOnce(&mut EffectiveSandboxPolicy),
) -> SandboxResult<()> {
    match setting.unsupported {
        UnsupportedPolicyBehavior::Error => {
            Err(SandboxError::UnsupportedPolicy(message.to_string()))
        }
        UnsupportedPolicyBehavior::WarnAndFallback => {
            fallback(effective);
            warnings.push(SandboxWarning {
                dimension,
                message: message.to_string(),
            });
            Ok(())
        }
    }
}

fn has_denied_filesystem_rule(policy: &FilesystemPolicy) -> bool {
    match policy {
        FilesystemPolicy::Workspace {
            default_access,
            rules,
        } => {
            *default_access == FilesystemAccess::Denied
                || rules
                    .iter()
                    .any(|rule| rule.access == FilesystemAccess::Denied)
        }
        FilesystemPolicy::Isolated | FilesystemPolicy::Host => false,
    }
}

fn soften_denied_filesystem(policy: &mut FilesystemPolicy) {
    if let FilesystemPolicy::Workspace {
        default_access,
        rules,
    } = policy
    {
        if *default_access == FilesystemAccess::Denied {
            *default_access = FilesystemAccess::ReadOnly;
        }
        for rule in rules {
            if rule.access == FilesystemAccess::Denied {
                rule.access = FilesystemAccess::ReadOnly;
            }
        }
    }
}

fn setup_linux_sandbox(policy: &EffectiveSandboxPolicy) -> std::io::Result<()> {
    if policy.identity == IdentityIsolation::Unprivileged {
        unshare(libc::CLONE_NEWUSER).map_err(|error| io_context("create user namespace", error))?;
        configure_user_mapping()?;
    }

    if !matches!(policy.filesystem, FilesystemPolicy::Host) {
        unshare(libc::CLONE_NEWNS).map_err(|error| io_context("create mount namespace", error))?;
        make_mounts_private().map_err(|error| io_context("make mounts private", error))?;
        make_root_read_only().map_err(|error| io_context("make root read-only", error))?;
        apply_filesystem_rules(&policy.filesystem)
            .map_err(|error| io_context("apply filesystem rules", error))?;
    }

    if policy.network != NetworkAccess::Host {
        unshare(libc::CLONE_NEWNET)
            .map_err(|error| io_context("create network namespace", error))?;
    }

    Ok(())
}

fn unshare(flags: libc::c_int) -> std::io::Result<()> {
    if unsafe { libc::unshare(flags) } == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn configure_user_mapping() -> std::io::Result<()> {
    let uid = unsafe { libc::getuid() };
    let gid = unsafe { libc::getgid() };
    let _ = std::fs::write("/proc/self/setgroups", "deny");
    std::fs::write("/proc/self/uid_map", format!("0 {uid} 1\n"))
        .map_err(|error| io_context("write uid mapping", error))?;
    std::fs::write("/proc/self/gid_map", format!("0 {gid} 1\n"))
        .map_err(|error| io_context("write gid mapping", error))?;
    if unsafe { libc::setresuid(0, 0, 0) } == -1 {
        return Err(io_context(
            "switch to mapped user identity",
            std::io::Error::last_os_error(),
        ));
    }
    if unsafe { libc::setresgid(0, 0, 0) } == -1 {
        return Err(io_context(
            "switch to mapped group identity",
            std::io::Error::last_os_error(),
        ));
    }
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } == -1 {
        return Err(io_context(
            "set no-new-privileges",
            std::io::Error::last_os_error(),
        ));
    }
    Ok(())
}

fn make_mounts_private() -> std::io::Result<()> {
    mount(
        None,
        Path::new("/"),
        None,
        libc::MS_REC | libc::MS_PRIVATE,
        None,
    )
}

fn make_root_read_only() -> std::io::Result<()> {
    mount(
        Some(Path::new("/")),
        Path::new("/"),
        None,
        libc::MS_BIND | libc::MS_REC,
        None,
    )?;
    mount(
        None,
        Path::new("/"),
        None,
        libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY,
        None,
    )
}

fn apply_filesystem_rules(policy: &FilesystemPolicy) -> std::io::Result<()> {
    let FilesystemPolicy::Workspace { rules, .. } = policy else {
        return Ok(());
    };
    let mut rules = rules.clone();
    rules.sort_by_key(|rule| rule.path.components().count());
    for rule in rules {
        let path = rule
            .path
            .canonicalize()
            .unwrap_or_else(|_| rule.path.clone());
        mount(
            Some(path.as_path()),
            path.as_path(),
            None,
            libc::MS_BIND | libc::MS_REC,
            None,
        )?;
        let remount_flags = match rule.access {
            FilesystemAccess::ReadOnly => libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY,
            FilesystemAccess::ReadWrite => libc::MS_BIND | libc::MS_REMOUNT,
            FilesystemAccess::Denied => libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY,
        };
        mount(None, path.as_path(), None, remount_flags, None)?;
    }
    Ok(())
}

fn mount(
    source: Option<&Path>,
    target: &Path,
    filesystem: Option<&str>,
    flags: libc::c_ulong,
    data: Option<&str>,
) -> std::io::Result<()> {
    mount_paths(
        source.map(path_to_cstring).transpose()?,
        Some(path_to_cstring(target)?),
        filesystem
            .map(CString::new)
            .transpose()
            .map_err(invalid_cstring)?,
        data.map(CString::new)
            .transpose()
            .map_err(invalid_cstring)?,
        flags,
    )
}

fn path_to_cstring(path: &Path) -> std::io::Result<CString> {
    CString::new(path.to_str().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Linux sandbox paths must be valid UTF-8",
        )
    })?)
    .map_err(invalid_cstring)
}

fn mount_paths(
    source: Option<CString>,
    target: Option<CString>,
    filesystem: Option<CString>,
    data: Option<CString>,
    flags: libc::c_ulong,
) -> std::io::Result<()> {
    if unsafe {
        libc::mount(
            source
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            target
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            filesystem
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            flags,
            data.as_ref().map_or(std::ptr::null_mut(), |value| {
                value.as_ptr() as *mut libc::c_void
            }),
        )
    } == -1
    {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn invalid_cstring(error: std::ffi::NulError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, error)
}

fn io_context(operation: &str, error: std::io::Error) -> std::io::Error {
    std::io::Error::new(error.kind(), format!("{operation}: {error}"))
}

fn set_process_group() -> std::io::Result<()> {
    if unsafe { libc::setpgid(0, 0) } == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

async fn terminate_child(
    child: &mut tokio::process::Child,
    child_pid: Option<i32>,
) -> SandboxResult<std::process::ExitStatus> {
    if let Some(pid) = child_pid {
        unsafe {
            libc::killpg(pid, libc::SIGKILL);
        }
    }
    let _ = child.kill().await;
    child.wait().await.map_err(SandboxError::Io)
}

fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[test]
    fn strict_pid_isolation_is_rejected() {
        let backend = LinuxNativeBackend;
        let mut policy = SandboxPolicy::workspace_read_write("/tmp");
        policy.process = PolicySetting::strict(crate::ProcessPolicy::default());
        assert!(backend.resolve_policy(&policy).is_err());
    }

    #[test]
    fn default_pid_and_identity_policies_fall_back() {
        let backend = LinuxNativeBackend;
        let policy = SandboxPolicy::workspace_read_write("/tmp");
        let resolution = backend.resolve_policy(&policy).expect("resolution");
        assert_eq!(resolution.warnings.len(), 1);
        assert_eq!(
            resolution.effective.process.visibility,
            ProcessVisibility::Host
        );
    }

    #[tokio::test]
    async fn enforces_workspace_write_boundary() {
        let workspace = tempfile::tempdir().expect("workspace");
        let outside = tempfile::tempdir().expect("outside");
        let workspace_file = workspace.path().join("created.txt");
        let outside_file = outside.path().join("blocked.txt");
        let command = format!(
            "printf workspace > '{}'; workspace_status=$?; printf outside > '{}'; outside_status=$?; test $workspace_status -eq 0 -a $outside_status -ne 0",
            workspace_file.display(),
            outside_file.display()
        );
        let request = SandboxRequest {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".to_string(), command],
            cwd: workspace.path().to_path_buf(),
            environment: BTreeMap::new(),
            cancellation: Default::default(),
            policy: SandboxPolicy::workspace_read_write(workspace.path()),
        };

        let output = LinuxNativeBackend
            .execute(request)
            .await
            .expect("sandbox execution");
        assert!(workspace_file.exists(), "allowed workspace write failed");
        assert!(
            !outside_file.exists(),
            "outside write unexpectedly succeeded"
        );
        assert!(matches!(output.status, SandboxStatus::Exited { code: 0 }));
    }
}
