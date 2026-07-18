use crate::{
    EffectiveSandboxPolicy, FilesystemPolicy, IdentityIsolation, NetworkAccess, PolicySetting,
    ProcessVisibility, SandboxBackend, SandboxError, SandboxOutput, SandboxPolicy,
    SandboxPolicyDimension, SandboxPolicyResolution, SandboxRequest, SandboxResult, SandboxStatus,
    SandboxWarning, UnsupportedPolicyBehavior,
};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::{sleep, Duration};

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// macOS Seatbelt backend using `/usr/bin/sandbox-exec`.
///
/// Seatbelt provides filesystem and network policy enforcement, but does not
/// provide Linux-style PID or user namespaces. Those unsupported dimensions
/// are resolved according to the policy setting attached to each dimension.
#[derive(Debug, Clone)]
pub struct MacOsSeatbeltBackend {
    sandbox_exec: PathBuf,
}

impl Default for MacOsSeatbeltBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MacOsSeatbeltBackend {
    pub fn new() -> Self {
        Self {
            sandbox_exec: PathBuf::from(SANDBOX_EXEC),
        }
    }

    pub fn with_sandbox_exec(path: impl Into<PathBuf>) -> Self {
        Self {
            sandbox_exec: path.into(),
        }
    }

    pub fn profile_for(policy: &EffectiveSandboxPolicy) -> SandboxResult<String> {
        render_profile(policy)
    }
}

impl SandboxBackend for MacOsSeatbeltBackend {
    fn name(&self) -> &str {
        "macos-seatbelt"
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

        resolve_process(&mut effective, &policy.process, &mut warnings)?;
        resolve_identity(&mut effective, &policy.identity, &mut warnings)?;

        if matches!(effective.filesystem, FilesystemPolicy::Isolated) {
            unsupported(
                SandboxPolicyDimension::Filesystem,
                &policy.filesystem,
                "Seatbelt backend requires an explicit workspace or host filesystem policy",
                &mut effective,
                &mut warnings,
                |effective| effective.filesystem = FilesystemPolicy::Host,
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

impl MacOsSeatbeltBackend {
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

        let profile = Self::profile_for(&resolution.effective)?;

        if !cfg!(target_os = "macos") {
            return Err(SandboxError::UnsupportedPolicy(
                "macOS Seatbelt is only available on macOS".to_string(),
            ));
        }

        let mut command = Command::new(&self.sandbox_exec);
        command
            .arg("-p")
            .arg(profile)
            .arg(&request.program)
            .args(&request.args)
            .current_dir(&request.cwd)
            .env_clear()
            .envs(&request.environment)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if resolution.effective.process.terminate_descendants {
            configure_process_group(&mut command);
        }

        let mut child = command.spawn().map_err(|error| {
            SandboxError::Launch(format!("{}: {error}", self.sandbox_exec.display()))
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

        let mut cancelled = request.cancellation.is_cancelled();
        let status = if cancelled {
            terminate_child(
                &mut child,
                child_pid,
                resolution.effective.process.terminate_descendants,
            )
            .await?
        } else {
            loop {
                tokio::select! {
                    result = child.wait() => break result?,
                    _ = sleep(Duration::from_millis(20)) => {
                        if request.cancellation.is_cancelled() {
                            cancelled = true;
                            break terminate_child(&mut child, child_pid, resolution.effective.process.terminate_descendants).await?;
                        }
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

        let status = if cancelled {
            SandboxStatus::Cancelled
        } else if let Some(signal) = exit_signal(&status) {
            SandboxStatus::Signaled { signal }
        } else {
            SandboxStatus::Exited {
                code: status.code().unwrap_or(-1),
            }
        };

        Ok(SandboxOutput {
            status,
            stdout,
            stderr,
            warnings: resolution.warnings,
            stdout_truncated: false,
            stderr_truncated: false,
            duration: started.elapsed(),
        })
    }
}

fn resolve_process(
    effective: &mut EffectiveSandboxPolicy,
    setting: &PolicySetting<crate::ProcessPolicy>,
    warnings: &mut Vec<SandboxWarning>,
) -> SandboxResult<()> {
    if setting.requested.visibility == ProcessVisibility::Isolated {
        return unsupported(
            SandboxPolicyDimension::Process,
            setting,
            "Seatbelt cannot isolate host process visibility",
            effective,
            warnings,
            |effective| effective.process.visibility = ProcessVisibility::Host,
        );
    }
    Ok(())
}

fn resolve_identity(
    effective: &mut EffectiveSandboxPolicy,
    setting: &PolicySetting<IdentityIsolation>,
    warnings: &mut Vec<SandboxWarning>,
) -> SandboxResult<()> {
    if setting.requested == IdentityIsolation::Unprivileged {
        return unsupported(
            SandboxPolicyDimension::Identity,
            setting,
            "Seatbelt cannot change the command's host identity",
            effective,
            warnings,
            |effective| effective.identity = IdentityIsolation::Host,
        );
    }
    Ok(())
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

fn render_profile(policy: &EffectiveSandboxPolicy) -> SandboxResult<String> {
    let mut profile = String::from("(version 1)\n(allow default)\n");

    match &policy.filesystem {
        FilesystemPolicy::Host => {}
        FilesystemPolicy::Workspace { roots } => {
            profile.push_str("(deny file-write*)\n");
            for root in roots {
                if root.access == crate::FilesystemAccess::ReadWrite {
                    let path = canonical_existing_path(&root.path);
                    profile.push_str(&format!(
                        "(allow file-write* (subpath \"{}\"))\n",
                        escape_profile_path(&path)?
                    ));
                }
            }
        }
        FilesystemPolicy::Isolated => {
            return Err(SandboxError::UnsupportedPolicy(
                "Seatbelt profile cannot represent isolated filesystem access".to_string(),
            ));
        }
    }

    match policy.network {
        NetworkAccess::Denied | NetworkAccess::Isolated => profile.push_str("(deny network*)\n"),
        NetworkAccess::Host => {}
    }

    Ok(profile)
}

fn escape_profile_path(path: &Path) -> SandboxResult<String> {
    let path = path.to_str().ok_or_else(|| {
        SandboxError::InvalidRequest("Seatbelt paths must be valid UTF-8".to_string())
    })?;
    Ok(path
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r"))
}

fn canonical_existing_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.as_std_mut().pre_exec(|| {
            if libc::setpgid(0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

async fn terminate_child(
    child: &mut tokio::process::Child,
    child_pid: Option<i32>,
    terminate_descendants: bool,
) -> SandboxResult<std::process::ExitStatus> {
    if terminate_descendants {
        if let Some(pid) = child_pid {
            #[cfg(unix)]
            unsafe {
                libc::killpg(pid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill().await;
    child.wait().await.map_err(SandboxError::Io)
}

#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn renders_read_only_workspace_profile() {
        let policy = EffectiveSandboxPolicy {
            filesystem: FilesystemPolicy::workspace(
                "/tmp/workspace",
                crate::FilesystemAccess::ReadOnly,
            ),
            network: NetworkAccess::Denied,
            process: crate::ProcessPolicy::default(),
            identity: IdentityIsolation::Host,
        };
        let profile = render_profile(&policy).expect("profile");
        assert!(profile.contains("(deny file-write*)"));
        assert!(profile.contains("(deny network*)"));
        assert!(!profile.contains("(allow file-write* (subpath \"/tmp/workspace\"))"));
    }

    #[test]
    fn renders_read_write_workspace_profile() {
        let policy = EffectiveSandboxPolicy {
            filesystem: FilesystemPolicy::workspace(
                "/tmp/workspace",
                crate::FilesystemAccess::ReadWrite,
            ),
            network: NetworkAccess::Host,
            process: crate::ProcessPolicy::default(),
            identity: IdentityIsolation::Host,
        };
        let profile = render_profile(&policy).expect("profile");
        assert!(profile.contains("(allow file-write* (subpath \"/tmp/workspace\"))"));
        assert!(!profile.contains("(deny network*)"));
    }

    #[test]
    fn rejects_unsupported_process_isolation_by_default() {
        let backend = MacOsSeatbeltBackend::new();
        let mut policy = SandboxPolicy::workspace_read_write("/tmp/workspace");
        policy.process = PolicySetting::strict(crate::ProcessPolicy::default());
        let error = backend
            .resolve_policy(&policy)
            .expect_err("policy must be rejected");
        assert!(error.to_string().contains("process visibility"));
    }

    #[test]
    fn records_process_and_identity_fallbacks() {
        let backend = MacOsSeatbeltBackend::new();
        let policy = SandboxPolicy::workspace_read_write("/tmp/workspace");
        let resolution = backend.resolve_policy(&policy).expect("fallback policy");
        assert_eq!(resolution.warnings.len(), 2);
        assert_eq!(
            resolution.effective.process.visibility,
            ProcessVisibility::Host
        );
        assert_eq!(resolution.effective.identity, IdentityIsolation::Host);
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn enforces_workspace_write_boundary() {
        let probe = std::process::Command::new(SANDBOX_EXEC)
            .args(["-p", "(version 1)(allow default)", "/usr/bin/true"])
            .output()
            .expect("sandbox-exec");
        if !probe.status.success() {
            eprintln!(
                "skipping Seatbelt integration test: sandbox-exec is unavailable: {}",
                String::from_utf8_lossy(&probe.stderr)
            );
            return;
        }

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
        let output = MacOsSeatbeltBackend::new()
            .execute(request)
            .await
            .expect("sandbox execution");

        println!(
            "sandbox output: status={:?}, stdout={}, stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        assert!(workspace_file.exists(), "allowed workspace write failed");
        assert!(
            !outside_file.exists(),
            "write outside configured workspace unexpectedly succeeded"
        );
        assert!(
            matches!(output.status, SandboxStatus::Exited { code } if code != 0),
            "unexpected sandbox status: {:?}",
            output.status
        );
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn cancelled_request_does_not_spawn_sandbox() {
        let cancellation = crate::CancellationToken::default();
        cancellation.cancel();
        let request = SandboxRequest {
            program: PathBuf::from("/bin/true"),
            args: Vec::new(),
            cwd: PathBuf::from("/tmp"),
            environment: BTreeMap::new(),
            cancellation,
            policy: SandboxPolicy::workspace_read_write("/tmp"),
        };

        let output = MacOsSeatbeltBackend::with_sandbox_exec("/path/that/must/not/be_spawned")
            .execute(request)
            .await
            .expect("pre-cancelled request");
        assert_eq!(output.status, SandboxStatus::Cancelled);
    }
}
