use crate::sandbox::{
    CancellationToken, SandboxBackend, SandboxOutput, SandboxPolicy, SandboxPreflight,
    SandboxRequest, SandboxStatus,
};
use lite_agent_kernel::events::{new_id, Suspension, SuspensionKind};
use lite_agent_runtime::{
    AgentError, AgentFunction, FunctionContext, FunctionExecution, FunctionSpec, Result,
    TurnAbortSignal,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ExecRequest {
    pub command: String,
    pub cwd: PathBuf,
    pub shell: Shell,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalHandling {
    Suspend,
    ReturnToolError,
}

#[derive(Debug, Clone)]
pub enum AuthorizationDecision {
    Allow { policy: SandboxPolicy },
    Deny { reason: String },
    RequireApproval { reason: String },
}

pub trait ExecAuthorizer: Send + Sync {
    fn authorize<'a>(
        &'a self,
        request: &'a ExecRequest,
        _current_policy: &'a SandboxPolicy,
        violation: &'a str,
        context: &'a FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AuthorizationDecision>> + Send + 'a>>;
}

/// In-memory command grants scoped by thread ID.
///
/// The host is responsible for persisting and restoring grants if they must
/// survive process restarts. A grant matches the command text exactly and
/// stores the policy approved for that thread and command.
#[derive(Default)]
pub struct ThreadExecAuthorizer {
    grants: std::sync::Mutex<BTreeMap<String, BTreeMap<String, SandboxPolicy>>>,
}

impl ThreadExecAuthorizer {
    pub fn grant(
        &self,
        thread_id: impl Into<String>,
        command: impl Into<String>,
        policy: SandboxPolicy,
    ) {
        let mut grants = self
            .grants
            .lock()
            .expect("thread exec grants mutex poisoned");
        grants
            .entry(thread_id.into())
            .or_default()
            .insert(command.into(), policy);
    }

    pub fn revoke(&self, thread_id: &str, command: &str) {
        let mut grants = self
            .grants
            .lock()
            .expect("thread exec grants mutex poisoned");
        if let Some(commands) = grants.get_mut(thread_id) {
            commands.remove(command);
            if commands.is_empty() {
                grants.remove(thread_id);
            }
        }
    }
}

impl ExecAuthorizer for ThreadExecAuthorizer {
    fn authorize<'a>(
        &'a self,
        request: &'a ExecRequest,
        _current_policy: &'a SandboxPolicy,
        _violation: &'a str,
        context: &'a FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AuthorizationDecision>> + Send + 'a>> {
        Box::pin(async move {
            let grants = self
                .grants
                .lock()
                .expect("thread exec grants mutex poisoned");
            let decision = if let Some(policy) = grants
                .get(&context.thread_id)
                .and_then(|commands| commands.get(&request.command))
            {
                AuthorizationDecision::Allow {
                    policy: policy.clone(),
                }
            } else {
                AuthorizationDecision::RequireApproval {
                    reason: "no grant exists for this command in the target thread".to_string(),
                }
            };
            Ok(decision)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shell {
    Sh,
    Bash,
    Zsh,
    Fish,
}

impl Shell {
    pub fn detect() -> Self {
        match std::env::var("SHELL").ok().as_deref() {
            Some(path) if path.ends_with("/bash") => Self::Bash,
            Some(path) if path.ends_with("/zsh") => Self::Zsh,
            Some(path) if path.ends_with("/fish") => Self::Fish,
            _ => Self::Sh,
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::Sh => "/bin/sh",
            Self::Bash => "/bin/bash",
            Self::Zsh => "/bin/zsh",
            Self::Fish => "/usr/bin/fish",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Sh => "sh",
            Self::Bash => "bash",
            Self::Zsh => "zsh",
            Self::Fish => "fish",
        }
    }
}

#[derive(Clone)]
pub struct ExecCommandConfig {
    pub cwd: PathBuf,
    pub shell: Shell,
    pub sandbox: Arc<dyn SandboxBackend>,
    pub policy: SandboxPolicy,
    pub authorizer: Arc<dyn ExecAuthorizer>,
    pub approval_handling: ApprovalHandling,
    pub environment: BTreeMap<String, String>,
    pub supported_commands: Option<Vec<String>>,
    pub max_output_bytes: usize,
    pub default_timeout: Duration,
}

impl ExecCommandConfig {
    pub fn new(
        cwd: impl Into<PathBuf>,
        sandbox: Arc<dyn SandboxBackend>,
        policy: SandboxPolicy,
        authorizer: Arc<dyn ExecAuthorizer>,
    ) -> Self {
        let mut environment = BTreeMap::new();
        if let Some(path) = std::env::var_os("PATH") {
            environment.insert("PATH".to_string(), path.to_string_lossy().into_owned());
        }
        Self {
            cwd: cwd.into(),
            shell: Shell::detect(),
            sandbox,
            policy,
            authorizer,
            approval_handling: ApprovalHandling::ReturnToolError,
            environment,
            supported_commands: None,
            max_output_bytes: 20 * 1024 * 1024,
            default_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_approval_handling(mut self, handling: ApprovalHandling) -> Self {
        self.approval_handling = handling;
        self
    }

    pub fn with_shell(mut self, shell: Shell) -> Self {
        self.shell = shell;
        self
    }

    pub fn with_supported_commands(mut self, commands: Vec<String>) -> Self {
        self.supported_commands = Some(commands);
        self
    }
}

pub struct ExecCommandTool {
    config: ExecCommandConfig,
}

impl ExecCommandTool {
    pub fn new(config: ExecCommandConfig) -> Self {
        Self { config }
    }
}

impl AgentFunction for ExecCommandTool {
    fn spec(&self) -> FunctionSpec {
        let supported_hint = self
            .config
            .supported_commands
            .as_ref()
            .filter(|commands| !commands.is_empty())
            .map(|commands| format!(" Supported command examples: {}.", commands.join(", ")))
            .unwrap_or_default();
        FunctionSpec {
            name: "exec_command".to_string(),
            description: format!(
                "Run a command using the {} shell in the configured working directory. Commands are subject to authorization and sandbox policy. A non-zero exit code is an execution result, not a tool failure.{}",
                self.config.shell.name(),
                supported_hint
            ),
            parameters: json!({
                "type": "object",
                "required": ["cmd"],
                "properties": {
                    "cmd": { "type": "string", "description": "Shell command to run." },
                    "timeout_ms": { "type": "integer", "minimum": 1, "description": "Optional timeout in milliseconds." }
                },
                "additionalProperties": false
            }),
        }
    }

    fn call<'a>(
        &'a self,
        args: Value,
        context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move { self.execute(args, context).await })
    }
}

impl ExecCommandTool {
    async fn execute(
        &self,
        args: Value,
        mut context: FunctionContext,
    ) -> Result<FunctionExecution> {
        let command = args.get("cmd").and_then(Value::as_str).ok_or_else(|| {
            AgentError::InvalidFunctionArguments {
                name: "exec_command".to_string(),
                message: "missing string field: cmd".to_string(),
            }
        })?;
        if command.trim().is_empty() {
            return Err(AgentError::InvalidFunctionArguments {
                name: "exec_command".to_string(),
                message: "cmd must not be empty".to_string(),
            });
        }
        let timeout = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map(Duration::from_millis)
            .unwrap_or(self.config.default_timeout);
        let request = ExecRequest {
            command: command.to_string(),
            cwd: self.config.cwd.clone(),
            shell: self.config.shell,
            timeout,
        };

        let preflight_request = self.sandbox_request(
            &request,
            self.config.policy.clone(),
            CancellationToken::new(),
        );
        let preflight = self
            .config
            .sandbox
            .preflight(&preflight_request)
            .map_err(|error| tool_error(format!("sandbox preflight failed: {error}")))?;
        if let SandboxPreflight::PolicyViolation { reason } = preflight {
            let decision = self
                .config
                .authorizer
                .authorize(&request, &self.config.policy, &reason, &context)
                .await
                .map_err(|error| tool_error(format!("authorization lookup failed: {error}")))?;
            return self
                .handle_policy_violation(request, reason, decision, &mut context)
                .await;
        }
        let output = self
            .run_sandbox(
                &request,
                self.config.policy.clone(),
                &mut context.abort_signal,
            )
            .await?;
        let output = self.limit_output(output);
        if let SandboxStatus::PolicyViolation { reason } = &output.status {
            return Err(tool_error(format!(
                "sandbox policy violation after command launch: {reason}; command was not retried"
            )));
        }
        Ok(self.completed_execution(&request, output))
    }
}

impl ExecCommandTool {
    async fn run_sandbox(
        &self,
        request: &ExecRequest,
        policy: SandboxPolicy,
        abort_signal: &mut TurnAbortSignal,
    ) -> Result<SandboxOutput> {
        let cancellation = CancellationToken::new();
        let sandbox_request = self.sandbox_request(request, policy, cancellation.clone());
        let sandbox = self.config.sandbox.clone();
        let execution = async move { sandbox.execute(sandbox_request).await };
        tokio::pin!(execution);
        let (result, timed_out) = tokio::select! {
            result = &mut execution => (result, false),
            () = abort_signal.wait_cancelled() => {
                cancellation.cancel();
                (execution.await, false)
            }
            () = tokio::time::sleep(request.timeout) => {
                cancellation.cancel();
                (execution.await, true)
            }
        };
        let mut output = result.map_err(|error| tool_error(error.to_string()))?;
        if timed_out && matches!(&output.status, SandboxStatus::Cancelled) {
            output.status = SandboxStatus::TimedOut;
        }
        Ok(output)
    }

    async fn handle_policy_violation(
        &self,
        request: ExecRequest,
        violation: String,
        decision: AuthorizationDecision,
        context: &mut FunctionContext,
    ) -> Result<FunctionExecution> {
        match decision {
            AuthorizationDecision::Deny { reason } => Err(tool_error(format!(
                "command denied by authorization policy: {reason}; sandbox violation: {violation}"
            ))),
            AuthorizationDecision::RequireApproval { reason } => {
                if self.config.approval_handling == ApprovalHandling::ReturnToolError {
                    return Err(tool_error(format!(
                        "command requires approval: {reason}; sandbox violation: {violation}"
                    )));
                }
                let suspension = Suspension {
                    id: new_id("suspension"),
                    kind: SuspensionKind::HumanApproval,
                    payload: json!({
                        "tool": "exec_command",
                        "thread_id": context.thread_id,
                        "call_id": context.call_id,
                        "command": request.command,
                        "cwd": request.cwd,
                        "shell": request.shell.name(),
                        "reason": reason,
                        "sandbox_violation": violation,
                        "deferred": true,
                        "attempted": false,
                        "executed": false
                    }),
                };
                Ok(FunctionExecution::SuspendedBeforeExecution { suspension })
            }
            AuthorizationDecision::Allow { policy } => {
                if policy == self.config.policy {
                    return Err(tool_error(
                        "authorization allowed the command without changing the sandbox policy, but the same policy already rejected it".to_string(),
                    ));
                }
                let retry = self
                    .run_sandbox(&request, policy, &mut context.abort_signal)
                    .await?;
                let retry = self.limit_output(retry);
                if let SandboxStatus::PolicyViolation { reason } = &retry.status {
                    return Err(tool_error(format!(
                        "command remained blocked after authorization: {reason}"
                    )));
                }
                Ok(self.completed_execution(&request, retry))
            }
        }
    }

    fn limit_output(&self, mut output: SandboxOutput) -> SandboxOutput {
        if output.stdout.len() > self.config.max_output_bytes {
            output.stdout.truncate(self.config.max_output_bytes);
            output.stdout_truncated = true;
        }
        if output.stderr.len() > self.config.max_output_bytes {
            output.stderr.truncate(self.config.max_output_bytes);
            output.stderr_truncated = true;
        }
        output
    }

    fn sandbox_request(
        &self,
        request: &ExecRequest,
        policy: SandboxPolicy,
        cancellation: CancellationToken,
    ) -> SandboxRequest {
        SandboxRequest {
            program: request.shell.path().into(),
            args: vec!["-lc".to_string(), request.command.clone()],
            cwd: request.cwd.clone(),
            environment: self.config.environment.clone(),
            cancellation,
            policy,
        }
    }

    fn completed_execution(
        &self,
        request: &ExecRequest,
        output: SandboxOutput,
    ) -> FunctionExecution {
        let status = status_json(output.status.clone());
        let (outcome, executed) = match &output.status {
            SandboxStatus::Exited { .. } | SandboxStatus::Signaled { .. } => ("executed", true),
            SandboxStatus::TimedOut => ("timed_out", false),
            SandboxStatus::Cancelled => ("cancelled", false),
            SandboxStatus::PolicyViolation { .. } => ("policy_violation", false),
        };
        FunctionExecution::Completed {
            output: json!({
                "outcome": outcome,
                "executed": executed,
                "cwd": request.cwd,
                "cmd": request.command,
                "shell": request.shell.name(),
                "status": status,
                "success": matches!(&output.status, SandboxStatus::Exited { code: 0 }),
                "stdout": String::from_utf8_lossy(&output.stdout),
                "stderr": String::from_utf8_lossy(&output.stderr),
                "stdout_truncated": output.stdout_truncated,
                "stderr_truncated": output.stderr_truncated,
                "warnings": output.warnings.iter().map(|warning| warning.message.clone()).collect::<Vec<_>>(),
                "duration_ms": output.duration.as_millis()
            }),
        }
    }
}

fn status_json(status: SandboxStatus) -> Value {
    match status {
        SandboxStatus::Exited { code } => json!({ "kind": "exited", "code": code }),
        SandboxStatus::Signaled { signal } => json!({ "kind": "signaled", "signal": signal }),
        SandboxStatus::TimedOut => json!({ "kind": "timed_out" }),
        SandboxStatus::Cancelled => json!({ "kind": "cancelled" }),
        SandboxStatus::PolicyViolation { reason } => {
            json!({ "kind": "policy_violation", "reason": reason })
        }
    }
}

fn tool_error(message: String) -> AgentError {
    AgentError::Function {
        name: "exec_command".to_string(),
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::{AuthorizationDecision, ExecAuthorizer, ExecCommandTool, ExecRequest};
    use crate::sandbox::{
        SandboxBackend, SandboxOutput, SandboxPolicy, SandboxPreflight, SandboxStatus,
    };
    use lite_agent_kernel::projection::ThreadProjection;
    use lite_agent_runtime::{
        turn_abort_pair, AgentFunction, FunctionContext, FunctionExecution, Result,
    };
    use serde_json::json;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::time::Duration;

    struct DenyAuthorizer;
    impl ExecAuthorizer for DenyAuthorizer {
        fn authorize<'a>(
            &'a self,
            _: &'a ExecRequest,
            _: &'a SandboxPolicy,
            _: &'a str,
            _: &'a FunctionContext,
        ) -> Pin<Box<dyn Future<Output = Result<AuthorizationDecision>> + Send + 'a>> {
            Box::pin(async {
                Ok(AuthorizationDecision::Deny {
                    reason: "test".to_string(),
                })
            })
        }
    }

    struct FakeBackend {
        violation: bool,
    }
    impl SandboxBackend for FakeBackend {
        fn name(&self) -> &str {
            "fake"
        }
        fn preflight(
            &self,
            _request: &crate::sandbox::SandboxRequest,
        ) -> crate::sandbox::SandboxResult<SandboxPreflight> {
            if self.violation {
                Ok(SandboxPreflight::PolicyViolation {
                    reason: "test policy violation".to_string(),
                })
            } else {
                Ok(SandboxPreflight::Allowed)
            }
        }
        fn resolve_policy(
            &self,
            policy: &SandboxPolicy,
        ) -> crate::sandbox::SandboxResult<crate::sandbox::SandboxPolicyResolution> {
            Ok(crate::sandbox::SandboxPolicyResolution {
                requested: policy.clone(),
                effective: crate::sandbox::EffectiveSandboxPolicy {
                    filesystem: policy.filesystem.requested.clone(),
                    network: policy.network.requested,
                    process: policy.process.requested,
                    identity: policy.identity.requested,
                },
                warnings: Vec::new(),
            })
        }
        fn execute<'a>(
            &'a self,
            _request: crate::sandbox::SandboxRequest,
        ) -> Pin<Box<dyn Future<Output = crate::sandbox::SandboxResult<SandboxOutput>> + Send + 'a>>
        {
            let violation = self.violation;
            Box::pin(async move {
                Ok(SandboxOutput {
                    status: if violation {
                        SandboxStatus::PolicyViolation {
                            reason: "operation not permitted".to_string(),
                        }
                    } else {
                        SandboxStatus::Exited { code: 7 }
                    },
                    stdout: b"out".to_vec(),
                    stderr: b"err".to_vec(),
                    warnings: Vec::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    duration: Duration::from_millis(1),
                })
            })
        }
    }

    fn context() -> FunctionContext {
        let (_, abort_signal) = turn_abort_pair();
        FunctionContext {
            thread_id: "thread".to_string(),
            turn_id: "turn".to_string(),
            call_id: "call".to_string(),
            projection: ThreadProjection::default(),
            abort_signal,
        }
    }

    #[tokio::test]
    async fn reports_non_zero_exit_as_execution_result() {
        let config = super::ExecCommandConfig::new(
            "/tmp",
            Arc::new(FakeBackend { violation: false }),
            SandboxPolicy::default(),
            Arc::new(DenyAuthorizer),
        );
        let execution = ExecCommandTool::new(config)
            .call(json!({"cmd": "false"}), context())
            .await
            .unwrap();
        let FunctionExecution::Completed { output } = execution else {
            panic!("expected completed")
        };
        assert_eq!(output["success"], false);
        assert_eq!(output["status"]["code"], 7);
    }

    #[tokio::test]
    async fn authorization_denial_prevents_execution() {
        let config = super::ExecCommandConfig::new(
            "/tmp",
            Arc::new(FakeBackend { violation: true }),
            SandboxPolicy::workspace_read_write("/tmp"),
            Arc::new(DenyAuthorizer),
        );
        let error = ExecCommandTool::new(config)
            .call(json!({"cmd": "echo no"}), context())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("command denied"));
    }
}
