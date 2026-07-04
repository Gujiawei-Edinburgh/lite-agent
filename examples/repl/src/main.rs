use lite_agent::functions::{FunctionExecution, SimpleFunction};
use lite_agent::model::FunctionSpec;
use lite_agent::FunctionContext;
use lite_agent::{
    builtin_registry, init_file_logging, Agent, AgentConfig, ChatCompletionsClient,
    FunctionRegistry, JsonFileThreadStore, ModelConfig, TurnOutcome, TurnStreamEvent,
};
use serde_json::json;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command as TokioCommand;
use tokio::time::{timeout, Duration};

#[derive(Debug)]
struct ReplArgs {
    thread: Option<String>,
    state_dir: PathBuf,
    model: String,
    base_url: String,
    api_key: String,
    command_cwd: PathBuf,
}

#[derive(Debug)]
enum Command {
    Help,
    Repl(ReplArgs),
}

#[tokio::main]
async fn main() -> lite_agent::Result<()> {
    let Command::Repl(args) = parse_args()? else {
        println!("{}", help_text());
        return Ok(());
    };
    let thread_id = args
        .thread
        .unwrap_or_else(|| lite_agent::events::new_id("thread"));
    let _logging_guard = init_file_logging(&args.state_dir)?;
    let store = Arc::new(JsonFileThreadStore::new(&args.state_dir));
    let model_client = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url: args.base_url,
        api_key: args.api_key,
        model: args.model,
    }));
    let agent = Agent::new(
        AgentConfig::default(),
        store,
        model_client,
        example_registry(args.command_cwd),
    );

    run_repl(agent, thread_id).await
}

fn example_registry(command_cwd: PathBuf) -> FunctionRegistry {
    let mut registry = builtin_registry();
    registry.register(exec_command_function(command_cwd));
    registry
}

fn exec_command_function(command_cwd: PathBuf) -> impl lite_agent::functions::AgentFunction {
    SimpleFunction::new(
        FunctionSpec {
            name: "exec_command".to_string(),
            description: concat!(
                "Run a shell command for local project testing. ",
                "Use it for multi-step tasks such as listing directories, creating files, ",
                "and writing markdown inside the configured working directory."
            )
            .to_string(),
            parameters: json!({
                "type": "object",
                "required": ["cmd"],
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to run."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds. Defaults to 30000."
                    }
                },
                "additionalProperties": false
            }),
        },
        move |args: serde_json::Value, _context: FunctionContext| {
            let cwd = command_cwd.clone();
            async move {
                let cmd = args
                    .get("cmd")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| lite_agent::AgentError::InvalidFunctionArguments {
                        name: "exec_command".to_string(),
                        message: "missing string field: cmd".to_string(),
                    })?
                    .to_string();
                let timeout_ms = args
                    .get("timeout_ms")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(30_000);

                let output = run_shell_command(&cwd, &cmd, timeout_ms).await?;
                Ok(FunctionExecution::Completed {
                    output,
                    thread_update: None,
                    extra_items: Vec::new(),
                })
            }
        },
    )
}

async fn run_shell_command(
    cwd: &Path,
    cmd: &str,
    timeout_ms: u64,
) -> lite_agent::Result<serde_json::Value> {
    let mut command = TokioCommand::new("/bin/zsh");
    command.arg("-lc").arg(cmd).current_dir(cwd);

    let output = timeout(Duration::from_millis(timeout_ms), command.output())
        .await
        .map_err(|_| lite_agent::AgentError::Function {
            name: "exec_command".to_string(),
            message: format!("command timed out after {timeout_ms}ms"),
        })?
        .map_err(|error| lite_agent::AgentError::Function {
            name: "exec_command".to_string(),
            message: error.to_string(),
        })?;

    Ok(json!({
        "cwd": cwd.display().to_string(),
        "cmd": cmd,
        "exit_code": output.status.code(),
        "success": output.status.success(),
        "stdout": truncate_output(&String::from_utf8_lossy(&output.stdout)),
        "stderr": truncate_output(&String::from_utf8_lossy(&output.stderr)),
    }))
}

fn truncate_output(output: &str) -> String {
    const MAX_CHARS: usize = 20_000;
    if output.chars().count() <= MAX_CHARS {
        return output.to_string();
    }
    let mut truncated: String = output.chars().take(MAX_CHARS).collect();
    truncated.push_str("\n...[truncated]");
    truncated
}

fn parse_args() -> lite_agent::Result<Command> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "repl".to_string());
    if command == "--help" || command == "-h" {
        return Ok(Command::Help);
    }
    if command != "repl" {
        return Err(lite_agent::AgentError::Model(format!(
            "unsupported command: {command}. expected: repl"
        )));
    }

    let mut parsed = ReplArgs {
        thread: None,
        state_dir: PathBuf::from(".lite-agent"),
        model: env::var("LITE_AGENT_MODEL").unwrap_or_default(),
        base_url: env::var("LITE_AGENT_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        api_key: env::var("LITE_AGENT_API_KEY").unwrap_or_default(),
        command_cwd: env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--thread" => parsed.thread = args.next(),
            "--state-dir" => parsed.state_dir = PathBuf::from(args.next().unwrap_or_default()),
            "--model" => parsed.model = args.next().unwrap_or_default(),
            "--base-url" => parsed.base_url = args.next().unwrap_or_default(),
            "--api-key" => parsed.api_key = args.next().unwrap_or_default(),
            "--command-cwd" => parsed.command_cwd = PathBuf::from(args.next().unwrap_or_default()),
            "--help" | "-h" => {
                return Ok(Command::Help);
            }
            other => {
                return Err(lite_agent::AgentError::Model(format!(
                    "unknown argument: {other}"
                )));
            }
        }
    }

    if parsed.model.is_empty() {
        return Err(lite_agent::AgentError::Model(
            "missing --model or LITE_AGENT_MODEL".to_string(),
        ));
    }
    if parsed.api_key.is_empty() {
        return Err(lite_agent::AgentError::Model(
            "missing --api-key or LITE_AGENT_API_KEY".to_string(),
        ));
    }

    Ok(Command::Repl(parsed))
}

fn help_text() -> String {
    concat!(
        "usage: lite-agent-repl repl [--thread ID] [--state-dir PATH] ",
        "[--model NAME] [--base-url URL] [--api-key KEY] [--command-cwd PATH]"
    )
    .to_string()
}

async fn run_repl(agent: Agent, thread_id: String) -> lite_agent::Result<()> {
    let stdin = BufReader::new(io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = io::stdout();

    stdout
        .write_all(format!("thread: {thread_id}\n").as_bytes())
        .await?;
    stdout.write_all(b"> ").await?;
    stdout.flush().await?;

    while let Some(line) = lines.next_line().await? {
        let input = line.trim();
        if input.eq_ignore_ascii_case("/exit") || input.eq_ignore_ascii_case("/quit") {
            break;
        }
        if input.is_empty() {
            stdout.write_all(b"> ").await?;
            stdout.flush().await?;
            continue;
        }

        match agent
            .run_turn_stream(&thread_id, input.to_string(), print_stream_event)
            .await?
        {
            TurnOutcome::AssistantMessage { text } => {
                stdout.write_all(text.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
            }
            TurnOutcome::WaitingForUser { prompt, .. } => {
                stdout.write_all(prompt.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
            }
            TurnOutcome::Failed { error } => {
                stdout
                    .write_all(format!("turn failed: {error}\n").as_bytes())
                    .await?;
            }
        }

        stdout.write_all(b"> ").await?;
        stdout.flush().await?;
    }

    Ok(())
}

fn print_stream_event(event: TurnStreamEvent) {
    match event {
        TurnStreamEvent::TurnStarted { turn_id, .. } => {
            println!("[turn started] {turn_id}");
        }
        TurnStreamEvent::ModelRequestStarted { iteration } => {
            println!("[model] iteration {iteration}");
        }
        TurnStreamEvent::FunctionCallsRequested { calls } => {
            for call in calls {
                println!("[function requested] {} {}", call.name, call.arguments);
            }
        }
        TurnStreamEvent::FunctionStarted { call_id, name } => {
            println!("[function started] {name} ({call_id})");
        }
        TurnStreamEvent::FunctionCompleted { call_id, name } => {
            println!("[function completed] {name} ({call_id})");
        }
        TurnStreamEvent::FunctionFailed {
            call_id,
            name,
            error,
        } => {
            println!("[function failed] {name} ({call_id}): {error}");
        }
        TurnStreamEvent::WaitingForUser { prompt, .. } => {
            println!("[waiting for user] {prompt}");
        }
        TurnStreamEvent::TurnFailed { error } => {
            println!("[turn failed] {error}");
        }
        TurnStreamEvent::ModelMessageDelta { text } => {
            print!("{text}");
        }
        TurnStreamEvent::ModelMessage { .. } | TurnStreamEvent::TurnCompleted { .. } => {
            println!();
        }
    }
}
