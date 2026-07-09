use lite_agent_core::functions::{FunctionExecution, SimpleFunction};
use lite_agent_core::model::FunctionSpec;
use lite_agent_core::FunctionContext;
use lite_agent_core::{
    builtin_registry, init_file_logging, turn_abort_pair, Agent, AgentConfig,
    ChatCompletionsClient, FunctionRegistry, JsonFileThreadStore, ModelConfig, ThreadStore,
    TurnModelEvent, TurnOutcome, TurnStateEvent, TurnStreamEvent,
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde_json::json;
use std::env;
use std::io::{self as std_io, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::process::Command as TokioCommand;
use tokio::time::{timeout, Duration};

#[derive(Debug)]
struct ReplArgs {
    thread: Option<String>,
    state_dir: PathBuf,
    model: String,
    base_url: String,
    api_key: String,
    reasoning_effort: String,
    command_cwd: PathBuf,
}

#[derive(Debug)]
enum Command {
    Help,
    Repl(ReplArgs),
}

#[tokio::main]
async fn main() -> lite_agent_core::Result<()> {
    let Command::Repl(args) = parse_args()? else {
        println!("{}", help_text());
        return Ok(());
    };
    let thread_id = args
        .thread
        .unwrap_or_else(|| lite_agent_core::events::new_id("thread"));
    let _logging_guard = init_file_logging(&args.state_dir)?;
    let store = Arc::new(JsonFileThreadStore::new(&args.state_dir));
    let model_client = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url: args.base_url,
        api_key: args.api_key,
        model: args.model,
        reasoning_effort: args.reasoning_effort,
    }));
    let agent = Agent::new(
        AgentConfig::default(),
        store.clone(),
        model_client,
        example_registry(args.command_cwd),
    );

    run_repl(agent, store, thread_id).await
}

fn example_registry(command_cwd: PathBuf) -> FunctionRegistry {
    let mut registry = builtin_registry();
    registry.register(exec_command_function(command_cwd));
    registry
}

fn exec_command_function(command_cwd: PathBuf) -> impl lite_agent_core::functions::AgentFunction {
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
                    .ok_or_else(|| lite_agent_core::AgentError::InvalidFunctionArguments {
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
) -> lite_agent_core::Result<serde_json::Value> {
    let mut command = TokioCommand::new("/bin/zsh");
    command.arg("-lc").arg(cmd).current_dir(cwd);

    let output = timeout(Duration::from_millis(timeout_ms), command.output())
        .await
        .map_err(|_| lite_agent_core::AgentError::Function {
            name: "exec_command".to_string(),
            message: format!("command timed out after {timeout_ms}ms"),
        })?
        .map_err(|error| lite_agent_core::AgentError::Function {
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

fn parse_args() -> lite_agent_core::Result<Command> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "repl".to_string());
    if command == "--help" || command == "-h" {
        return Ok(Command::Help);
    }
    if command != "repl" {
        return Err(lite_agent_core::AgentError::Model(format!(
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
        reasoning_effort: env::var("LITE_AGENT_REASONING_EFFORT")
            .unwrap_or_else(|_| ModelConfig::default_reasoning_effort()),
        command_cwd: env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--thread" => parsed.thread = args.next(),
            "--state-dir" => parsed.state_dir = PathBuf::from(args.next().unwrap_or_default()),
            "--model" => parsed.model = args.next().unwrap_or_default(),
            "--base-url" => parsed.base_url = args.next().unwrap_or_default(),
            "--api-key" => parsed.api_key = args.next().unwrap_or_default(),
            "--reasoning-effort" => parsed.reasoning_effort = args.next().unwrap_or_default(),
            "--command-cwd" => parsed.command_cwd = PathBuf::from(args.next().unwrap_or_default()),
            "--help" | "-h" => {
                return Ok(Command::Help);
            }
            other => {
                return Err(lite_agent_core::AgentError::Model(format!(
                    "unknown argument: {other}"
                )));
            }
        }
    }

    if parsed.model.is_empty() {
        return Err(lite_agent_core::AgentError::Model(
            "missing --model or LITE_AGENT_MODEL".to_string(),
        ));
    }
    if parsed.api_key.is_empty() {
        return Err(lite_agent_core::AgentError::Model(
            "missing --api-key or LITE_AGENT_API_KEY".to_string(),
        ));
    }
    if parsed.reasoning_effort.is_empty() {
        parsed.reasoning_effort = ModelConfig::default_reasoning_effort();
    }

    Ok(Command::Repl(parsed))
}

fn help_text() -> String {
    concat!(
        "usage: lite-agent-repl repl [--thread ID] [--state-dir PATH] ",
        "[--model NAME] [--base-url URL] [--api-key KEY] ",
        "[--reasoning-effort VALUE] [--command-cwd PATH]"
    )
    .to_string()
}

async fn run_repl(
    agent: Agent,
    store: Arc<JsonFileThreadStore>,
    thread_id: String,
) -> lite_agent_core::Result<()> {
    let mut editor = DefaultEditor::new().map_err(|error| {
        lite_agent_core::AgentError::Model(format!("failed to initialize REPL: {error}"))
    })?;

    println!("thread: {thread_id}");

    loop {
        let line = match editor.readline("> ") {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => break,
            Err(ReadlineError::Eof) => break,
            Err(error) => {
                return Err(lite_agent_core::AgentError::Model(format!(
                    "failed to read input: {error}"
                )));
            }
        };
        let input = line.trim();
        if input.eq_ignore_ascii_case("/exit") || input.eq_ignore_ascii_case("/quit") {
            break;
        }
        if input.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(input);

        let render_state = Arc::new(Mutex::new(StreamRenderState::default()));
        let render_state_for_events = render_state.clone();
        let (abort_handle, abort_signal) = turn_abort_pair();
        let turn = agent.run_turn_stream_abortable(
            &thread_id,
            input.to_string(),
            abort_signal,
            move |event| {
                if let Ok(mut state) = render_state_for_events.lock() {
                    print_stream_event(event, &mut state);
                }
            },
        );
        tokio::pin!(turn);
        let outcome = tokio::select! {
            result = &mut turn => result?,
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(lite_agent_core::AgentError::Io)?;
                abort_handle.abort();
                let outcome = turn.await?;
                println!();
                outcome
            }
        };

        match outcome {
            TurnOutcome::AssistantMessage { text } => {
                let state = render_state.lock().expect("render state");
                if !state.assistant_started {
                    drop(state);
                    println!("{text}");
                } else if state.line_open {
                    drop(state);
                    println!();
                }
            }
            TurnOutcome::WaitingForUser { prompt, .. } => {
                println!("{prompt}");
            }
            TurnOutcome::Failed { error } => {
                println!("turn failed: {error}");
            }
            TurnOutcome::Aborted { reason: _ } => {
                continue;
            }
        }
    }

    print_thread_token_usage(store, &thread_id).await?;
    Ok(())
}

async fn print_thread_token_usage(
    store: Arc<JsonFileThreadStore>,
    thread_id: &str,
) -> lite_agent_core::Result<()> {
    match store.load(thread_id).await {
        Ok(thread) => {
            println!("[thread tokens] {}", thread.token_usage);
        }
        Err(lite_agent_core::AgentError::ThreadNotFound(_)) => {
            println!("[thread tokens] input=0, cached_input=0, output=0, total=0");
        }
        Err(error) => return Err(error),
    }
    Ok(())
}

#[derive(Default)]
struct StreamRenderState {
    assistant_started: bool,
    line_open: bool,
}

fn print_stream_event(event: TurnStreamEvent, state: &mut StreamRenderState) {
    match event {
        TurnStreamEvent::State(TurnStateEvent::TurnStarted { turn_id, .. }) => {
            print_process_line(state, &format!("[turn started] {turn_id}"));
        }
        TurnStreamEvent::Model(TurnModelEvent::RequestStarted { iteration }) => {
            print_process_line(state, &format!("[model] iteration {iteration}"));
        }
        TurnStreamEvent::State(TurnStateEvent::FunctionCallsRequested { calls }) => {
            for call in calls {
                print_process_line(
                    state,
                    &format!("[function requested] {} {}", call.name, call.arguments),
                );
            }
        }
        TurnStreamEvent::State(TurnStateEvent::FunctionStarted { call_id, name }) => {
            print_process_line(state, &format!("[function started] {name} ({call_id})"));
        }
        TurnStreamEvent::State(TurnStateEvent::FunctionCompleted { call_id, name }) => {
            print_process_line(state, &format!("[function completed] {name} ({call_id})"));
        }
        TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
            call_id,
            name,
            error,
        }) => {
            print_process_line(
                state,
                &format!("[function failed] {name} ({call_id}): {error}"),
            );
        }
        TurnStreamEvent::State(TurnStateEvent::WaitingForUser { prompt, .. }) => {
            print_process_line(state, &format!("[waiting for user] {prompt}"));
        }
        TurnStreamEvent::State(TurnStateEvent::TurnFailed { error }) => {
            print_process_line(state, &format!("[turn failed] {error}"));
        }
        TurnStreamEvent::State(TurnStateEvent::TurnAborted { reason }) => {
            print_process_line(state, &format!("[turn aborted] {reason}"));
        }
        TurnStreamEvent::Model(TurnModelEvent::AssistantDelta { text }) => {
            if !text.is_empty() {
                state.assistant_started = true;
                state.line_open = !text.ends_with('\n');
                print!("{text}");
                let _ = std_io::stdout().flush();
            }
        }
        TurnStreamEvent::Runtime(event) => {
            print_process_line(state, &format!("[{}] {}", event.source, event.message));
        }
        TurnStreamEvent::Model(TurnModelEvent::AssistantMessage { .. })
        | TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage { .. })
        | TurnStreamEvent::State(TurnStateEvent::TurnCompleted { .. }) => {}
    }
}

fn print_process_line(state: &mut StreamRenderState, line: &str) {
    if state.line_open {
        println!();
        state.line_open = false;
    }
    println!("{line}");
    let _ = std_io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::{print_stream_event, StreamRenderState};
    use lite_agent_core::{TurnModelEvent, TurnStreamEvent};

    #[test]
    fn assistant_delta_marks_line_open_until_newline() {
        let mut state = StreamRenderState::default();

        print_stream_event(
            TurnStreamEvent::Model(TurnModelEvent::AssistantDelta {
                text: "你好".to_string(),
            }),
            &mut state,
        );
        assert!(state.assistant_started);
        assert!(state.line_open);

        print_stream_event(
            TurnStreamEvent::Model(TurnModelEvent::AssistantDelta {
                text: "\n".to_string(),
            }),
            &mut state,
        );
        assert!(state.assistant_started);
        assert!(!state.line_open);
    }
}
