use lite_agent::{
    builtin_registry, Agent, AgentConfig, ChatCompletionsClient, JsonFileThreadStore, ModelConfig,
    TurnOutcome,
};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Debug)]
struct ReplArgs {
    thread: Option<String>,
    state_dir: PathBuf,
    model: String,
    base_url: String,
    api_key: String,
}

#[tokio::main]
async fn main() -> lite_agent::Result<()> {
    let args = parse_args()?;
    let thread_id = args
        .thread
        .unwrap_or_else(|| lite_agent::events::new_id("thread"));
    let store = Arc::new(JsonFileThreadStore::new(args.state_dir));
    let model_client = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url: args.base_url,
        api_key: args.api_key,
        model: args.model,
    }));
    let agent = Agent::new(
        AgentConfig::default(),
        store,
        model_client,
        builtin_registry(),
    );

    run_repl(agent, thread_id).await
}

fn parse_args() -> lite_agent::Result<ReplArgs> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "repl".to_string());
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
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--thread" => parsed.thread = args.next(),
            "--state-dir" => parsed.state_dir = PathBuf::from(args.next().unwrap_or_default()),
            "--model" => parsed.model = args.next().unwrap_or_default(),
            "--base-url" => parsed.base_url = args.next().unwrap_or_default(),
            "--api-key" => parsed.api_key = args.next().unwrap_or_default(),
            "--help" | "-h" => {
                return Err(lite_agent::AgentError::Model(help_text()));
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

    Ok(parsed)
}

fn help_text() -> String {
    concat!(
        "usage: lite-agent repl [--thread ID] [--state-dir PATH] ",
        "[--model NAME] [--base-url URL] [--api-key KEY]"
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

        match agent.run_turn(&thread_id, input.to_string()).await? {
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
