use crate::error::{AgentError, Result};
use crate::projection::{ChatMessage, ThreadProjection};
use chrono::{Local, Utc};
use std::sync::Arc;

pub struct ContextBuildInput<'a> {
    pub projection: &'a ThreadProjection,
    pub system_prompt: &'a str,
    pub runtime_context: Option<&'a str>,
}

pub trait ContextBuilder: Send + Sync {
    fn build(&self, input: ContextBuildInput<'_>) -> Result<Vec<ChatMessage>>;
}

pub trait TokenEstimator: Send + Sync {
    fn estimate(&self, messages: &[ChatMessage]) -> usize;
}

#[derive(Debug, Default)]
pub struct ApproximateTokenEstimator;

impl TokenEstimator for ApproximateTokenEstimator {
    fn estimate(&self, messages: &[ChatMessage]) -> usize {
        serde_json::to_string(messages)
            .map(|encoded| encoded.len().div_ceil(4))
            .unwrap_or(usize::MAX)
    }
}

pub trait ContextCompactor: Send + Sync {
    fn compact(&self, omitted: &[ChatMessage]) -> Result<Option<ChatMessage>>;
}

pub struct CompactingContextBuilder {
    pub max_context_tokens: usize,
    pub estimator: Arc<dyn TokenEstimator>,
    pub compactor: Option<Arc<dyn ContextCompactor>>,
}

impl Default for CompactingContextBuilder {
    fn default() -> Self {
        Self {
            max_context_tokens: 32_000,
            estimator: Arc::new(ApproximateTokenEstimator),
            compactor: None,
        }
    }
}

impl CompactingContextBuilder {
    pub fn with_compactor<C>(mut self, compactor: C) -> Self
    where
        C: ContextCompactor + 'static,
    {
        self.compactor = Some(Arc::new(compactor));
        self
    }
}

impl ContextBuilder for CompactingContextBuilder {
    fn build(&self, input: ContextBuildInput<'_>) -> Result<Vec<ChatMessage>> {
        let mut system_content = input.system_prompt.to_string();
        if let Some(runtime_context) = input.runtime_context {
            let runtime_context = runtime_context.trim();
            if !runtime_context.is_empty() {
                system_content.push_str(
                    "\nCurrent turn runtime context. This host-supplied context applies only to this model request and is not durable thread state:\n",
                );
                system_content.push_str(runtime_context);
            }
        }
        system_content.push_str(&format!(
            "\nCurrent time context: local={}, utc={}. Use this as the current date/time for time-sensitive answers.",
            Local::now().format("%Y-%m-%d %H:%M:%S %:z"),
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        if let Some(goal) = &input.projection.goal {
            system_content.push_str(&format!(
                "\nCurrent thread goal: objective={}, status={:?}, notes={}",
                goal.objective,
                goal.status,
                goal.notes.as_deref().unwrap_or("")
            ));
        }

        let system = ChatMessage::System {
            content: system_content,
        };
        let system_tokens = self.estimator.estimate(std::slice::from_ref(&system));
        if system_tokens >= self.max_context_tokens {
            return Err(AgentError::ContextWindowExceeded {
                estimated: system_tokens,
                limit: self.max_context_tokens,
            });
        }
        let budget = self.max_context_tokens - system_tokens;
        let groups = message_groups(&input.projection.conversation);
        let mut selected = Vec::new();
        let mut selected_tokens: usize = 0;
        let mut first_omitted_group = groups.len();

        for (index, group) in groups.iter().enumerate().rev() {
            let group_tokens = self.estimator.estimate(group);
            if selected.is_empty() && group_tokens > budget {
                return Err(AgentError::ContextWindowExceeded {
                    estimated: group_tokens + system_tokens,
                    limit: self.max_context_tokens,
                });
            }
            if selected_tokens.saturating_add(group_tokens) > budget {
                first_omitted_group = index;
                break;
            }
            selected.splice(0..0, group.iter().cloned());
            selected_tokens = selected_tokens.saturating_add(group_tokens);
            first_omitted_group = index;
        }

        let omitted: Vec<ChatMessage> = groups[..first_omitted_group]
            .iter()
            .flatten()
            .cloned()
            .collect();
        let summary = match (&self.compactor, omitted.is_empty()) {
            (Some(compactor), false) => compactor.compact(&omitted)?,
            _ => None,
        };

        let mut messages = vec![system];
        if let Some(summary) = summary {
            let summary_tokens = self.estimator.estimate(std::slice::from_ref(&summary));
            if selected_tokens.saturating_add(summary_tokens) <= budget {
                messages.push(summary);
            }
        }
        messages.extend(selected);
        Ok(messages)
    }
}

fn message_groups(messages: &[ChatMessage]) -> Vec<Vec<ChatMessage>> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let mut group = vec![messages[index].clone()];
        index += 1;
        if matches!(
            group[0],
            ChatMessage::Assistant { ref tool_calls, .. } if !tool_calls.is_empty()
        ) {
            while index < messages.len() && matches!(messages[index], ChatMessage::Tool { .. }) {
                group.push(messages[index].clone());
                index += 1;
            }
        }
        groups.push(group);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, CompactingContextBuilder, ContextBuildInput, ContextBuilder};
    use crate::projection::ThreadProjection;

    #[test]
    fn keeps_recent_messages_within_budget() {
        let projection = ThreadProjection {
            conversation: vec![
                ChatMessage::User {
                    content: "old".to_string(),
                },
                ChatMessage::User {
                    content: "new".to_string(),
                },
            ],
            ..ThreadProjection::default()
        };
        let builder = CompactingContextBuilder {
            max_context_tokens: 100,
            ..Default::default()
        };
        let messages = builder
            .build(ContextBuildInput {
                projection: &projection,
                system_prompt: "system",
                runtime_context: None,
            })
            .expect("context");
        assert!(messages.iter().any(|message| matches!(
            message,
            ChatMessage::User { content } if content == "new"
        )));
    }
}
