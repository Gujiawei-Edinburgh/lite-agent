use crate::error::{AgentError, Result};
use crate::projection::{ChatMessage, ThreadProjection};
use crate::store::ThreadContextCache;
use chrono::{Local, Utc};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct ContextBuildInput<'a> {
    pub thread_id: &'a str,
    pub thread_version: u64,
    pub projection: &'a ThreadProjection,
    pub system_prompt: &'a str,
    pub runtime_context: Option<&'a str>,
    pub cached_context: Option<&'a ThreadContextCache>,
}

pub trait ContextBuilder: Send + Sync {
    fn build<'a>(
        &'a self,
        input: ContextBuildInput<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ContextBuildOutput>> + Send + 'a>>;
}

#[derive(Debug)]
pub struct ContextBuildOutput {
    pub messages: Vec<ChatMessage>,
    pub cache: Option<ThreadContextCache>,
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
    fn compact<'a>(
        &'a self,
        input: CompactionInput,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

pub struct CompactionInput {
    pub messages: Vec<ChatMessage>,
    pub previous_summary: Option<String>,
}

pub struct CompactingContextBuilder {
    pub max_context_tokens: usize,
    pub policy_version: String,
    pub estimator: Arc<dyn TokenEstimator>,
    pub compactor: Option<Arc<dyn ContextCompactor>>,
}

impl Default for CompactingContextBuilder {
    fn default() -> Self {
        Self {
            max_context_tokens: 32_000,
            policy_version: "compacting-v1".to_string(),
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
    fn build<'a>(
        &'a self,
        input: ContextBuildInput<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ContextBuildOutput>> + Send + 'a>> {
        Box::pin(async move {
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
            let groups = tool_call_blocks(&input.projection.conversation);
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
            let cache_is_current = input.cached_context.is_some_and(|cache| {
                cache.source_version == input.thread_version
                    && cache.policy_version == self.policy_version
                    && cache.covered_message_count == omitted.len()
            });
            let summary = if cache_is_current {
                input.cached_context.map(|cache| cache.summary.clone())
            } else if let Some(compactor) = &self.compactor {
                let previous_summary = input.cached_context.and_then(|cache| {
                    (cache.policy_version == self.policy_version
                        && cache.covered_message_count <= omitted.len())
                    .then(|| cache.summary.clone())
                });
                let start = input
                    .cached_context
                    .filter(|cache| cache.policy_version == self.policy_version)
                    .map(|cache| cache.covered_message_count.min(omitted.len()))
                    .unwrap_or(0);
                Some(
                    compactor
                        .compact(CompactionInput {
                            messages: omitted[start..].to_vec(),
                            previous_summary,
                        })
                        .await?,
                )
            } else {
                None
            };

            let mut messages = vec![system];
            if let Some(summary) = &summary {
                let summary_message = ChatMessage::System {
                    content: summary.clone(),
                };
                let summary_tokens = self
                    .estimator
                    .estimate(std::slice::from_ref(&summary_message));
                if selected_tokens.saturating_add(summary_tokens) <= budget {
                    messages.push(summary_message);
                }
            }
            messages.extend(selected);
            Ok(ContextBuildOutput {
                messages,
                cache: summary.map(|summary| ThreadContextCache {
                    thread_id: input.thread_id.to_string(),
                    source_version: input.thread_version,
                    policy_version: self.policy_version.clone(),
                    covered_message_count: omitted.len(),
                    summary,
                }),
            })
        })
    }
}

fn tool_call_blocks(messages: &[ChatMessage]) -> Vec<Vec<ChatMessage>> {
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

    #[tokio::test]
    async fn keeps_recent_messages_within_budget() {
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
                thread_id: "t",
                thread_version: 1,
                system_prompt: "system",
                runtime_context: None,
                cached_context: None,
            })
            .await
            .expect("context");
        assert!(messages.messages.iter().any(|message| matches!(
            message,
            ChatMessage::User { content } if content == "new"
        )));
    }
}
