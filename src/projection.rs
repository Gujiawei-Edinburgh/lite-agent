use crate::events::{GoalState, Thread, ToolResult, TurnId, TurnItemKind, TurnStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserInputRequest {
    pub request_id: String,
    pub prompt: String,
    pub turn_id: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletedFunctionCall {
    pub call_id: String,
    pub name: String,
    pub output: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ThreadProjection {
    pub messages_for_model: Vec<ChatMessage>,
    pub goal: Option<GoalState>,
    pub pending_user_input_request: Option<UserInputRequest>,
    pub completed_function_results: Vec<CompletedFunctionCall>,
    pub last_assistant_message: Option<String>,
    pub latest_turn_id: Option<TurnId>,
    pub active_turn_id: Option<TurnId>,
}

impl ThreadProjection {
    pub fn from_thread(thread: &Thread) -> Self {
        let mut projection = Self {
            goal: thread.goal.clone(),
            latest_turn_id: thread.turns.last().map(|turn| turn.id.clone()),
            active_turn_id: thread
                .turns
                .iter()
                .rev()
                .find(|turn| {
                    matches!(
                        turn.status,
                        TurnStatus::Running | TurnStatus::WaitingForUser
                    )
                })
                .map(|turn| turn.id.clone()),
            ..Self::default()
        };

        for turn in &thread.turns {
            for item in &turn.items {
                match &item.kind {
                    TurnItemKind::UserInput { text, response_to } => {
                        projection.messages_for_model.push(ChatMessage {
                            role: ChatRole::User,
                            content: text.clone(),
                            name: None,
                            tool_call_id: None,
                        });
                        if let Some(response_to) = response_to {
                            if projection
                                .pending_user_input_request
                                .as_ref()
                                .is_some_and(|request| request.request_id == *response_to)
                            {
                                projection.pending_user_input_request = None;
                            }
                        } else {
                            projection.pending_user_input_request = None;
                        }
                    }
                    TurnItemKind::ModelMessage { text } => {
                        projection.last_assistant_message = Some(text.clone());
                        projection.messages_for_model.push(ChatMessage {
                            role: ChatRole::Assistant,
                            content: text.clone(),
                            name: None,
                            tool_call_id: None,
                        });
                    }
                    TurnItemKind::ModelFunctionCall {
                        call_id,
                        name,
                        arguments,
                    } => {
                        projection.messages_for_model.push(ChatMessage {
                            role: ChatRole::Assistant,
                            content: format!("Function call requested: {name}({arguments})"),
                            name: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                    }
                    TurnItemKind::ToolOutput {
                        call_id,
                        name,
                        result,
                    } => match result {
                        ToolResult::Success { output } => {
                            projection
                                .completed_function_results
                                .push(CompletedFunctionCall {
                                    call_id: call_id.clone(),
                                    name: name.clone(),
                                    output: output.clone(),
                                });
                            projection.messages_for_model.push(ChatMessage {
                                role: ChatRole::User,
                                content: format!("Function {name} completed with output: {output}"),
                                name: None,
                                tool_call_id: None,
                            });
                        }
                        ToolResult::Error { error } => {
                            projection.messages_for_model.push(ChatMessage {
                                role: ChatRole::User,
                                content: format!(
                                    "Function {name} with call id {call_id} failed: {error}"
                                ),
                                name: None,
                                tool_call_id: None,
                            });
                        }
                    },
                    TurnItemKind::UserInputRequested { request_id, prompt } => {
                        projection.pending_user_input_request = Some(UserInputRequest {
                            request_id: request_id.clone(),
                            prompt: prompt.clone(),
                            turn_id: turn.id.clone(),
                        });
                    }
                    TurnItemKind::GoalUpdated { current, .. } => {
                        projection.goal = Some(current.clone());
                    }
                    TurnItemKind::TurnFailed { .. } | TurnItemKind::TurnAborted { .. } => {}
                }
            }
        }

        projection
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{
        GoalState, GoalStatus, Thread, Turn, TurnItem, TurnItemKind, TurnItemSource,
    };

    use super::ThreadProjection;

    #[test]
    fn copies_thread_goal() {
        let mut thread = Thread::new("t");
        thread.goal = Some(GoalState {
            objective: "ship".to_string(),
            status: GoalStatus::Active,
            notes: None,
        });

        let projection = ThreadProjection::from_thread(&thread);
        assert_eq!(
            projection.goal.as_ref().map(|goal| goal.objective.as_str()),
            Some("ship")
        );
    }

    #[test]
    fn tracks_pending_user_input_request() {
        let mut thread = Thread::new("t");
        let mut first_turn = Turn::new();
        first_turn.push_item(TurnItem::new(
            TurnItemSource::Runtime,
            TurnItemKind::UserInputRequested {
                request_id: "r1".to_string(),
                prompt: "Which one?".to_string(),
            },
        ));
        thread.turns.push(first_turn);
        assert!(ThreadProjection::from_thread(&thread)
            .pending_user_input_request
            .is_some());

        let mut second_turn = Turn::new();
        second_turn.push_item(TurnItem::new(
            TurnItemSource::User,
            TurnItemKind::UserInput {
                text: "A".to_string(),
                response_to: Some("r1".to_string()),
            },
        ));
        thread.turns.push(second_turn);
        assert!(ThreadProjection::from_thread(&thread)
            .pending_user_input_request
            .is_none());
    }
}
