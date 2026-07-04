use crate::events::{GoalState, Thread, ToolResult, TurnId, TurnItemKind, TurnStatus};
use crate::model::ModelFunctionCall;
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
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ModelFunctionCall>,
    },
    Tool {
        tool_call_id: String,
        name: String,
        content: Value,
    },
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
                        projection.messages_for_model.push(ChatMessage::User {
                            content: text.clone(),
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
                        projection.messages_for_model.push(ChatMessage::Assistant {
                            content: Some(text.clone()),
                            tool_calls: Vec::new(),
                        });
                    }
                    TurnItemKind::ModelFunctionCall {
                        call_id,
                        name,
                        arguments,
                    } => {
                        projection.messages_for_model.push(ChatMessage::Assistant {
                            content: None,
                            tool_calls: vec![ModelFunctionCall {
                                call_id: call_id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            }],
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
                            projection.messages_for_model.push(ChatMessage::Tool {
                                tool_call_id: call_id.clone(),
                                name: name.clone(),
                                content: output.clone(),
                            });
                        }
                        ToolResult::Error { error } => {
                            projection.messages_for_model.push(ChatMessage::Tool {
                                tool_call_id: call_id.clone(),
                                name: name.clone(),
                                content: Value::String(error.clone()),
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
        GoalState, GoalStatus, Thread, ToolResult, Turn, TurnItem, TurnItemKind, TurnItemSource,
    };
    use serde_json::json;

    use super::{ChatMessage, ThreadProjection};

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

    #[test]
    fn projects_function_call_and_tool_output_as_structured_messages() {
        let mut thread = Thread::new("t");
        let mut turn = Turn::new();
        turn.push_item(TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelFunctionCall {
                call_id: "call_1".to_string(),
                name: "exec_command".to_string(),
                arguments: json!({ "cmd": "ls" }),
            },
        ));
        turn.push_item(TurnItem::new(
            TurnItemSource::Tool,
            TurnItemKind::ToolOutput {
                call_id: "call_1".to_string(),
                name: "exec_command".to_string(),
                result: ToolResult::Success {
                    output: json!({ "stdout": "src\n" }),
                },
            },
        ));
        thread.turns.push(turn);

        let projection = ThreadProjection::from_thread(&thread);
        assert!(matches!(
            &projection.messages_for_model[0],
            ChatMessage::Assistant { tool_calls, .. }
                if tool_calls.len() == 1 && tool_calls[0].call_id == "call_1"
        ));
        assert!(matches!(
            &projection.messages_for_model[1],
            ChatMessage::Tool { tool_call_id, content, .. }
                if tool_call_id == "call_1" && content == &json!({ "stdout": "src\n" })
        ));
    }
}
