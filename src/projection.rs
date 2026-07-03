use crate::events::{GoalStatus, Thread, ThreadEventKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoalState {
    pub objective: String,
    pub status: GoalStatus,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserInputRequest {
    pub request_id: String,
    pub prompt: String,
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
    pub latest_goal: Option<GoalState>,
    pub pending_user_input_request: Option<UserInputRequest>,
    pub completed_function_results: Vec<CompletedFunctionCall>,
    pub last_assistant_message: Option<String>,
}

impl ThreadProjection {
    pub fn from_thread(thread: &Thread) -> Self {
        let mut projection = Self::default();

        for event in &thread.events {
            match &event.kind {
                ThreadEventKind::UserInputReceived { text, response_to } => {
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
                ThreadEventKind::AssistantMessageEmitted { text } => {
                    projection.last_assistant_message = Some(text.clone());
                    projection.messages_for_model.push(ChatMessage {
                        role: ChatRole::Assistant,
                        content: text.clone(),
                        name: None,
                        tool_call_id: None,
                    });
                }
                ThreadEventKind::FunctionCallRequested {
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
                ThreadEventKind::FunctionCallCompleted {
                    call_id,
                    name,
                    output,
                } => {
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
                ThreadEventKind::FunctionCallFailed {
                    call_id,
                    name,
                    error,
                } => {
                    projection.messages_for_model.push(ChatMessage {
                        role: ChatRole::User,
                        content: format!("Function {name} with call id {call_id} failed: {error}"),
                        name: None,
                        tool_call_id: None,
                    });
                }
                ThreadEventKind::GoalUpdated {
                    objective,
                    status,
                    notes,
                } => {
                    projection.latest_goal = Some(GoalState {
                        objective: objective.clone(),
                        status: *status,
                        notes: notes.clone(),
                    });
                }
                ThreadEventKind::UserInputRequested { request_id, prompt } => {
                    projection.pending_user_input_request = Some(UserInputRequest {
                        request_id: request_id.clone(),
                        prompt: prompt.clone(),
                    });
                }
                ThreadEventKind::TurnFailed { .. } => {}
            }
        }

        projection
    }
}

#[cfg(test)]
mod tests {
    use crate::events::{GoalStatus, Thread, ThreadEvent, ThreadEventKind};

    use super::ThreadProjection;

    #[test]
    fn derives_latest_goal() {
        let mut thread = Thread::new("t");
        thread
            .events
            .push(ThreadEvent::new(ThreadEventKind::GoalUpdated {
                objective: "first".to_string(),
                status: GoalStatus::Active,
                notes: None,
            }));
        thread
            .events
            .push(ThreadEvent::new(ThreadEventKind::GoalUpdated {
                objective: "second".to_string(),
                status: GoalStatus::Complete,
                notes: Some("done".to_string()),
            }));

        let projection = ThreadProjection::from_thread(&thread);
        let goal = projection.latest_goal.expect("goal");
        assert_eq!(goal.objective, "second");
        assert_eq!(goal.status, GoalStatus::Complete);
        assert_eq!(goal.notes.as_deref(), Some("done"));
    }

    #[test]
    fn tracks_pending_user_input_request() {
        let mut thread = Thread::new("t");
        thread
            .events
            .push(ThreadEvent::new(ThreadEventKind::UserInputRequested {
                request_id: "r1".to_string(),
                prompt: "Which one?".to_string(),
            }));
        assert!(ThreadProjection::from_thread(&thread)
            .pending_user_input_request
            .is_some());

        thread
            .events
            .push(ThreadEvent::new(ThreadEventKind::UserInputReceived {
                text: "A".to_string(),
                response_to: Some("r1".to_string()),
            }));
        assert!(ThreadProjection::from_thread(&thread)
            .pending_user_input_request
            .is_none());
    }
}
