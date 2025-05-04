use crate::config::prompt::{Message, Prompt};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct OpenAiPrompt {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct AnthropicPrompt {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub max_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone)]
pub(super) struct GoogleMessagePart {
    pub text: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone)]
pub(super) struct GoogleMessage {
    pub role: String,
    pub parts: Vec<GoogleMessagePart>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u16>,
    pub response_mime_type: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct Unit;

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct GoogleTools {
    pub google_search: Unit
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct GooglePrompt {
    pub contents: Vec<GoogleMessage>,
    pub generation_config: GoogleGenerationConfig,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<GoogleTools>
}

impl From<Prompt> for OpenAiPrompt {
    fn from(prompt: Prompt) -> OpenAiPrompt {
        OpenAiPrompt {
            model: prompt
                .model
                .expect("model must be specified either in the api config or in the prompt config"),
            messages: prompt.messages,
            temperature: prompt.temperature,
            stream: prompt.stream,
        }
    }
}

impl From<Prompt> for AnthropicPrompt {
    fn from(prompt: Prompt) -> Self {
        let merged_messages =
            prompt
                .messages
                .into_iter()
                .fold(Vec::new(), |mut acc: Vec<Message>, mut message| {
                    if message.role == "system" {
                        message.role = "user".to_string();
                    }
                    match acc.last_mut() {
                        Some(last_message) if last_message.role == message.role => {
                            last_message.content.push_str("\n\n");
                            last_message.content.push_str(&message.content);
                        }
                        _ => acc.push(message),
                    }
                    acc
                });

        AnthropicPrompt {
            model: prompt.model.expect("model must be specified"),
            messages: merged_messages,
            temperature: prompt.temperature,
            stream: prompt.stream,
            max_tokens: 4096,
        }
    }
}

impl From<Prompt> for GooglePrompt {
    fn from(prompt: Prompt) -> Self {
        let merged_messages: Vec<GoogleMessage> =
            prompt
                .messages
                .into_iter()
                .fold(Vec::new(), |mut acc: Vec<GoogleMessage>, mut message| {
                    if message.role == "system" {
                        message.role = "user".to_string();
                    }
                    match acc.last_mut() {
                        Some(last_message) if last_message.role == message.role => {
                            last_message.parts[0].text.push_str("\n\n");
                            last_message.parts[0].text.push_str(&message.content);
                        }
                        _ => acc.push(GoogleMessage {
                            role: message.role,
                            parts: vec![GoogleMessagePart { text: message.content }]
                        }),
                    }
                    acc
                });
        
        GooglePrompt {
            contents: merged_messages,
            generation_config: GoogleGenerationConfig {
                temperature: prompt.temperature,
                max_output_tokens: None,
                response_mime_type: "text/plain".to_string(),
            },
            // tools: vec![GoogleTools { google_search: Unit }],
            tools: vec![]
        }
    }
}
