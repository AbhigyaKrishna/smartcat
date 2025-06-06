use crate::config::prompt::Message;
use serde::Deserialize;
use std::fmt::Debug;

// OpenAi
#[derive(Debug, Deserialize)]
pub(super) struct OpenAiResponse {
    pub choices: Vec<MessageWrapper>,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageWrapper {
    pub message: Message,
}

impl From<OpenAiResponse> for String {
    fn from(value: OpenAiResponse) -> Self {
        value.choices.first().unwrap().message.content.to_owned()
    }
}

// Anthropic
#[derive(Debug, Deserialize)]
pub(super) struct AnthropicMessage {
    pub text: String,
    #[serde(rename(serialize = "type", deserialize = "type"))]
    pub _type: String,
}

impl From<AnthropicResponse> for String {
    fn from(value: AnthropicResponse) -> Self {
        value.content.first().unwrap().text.to_owned()
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicResponse {
    pub content: Vec<AnthropicMessage>,
}

// Ollama
#[derive(Debug, Deserialize)]
pub(super) struct OllamaResponse {
    pub message: Message,
}

impl From<OllamaResponse> for String {
    fn from(value: OllamaResponse) -> Self {
        value.message.content
    }
}

// Google
#[derive(Debug, Deserialize)]
pub(super) struct GoogleContentParts {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct GoogleResponseContent {
    pub parts: Vec<GoogleContentParts>,
}

#[derive(Debug, Deserialize)]
pub(super) struct GoogleResponseCandidate {
    pub content: GoogleResponseContent,
}

#[derive(Debug, Deserialize)]
pub(super) struct GoogleResponse {
    pub candidates: Vec<GoogleResponseCandidate>,
}

impl From<GoogleResponse> for String {
    fn from(value: GoogleResponse) -> Self {
        value.candidates.iter().fold(String::new(), |acc, c| {
            acc + &c.content.parts.iter().fold(String::new(), |acc, c| {
                acc + &c.text
            })
        })
    }
}
