//! Problem 1: Converting UltraChat-style message lists into tagged chat strings.

use crate::chat::Message;

/// Convert a slice of chat [`Message`]s into a single tagged text string using
/// `<USER>`, `</USER>`, `<ASSISTANT>`, and `</ASSISTANT>` tags with no extra
/// whitespace or newlines between messages.
///
/// # Example
/// ```text
/// [Message::user("Hello"), Message::assistant("Hi!")]
///   -> "<USER>Hello</USER><ASSISTANT>Hi!</ASSISTANT>"
/// ```
///
/// # Requirements
/// - Wrap each `"user"` message as `<USER>{content}</USER>`.
/// - Wrap each `"assistant"` message as `<ASSISTANT>{content}</ASSISTANT>`.
/// - In general, for a role string `r`, format `<{R}>{content}</{R}>` where `R = r.to_uppercase()`.
/// - Preserve all characters inside `content` (including newlines, tabs, or empty strings)
///   and insert no extra separators between adjacent messages.
#[allow(unused_variables)]
pub fn convert_to_chat_format(messages: &[Message]) -> String {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
