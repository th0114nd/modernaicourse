//! Problem 5: Reasoning & tool-response loss masking (`get_loss_mask`).

#[allow(unused_imports)]
use crate::{ANSWER_CLOSE, RESPONSE_CLOSE, THINK_OPEN, TOOL_CLOSE, Tokenizer};

/// Build a boolean mask selecting only the model-generated reasoning and answer tokens
/// to train on for GSM8K SFT and RL.
///
/// # Rules (matching `homework/hw7/hw7.py`)
/// - Start with `in_loss = false` (do not train on `<QUESTION>...</QUESTION><THINK>`).
/// - For each token `tok` in `tokens`:
///   1. Push the current `in_loss` value into the output mask.
///   2. Update `in_loss` for the *next* token:
///      - After `<THINK>` or `</RESPONSE>`: set `in_loss = true` (model resumes generation).
///      - After `</TOOL>` or `</ANSWER>`: set `in_loss = false` (tool response or post-answer padding).
#[allow(unused_variables)]
pub fn get_loss_mask<T: Tokenizer>(tokens: &[u32], tokenizer: &T) -> Vec<bool> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
