//! Problem 3: Building the assistant-response loss mask for chat SFT and DPO.

#[allow(unused_imports)]
use crate::chat::{ASSISTANT_CLOSE, ASSISTANT_OPEN, Tokenizer};

/// Build a boolean mask selecting only the assistant-response tokens for training.
///
/// Specifically, `mask[i]` is `true` **only** for tokens that occur strictly *after*
/// an `<ASSISTANT>` token up through (and **including**) the matching `</ASSISTANT>` token.
///
/// # Inputs
/// - `tokens`: tokenized chat conversation (`&[u32]`)
/// - `tokenizer`: tokenizer providing `tokenizer.special_token("<ASSISTANT>")` and
///   `tokenizer.special_token("</ASSISTANT>")`
///
/// # Output
/// `Vec<bool>` of the same length as `tokens`.
///
/// # Example
/// Given `[7, <ASSISTANT>, 1, 2, </ASSISTANT>, 8]`:
/// returns `[false, false, true, true, true, false]`.
#[allow(unused_variables)]
pub fn get_loss_mask<T: Tokenizer>(tokens: &[u32], tokenizer: &T) -> Vec<bool> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
