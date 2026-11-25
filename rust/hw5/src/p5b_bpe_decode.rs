//! Problem 5b: Decoding token IDs with a trained BPE tokenizer.

use std::collections::HashMap;

/// Decode a sequence of BPE token IDs back into raw text.
///
/// # Inputs
/// - `seq`: slice of token IDs
/// - `tokens`: mapping from token string to token ID
///
/// # Output
/// Decoded string obtained by concatenating the token string for each ID in `seq`.
#[allow(unused_variables)]
pub fn bpe_decode(seq: &[usize], tokens: &HashMap<String, usize>) -> String {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
