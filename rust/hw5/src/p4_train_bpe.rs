//! Problem 4: Training a Byte-Pair Encoding (BPE) tokenizer.

use std::collections::HashMap;

#[allow(unused_imports)]
use crate::{merge_pair, most_common_pair, text_to_corpus};

/// Train a simple BPE tokenizer from raw text up to `num_tokens` total vocabulary entries.
///
/// # Algorithm
/// 1. Initialize the vocabulary with the 256 base Latin-1 characters
///    `char::from(b).to_string() -> b as usize` for `b in 0u8..=255`
///    (matching Python's `[chr(i) for i in range(256)]`).
/// 2. Build `(corpus, counts)` with [`text_to_corpus`] and initialize an empty `merges` list.
/// 3. Repeat while `tokens.len() < num_tokens`:
///    - Find the most frequent pair using [`most_common_pair`].
///    - Append the pair to `merges` and insert `format!("{}{}", pair.0, pair.1)` into `tokens`
///      at index `tokens.len()`.
///    - Merge the pair in place across `corpus` using [`merge_pair`].
///
/// # Inputs
/// - `text`: training text used to learn BPE merges
/// - `num_tokens`: target vocabulary size (`>= 256`)
///
/// # Output
/// `(tokens, merges)` — mapping from token string to token ID, and ordered list of learned merges.
#[allow(unused_variables)]
pub fn train_bpe(text: &str, num_tokens: usize) -> (HashMap<String, usize>, Vec<(String, String)>) {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
