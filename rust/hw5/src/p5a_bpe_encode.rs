//! Problem 5a: Encoding text with a trained BPE tokenizer.

use std::collections::HashMap;

#[allow(unused_imports)]
use crate::{merge_pair, split_keep_whitespace};

/// Encode `text` into a flat sequence of token IDs using learned BPE `merges` and `tokens`.
///
/// # Algorithm
/// 1. Split `text` with [`split_keep_whitespace`] (preserving duplicates and original order)
///    and convert each word to a `Vec<String>` of single-character tokens.
/// 2. For each `(left, right)` in `merges` (in order), call [`merge_pair`] on the corpus.
/// 3. Flatten all tokens across all words and map each token string through `tokens`.
///
/// # Inputs
/// - `text`: input string to encode
/// - `merges`: ordered list of learned merge pairs
/// - `tokens`: mapping from token string to token ID
///
/// # Output
/// Flat `Vec<usize>` of encoded token IDs.
#[allow(unused_variables)]
pub fn bpe_encode(
    text: &str,
    merges: &[(String, String)],
    tokens: &HashMap<String, usize>,
) -> Vec<usize> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
