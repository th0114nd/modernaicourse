//! Problem 1: Splitting a document into a unique-word corpus with counts.

#[allow(unused_imports)]
use std::collections::HashMap;

#[allow(unused_imports)]
use crate::split_keep_whitespace;

/// Convert raw text into a deduplicated whitespace-delimited corpus and occurrence counts.
///
/// Each unique whitespace-prefixed word (as produced by [`split_keep_whitespace`], the Rust
/// counterpart to Python's `re.split(r'(?=\s)', text)`) is represented as a `Vec<String>` of
/// single-character tokens, paired with the number of times that word occurs in `text`.
///
/// # Requirements
/// - Preserve **first-appearance order** for unique words (matching Python's `Counter`).
/// - Omit empty segments (e.g. a leading space `" test"` produces `[" ", "t", "e", "s", "t"]`,
///   not an empty word before it).
///
/// # Inputs
/// - `text`: raw input document
///
/// # Output
/// `(corpus, counts)` where `corpus[i]` is the `i`-th unique word as single-character tokens
/// and `counts[i]` is its occurrence count.
#[allow(unused_variables)]
pub fn text_to_corpus(text: &str) -> (Vec<Vec<String>>, Vec<usize>) {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
