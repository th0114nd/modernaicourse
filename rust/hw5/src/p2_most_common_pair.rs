//! Problem 2: Finding the most common adjacent token pair.

#[allow(unused_imports)]
use std::collections::HashMap;

/// Find the most frequent adjacent token pair in a weighted corpus.
///
/// Each occurrence of an adjacent pair `(word[j], word[j + 1])` inside `corpus[i]`
/// contributes `counts[i]` to that pair's total frequency.
///
/// # Tie-breaking
/// Break ties in favor of the pair that appears **first** when scanning `corpus` from left
/// to right (matching Python's `Counter.most_common(1)` insertion-order tie-breaking).
///
/// # Inputs
/// - `corpus`: slice of unique words, each represented as a `Vec<String>` of tokens
/// - `counts`: slice of occurrence counts aligned with `corpus`
///
/// # Output
/// `(left, right)` — the highest-frequency adjacent token pair.
#[allow(unused_variables)]
pub fn most_common_pair(corpus: &[Vec<String>], counts: &[usize]) -> (String, String) {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
