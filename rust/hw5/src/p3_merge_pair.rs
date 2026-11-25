//! Problem 3: Merging an adjacent token pair in place across the corpus.

/// Merge every non-overlapping left-to-right occurrence of `pair` in `corpus` in place.
///
/// For example, merging `(" ", "t")` in `[[" ", "t", "e", "s", "t"]]` mutates it into
/// `[[" t", "e", "s", "t"]]`.
///
/// # Inputs
/// - `corpus`: mutable slice of tokenized words to modify in place
/// - `pair`: `(left, right)` adjacent token pair to replace with `format!("{left}{right}")`
#[allow(unused_variables)]
pub fn merge_pair(corpus: &mut [Vec<String>], pair: (&str, &str)) {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
