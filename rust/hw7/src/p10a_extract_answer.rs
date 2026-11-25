//! Problem 10a: Extracting the integer final answer from a token sequence (`extract_answer`).

#[allow(unused_imports)]
use crate::{ANSWER_CLOSE, ANSWER_OPEN, Tokenizer};

/// Extract the integer answer contained between `<ANSWER>` and `</ANSWER>` tags.
///
/// Returns `None` if either tag is missing, if `</ANSWER>` does not appear after `<ANSWER>`,
/// or if the decoded text between the tags does not parse as a valid integer (`i64`).
#[allow(unused_variables)]
pub fn extract_answer<T: Tokenizer>(tokenizer: &T, tokens: &[u32]) -> Option<i64> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
