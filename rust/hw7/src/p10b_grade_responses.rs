//! Problem 10b: Grading generated completions against a ground-truth answer (`grade_responses`).

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor};

#[allow(unused_imports)]
use crate::{Tokenizer, extract_answer};

/// Score multiple generated completions (`tokens` of shape `[num_completions, seq_len]`,
/// `DType::U32`) against a reference sequence `ground_truth` (`[seq_len]`, `DType::U32`).
///
/// For each completion row:
/// - If [`extract_answer`] returns `Some(ans)`: add `format_weight`, and if `Some(ans) == gt_ans`,
///   also add `correct_weight`.
/// - Otherwise (invalid or missing `<ANSWER>...</ANSWER>`): score is `0.0`.
#[allow(unused_variables)]
pub fn grade_responses<T: Tokenizer>(
    tokenizer: &T,
    tokens: &Tensor,
    ground_truth: &Tensor,
    correct_weight: f64,
    format_weight: f64,
) -> Result<Vec<f64>> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
