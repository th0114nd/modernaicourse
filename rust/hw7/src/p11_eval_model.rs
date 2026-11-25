//! Problem 11: Evaluating a reasoning model on GSM8K (`eval_model`).

#[allow(unused_imports)]
use candle_core::{Device, Error, IndexOp, Result, Tensor};

#[allow(unused_imports)]
use crate::{
    ANSWER_CLOSE, CausalLM, THINK_OPEN, Tokenizer, generate_parallel_tool, grade_responses,
};

/// Evaluate a reasoning model on a batch-size-1 GSM8K loader by sampling `num_completions`
/// completions per example (with tool calls) for up to `max_cases` examples.
///
/// # Algorithm
/// For each example `(x, y, _mask)` (up to `max_cases`):
/// 1. Extract the prompt tokens from `x[0]` up to and **including** `<THINK>`.
/// 2. Call [`generate_parallel_tool`] with `eot_token = tokenizer.special_token("</ANSWER>")`.
/// 3. Grade completions against ground truth `y[0]` using [`grade_responses`] to count:
///    - Correct answers (`correct_weight = 1.0, format_weight = 0.0`).
///    - Properly formatted answers (`correct_weight = 0.0, format_weight = 1.0`).
///    - Whether at least one completion was correct (`pass@k`).
///
/// # Output
/// `(pass_at_1, format_rate, pass_at_k)`:
/// 1. Fraction of all `n_cases * num_completions` generations that were correct (Pass@1).
/// 2. Fraction of all `n_cases * num_completions` generations that were properly formatted.
/// 3. Fraction of the `n_cases` examples with at least one correct completion (Pass@k).
#[allow(unused_variables)]
pub fn eval_model<M, I, T>(
    loader: I,
    model: &mut M,
    tokenizer: &T,
    num_completions: usize,
    max_tokens: usize,
    temp: f64,
    max_cases: usize,
) -> Result<(f64, f64, f64)>
where
    M: CausalLM,
    I: IntoIterator<Item = Result<(Tensor, Tensor, Tensor)>>,
    T: Tokenizer,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
