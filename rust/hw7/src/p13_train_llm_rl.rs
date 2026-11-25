//! Problem 13: Reinforcement learning training loop (`train_llm_rl`).

#[allow(unused_imports)]
use candle_core::{Device, Error, IndexOp, Result, Tensor};

#[allow(unused_imports)]
use crate::{
    ANSWER_CLOSE, CausalLM, Optimizer, THINK_OPEN, Tokenizer, generate_parallel_tool,
    grade_responses, rl_loss,
};

/// Run one pass of reinforcement learning over a batch-size-1 GSM8K data loader.
///
/// # Algorithm (matching `homework/hw7/hw7.py`)
/// For each `(step, batch)` in `loader`:
/// 1. If `max_iter` is `Some(limit)` and `step >= limit`, stop (`break`).
/// 2. Extract prompt tokens from `x[0]` up to and **including** `<THINK>`, and ground truth
///    from `y[0]`.
/// 3. Sample `num_completions` completions with tool execution via [`generate_parallel_tool`]
///    (passing `eot_token = tokenizer.special_token("</ANSWER>")`).
/// 4. Grade completions via [`grade_responses`] using `correct_weight` and `format_weight`,
///    and pack the scores into a 1D `DType::F32` reward tensor on `model.device()`.
/// 5. Compute `loss = rl_loss(model, tokenizer, &tokens, &rewards)?`.
/// 6. Update parameters: `opt.zero_grad(); let grads = loss.backward()?; opt.step(&grads)?;`.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn train_llm_rl<M, I, O, T>(
    model: &mut M,
    loader: I,
    opt: &mut O,
    tokenizer: &T,
    num_completions: usize,
    temp: f64,
    max_tokens: usize,
    max_iter: Option<usize>,
    correct_weight: f64,
    format_weight: f64,
) -> Result<()>
where
    M: CausalLM,
    I: IntoIterator<Item = Result<(Tensor, Tensor, Tensor)>>,
    O: Optimizer,
    T: Tokenizer,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
