//! Problem 2: Batched parallel autoregressive generation (`generate_parallel`).

#[allow(unused_imports)]
use candle_core::{D, DType, Result, Tensor};

#[allow(unused_imports)]
use crate::{CausalLM, multinomial_batch};

/// Autoregressively sample `num_completions` independent completions in parallel for a
/// single `prompt_tokens` sequence using the model's batched KV cache.
///
/// # Conventions (matching `homework/hw7/hw7.py`)
/// - `max_tokens` is the **total** maximum sequence length per completion, **including** the prompt.
/// - Stop early only once **all** `num_completions` sequences have generated `eot_token`
///   (when `eot_token` is `Some(id)`). Individual completions that already hit `eot_token`
///   keep generating normally while waiting for the rest of the batch.
/// - Returns a `DType::U32` tensor of shape `[num_completions, max_tokens]` on `model.device()`
///   (if early stopping triggers before `max_tokens`, any remaining columns are `0`).
/// - Uses temperature-scaled softmax sampling via [`multinomial_batch`] (or greedy argmax if
///   `temp <= 0.0`).
#[allow(unused_variables)]
pub fn generate_parallel<M: CausalLM>(
    model: &mut M,
    prompt_tokens: &[u32],
    num_completions: usize,
    eot_token: Option<u32>,
    temp: f64,
    max_tokens: usize,
) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
