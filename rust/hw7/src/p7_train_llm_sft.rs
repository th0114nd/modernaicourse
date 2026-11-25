//! Problem 7: Supervised reasoning finetuning loop (`train_llm_sft`) with masked cross-entropy.

#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor};

#[allow(unused_imports)]
use crate::{CausalLM, Optimizer, cross_entropy_loss, mask_indices};

/// Run one pass of supervised finetuning over `(x, y, mask)` minibatches.
///
/// Identical to Homework 6's `train_llm_chat`:
/// 1. Stop once `step >= limit` if `max_iter == Some(limit)`.
/// 2. Compute `logits = model.forward(&x, 0, false)?.to_dtype(DType::F32)?` (`[B, S, V]`).
/// 3. Extract masked entries via [`mask_indices`] (skip batch if 0 masked entries).
/// 4. Compute `loss = cross_entropy_loss(&masked_logits, &masked_y)?`.
/// 5. Run `opt.zero_grad(); let grads = loss.backward()?; opt.step(&grads)?;`.
#[allow(unused_variables)]
pub fn train_llm_sft<M, I, O>(
    model: &mut M,
    loader: I,
    opt: &mut O,
    max_iter: Option<usize>,
) -> Result<()>
where
    M: CausalLM,
    I: IntoIterator<Item = Result<(Tensor, Tensor, Tensor)>>,
    O: Optimizer,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
