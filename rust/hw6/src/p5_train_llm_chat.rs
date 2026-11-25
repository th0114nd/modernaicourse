//! Problem 5: Supervised chat finetuning loop (`train_llm_chat`) with masked cross-entropy loss.

#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor};

#[allow(unused_imports)]
use crate::{CausalLM, Optimizer, cross_entropy_loss, mask_indices};

/// Run one pass of supervised chat finetuning over `(x, y, mask)` minibatches.
///
/// # Algorithm
/// For each `(step, batch)` in `chat_loader`:
/// 1. If `max_iter` is `Some(limit)` and `step >= limit`, stop immediately (`break`).
/// 2. Compute `logits = model.forward(&x, 0, false)?.to_dtype(DType::F32)?` (shape `[B, S, V]`).
/// 3. Extract only the positions where `mask != 0` using [`mask_indices`]:
///    - `let idx = mask_indices(&mask)?;`
///    - If `idx.elem_count() == 0`, skip this batch (`continue`) to avoid `0/0` NaN loss.
///    - `let masked_logits = logits.reshape((b * s, v))?.index_select(&idx, 0)?;`
///    - `let masked_y = y.reshape((b * s,))?.index_select(&idx, 0)?;`
/// 4. Compute `let loss = cross_entropy_loss(&masked_logits, &masked_y)?;`.
/// 5. Call `opt.zero_grad();`, `let grads = loss.backward()?;`, and `opt.step(&grads)?;`.
#[allow(unused_variables)]
pub fn train_llm_chat<M, I, O>(
    model: &mut M,
    chat_loader: I,
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
