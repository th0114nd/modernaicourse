//! Problem 11: LLM pretraining loop.

#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor};

#[allow(unused_imports)]
use crate::{CausalLM, Optimizer, cross_entropy_loss};

/// Run one full training pass over all `(x, y)` minibatches in `loader`.
///
/// # Algorithm
/// For each batch `(x, y)` from `loader`:
/// 1. Run the model without KV caching: `let logits = model.forward(&x, 0, false)?.to_dtype(DType::F32)?;`
///    (converting logits to `DType::F32` ensures numerically stable cross-entropy even if the
///    model computes in lower precision such as `BF16` or `F16`).
/// 2. Compute the scalar loss via [`cross_entropy_loss`]`(&logits, &y)?`.
/// 3. Call `opt.zero_grad();`, compute `let grads = loss.backward()?;`, and update parameters
///    with `opt.step(&grads)?;`.
///
/// # Inputs
/// - `model`: causal language model implementing [`CausalLM`]
/// - `loader`: iterable yielding `Result<(Tensor, Tensor)>` (e.g. `&DataLoader`)
/// - `opt`: parameter optimizer implementing [`Optimizer`] (e.g. `&mut Adam`)
#[allow(unused_variables)]
pub fn train_llm<M, I, O>(model: &mut M, loader: I, opt: &mut O) -> Result<()>
where
    M: CausalLM,
    I: IntoIterator<Item = Result<(Tensor, Tensor)>>,
    O: Optimizer,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
