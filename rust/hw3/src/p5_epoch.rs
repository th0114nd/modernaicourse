//! Problem 5: Single training/evaluation `epoch`.

#[allow(unused_imports)]
use candle_core::{D, DType, Result, Tensor};

use crate::nn::{Loss, Module, Optimizer};

/// Run one full pass through a dataset, with optional parameter optimization.
///
/// For each minibatch `(x, y)` yielded by `loader`:
/// 1. Compute `logits = model.forward(&x)?`.
/// 2. Compute `batch_loss = loss_fn.forward(&logits, &y)?`.
/// 3. Accumulate the example-weighted total loss (`batch_loss * batch_len`) and total
///    misclassification count (`(logits.argmax(D::Minus1)? != y)` summed over the batch).
/// 4. If `opt` is `Some(opt)`, run `let grads = batch_loss.backward()?; opt.step(&grads)?;`
///    (use `if let Some(opt) = opt.as_mut() { ... }` inside the loop).
///
/// Returns `(avg_loss, avg_error)` averaged across all examples seen in the epoch.
///
/// # Note on uneven final batches
/// Weight each minibatch's mean loss by its actual batch size `y.dim(0)?` before dividing
/// by the total number of examples `N`, so uneven final batches are averaged exactly.
#[allow(unused_variables, unused_mut)]
pub fn epoch<I>(
    model: &dyn Module,
    loader: I,
    loss_fn: &dyn Loss,
    mut opt: Option<&mut dyn Optimizer>,
) -> Result<(f64, f64)>
where
    I: IntoIterator<Item = Result<(Tensor, Tensor)>>,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
