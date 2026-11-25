//! Problem 4: Minibatch stochastic gradient descent for a linear classifier.

use crate::cross_entropy_loss;
use candle_core::error::Error;
use candle_core::{bail, DType, Result, Tensor, Var};

/// Train a linear classifier $W \in \mathbb{R}^{k \times n}$ via minibatch SGD
/// to minimize [`cross_entropy_loss`].
///
/// Details:
/// - Initialize $W$ as a trainable [`Var`] of zeros with shape `[k, n]` (`DType::F32`),
///   where $n$ is the feature dimension (`x.dim(1)`) and $k = \max(y) + 1$.
/// - For each of `epochs` passes over `(x, y)`, iterate sequentially over contiguous
///   minibatches of size `batch_size` (the final batch may be smaller if `N` is not
///   divisible by `batch_size`, matching `torch.split`).
/// - For each minibatch `(x_batch, y_batch)`:
///   1. Compute logits `x_batch @ W^T` (shape `[B, k]`).
///   2. Compute `loss = cross_entropy_loss(&logits, &y_batch)?`.
///   3. Compute gradients via `let grads = loss.backward()?;`.
///   4. Update `w.set(&(w.as_tensor() - (grad_w * step_size)?)?)?;`.
/// - Return the trained weight matrix `W` as a detached `[k, n]` `f32` [`Tensor`].
///
/// # Inputs
/// - `x`: 2D `f32` tensor of shape `[N, n]` — training inputs (one per row)
/// - `y`: 1D `u32` tensor of shape `[N]` — target class labels in `0..k`
/// - `epochs`: number of passes over the training set
/// - `step_size`: learning rate for parameter updates
/// - `batch_size`: number of examples per minibatch (must be `>= 1`)
///
/// # Output
/// - 2D `f32` tensor of shape `[k, n]` containing the trained linear classifier weights
pub fn train_sgd(
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    step_size: f64,
    batch_size: usize,
) -> Result<Tensor> {
    // BEGIN YOUR CODE
    let (all_batch_size, n) = x.dims2()?;
    let all_batch_size_2 = y.dims1()?;
    if all_batch_size != all_batch_size_2 {
        bail!("all batch size mismatch")
    }
    if step_size == 0.0 {
        bail!("bad step size");
    }
    if batch_size == 0 {
        bail!("batch size");
    }
    let k = y.max_all()?.to_scalar::<u32>()? + 1;
    let w = Var::zeros((k as usize, n), DType::F32, x.device())?;
    for _ in 0..epochs {
        for start in (0..all_batch_size).step_by(batch_size) {
            let len = batch_size.min(all_batch_size - start);

            let x_batch = x.narrow(0, start, len)?;
            let y_batch = y.narrow(0, start, len)?;
            let wt = w.transpose(0, 1)?;
            let y_preds = x_batch.matmul(&wt)?;
            let loss_batch = cross_entropy_loss(&y_preds, &y_batch)?;
            let grads = loss_batch.backward()?;
            let g = grads.get(w.as_tensor()).ok_or(Error::msg("no grads get"))?;
            w.set(&(w.as_tensor() - (g * step_size)?)?)?;
        }
    }
    Ok(w.as_detached_tensor())
    // END YOUR CODE
}
