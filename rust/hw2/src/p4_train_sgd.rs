//! Problem 4: Minibatch stochastic gradient descent for a linear classifier.

#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::cross_entropy_loss;

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
#[allow(unused_variables)]
pub fn train_sgd(
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    step_size: f64,
    batch_size: usize,
) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
