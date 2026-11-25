//! Part 0k: Per-example masked sequence log probabilities (`log_probs`).
//!
//! Identical to HW6 (`log_probs`); copy your implementation from `hw6`.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Compute the summed masked log probability for each sequence in a minibatch:
///
/// $$\log p(y_b \mid x_b) = \sum_{t=0}^{S-1} \mathbb{1}[\text{mask}_{b,t} \ne 0] \cdot \left( z_{b, t, y_{b,t}} - \log \sum_{j=0}^{V-1} e^{z_{b, t, j}} \right)$$
///
/// # Inputs
/// - `logits`: `[batch_size, seq_len, num_tokens]` float tensor
/// - `y`: `[batch_size, seq_len]` integer (`DType::U32`) tensor of target token IDs
/// - `mask`: `[batch_size, seq_len]` mask tensor (`DType::U8`, non-zero entries included)
///
/// # Output
/// 1D float tensor of shape `[batch_size]` containing the summed masked log probabilities
/// for each example in the batch.
///
/// # Candle Note
/// `Tensor::gather` requires contiguous inputs. Call `.contiguous()?` on `logits` and `y`
/// so strided slices such as `tokens.narrow(1, 1, s - 1)?` work cleanly.
#[allow(unused_variables)]
pub fn log_probs(logits: &Tensor, y: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
