//! Part 0i: Arbitrary-rank cross-entropy loss.
//!
//! Identical to HW5 (`cross_entropy_loss`); copy your implementation from `hw5`.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Compute the average cross-entropy loss over an arbitrary-rank batch of logits and targets:
///
/// $$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \left( \log \sum_{j=1}^{k} e^{z_{i,j}} - z_{i, y_i} \right)$$
#[allow(unused_variables)]
pub fn cross_entropy_loss(logits: &Tensor, y: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
