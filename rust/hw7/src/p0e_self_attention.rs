//! Part 0e: Scaled dot-product self-attention.
//!
//! Identical to HW6 (`self_attention`); copy your implementation from `hw6`.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Compute scaled dot-product attention with an optional additive mask:
///
/// $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$
#[allow(unused_variables)]
pub fn self_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
