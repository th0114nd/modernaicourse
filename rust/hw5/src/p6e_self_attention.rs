//! Problem 6e: Scaled dot-product self-attention.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Compute scaled dot-product attention with an optional additive mask:
///
/// $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$
///
/// Supports arbitrary leading batch/head dimensions:
/// - `q`: `[..., query_len, d]`
/// - `k`: `[..., key_len, d]`
/// - `v`: `[..., key_len, d_v]`
/// - `mask`: optional additive mask broadcastable to `[..., query_len, key_len]`
///
/// # Output
/// Attention output tensor of shape `[..., query_len, d_v]`.
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
