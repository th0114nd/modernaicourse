//! Problem 5: Scaled dot-product self-attention.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Compute scaled dot-product self-attention:
///
/// $$\text{self\_attention}(Q, K, V, \text{mask}) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}} + \text{mask}\right) V$$
///
/// where $d$ is the last dimension of $Q$ (and $K$).
///
/// # Inputs
/// - `q`: queries of shape `[..., seq_q, d]`
/// - `k`: keys of shape `[..., seq_k, d]`
/// - `v`: values of shape `[..., seq_k, d_v]`
/// - `mask`: optional additive mask broadcastable to `[..., seq_q, seq_k]` (e.g. `[seq_q, seq_k]`
///   with `0.0` for allowed positions and `-inf` for masked positions)
///
/// # Output
/// Attention output of shape `[..., seq_q, d_v]`.
///
/// # Candle tips
/// - Transpose the last two axes of `k` with `k.transpose(D::Minus2, D::Minus1)?.contiguous()?`
///   before `q.matmul(...)` so `matmul` receives contiguous memory.
/// - Use `broadcast_add` when adding `mask` so a 2D `[seq_q, seq_k]` mask broadcasts over
///   leading `[batch, n_heads]` dimensions.
/// - Implement softmax along `D::Minus1` using the max-shift trick (`scores - max`) for
///   numerical stability; `-inf - finite = -inf` and `exp(-inf) = 0.0`.
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
