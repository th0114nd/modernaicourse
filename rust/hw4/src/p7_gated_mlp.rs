//! Problem 7: Gated MLP (SwiGLU feed-forward network as used in Llama).

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor};

#[allow(unused_imports)]
use crate::{Linear, silu};

/// Llama-style Gated MLP (SwiGLU):
///
/// $$\text{GatedMLP}(X) = \left(\text{silu}(X W_1^T) \odot X W_3^T\right) W_2^T$$
///
/// where $W_1, W_3 \in \mathbb{R}^{d_{\text{ffn}} \times d}$ and
/// $W_2 \in \mathbb{R}^{d \times d_{\text{ffn}}}$.
#[derive(Debug, Clone)]
pub struct GatedMLP {
    /// Gate projection (`dim -> ffn_dim`).
    pub w1: Linear,
    /// Down projection (`ffn_dim -> dim`).
    pub w2: Linear,
    /// Up projection (`dim -> ffn_dim`).
    pub w3: Linear,
}

impl GatedMLP {
    /// Initialize the Gated MLP layers `w1`, `w2`, `w3`.
    ///
    /// # Inputs
    /// - `dim`: model dimension
    /// - `ffn_dim`: feed-forward hidden dimension
    /// - `device`: Candle device on which to allocate weights
    #[allow(unused_variables)]
    pub fn new(dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Compute the forward pass $(\text{silu}(X W_1^T) \odot X W_3^T) W_2^T$.
    ///
    /// # Inputs
    /// - `x`: input tensor of shape `[..., dim]`
    ///
    /// # Output
    /// Output tensor of shape `[..., dim]`.
    #[allow(unused_variables)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
