//! Part 0g: Two-layer `MLP` feed-forward network.
//!
//! Identical to HW5 (`MLP`); copy your implementation from `hw5`.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{Linear, silu};

/// Standard two-layer feed-forward network with SiLU nonlinearity:
///
/// $$\text{MLP}(X) = \text{SiLU}(X W_1^T) W_2^T$$
#[derive(Debug, Clone)]
pub struct MLP {
    /// Up-projection (`dim -> ffn_dim`).
    pub w1: Linear,
    /// Down-projection (`ffn_dim -> dim`).
    pub w2: Linear,
}

impl MLP {
    /// Initialize the two-layer `MLP`.
    #[allow(unused_variables)]
    pub fn new(dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Apply $\text{SiLU}(X W_1^T) W_2^T$ to `x` of shape `[..., dim]`.
    #[allow(unused_variables)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Return all trainable parameters owned by this layer.
    pub fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
