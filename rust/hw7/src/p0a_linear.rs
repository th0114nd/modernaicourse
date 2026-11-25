//! Part 0a: `Linear` layer (no bias) with Kaiming-style random initialization.
//!
//! Identical to HW6 (`Linear`); copy your implementation from `hw6`.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

/// A trainable linear (fully connected) layer without bias, implementing $f(X) = X W^T$.
///
/// # Requirements
/// - Store `pub weight: Var` of shape `[out_dim, in_dim]` (`DType::F32`).
/// - Initialize `weight` with random normal entries scaled by $\sqrt{2 / \text{in\_dim}}$.
/// - `forward(&self, x)` computes $X W^T$ for arbitrary leading dimensions
///   `[..., in_dim]` $\to$ `[..., out_dim]`.
/// - `parameters(&self)` returns `vec![self.weight.clone()]`.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix of shape `[out_dim, in_dim]`.
    pub weight: Var,
}

impl Linear {
    /// Initialize the `Linear` layer with $\mathcal{N}(0, 2 / \text{in\_dim})$ weights.
    #[allow(unused_variables)]
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Apply $X W^T$ to an input tensor of shape `[..., in_dim]`.
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
