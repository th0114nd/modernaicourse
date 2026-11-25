//! Problem 1: `Linear` layer (no bias) with Kaiming/He initialization.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

use crate::nn::Module;

/// A bias-free linear transformation $y = x W^T$ with trainable weight matrix
/// $W \in \mathbb{R}^{\text{out\_dim} \times \text{in\_dim}}$.
///
/// # Requirements
/// - Store the trainable weights in the public `weight` field as a [`Var`] of shape
///   `[out_dim, in_dim]` (`DType::F32`).
/// - Initialize `weight` from $\mathcal{N}(0, 2 / \text{in\_dim})$, i.e., standard
///   deviation $\sqrt{2 / \text{in\_dim}}$ (Kaiming/He initialization for ReLU networks).
/// - `forward` must support both 2D inputs `[batch_size, in_dim]` and higher-rank
///   batched inputs `[..., in_dim]`, producing `[..., out_dim]`.
///   Hint: use [`Tensor::broadcast_matmul`] rather than [`Tensor::matmul`] so 2D
///   `weight^T` broadcasts over leading batch dimensions.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Trainable weight matrix of shape `[out_dim, in_dim]`.
    pub weight: Var,
}

impl Linear {
    /// Initialize a linear layer with Gaussian weights scaled by $\sqrt{2 / \text{in\_dim}}$.
    ///
    /// # Inputs
    /// - `in_dim`: input feature dimension (must be `>= 1`)
    /// - `out_dim`: output feature dimension (must be `>= 1`)
    /// - `device`: Candle device on which to allocate `weight`
    #[allow(unused_variables)]
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl Module for Linear {
    /// Apply the linear layer $x \mapsto x W^T$ to an input tensor of shape `[..., in_dim]`.
    #[allow(unused_variables)]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Return the single trainable parameter `[self.weight]`.
    fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
