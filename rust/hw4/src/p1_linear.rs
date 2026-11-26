//! Problem 1: `Linear` layer (no bias).

use candle_core::{bail, DType, Device, Result, Tensor, Var};

/// A linear (fully connected) layer without bias, implementing $f(X) = X W^T$.
///
/// # Requirements
/// - Store the weight as a public `weight: Var` of shape `[out_dim, in_dim]` (`DType::F32`).
///   Initialize with `Var::zeros((out_dim, in_dim), DType::F32, device)?` (Candle's safe
///   counterpart to PyTorch's `torch.empty` when weights will be loaded from a checkpoint).
/// - `forward(&self, x)` computes $X W^T$ for arbitrary leading batch dimensions:
///   input `[..., in_dim]` $\to$ output `[..., out_dim]`.
///   Hint: use [`Tensor::broadcast_matmul`] so the 2D matrix $W^T$ broadcasts across
///   leading dimensions.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix of shape `[out_dim, in_dim]`.
    pub weight: Var,
}

impl Linear {
    /// Initialize the `Linear` layer with shape `[out_dim, in_dim]`.
    ///
    /// # Inputs
    /// - `in_dim`: input feature dimension
    /// - `out_dim`: output feature dimension
    /// - `device`: Candle device on which to allocate `weight`
    #[allow(unused_variables)]
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        if in_dim == 0 || out_dim == 0 {
            bail!("degenerate dims");
        }
        Ok(Self {
            weight: Var::zeros((out_dim, in_dim), DType::F32, device)?,
        })
        // END YOUR CODE
    }

    /// Compute the forward pass $X W^T$.
    ///
    /// # Inputs
    /// - `x`: input tensor of shape `[..., in_dim]`
    ///
    /// # Output
    /// Output tensor of shape `[..., out_dim]`.
    #[allow(unused_variables)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        x.broadcast_matmul(&self.weight.t()?)
        // END YOUR CODE
    }
}
