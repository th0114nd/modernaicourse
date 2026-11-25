//! Problem 4: RMSNorm (Root Mean Square Layer Normalization).

#[allow(unused_imports)]
use candle_core::{D, DType, Device, Result, Tensor, Var};

/// Root Mean Square Layer Normalization:
///
/// $$\text{RMSNorm}(X) = \frac{X}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} X_j^2 + \epsilon}} \odot w$$
///
/// # Requirements
/// - Store the learnable scale parameter in the public `weight: Var` field of shape
///   `[dim]` (`DType::F32`), initialized to all ones (`Var::ones`).
/// - Store `eps: f64` in the public `eps` field.
/// - Normalize over the last dimension (`D::Minus1`) for arbitrary input shape `[..., dim]`.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    /// Learnable scale vector of shape `[dim]`, initialized to ones.
    pub weight: Var,
    /// Small constant added inside the square root for numerical stability.
    pub eps: f64,
}

impl RMSNorm {
    /// Initialize an `RMSNorm` layer with `weight = ones(dim)` and the given `eps`.
    ///
    /// # Inputs
    /// - `dim`: feature dimension (last axis of inputs)
    /// - `eps`: epsilon for numerical stability (typically `1e-5`)
    /// - `device`: Candle device on which to allocate `weight`
    #[allow(unused_variables)]
    pub fn new(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Apply RMS normalization along the last dimension.
    ///
    /// # Inputs
    /// - `x`: input tensor of shape `[..., dim]`
    ///
    /// # Output
    /// Normalized tensor of shape `[..., dim]`.
    #[allow(unused_variables)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
