//! Problem 3: SiLU (Sigmoid Linear Unit) activation function.

use candle_core::{DType, Result, Tensor};

/// Compute the SiLU (Sigmoid Linear Unit) activation function element-wise:
///
/// $$\text{silu}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
///
/// > **Honor system**: Implement using primitive tensor arithmetic (`neg`, `exp`, `recip`,
/// > multiplication/division) rather than calling `Tensor::silu`.
///
/// # Inputs
/// - `x`: input tensor of arbitrary shape
///
/// # Output
/// Tensor of the same shape as `x`.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    x / (x.neg()?.exp()?.affine(1.0, 1.0)?)
    // END YOUR CODE
}
