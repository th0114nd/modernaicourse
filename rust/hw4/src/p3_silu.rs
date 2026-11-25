//! Problem 3: SiLU (Sigmoid Linear Unit) activation function.

#[allow(unused_imports)]
use candle_core::{Result, Tensor};

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
#[allow(unused_variables)]
pub fn silu(x: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
