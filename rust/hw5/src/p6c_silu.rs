//! Problem 6c: SiLU (Swish) activation function.

#[allow(unused_imports)]
use candle_core::{Result, Tensor};

/// Compute the elementwise SiLU activation:
///
/// $$\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$$
///
/// Implemented using primitive tensor ops (do not call `Tensor::silu`).
#[allow(unused_variables)]
pub fn silu(x: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
