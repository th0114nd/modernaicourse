//! Part 0d: Functional RMS normalization (no learned weight).
//!
//! Identical to HW5 (`rms_norm`); copy your implementation from `hw5`.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Apply RMS normalization along the final dimension of `x` without a learned scaling weight:
///
/// $$\text{rms\_norm}(X) = \frac{X}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} X_{\dots, j}^2 + \epsilon}}$$
#[allow(unused_variables)]
pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
