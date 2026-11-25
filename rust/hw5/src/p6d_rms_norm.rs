//! Problem 6d: Functional RMS normalization (no learned weight).

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Apply RMS normalization along the final dimension of `x` without a learned scaling weight:
///
/// $$\text{rms\_norm}(X) = \frac{X}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} X_{\dots, j}^2 + \epsilon}}$$
///
/// # Inputs
/// - `x`: input tensor of shape `[..., dim]`
/// - `eps`: numerical stability constant (typically `1e-5`)
///
/// # Output
/// RMS-normalized tensor of the same shape as `x`.
#[allow(unused_variables)]
pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
