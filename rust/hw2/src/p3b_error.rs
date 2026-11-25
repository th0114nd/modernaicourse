//! Problem 3b: Average classification error rate.

use candle_core::{bail, DType, Result, Tensor};

/// Compute the average hard-prediction classification error rate.
///
/// Each row of `y_pred` predicts the class with the largest logit (`argmax` along
/// dimension `1`). Returns the fraction of examples where the predicted class
/// differs from `y`.
///
/// # Inputs
/// - `y_pred`: 2D `f32` tensor of shape `[N, k]` containing predicted logits per example
/// - `y`: 1D `u32` tensor of shape `[N]` containing target class indices in `0..k`
///
/// # Output
/// - Scalar (0D, shape `[]`) `f32` tensor in `[0.0, 1.0]` equal to the mean error rate
pub fn error(y_pred: &Tensor, y: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    let (n, _k) = y_pred.dims2()?;
    let n2 = y.dims1()?;
    if n != n2 {
        bail!("dim mismatch");
    }
    y_pred.argmax(1)?.ne(y)?.to_dtype(DType::F32)?.mean_all()

    // END YOUR CODE
}
