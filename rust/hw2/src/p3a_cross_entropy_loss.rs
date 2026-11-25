//! Problem 3a: Average cross-entropy loss.

use candle_core::bail;
use candle_core::{Result, Tensor};

/// Compute the average cross-entropy loss between predicted logits and integer class labels.
///
/// For each row $\hat{y} \in \mathbb{R}^k$ and target class $y \in \{0, \dots, k-1\}$:
///
/// $$L_{ce}(\hat{y}, y) = -\hat{y}_y + \log \sum_{j=0}^{k-1} \exp(\hat{y}_j)$$
///
/// Hint: use [`Tensor::log_sum_exp`] for numerical stability and [`Tensor::gather`]
/// to index $\hat{y}_y$ differentiably without loops.
///
/// # Inputs
/// - `y_pred`: 2D `f32` tensor of shape `[N, k]` containing predicted logits per example
/// - `y`: 1D `u32` tensor of shape `[N]` containing target class indices in `0..k`
///
/// # Output
/// - Scalar (0D, shape `[]`) `f32` tensor equal to the mean cross-entropy loss
#[allow(unused_variables)]
pub fn cross_entropy_loss(y_pred: &Tensor, y: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    let (n, k) = y_pred.dims2()?;
    let np = y.dims1()?;
    if n != np {
        bail!("mismatched dims");
    }
    let y_unsqueezed = y.unsqueeze(1)?;
    assert_eq!((n, 1), y_unsqueezed.dims2()?);
    // y_pred.gather: N x 1; contracting kind of?
    let gathered = y_pred.gather(&y_unsqueezed, 1)?;
    let ypred_y = gathered.squeeze(1)?;
    assert_eq!(n, ypred_y.dims1()?);
    let denom = y_pred.log_sum_exp(1)?;
    let per_item_loss = denom.sub(&ypred_y)?;
    per_item_loss.mean_all()
    // END YOUR CODE
}
