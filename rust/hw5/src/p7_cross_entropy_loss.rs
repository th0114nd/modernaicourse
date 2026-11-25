//! Problem 7: Arbitrary-rank cross-entropy loss.

#[allow(unused_imports)]
use candle_core::{D, Result, Tensor};

/// Compute the average cross-entropy loss over an arbitrary-rank batch of logits and targets:
///
/// $$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \left( \log \sum_{j=1}^{k} e^{z_{i,j}} - z_{i, y_i} \right)$$
///
/// # Inputs
/// - `logits`: $N$-D float tensor of shape `[..., k]`
/// - `y`: $(N-1)$-D integer (`DType::U32`) tensor of target class indices in `[0, k)` with shape `[...]`
///
/// # Output
/// Scalar (rank-0 `[]`) `Tensor` containing the mean cross-entropy loss.
///
/// # Implementation hints
/// - Reshape `logits` to `[M, k]` and `y` to `[M]` (where `M = y.elem_count()`).
/// - Subtract row-wise maxima (`logits_2d.max_keepdim(D::Minus1)?`) before `exp()` for numerical stability.
/// - Extract target logits with `logits_2d.gather(&y_1d.unsqueeze(1)?, 1)?.squeeze(1)?`.
#[allow(unused_variables)]
pub fn cross_entropy_loss(logits: &Tensor, y: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
