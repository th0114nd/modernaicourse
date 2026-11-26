//! Problem 2: Numerically stable `CrossEntropyLoss`.

#[allow(unused_imports)]
use candle_core::{Result, Tensor};

use crate::nn::Loss;

/// Average cross-entropy loss over a minibatch of predicted logits and class targets.
///
/// For each example with logits $\hat{y} \in \mathbb{R}^k$ and target class
/// $y \in \{0, \dots, k-1\}$:
///
/// $$L_{ce}(\hat{y}, y) = -\hat{y}_y + \log \sum_{j=0}^{k-1} \exp(\hat{y}_j)$$
///
/// Hint: use [`Tensor::log_sum_exp`] (or subtract the row maximum before `exp`) for
/// numerical stability on large logits, and [`Tensor::gather`] to index $\hat{y}_y$
/// differentiably without loops.
#[derive(Debug, Default, Clone, Copy)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new `CrossEntropyLoss` instance.
    pub fn new() -> Self {
        Self
    }
}

impl Loss for CrossEntropyLoss {
    /// Compute average cross-entropy loss over a minibatch.
    ///
    /// # Inputs
    /// - `logits`: 2D `f32` tensor of shape `[N, k]` — predicted logits per example
    /// - `y`: 1D `u32` tensor of shape `[N]` — target class index in `0..k` for each example
    ///
    /// # Output
    /// - Scalar (0D, shape `[]`) `f32` tensor equal to the mean cross-entropy loss
    #[allow(unused_variables)]
    fn forward(&self, logits: &Tensor, y: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        let y = y.unsqueeze(1)?;
        let gathered = logits.gather(&y, 1)?.squeeze(1)?;
        logits.log_sum_exp(1)?.sub(&gathered)?.mean_all()
        // END YOUR CODE
    }
}
