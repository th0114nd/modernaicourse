//! Part 0b: `Embedding` layer with standard normal random initialization.
//!
//! Identical to HW6 (`Embedding`); copy your implementation from `hw6`.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

/// Trainable token embedding lookup table mapping integer token IDs to dense vectors.
///
/// # Requirements
/// - Store `pub weight: Var` of shape `[num_tokens, dim]` (`DType::F32`).
/// - Initialize `weight` with standard normal entries $\mathcal{N}(0, 1)$.
/// - `forward(&self, y)` maps integer token IDs of arbitrary shape `[...]` (`DType::U32`)
///   to embeddings of shape `[..., dim]`.
/// - `parameters(&self)` returns `vec![self.weight.clone()]`.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding table of shape `[num_tokens, dim]`.
    pub weight: Var,
}

impl Embedding {
    /// Initialize the `Embedding` table with $\mathcal{N}(0, 1)$ weights.
    #[allow(unused_variables)]
    pub fn new(num_tokens: usize, dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Look up embeddings for an arbitrary-rank `u32` token ID tensor `y`.
    #[allow(unused_variables)]
    pub fn forward(&self, y: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Return all trainable parameters owned by this layer.
    pub fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
