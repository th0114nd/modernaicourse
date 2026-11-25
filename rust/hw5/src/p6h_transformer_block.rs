//! Problem 6h: Pre-norm `TransformerBlock` with functional `rms_norm` and `MLP`.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{MLP, MultiHeadAttentionKVCache, rms_norm};

/// Pre-norm Transformer decoder block with residual connections:
///
/// $$Z = X + \text{Attention}(\text{rms\_norm}(X))$$
/// $$\text{out} = Z + \text{MLP}(\text{rms\_norm}(Z))$$
///
/// # Requirements (differences from HW4)
/// - Uses functional [`rms_norm`] with `eps = 1e-5` (no learned norm weights).
/// - Uses the two-layer [`MLP`] instead of `GatedMLP`.
/// - `parameters(&self)` returns all trainable parameters from `attn` and `mlp`.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttentionKVCache,
    pub mlp: MLP,
}

impl TransformerBlock {
    /// Initialize a `TransformerBlock`.
    #[allow(unused_variables)]
    pub fn new(
        dim: usize,
        n_heads: usize,
        ffn_dim: usize,
        max_cache_size: usize,
        device: &Device,
    ) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Apply one Transformer block to `x` of shape `[batch, seq_len, dim]`.
    #[allow(unused_variables)]
    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seq_pos: usize,
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Return all trainable parameters owned by this block.
    pub fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
