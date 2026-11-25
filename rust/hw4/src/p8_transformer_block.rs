//! Problem 8: Pre-norm Transformer block.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor};

#[allow(unused_imports)]
use crate::{GatedMLP, MultiHeadAttentionKVCache, RMSNorm};

/// A single pre-norm Transformer block with residual connections:
///
/// $$Z = X + \text{MultiHeadAttentionKVCache}(\text{RMSNorm}_1(X))$$
/// $$Y = Z + \text{GatedMLP}(\text{RMSNorm}_2(Z))$$
///
/// Initialize `norm1` and `norm2` with `eps = 1e-5`.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttentionKVCache,
    pub norm1: RMSNorm,
    pub norm2: RMSNorm,
    pub mlp: GatedMLP,
}

impl TransformerBlock {
    /// Initialize the Transformer block.
    ///
    /// # Inputs
    /// - `dim`: model dimension
    /// - `n_heads`: number of attention heads
    /// - `ffn_dim`: feed-forward hidden dimension
    /// - `max_cache_size`: maximum sequence length for the attention KV cache
    /// - `device`: Candle device on which to allocate weights and caches
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

    /// Compute the forward pass of the Transformer block.
    ///
    /// # Inputs
    /// - `x`: input tensor of shape `[batch, seq_len, dim]`
    /// - `mask`: optional additive attention mask
    /// - `seq_pos`: starting position in the sequence (for KV caching)
    /// - `use_kv_cache`: whether to use the attention KV cache
    ///
    /// # Output
    /// Output tensor of shape `[batch, seq_len, dim]`.
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
}
