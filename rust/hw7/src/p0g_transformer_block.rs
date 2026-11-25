//! Part 0g: Pre-norm `TransformerBlock` with functional `rms_norm`, batched
//! `MultiHeadAttentionKVCache`, and `MLP`.
//!
//! Same logic as HW6 (`TransformerBlock`), now using Homework 7's multi-batch
//! [`MultiHeadAttentionKVCache`] (initialized with [`DEFAULT_MAX_CACHE_BATCHES`] = `64`).

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{DEFAULT_MAX_CACHE_BATCHES, MLP, MultiHeadAttentionKVCache, rms_norm};

/// Pre-norm Transformer decoder block with residual connections:
///
/// $$Z = X + \text{Attention}(\text{rms\_norm}(X))$$
/// $$\text{out} = Z + \text{MLP}(\text{rms\_norm}(Z))$$
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttentionKVCache,
    pub mlp: MLP,
}

impl TransformerBlock {
    /// Initialize a `TransformerBlock` whose KV cache supports up to
    /// [`DEFAULT_MAX_CACHE_BATCHES`] (`64`) parallel completions.
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
