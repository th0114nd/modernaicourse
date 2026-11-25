//! Problem 1: Multi-batch KV-cached multi-head self-attention (`MultiHeadAttentionKVCache`).

#[allow(unused_imports)]
use candle_core::{DType, Device, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{Linear, self_attention};

/// Default batch capacity of the KV cache (matches `max_cache_batches=64` in `hw7.py`).
pub const DEFAULT_MAX_CACHE_BATCHES: usize = 64;

/// Multi-head self-attention with a pre-allocated multi-batch KV cache for parallel decoding.
///
/// Unlike Homework 4–6's `[1, max_cache_size, dim]` cache, Homework 7 allocates caches of shape
/// `[max_cache_batches, max_cache_size, dim]` so a single prompt can generate up to
/// `max_cache_batches` completions simultaneously.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionKVCache {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wp: Linear,
    /// Cached keys of shape `[max_cache_batches, max_cache_size, dim]`.
    pub k_cache: Tensor,
    /// Cached values of shape `[max_cache_batches, max_cache_size, dim]`.
    pub v_cache: Tensor,
    pub dim: usize,
    pub n_heads: usize,
    pub max_cache_size: usize,
    pub max_cache_batches: usize,
}

impl MultiHeadAttentionKVCache {
    /// Initialize multi-head attention with zero-filled KV caches of shape
    /// `[max_cache_batches, max_cache_size, dim]`.
    #[allow(unused_variables)]
    pub fn new(
        dim: usize,
        n_heads: usize,
        max_cache_size: usize,
        max_cache_batches: usize,
        device: &Device,
    ) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Compute multi-head self-attention on `x` of shape `[batch_size, seq_len, dim]`,
    /// optionally updating and attending over the first `batch_size` rows of the KV cache.
    ///
    /// When `use_kv_cache` is `true`:
    /// - Write the newly projected keys/values into
    ///   `k_cache[0..batch_size, seq_pos..seq_pos + seq_len, 0..dim]` (detached).
    /// - Read back `k_cache[0..batch_size, 0..seq_pos + seq_len, 0..dim]` and similarly for `v_cache`.
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

    /// Return all trainable parameters owned by this layer.
    pub fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
