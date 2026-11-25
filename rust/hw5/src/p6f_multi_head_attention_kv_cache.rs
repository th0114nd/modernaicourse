//! Problem 6f: Multi-head attention with KV caching.

#[allow(unused_imports)]
use candle_core::{DType, Device, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{Linear, self_attention};

/// Multi-head self-attention with a pre-allocated KV cache for fast autoregressive decoding.
///
/// # Requirements
/// - Contains linear projections `wq`, `wk`, `wv`, `wp` (each `dim -> dim`).
/// - Contains `k_cache` and `v_cache` of shape `[1, max_cache_size, dim]` (`DType::F32`),
///   initialized to all zeros.
/// - When `use_kv_cache == true`:
///   write new `K = wk(x)` and `V = wv(x)` into `k_cache` / `v_cache` at `seq_pos .. seq_pos + seq_len`,
///   and attend over cached positions `0 .. seq_pos + seq_len`.
/// - When `use_kv_cache == false`:
///   run standard multi-head self-attention for any `batch_size >= 1` without mutating the caches
///   (used during training).
/// - `parameters(&self)` returns all trainable `Var`s from `wq`, `wk`, `wv`, and `wp`.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionKVCache {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wp: Linear,
    /// Cached keys of shape `[1, max_cache_size, dim]`.
    pub k_cache: Tensor,
    /// Cached values of shape `[1, max_cache_size, dim]`.
    pub v_cache: Tensor,
    pub dim: usize,
    pub n_heads: usize,
    pub max_cache_size: usize,
}

impl MultiHeadAttentionKVCache {
    /// Initialize multi-head attention with zero-filled KV caches of shape `[1, max_cache_size, dim]`.
    #[allow(unused_variables)]
    pub fn new(dim: usize, n_heads: usize, max_cache_size: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Compute multi-head self-attention, optionally updating and attending over the KV cache.
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
