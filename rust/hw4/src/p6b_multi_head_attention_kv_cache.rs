//! Problem 6b: Multi-head attention with KV caching.

#[allow(unused_imports)]
use candle_core::{bail, DType, Device, Result, Tensor};

#[allow(unused_imports)]
use crate::{self_attention, Linear};

/// Multi-head self-attention with a pre-allocated KV cache for fast autoregressive decoding.
///
/// # Requirements
/// - Contains linear projections `wq`, `wk`, `wv`, `wp` (each `dim -> dim`).
/// - Contains `k_cache: Tensor` and `v_cache: Tensor` of shape `[1, max_cache_size, dim]`
///   (`DType::F32`), initialized to all zeros.
/// - Because Candle [`Tensor`]s are immutable, `forward` takes `&mut self` so it can update
///   `self.k_cache` and `self.v_cache` cleanly without interior mutability (`RefCell`).
///
/// # Behavior
/// - When `use_kv_cache == true`:
///   1. Project `k = self.wk.forward(x)?` and `v = self.wv.forward(x)?` for the new tokens.
///   2. Write `k` and `v` into `self.k_cache` and `self.v_cache` at positions
///      `seq_pos .. seq_pos + seq_len` (hint: `Tensor::slice_assign`).
///   3. Slice the full cached keys and values `0 .. seq_pos + seq_len` (hint: `Tensor::narrow`)
///      for multi-head attention against the new queries `q`.
/// - When `use_kv_cache == false`:
///   Behave identically to [`crate::MultiHeadAttention`] without modifying `k_cache`/`v_cache`
///   (and support any `batch >= 1`).
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
    ///
    /// # Inputs
    /// - `dim`: model dimension (must be divisible by `n_heads`)
    /// - `n_heads`: number of attention heads
    /// - `max_cache_size`: maximum sequence length stored in `k_cache` / `v_cache`
    /// - `device`: Candle device on which to allocate weights and caches
    pub fn new(dim: usize, n_heads: usize, max_cache_size: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        if dim == 0 || n_heads == 0 || max_cache_size == 0 || !dim.is_multiple_of(n_heads) {
            bail!("bad input");
        }
        Ok(Self {
            wq: Linear::new(dim, dim, device)?,
            wk: Linear::new(dim, dim, device)?,
            wv: Linear::new(dim, dim, device)?,
            wp: Linear::new(dim, dim, device)?,
            k_cache: Tensor::zeros((1, max_cache_size, dim), DType::F32, device)?,
            v_cache: Tensor::zeros((1, max_cache_size, dim), DType::F32, device)?,
            dim,
            n_heads,
            max_cache_size,
        })
        // END YOUR CODE
    }

    /// Compute multi-head attention, optionally reading/updating the KV cache.
    ///
    /// # Inputs
    /// - `x`: input tensor of shape `[batch, seq_len, dim]` (`batch == 1` when `use_kv_cache == true`)
    /// - `mask`: optional additive mask of shape `[seq_len, total_len]` (where `total_len` is
    ///   `seq_pos + seq_len` when caching, or `seq_len` when not caching)
    /// - `seq_pos`: starting sequence index for writing into the KV cache
    /// - `use_kv_cache`: whether to update and attend over the cached keys/values
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
        // - When `use_kv_cache == true`:
        //   1. Project `k = self.wk.forward(x)?` and `v = self.wv.forward(x)?` for the new tokens.
        //   2. Write `k` and `v` into `self.k_cache` and `self.v_cache` at positions
        //      `seq_pos .. seq_pos + seq_len` (hint: `Tensor::slice_assign`).
        //   3. Slice the full cached keys and values `0 .. seq_pos + seq_len` (hint: `Tensor::narrow`)
        //      for multi-head attention against the new queries `q`.
        let (batch, seq_len, dim) = x.dims3()?;
        if dim != self.dim {
            bail!("dim mismatch");
        }
        let head_dim = self.dim / self.n_heads;
        let new_dims = &[batch, seq_len, self.n_heads, head_dim];
        if use_kv_cache {
            if batch != 1 {
                bail!("bad batch");
            }
            // k_cache: Tensor::zeros((1, max_cache_size, dim), DType::F32, device)?,
            // k: [1, seq_len, dim]
            let k = self.wk.forward(x)?;
            let v = self.wv.forward(x)?;
            let target = &[0..1, seq_pos..seq_pos + seq_len, 0..dim];
            self.k_cache = self.k_cache.slice_assign(target, &k)?;
            self.v_cache = self.v_cache.slice_assign(target, &v)?;
            let sa = {
                let q = self.wq.forward(x)?.reshape(new_dims)?.transpose(1, 2)?;
                let k = self.k_cache.narrow(1, 0, seq_pos + seq_len)?;
                let v = self.v_cache.narrow(1, 0, seq_pos + seq_len)?;
                self_attention(&q, &k, &v, mask)?
            };
        }
        let sa = {
            let q = self.wq.forward(x)?.reshape(new_dims)?.transpose(1, 2)?;
            let k = self.wk.forward(x)?.reshape(new_dims)?.transpose(1, 2)?;
            let v = self.wv.forward(x)?.reshape(new_dims)?.transpose(1, 2)?;
            self_attention(&q, &k, &v, mask)?
        };
        self.wp.forward(
            &sa.transpose(1, 2)?
                .contiguous()?
                .reshape(&[batch, seq_len, dim])?,
        )
        // END YOUR CODE
    }
}
