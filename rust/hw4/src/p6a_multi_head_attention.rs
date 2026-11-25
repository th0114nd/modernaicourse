//! Problem 6a: Multi-head attention (without KV caching).

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor};

#[allow(unused_imports)]
use crate::{Linear, self_attention};

/// Multi-head self-attention without KV caching.
///
/// # Steps
/// 1. Compute $Q = X W_q^T$, $K = X W_k^T$, $V = X W_v^T$ using [`Linear`] layers
///    `wq`, `wk`, `wv`.
/// 2. Split the last dimension `dim` into `n_heads` heads of size `head_dim = dim / n_heads`:
///    reshape `[batch, seq_len, dim]` $\to$ `[batch, seq_len, n_heads, head_dim]`,
///    then transpose axes `(1, 2)` $\to$ `[batch, n_heads, seq_len, head_dim]`.
/// 3. Apply [`self_attention`] to `(Q, K, V, mask)`.
/// 4. Transpose axes `(1, 2)` back, make contiguous, reshape to `[batch, seq_len, dim]`,
///    and apply the output projection `wp`.
///
/// For this first version, ignore `seq_pos` and `use_kv_cache`.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wp: Linear,
    pub dim: usize,
    pub n_heads: usize,
}

impl MultiHeadAttention {
    /// Initialize multi-head attention with `dim`-to-`dim` linear projections `wq`, `wk`, `wv`, `wp`.
    ///
    /// # Inputs
    /// - `dim`: model dimension (must be divisible by `n_heads`)
    /// - `n_heads`: number of attention heads
    /// - `device`: Candle device on which to allocate weights
    #[allow(unused_variables)]
    pub fn new(dim: usize, n_heads: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Compute multi-head self-attention for input `x` of shape `[batch, seq_len, dim]`.
    ///
    /// # Inputs
    /// - `x`: input tensor of shape `[batch, seq_len, dim]`
    /// - `mask`: optional additive attention mask (e.g. `[seq_len, seq_len]`)
    /// - `seq_pos`: unused in this non-caching version
    /// - `use_kv_cache`: unused in this non-caching version
    ///
    /// # Output
    /// Output tensor of shape `[batch, seq_len, dim]`.
    #[allow(unused_variables)]
    pub fn forward(
        &self,
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
