//! Problem 6a: Multi-head attention (without KV caching).

use candle_core::{bail, Device, Result, Tensor};

use crate::{self_attention, Linear};

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
        if dim == 0 || n_heads == 0 {
            bail!("degeneracy");
        }
        if !dim.is_multiple_of(n_heads) {
            bail!("dim heads alignment");
        }
        Ok(Self {
            wq: Linear::new(dim, dim, device)?,
            wk: Linear::new(dim, dim, device)?,
            wv: Linear::new(dim, dim, device)?,
            wp: Linear::new(dim, dim, device)?,
            dim,
            n_heads,
        })
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
        let (batch, seq_len, dim) = x.dims3()?;
        if dim != self.dim {
            bail!("dim mismatch");
        }
        let head_dim = self.dim / self.n_heads;
        let new_dims = &[batch, seq_len, self.n_heads, head_dim];
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
