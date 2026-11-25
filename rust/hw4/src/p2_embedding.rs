//! Problem 2: `Embedding` layer.

#[allow(unused_imports)]
use candle_core::{DType, Device, Result, Tensor, Var};

/// An embedding lookup table that maps integer token indices to dense vectors.
///
/// # Requirements
/// - Store the embedding table in the public `weight: Var` field of shape
///   `[num_tokens, dim]` (`DType::F32`), initialized with `Var::zeros`.
/// - `forward(&self, y)` takes integer token indices `y` (`DType::U32`) of arbitrary
///   shape `[...]` (e.g., `[seq_len]` or `[batch, seq_len]`) and returns embeddings
///   of shape `[..., dim]`.
///   Hint: in Candle, [`Tensor::index_select`] requires a 1D index tensor, so flatten `y`
///   with `y.flatten_all()?`, select rows along dimension `0`, and reshape to `[...y.dims(), dim]`.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding weight matrix of shape `[num_tokens, dim]`.
    pub weight: Var,
}

impl Embedding {
    /// Initialize the `Embedding` layer.
    ///
    /// # Inputs
    /// - `num_tokens`: vocabulary size
    /// - `dim`: embedding dimension
    /// - `device`: Candle device on which to allocate `weight`
    #[allow(unused_variables)]
    pub fn new(num_tokens: usize, dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Look up embeddings for arbitrary-shape integer token indices `y`.
    ///
    /// # Inputs
    /// - `y`: token index tensor (`DType::U32`) of arbitrary shape `[...]`
    ///
    /// # Output
    /// Dense embeddings of shape `[..., dim]`.
    #[allow(unused_variables)]
    pub fn forward(&self, y: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
