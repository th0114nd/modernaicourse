//! Problem 6i: Complete trainable `LLM` decoder-only Transformer.

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{CausalLM, Embedding, Linear, TransformerBlock, rms_norm};

/// Complete decoder-only Transformer language model trained from scratch on TinyStories.
///
/// # Requirements (differences from HW4's `Llama3Simplified`)
/// - `embedding: Embedding` (`num_tokens`, `dim`) — initialized $\mathcal{N}(0, 1)$.
/// - `pos_embeddings: Var` of shape `[max_seq_len, dim]` — trainable, initialized $\mathcal{N}(0, 1)$.
/// - `layers: Vec<TransformerBlock>` of length `num_layers`.
/// - `output: Linear` (`dim -> num_tokens`).
/// - `mask: Tensor` of shape `[max_seq_len, max_seq_len]` (`DType::F32`), causal upper-triangular
///   mask (`0.0` on and below diagonal, `-inf` strictly above diagonal).
/// - Final normalization uses functional [`rms_norm`] with `eps = 1e-5`.
/// - `parameters(&self)` collects all trainable [`Var`]s across `embedding`, `pos_embeddings`,
///   all `layers`, and `output`.
#[derive(Debug, Clone)]
pub struct LLM {
    pub embedding: Embedding,
    /// Trainable positional embeddings of shape `[max_seq_len, dim]`.
    pub pos_embeddings: Var,
    pub layers: Vec<TransformerBlock>,
    pub output: Linear,
    /// Precomputed causal attention mask of shape `[max_seq_len, max_seq_len]`.
    pub mask: Tensor,
}

impl LLM {
    /// Initialize the `LLM` model.
    #[allow(unused_variables)]
    pub fn new(
        num_tokens: usize,
        dim: usize,
        n_heads: usize,
        max_seq_len: usize,
        ffn_dim: usize,
        num_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Run the forward pass over `tokens` (`[batch, seq_len]`, `DType::U32`).
    ///
    /// Returns logits of shape `[batch, seq_len, num_tokens]`.
    #[allow(unused_variables)]
    pub fn forward(
        &mut self,
        tokens: &Tensor,
        seq_pos: usize,
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Return all trainable parameters owned by the model.
    pub fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

/// Provided trait impl (do not modify): wires [`LLM`] into [`crate::train_llm`] and [`crate::generate`].
impl CausalLM for LLM {
    fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor> {
        LLM::forward(self, tokens, seq_pos, use_kv_cache)
    }

    fn device(&self) -> &Device {
        self.embedding.weight.device()
    }
}
