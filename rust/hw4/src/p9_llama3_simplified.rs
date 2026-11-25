//! Problem 9: `Llama3Simplified` language model.

use std::collections::HashMap;

#[allow(unused_imports)]
use candle_core::{DType, Device, Result, Tensor};

use crate::llama::CausalLM;
#[allow(unused_imports)]
use crate::{Embedding, Linear, RMSNorm, TransformerBlock};

/// Simplified Llama 3 model with learned positional embeddings (instead of RoPE).
///
/// # Components
/// - `embedding`: [`Embedding`] mapping token IDs to `dim`-dimensional vectors.
/// - `pos_embeddings`: [`Tensor`] of shape `[max_seq_len, dim]` (`DType::F32`), initialized to zeros.
/// - `layers`: `Vec<TransformerBlock>` of length `num_layers` (each with `max_cache_size = max_seq_len`).
/// - `norm`: final [`RMSNorm`] (`eps = 1e-5`).
/// - `output`: final [`Linear`] projection (`dim -> num_tokens`).
/// - `mask`: causal attention mask [`Tensor`] of shape `[max_seq_len, max_seq_len]` (`DType::F32`),
///   with `0.0` on and below the main diagonal (`col <= row`) and `f32::NEG_INFINITY` strictly
///   above the main diagonal (`col > row`).
///
/// # Forward Pass
/// 1. Look up token embeddings for `tokens` (`[batch, seq_len]`) and add positional embeddings
///    for positions `seq_pos .. seq_pos + seq_len`.
/// 2. Slice the causal mask:
///    - When `use_kv_cache == true`: rows `seq_pos .. seq_pos + seq_len` and columns
///      `0 .. seq_pos + seq_len` (so each new query attends to all past + current cached keys).
///    - When `use_kv_cache == false`: rows `0 .. seq_len` and columns `0 .. seq_len`.
/// 3. Pass sequentially through every block in `self.layers`.
/// 4. Apply `self.norm`, then `self.output`, returning logits of shape `[batch, seq_len, num_tokens]`.
#[derive(Debug, Clone)]
pub struct Llama3Simplified {
    pub embedding: Embedding,
    pub pos_embeddings: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub output: Linear,
    pub mask: Tensor,
}

impl Llama3Simplified {
    /// Initialize the `Llama3Simplified` model.
    ///
    /// # Inputs
    /// - `num_tokens`: vocabulary size
    /// - `dim`: model hidden dimension
    /// - `n_heads`: number of attention heads
    /// - `max_seq_len`: maximum sequence length (also used as KV cache size and mask size)
    /// - `ffn_dim`: feed-forward hidden dimension
    /// - `num_layers`: number of Transformer blocks
    /// - `device`: Candle device on which to allocate weights and buffers
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

    /// Compute logits for token indices `tokens` of shape `[batch, seq_len]`.
    ///
    /// # Inputs
    /// - `tokens`: token IDs (`DType::U32`) of shape `[batch, seq_len]`
    /// - `seq_pos`: starting position in the sequence
    /// - `use_kv_cache`: whether to use each layer's KV cache
    ///
    /// # Output
    /// Logits tensor of shape `[batch, seq_len, num_tokens]`.
    #[allow(unused_variables)]
    pub fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Provided helper (do not modify): load pretrained Llama 3 Simplified weights
    /// from a checkpoint map (`name -> Tensor`) into this model.
    pub fn load_llama_weights(&mut self, checkpoint: &HashMap<String, Tensor>) -> Result<()> {
        let device = self.embedding.weight.device().clone();
        let get = |key: &str| -> Result<Tensor> {
            let t = checkpoint
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing checkpoint key: {key}")))?;
            t.to_device(&device)?.to_dtype(DType::F32)
        };

        self.embedding.weight.set(&get("tok_embeddings.weight")?)?;
        self.pos_embeddings = get("pos_embeddings.weight")?;
        self.norm.weight.set(&get("norm.weight")?)?;
        self.output.weight.set(&get("output.weight")?)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer
                .attn
                .wq
                .weight
                .set(&get(&format!("layers.{i}.attention.wq.weight"))?)?;
            layer
                .attn
                .wk
                .weight
                .set(&get(&format!("layers.{i}.attention.wk.weight"))?)?;
            layer
                .attn
                .wv
                .weight
                .set(&get(&format!("layers.{i}.attention.wv.weight"))?)?;
            layer
                .attn
                .wp
                .weight
                .set(&get(&format!("layers.{i}.attention.wo.weight"))?)?;

            layer
                .mlp
                .w1
                .weight
                .set(&get(&format!("layers.{i}.feed_forward.w1.weight"))?)?;
            layer
                .mlp
                .w2
                .weight
                .set(&get(&format!("layers.{i}.feed_forward.w2.weight"))?)?;
            layer
                .mlp
                .w3
                .weight
                .set(&get(&format!("layers.{i}.feed_forward.w3.weight"))?)?;

            layer
                .norm1
                .weight
                .set(&get(&format!("layers.{i}.attention_norm.weight"))?)?;
            layer
                .norm2
                .weight
                .set(&get(&format!("layers.{i}.ffn_norm.weight"))?)?;
        }

        Ok(())
    }
}

impl CausalLM for Llama3Simplified {
    fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor> {
        Llama3Simplified::forward(self, tokens, seq_pos, use_kv_cache)
    }
}
