//! Part 0j: Complete trainable `LLM` decoder-only Transformer with checkpoint I/O
//! (`load_weights`, `save_weights`) and independent snapshotting (`deep_clone`).
//!
//! Copy your `new`, `forward`, and `parameters` implementations from HW6.
//! Leave the provided `load_weights`, `save_weights`, `save_weights_bf16`, `deep_clone`,
//! and `CausalLM` methods untouched.

use std::collections::HashMap;
use std::path::Path;

#[allow(unused_imports)]
use candle_core::{DType, Device, Error, Result, Tensor, Var};

#[allow(unused_imports)]
use crate::{
    CausalLM, Embedding, Linear, MLP, MultiHeadAttentionKVCache, TransformerBlock,
    load_checkpoint_tensors, rms_norm,
};

/// Complete decoder-only Transformer language model used for SFT reasoning and RL on GSM8K.
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
    /// Initialize the `LLM` model (copy your HW6 implementation).
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
    /// Returns logits of shape `[batch, seq_len, num_tokens]` (copy your HW6 implementation).
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

    /// Return all trainable parameters owned by the model (copy your HW6 implementation).
    pub fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Provided method (do not modify): load weights from either a PyTorch `.pth` checkpoint
    /// (`candle_core::pickle::read_all`) or a `.safetensors` checkpoint
    /// (`candle_core::safetensors::load`), using the Llama-like key layout from `hw7.py`.
    ///
    /// All tensors are converted to `DType::F32` on the model's current device.
    pub fn load_weights<P: AsRef<Path>>(&mut self, filename: P) -> Result<()> {
        let device = self.embedding.weight.device().clone();
        let checkpoint = load_checkpoint_tensors(filename, &device)?;

        let get = |key: &str| -> Result<&Tensor> {
            checkpoint
                .get(key)
                .ok_or_else(|| Error::Msg(format!("missing checkpoint tensor '{key}'")))
        };

        self.embedding.weight.set(get("tok_embeddings.weight")?)?;
        self.pos_embeddings.set(get("pos_embeddings.weight")?)?;
        self.output.weight.set(get("output.weight")?)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer
                .attn
                .wq
                .weight
                .set(get(&format!("layers.{i}.attention.wq.weight"))?)?;
            layer
                .attn
                .wk
                .weight
                .set(get(&format!("layers.{i}.attention.wk.weight"))?)?;
            layer
                .attn
                .wv
                .weight
                .set(get(&format!("layers.{i}.attention.wv.weight"))?)?;
            // Note: Llama checkpoints store the output projection under `.wo.weight`.
            layer
                .attn
                .wp
                .weight
                .set(get(&format!("layers.{i}.attention.wo.weight"))?)?;

            layer
                .mlp
                .w1
                .weight
                .set(get(&format!("layers.{i}.feed_forward.w1.weight"))?)?;
            layer
                .mlp
                .w2
                .weight
                .set(get(&format!("layers.{i}.feed_forward.w2.weight"))?)?;
        }
        Ok(())
    }

    /// Provided method (do not modify): save weights to a `.safetensors` file using the
    /// exact Llama-like key layout from `hw7.py` (readable back via [`LLM::load_weights`]).
    pub fn save_weights<P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        self.save_weights_with_dtype(filename, DType::F32)
    }

    /// Provided method (do not modify): save weights in `DType::BF16` `.safetensors` format
    /// (matching `hw7.py`'s `.to(torch.bfloat16).cpu()` checkpoint compression).
    pub fn save_weights_bf16<P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        self.save_weights_with_dtype(filename, DType::BF16)
    }

    fn save_weights_with_dtype<P: AsRef<Path>>(&self, filename: P, dtype: DType) -> Result<()> {
        let prep = |v: &Var| -> Result<Tensor> {
            v.as_detached_tensor()
                .to_device(&Device::Cpu)?
                .to_dtype(dtype)
        };

        let mut checkpoint: HashMap<String, Tensor> =
            HashMap::with_capacity(3 + self.layers.len() * 6);
        checkpoint.insert(
            "tok_embeddings.weight".to_string(),
            prep(&self.embedding.weight)?,
        );
        checkpoint.insert(
            "pos_embeddings.weight".to_string(),
            prep(&self.pos_embeddings)?,
        );
        checkpoint.insert("output.weight".to_string(), prep(&self.output.weight)?);

        for (i, layer) in self.layers.iter().enumerate() {
            checkpoint.insert(
                format!("layers.{i}.attention.wq.weight"),
                prep(&layer.attn.wq.weight)?,
            );
            checkpoint.insert(
                format!("layers.{i}.attention.wk.weight"),
                prep(&layer.attn.wk.weight)?,
            );
            checkpoint.insert(
                format!("layers.{i}.attention.wv.weight"),
                prep(&layer.attn.wv.weight)?,
            );
            checkpoint.insert(
                format!("layers.{i}.attention.wo.weight"),
                prep(&layer.attn.wp.weight)?,
            );
            checkpoint.insert(
                format!("layers.{i}.feed_forward.w1.weight"),
                prep(&layer.mlp.w1.weight)?,
            );
            checkpoint.insert(
                format!("layers.{i}.feed_forward.w2.weight"),
                prep(&layer.mlp.w2.weight)?,
            );
        }

        candle_core::safetensors::save(&checkpoint, filename)
    }

    /// Provided method (do not modify): create a **deep** independent copy of this `LLM`
    /// with freshly allocated [`Var`]s and distinct [`TensorId`](candle_core::TensorId)s.
    ///
    /// Useful for keeping a frozen reference snapshot (e.g., comparing SFT vs RL policies or
    /// adding an optional KL penalty) without Candle's shallow `Var::clone()` aliasing.
    pub fn deep_clone(&self) -> Result<Self> {
        let clone_var = |v: &Var| -> Result<Var> {
            let copied = v.as_detached_tensor().copy()?;
            Var::from_tensor(&copied)
        };
        let clone_linear = |l: &Linear| -> Result<Linear> {
            Ok(Linear {
                weight: clone_var(&l.weight)?,
            })
        };

        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(TransformerBlock {
                attn: MultiHeadAttentionKVCache {
                    wq: clone_linear(&layer.attn.wq)?,
                    wk: clone_linear(&layer.attn.wk)?,
                    wv: clone_linear(&layer.attn.wv)?,
                    wp: clone_linear(&layer.attn.wp)?,
                    k_cache: layer.attn.k_cache.copy()?,
                    v_cache: layer.attn.v_cache.copy()?,
                    dim: layer.attn.dim,
                    n_heads: layer.attn.n_heads,
                    max_cache_size: layer.attn.max_cache_size,
                    max_cache_batches: layer.attn.max_cache_batches,
                },
                mlp: MLP {
                    w1: clone_linear(&layer.mlp.w1)?,
                    w2: clone_linear(&layer.mlp.w2)?,
                },
            });
        }

        Ok(Self {
            embedding: Embedding {
                weight: clone_var(&self.embedding.weight)?,
            },
            pos_embeddings: clone_var(&self.pos_embeddings)?,
            layers,
            output: clone_linear(&self.output)?,
            mask: self.mask.copy()?,
        })
    }
}

/// Provided trait impl (do not modify): wires [`LLM`] into [`crate::generate_parallel`],
/// [`crate::generate_parallel_tool`], [`crate::train_llm_sft`], [`crate::eval_model`],
/// [`crate::rl_loss`], and [`crate::train_llm_rl`].
impl CausalLM for LLM {
    fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor> {
        LLM::forward(self, tokens, seq_pos, use_kv_cache)
    }

    fn device(&self) -> &Device {
        self.embedding.weight.device()
    }
}
