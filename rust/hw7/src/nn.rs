//! Provided neural-network traits, sampling (1D and batched 2D), and masking helpers
//! for Homework 7.
//!
//! **Do not modify this file.**
//!
//! Provides:
//! - [`CausalLM`]: trait implemented by [`crate::LLM`] (and lightweight mock models in tests)
//!   so [`crate::generate_parallel`], [`crate::generate_parallel_tool`], [`crate::train_llm_sft`],
//!   [`crate::eval_model`], [`crate::rl_loss`], and [`crate::train_llm_rl`] work generically.
//! - [`Optimizer`]: trait implemented by [`crate::Adam`] (and test optimizers like SGD)
//!   for updating parameters from a Candle [`GradStore`].
//! - [`multinomial`]: samples a single token index from a 1D probability tensor.
//! - [`multinomial_batch`]: samples one token index per row from a 2D probability tensor of
//!   shape `[batch_size, vocab_size]` (Candle counterpart to `torch.multinomial(probs, 1)`).
//! - [`mask_indices`]: extracts flattened `DType::U32` indices of non-zero (`true`) entries
//!   in a `DType::U8` mask tensor, enabling PyTorch-style boolean indexing (`logits[mask]`,
//!   `y[mask]`) via `reshape` + `Tensor::index_select`.

use candle_core::{DType, Device, Result, Tensor, backprop::GradStore};
use rand::Rng;

/// Trait for causal language models supporting both training and batched KV-cached inference.
///
/// Implemented by [`crate::LLM`] and by lightweight mock models in the test suites.
pub trait CausalLM {
    /// Compute next-token logits of shape `[batch, seq_len, vocab_size]` from token IDs
    /// of shape `[batch, seq_len]` (`DType::U32`).
    fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor>;

    /// Return the Candle [`Device`] on which this model's parameters reside.
    ///
    /// Counterpart to `next(model.parameters()).device` in PyTorch.
    fn device(&self) -> &Device {
        &Device::Cpu
    }
}

/// Trait for parameter optimizers updated from a Candle [`GradStore`].
pub trait Optimizer {
    /// Apply one parameter update using gradients returned by `loss.backward()?`.
    fn step(&mut self, grads: &GradStore) -> Result<()>;

    /// No-op in Candle: unlike PyTorch's persistent `.grad` buffers,
    /// `Tensor::backward` returns a fresh [`GradStore`] on every step.
    fn zero_grad(&mut self) {}
}

fn sample_row(vals: &[f32], rng: &mut impl Rng) -> Result<u32> {
    if vals.is_empty() {
        candle_core::bail!("multinomial expects a non-empty probability distribution");
    }
    let total: f64 = vals.iter().map(|&p| p.max(0.0) as f64).sum();
    if !total.is_finite() || total <= 0.0 {
        candle_core::bail!("multinomial probabilities must have a finite positive sum");
    }

    let threshold = rng.random::<f64>() * total;
    let mut acc = 0.0_f64;
    for (idx, &p) in vals.iter().enumerate() {
        acc += p.max(0.0) as f64;
        if acc >= threshold {
            return Ok(idx as u32);
        }
    }
    Ok((vals.len() - 1) as u32)
}

/// Sample a single index from a 1D probability distribution tensor (`f32`).
///
/// Candle counterpart to `torch.multinomial(probs, num_samples=1).item()`.
/// Probabilities should be non-negative; they are normalized by their sum for
/// numerical robustness.
pub fn multinomial(probs: &Tensor) -> Result<u32> {
    if probs.rank() != 1 {
        candle_core::bail!(
            "multinomial expects a 1D probability tensor, got shape {:?}",
            probs.dims()
        );
    }
    let vals = probs
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut rng = rand::rng();
    sample_row(&vals, &mut rng)
}

/// Sample one token index per row from a 2D probability tensor of shape
/// `[batch_size, vocab_size]`, returning a `[batch_size, 1]` tensor (`DType::U32`)
/// on `probs.device()`.
///
/// Candle counterpart to `torch.multinomial(probs, 1)` used in [`crate::generate_parallel`]
/// and [`crate::generate_parallel_tool`].
pub fn multinomial_batch(probs: &Tensor) -> Result<Tensor> {
    let (batch_size, vocab_size) = probs.dims2()?;
    let flat = probs
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut rng = rand::rng();
    let mut out = Vec::with_capacity(batch_size);
    for row in flat.chunks_exact(vocab_size) {
        out.push(sample_row(row, &mut rng)?);
    }
    Tensor::from_vec(out, (batch_size, 1), probs.device())
}

/// Return a 1D `DType::U32` tensor containing the flattened row-major positions where
/// `mask` (`DType::U8`) is non-zero (`!= 0`), placed on `mask.device()`.
///
/// Candle does not support PyTorch's `logits[mask]` boolean indexing directly.
/// Combine `mask_indices` with `reshape` and [`Tensor::index_select`] instead:
///
/// ```ignore
/// let (b, s, v) = logits.dims3()?;
/// let idx = mask_indices(&mask)?;
/// let flat_logits = logits.reshape((b * s, v))?.index_select(&idx, 0)?;
/// let flat_y = y.reshape((b * s,))?.index_select(&idx, 0)?;
/// ```
pub fn mask_indices(mask: &Tensor) -> Result<Tensor> {
    let flat = mask
        .to_device(&Device::Cpu)?
        .to_dtype(DType::U8)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let indices: Vec<u32> = flat
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v != 0 { Some(i as u32) } else { None })
        .collect();
    let len = indices.len();
    Tensor::from_vec(indices, (len,), mask.device())
}
