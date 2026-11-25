//! Provided Llama 3.2 1B Simplified scaffolding, checkpoint loader, and sampling helper.
//!
//! **Do not modify this file.**
//!
//! Provides:
//! - [`CausalLM`]: trait implemented by [`Llama3Simplified`] (and mock models in tests)
//!   so [`crate::generate`] can run generically.
//! - [`LlamaConfig`]: serde model matching `params.json` in
//!   `zkolter/Llama-3.2-1B-Instruct-Simplified`.
//! - [`download_llama_files`]: fetches `consolidated.00.pth`, `params.json`, and
//!   `tokenizer.model` via `hf-hub` (or uses `LLAMA_DIR` if set).
//! - [`load_pth_checkpoint`]: reads a PyTorch `.pth` checkpoint using
//!   `candle_core::pickle::read_all`.
//! - [`load_pretrained_llama3`]: downloads, constructs, and populates the full
//!   Llama 3.2 1B Simplified model in `f32`.
//! - [`multinomial`]: samples a single token index from a 1D probability tensor
//!   (Candle counterpart to `torch.multinomial(probs, num_samples=1)`).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Error, Result, Tensor};
use rand::Rng;
use serde::Deserialize;

use crate::Llama3Simplified;

/// HuggingFace repository hosting the course's simplified Llama 3.2 1B checkpoint.
pub const LLAMA_REPO_ID: &str = "zkolter/Llama-3.2-1B-Instruct-Simplified";

/// Trait for causal language models supporting optional KV-cached incremental decoding.
///
/// Implemented by [`Llama3Simplified`] and by lightweight mock models in the test suites.
pub trait CausalLM {
    /// Compute next-token logits of shape `[batch, seq_len, vocab_size]` from token IDs
    /// of shape `[batch, seq_len]` (`DType::U32`).
    fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor>;
}

/// Model hyperparameters parsed from `params.json`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub ffn_dim_multiplier: usize,
    pub norm_eps: f64,
    pub max_seq_len: usize,
}

impl LlamaConfig {
    /// Parse a `params.json` file from disk.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref()).map_err(Error::wrap)?;
        serde_json::from_str(&content)
            .map_err(|e| Error::Msg(format!("failed to parse params.json: {e}")))
    }

    /// Feed-forward hidden dimension (`dim * ffn_dim_multiplier`).
    pub fn ffn_dim(&self) -> usize {
        self.dim * self.ffn_dim_multiplier
    }
}

/// Local paths to the downloaded Llama 3.2 1B Simplified artifacts.
#[derive(Debug, Clone)]
pub struct LlamaFiles {
    pub checkpoint: PathBuf,
    pub params: PathBuf,
    pub tokenizer_model: PathBuf,
}

/// Locate or download `consolidated.00.pth`, `params.json`, and `tokenizer.model`
/// from `zkolter/Llama-3.2-1B-Instruct-Simplified`.
///
/// If the `LLAMA_DIR` environment variable is set and contains all three files,
/// those local files are used directly without contacting HuggingFace Hub.
pub fn download_llama_files() -> Result<LlamaFiles> {
    if let Ok(dir) = std::env::var("LLAMA_DIR") {
        let base = PathBuf::from(dir);
        let checkpoint = base.join("consolidated.00.pth");
        let params = base.join("params.json");
        let tokenizer_model = base.join("tokenizer.model");
        if checkpoint.is_file() && params.is_file() && tokenizer_model.is_file() {
            return Ok(LlamaFiles {
                checkpoint,
                params,
                tokenizer_model,
            });
        }
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Msg(format!("failed to initialize hf-hub API: {e}")))?;
    let repo = api.model(LLAMA_REPO_ID.to_string());

    let checkpoint = repo
        .get("consolidated.00.pth")
        .map_err(|e| Error::Msg(format!("failed to download consolidated.00.pth: {e}")))?;
    let params = repo
        .get("params.json")
        .map_err(|e| Error::Msg(format!("failed to download params.json: {e}")))?;
    let tokenizer_model = repo
        .get("tokenizer.model")
        .map_err(|e| Error::Msg(format!("failed to download tokenizer.model: {e}")))?;

    Ok(LlamaFiles {
        checkpoint,
        params,
        tokenizer_model,
    })
}

/// Load all tensors from a PyTorch `.pth` checkpoint into a map of `name -> Tensor`.
///
/// Wraps [`candle_core::pickle::read_all`] and casts tensors to `DType::F32` on `device`
/// (matching `model = model.float()` in `homework/hw4/hw4.py`).
pub fn load_pth_checkpoint<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let entries = candle_core::pickle::read_all(path.as_ref())?;
    let mut map = HashMap::with_capacity(entries.len());
    for (name, tensor) in entries {
        let tensor = tensor.to_device(device)?.to_dtype(DType::F32)?;
        map.insert(name, tensor);
    }
    Ok(map)
}

/// Download (if not cached) and load the pretrained Llama 3.2 1B Simplified model.
///
/// Mirrors the HuggingFace loading + `eval_llama3` cell in `homework/hw4/hw4.py`.
pub fn load_pretrained_llama3(device: &Device) -> Result<(Llama3Simplified, LlamaFiles)> {
    let files = download_llama_files()?;
    let config = LlamaConfig::from_file(&files.params)?;
    let checkpoint = load_pth_checkpoint(&files.checkpoint, device)?;

    let mut model = Llama3Simplified::new(
        config.vocab_size,
        config.dim,
        config.n_heads,
        config.max_seq_len,
        config.ffn_dim(),
        config.n_layers,
        device,
    )?;
    model.load_llama_weights(&checkpoint)?;
    Ok((model, files))
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
    let vals = probs.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if vals.is_empty() {
        candle_core::bail!("multinomial expects a non-empty probability tensor");
    }

    let total: f64 = vals.iter().map(|&p| p.max(0.0) as f64).sum();
    if !total.is_finite() || total <= 0.0 {
        candle_core::bail!("multinomial probabilities must have a finite positive sum");
    }

    let mut rng = rand::rng();
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
