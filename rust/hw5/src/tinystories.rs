//! Provided TinyStories dataset downloader and GPT-2 (`tiktoken`) tokenizer.
//!
//! **Do not modify this file.**
//!
//! Provides:
//! - [`Tokenizer`]: trait implemented by [`Gpt2Tokenizer`] and mock tokenizers in tests
//!   for [`crate::pretokenize_data`] and [`crate::generate_with_writer`].
//! - [`Gpt2Tokenizer`]: wrapper around [`tiktoken_rs::r50k_base`] (matching Python's
//!   `tiktoken.get_encoding("gpt2")`).
//! - [`download_tinystories`]: fetches `TinyStoriesV2-GPT4-train.txt` from
//!   `roneneldan/TinyStories` via `hf-hub` (or uses `TINYSTORIES_PATH` if set).
//! - [`padded_vocab_size`]: rounds a vocabulary size up to the next multiple of 256
//!   (`(n_vocab / 256 + 1) * 256`, matching `hw5.py`).

use std::path::PathBuf;

use candle_core::{Error, Result};
use tiktoken_rs::CoreBPE;

/// HuggingFace dataset repository hosting TinyStories.
pub const TINYSTORIES_REPO_ID: &str = "roneneldan/TinyStories";

/// Training split filename in `roneneldan/TinyStories`.
pub const TINYSTORIES_TRAIN_FILE: &str = "TinyStoriesV2-GPT4-train.txt";

/// Trait for tokenizers used by [`crate::pretokenize_data`] and [`crate::generate_with_writer`].
pub trait Tokenizer {
    /// Encode `text` into token IDs (allowing special tokens such as `<|endoftext|>`).
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode a slice of token IDs back into text.
    fn decode(&self, tokens: &[u32]) -> Result<String>;

    /// End-of-text token ID (e.g. `50256` for GPT-2 `<|endoftext|>`).
    fn eot_token(&self) -> u32;

    /// Total vocabulary size.
    fn n_vocab(&self) -> usize;
}

/// GPT-2 BPE tokenizer backed by `tiktoken-rs` (`r50k_base`, identical to `tiktoken.get_encoding("gpt2")`).
pub struct Gpt2Tokenizer {
    bpe: CoreBPE,
    eot_token: u32,
    n_vocab: usize,
}

impl Gpt2Tokenizer {
    /// Construct the GPT-2 (`r50k_base`) tokenizer.
    pub fn new() -> Result<Self> {
        let bpe = tiktoken_rs::r50k_base()
            .map_err(|e| Error::Msg(format!("failed to load GPT-2 tiktoken encoding: {e}")))?;
        Ok(Self {
            bpe,
            eot_token: 50_256,
            n_vocab: 50_257,
        })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self
            .bpe
            .decode_bytes(tokens)
            .map_err(|e| Error::Msg(format!("failed to decode tokens: {e}")))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn eot_token(&self) -> u32 {
        self.eot_token
    }

    fn n_vocab(&self) -> usize {
        self.n_vocab
    }
}

/// Round `n_vocab` up to the next multiple of 256 (`(n_vocab / 256 + 1) * 256`),
/// matching `hw5.py`'s embedding/output padding for GPU efficiency.
pub fn padded_vocab_size(n_vocab: usize) -> usize {
    (n_vocab / 256 + 1) * 256
}

/// Locate or download `TinyStoriesV2-GPT4-train.txt` from `roneneldan/TinyStories`.
///
/// Resolution order:
/// 1. `TINYSTORIES_PATH` environment variable (path to the `.txt` file) if set and exists.
/// 2. `./TinyStoriesV2-GPT4-train.txt` in the current working directory if present.
/// 3. Download/cache via `hf-hub` from `roneneldan/TinyStories`.
pub fn download_tinystories() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("TINYSTORIES_PATH") {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Ok(p);
        }
    }

    let local = PathBuf::from(TINYSTORIES_TRAIN_FILE);
    if local.is_file() {
        return Ok(local);
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Msg(format!("failed to initialize hf-hub API: {e}")))?;
    let repo = api.dataset(TINYSTORIES_REPO_ID.to_string());
    repo.get(TINYSTORIES_TRAIN_FILE)
        .map_err(|e| Error::Msg(format!("failed to download {TINYSTORIES_TRAIN_FILE}: {e}")))
}
