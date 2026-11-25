//! Provided chat message types, chat-extended GPT-2 tokenizer, HuggingFace dataset/checkpoint
//! downloader, and `.pth`/`.safetensors` weight I/O helpers for Homework 6.
//!
//! **Do not modify this file.**
//!
//! Provides:
//! - [`Message`]: UltraChat-style `{ "role", "content" }` JSON message.
//! - [`Tokenizer`]: trait implemented by [`ChatTokenizer`], [`Gpt2Tokenizer`], and mock
//!   tokenizers in the test suites.
//! - [`Gpt2Tokenizer`]: standard GPT-2 (`r50k_base`) tokenizer for base-model generation.
//! - [`ChatTokenizer`]: GPT-2 (`r50k_base`) extended with the exact Homework 6 special tokens:
//!   - `"<|endoftext|>"` $\to$ `50256`
//!   - `"<USER>"` $\to$ `50257`
//!   - `"</USER>"` $\to$ `50258`
//!   - `"<ASSISTANT>"` $\to$ `50259`
//!   - `"</ASSISTANT>"` $\to$ `50300`
//! - [`ModelConfig`]: hyperparameter struct matching `params.json` in `zkolter/Chat-Tuning-Homework`.
//! - [`download_hw6_files`]: locates or downloads `model_base.pth`, `model_chat.pth`,
//!   `params.json`, `ultrachat_short.json`, `ultrachat_dpo_pos.json`, and `ultrachat_dpo_neg.json`.
//! - [`load_checkpoint_tensors`] & [`load_pretrained_llm`]: loads either PyTorch `.pth`
//!   checkpoints (via `candle_core::pickle::read_all`) or `.safetensors` checkpoints
//!   (via `candle_core::safetensors::load`) in `DType::F32`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Error, Result, Tensor};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tiktoken_rs::{CoreBPE, Rank};

use crate::LLM;

/// HuggingFace model repository hosting the Homework 6 pretrained checkpoints and datasets.
pub const HW6_REPO_ID: &str = "zkolter/Chat-Tuning-Homework";

/// Chat tag strings matching `homework/hw6/hw6.py`.
pub const USER_OPEN: &str = "<USER>";
pub const USER_CLOSE: &str = "</USER>";
pub const ASSISTANT_OPEN: &str = "<ASSISTANT>";
pub const ASSISTANT_CLOSE: &str = "</ASSISTANT>";

/// Exact special token IDs matching `homework/hw6/hw6.py` (`gpt2_chat`).
pub const EOT_ID: u32 = 50_256;
pub const USER_OPEN_ID: u32 = 50_257;
pub const USER_CLOSE_ID: u32 = 50_258;
pub const ASSISTANT_OPEN_ID: u32 = 50_259;
/// Note: intentionally `50300` (not `50260`), matching `hw6.py` and `model_chat.pth`.
pub const ASSISTANT_CLOSE_ID: u32 = 50_300;

/// GPT-2 (`r50k_base`) pre-tokenization regex pattern (exact match to `tiktoken_rs::r50k_base`).
const R50K_PAT_STR: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

/// A single UltraChat-style chat message (`{"role": "...", "content": "..."}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Trait for tokenizers used by [`crate::pretokenize_chat`], [`crate::get_loss_mask`],
/// [`crate::DataLoaderChat`], and [`crate::generate_with_writer`].
pub trait Tokenizer {
    /// Encode `text` into token IDs, allowing all special tokens
    /// (counterpart to `tokenizer.encode(text, allowed_special="all")`).
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode a slice of token IDs back into text.
    fn decode(&self, tokens: &[u32]) -> Result<String>;

    /// Look up a special token's ID by name (e.g., `"<ASSISTANT>"`, `"</ASSISTANT>"`).
    ///
    /// Rust counterpart to `tokenizer._special_tokens[name]`.
    fn special_token(&self, name: &str) -> Option<u32>;

    /// End-of-text token ID (`50256` for GPT-2 `<|endoftext|>`).
    fn eot_token(&self) -> u32 {
        self.special_token("<|endoftext|>").unwrap_or(EOT_ID)
    }
}

/// Standard GPT-2 (`r50k_base`) tokenizer matching `tiktoken.get_encoding("gpt2")`.
pub struct Gpt2Tokenizer {
    bpe: CoreBPE,
    special_tokens: HashMap<String, u32>,
}

impl Gpt2Tokenizer {
    /// Construct the standard GPT-2 tokenizer.
    pub fn new() -> Result<Self> {
        let bpe = tiktoken_rs::r50k_base()
            .map_err(|e| Error::Msg(format!("failed to load GPT-2 tiktoken encoding: {e}")))?;
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|endoftext|>".to_string(), EOT_ID);
        Ok(Self {
            bpe,
            special_tokens,
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

    fn special_token(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }
}

/// GPT-2 (`r50k_base`) tokenizer augmented with Homework 6's chat special tokens
/// (`<USER>`, `</USER>`, `<ASSISTANT>`, `</ASSISTANT>`).
///
/// Exact Rust counterpart to the `gpt2_chat` `tiktoken.Encoding` built in `homework/hw6/hw6.py`.
pub struct ChatTokenizer {
    bpe: CoreBPE,
    special_tokens: HashMap<String, u32>,
}

impl ChatTokenizer {
    /// Construct the Homework 6 chat tokenizer (`gpt2_chat`).
    pub fn new() -> Result<Self> {
        let base = tiktoken_rs::r50k_base()
            .map_err(|e| Error::Msg(format!("failed to load GPT-2 tiktoken encoding: {e}")))?;

        // Reconstruct GPT-2's 50,256 mergeable BPE ranks directly from `r50k_base`.
        let mut encoder: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
        encoder.reserve(EOT_ID as usize);
        for rank in 0..EOT_ID {
            let bytes = base
                .decode_bytes(&[rank])
                .map_err(|e| Error::Msg(format!("failed to extract r50k rank {rank}: {e}")))?;
            encoder.insert(bytes, rank);
        }

        let specials = [
            ("<|endoftext|>", EOT_ID),
            (USER_OPEN, USER_OPEN_ID),
            (USER_CLOSE, USER_CLOSE_ID),
            (ASSISTANT_OPEN, ASSISTANT_OPEN_ID),
            (ASSISTANT_CLOSE, ASSISTANT_CLOSE_ID),
        ];

        let mut special_fx: FxHashMap<String, Rank> = FxHashMap::default();
        let mut special_tokens = HashMap::with_capacity(specials.len());
        for (name, id) in specials {
            special_fx.insert(name.to_string(), id);
            special_tokens.insert(name.to_string(), id);
        }

        let bpe = CoreBPE::new(encoder, special_fx, R50K_PAT_STR)
            .map_err(|e| Error::Msg(format!("failed to construct gpt2_chat CoreBPE: {e}")))?;

        Ok(Self {
            bpe,
            special_tokens,
        })
    }

    /// Map of special token strings to their integer IDs.
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }
}

impl Tokenizer for ChatTokenizer {
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

    fn special_token(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }
}

/// Model hyperparameters parsed from `params.json` in `zkolter/Chat-Tuning-Homework`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub num_tokens: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
    pub ffn_dim: usize,
    pub n_layers: usize,
}

impl ModelConfig {
    /// Parse a `params.json` file from disk.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref()).map_err(Error::wrap)?;
        serde_json::from_str(&content)
            .map_err(|e| Error::Msg(format!("failed to parse params.json: {e}")))
    }
}

/// Local paths to all Homework 6 checkpoints and UltraChat datasets.
#[derive(Debug, Clone)]
pub struct ChatFiles {
    pub model_base: PathBuf,
    pub model_chat: PathBuf,
    pub params: PathBuf,
    pub ultrachat_short: PathBuf,
    pub ultrachat_dpo_pos: PathBuf,
    pub ultrachat_dpo_neg: PathBuf,
}

fn try_files_in_dir(base: &Path) -> Option<ChatFiles> {
    let files = ChatFiles {
        model_base: base.join("model_base.pth"),
        model_chat: base.join("model_chat.pth"),
        params: base.join("params.json"),
        ultrachat_short: base.join("ultrachat_short.json"),
        ultrachat_dpo_pos: base.join("ultrachat_dpo_pos.json"),
        ultrachat_dpo_neg: base.join("ultrachat_dpo_neg.json"),
    };
    if files.model_base.is_file()
        && files.model_chat.is_file()
        && files.params.is_file()
        && files.ultrachat_short.is_file()
        && files.ultrachat_dpo_pos.is_file()
        && files.ultrachat_dpo_neg.is_file()
    {
        Some(files)
    } else {
        None
    }
}

/// Locate or download all 6 Homework 6 files from `zkolter/Chat-Tuning-Homework`.
///
/// Resolution order:
/// 1. `HW6_DIR` environment variable if set and containing all 6 files.
/// 2. Current working directory (`.`) if all 6 files are already present.
/// 3. Download/cache via `hf-hub` from `zkolter/Chat-Tuning-Homework`.
pub fn download_hw6_files() -> Result<ChatFiles> {
    if let Ok(dir) = std::env::var("HW6_DIR")
        && let Some(files) = try_files_in_dir(Path::new(&dir))
    {
        return Ok(files);
    }

    if let Some(files) = try_files_in_dir(Path::new(".")) {
        return Ok(files);
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Msg(format!("failed to initialize hf-hub API: {e}")))?;
    let repo = api.model(HW6_REPO_ID.to_string());

    let get = |name: &str| -> Result<PathBuf> {
        repo.get(name)
            .map_err(|e| Error::Msg(format!("failed to download {name} from {HW6_REPO_ID}: {e}")))
    };

    Ok(ChatFiles {
        model_base: get("model_base.pth")?,
        model_chat: get("model_chat.pth")?,
        params: get("params.json")?,
        ultrachat_short: get("ultrachat_short.json")?,
        ultrachat_dpo_pos: get("ultrachat_dpo_pos.json")?,
        ultrachat_dpo_neg: get("ultrachat_dpo_neg.json")?,
    })
}

/// Load a weight dictionary from either a `.safetensors` file or a PyTorch `.pth` pickle
/// checkpoint, converting every tensor to `DType::F32` on `device`.
pub fn load_checkpoint_tensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let is_safetensors = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"));

    let mut map = HashMap::new();
    if is_safetensors {
        let raw = candle_core::safetensors::load(path, device)?;
        map.reserve(raw.len());
        for (name, tensor) in raw {
            map.insert(name, tensor.to_dtype(DType::F32)?);
        }
    } else {
        let entries = candle_core::pickle::read_all(path)?;
        map.reserve(entries.len());
        for (name, tensor) in entries {
            let tensor = tensor.to_device(device)?.to_dtype(DType::F32)?;
            map.insert(name, tensor);
        }
    }
    Ok(map)
}

/// Construct an [`LLM`] from `params.json` and populate its weights from a `.pth` or
/// `.safetensors` checkpoint in `DType::F32`.
pub fn load_pretrained_llm<P1: AsRef<Path>, P2: AsRef<Path>>(
    params_path: P1,
    weights_path: P2,
    device: &Device,
) -> Result<LLM> {
    let cfg = ModelConfig::from_file(params_path)?;
    let mut model = LLM::new(
        cfg.num_tokens,
        cfg.dim,
        cfg.n_heads,
        cfg.max_seq_len,
        cfg.ffn_dim,
        cfg.n_layers,
        device,
    )?;
    model.load_weights(weights_path)?;
    Ok(model)
}
