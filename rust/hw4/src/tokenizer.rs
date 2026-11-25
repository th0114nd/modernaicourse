//! Provided Llama 3 BPE tokenizer and Instruct chat template helpers.
//!
//! **Do not modify this file.**
//!
//! Direct Rust port of Meta's `tokenizer.py` distributed in
//! `zkolter/Llama-3.2-1B-Instruct-Simplified` (`Tokenizer`, `Message`, `ChatFormat`),
//! backed by [`tiktoken_rs::CoreBPE`].

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use candle_core::{Error, Result};
use rustc_hash::FxHashMap;
use tiktoken_rs::{CoreBPE, Rank};

/// Trait for tokenizers used by [`crate::generate`].
///
/// Implemented by [`LlamaTokenizer`] and by lightweight mock tokenizers in the test suites.
pub trait Tokenizer {
    /// Decode a slice of token IDs into text.
    fn decode(&self, tokens: &[u32]) -> Result<String>;

    /// Return `true` if `token` terminates autoregressive generation.
    fn is_stop_token(&self, token: u32) -> bool;
}

/// Llama 3 pre-tokenization regex pattern (exact match to Meta's `tokenizer.py`).
const LLAMA3_PAT_STR: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;

/// Llama 3 Tiktoken BPE tokenizer loaded from Meta's `tokenizer.model`.
pub struct LlamaTokenizer {
    bpe: CoreBPE,
    /// Mapping from special token strings (e.g. `"<|begin_of_text|>"`) to IDs.
    pub special_tokens: HashMap<String, u32>,
    /// Beginning-of-sequence token ID (`<|begin_of_text|>`).
    pub bos_id: u32,
    /// End-of-sequence token ID (`<|end_of_text|>`).
    pub eos_id: u32,
    /// Set of stop token IDs (`<|end_of_text|>` and `<|eot_id|>`).
    pub stop_tokens: HashSet<u32>,
}

impl LlamaTokenizer {
    /// Load a Llama 3 tokenizer from a tiktoken `.model` file (`<base64_bytes> <rank>` per line).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref()).map_err(Error::wrap)?;
        let mut mergeable_ranks: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let b64 = parts
                .next()
                .ok_or_else(|| Error::Msg(format!("malformed tiktoken line: {line}")))?;
            let rank_str = parts
                .next()
                .ok_or_else(|| Error::Msg(format!("missing rank on tiktoken line: {line}")))?;
            let bytes = STANDARD
                .decode(b64)
                .map_err(|e| Error::Msg(format!("invalid base64 token '{b64}': {e}")))?;
            let rank: Rank = rank_str
                .parse()
                .map_err(|e| Error::Msg(format!("invalid rank '{rank_str}': {e}")))?;
            mergeable_ranks.insert(bytes, rank);
        }

        let num_base_tokens = mergeable_ranks.len() as u32;
        let mut special_names: Vec<String> = vec![
            "<|begin_of_text|>".to_string(),
            "<|end_of_text|>".to_string(),
            "<|reserved_special_token_0|>".to_string(),
            "<|reserved_special_token_1|>".to_string(),
            "<|reserved_special_token_2|>".to_string(),
            "<|reserved_special_token_3|>".to_string(),
            "<|start_header_id|>".to_string(),
            "<|end_header_id|>".to_string(),
            "<|reserved_special_token_4|>".to_string(),
            "<|eot_id|>".to_string(),
        ];
        for i in 5..(NUM_RESERVED_SPECIAL_TOKENS - 5) {
            special_names.push(format!("<|reserved_special_token_{i}|>"));
        }

        let mut special_tokens = HashMap::with_capacity(special_names.len());
        let mut special_fx: FxHashMap<String, Rank> = FxHashMap::default();
        for (i, name) in special_names.into_iter().enumerate() {
            let id = num_base_tokens + i as u32;
            special_tokens.insert(name.clone(), id);
            special_fx.insert(name, id);
        }

        let bos_id = special_tokens["<|begin_of_text|>"];
        let eos_id = special_tokens["<|end_of_text|>"];
        let eot_id = special_tokens["<|eot_id|>"];
        let mut stop_tokens = HashSet::new();
        stop_tokens.insert(eos_id);
        stop_tokens.insert(eot_id);

        let bpe = CoreBPE::new(mergeable_ranks, special_fx, LLAMA3_PAT_STR)
            .map_err(|e| Error::Msg(format!("failed to construct CoreBPE: {e}")))?;

        Ok(Self {
            bpe,
            special_tokens,
            bos_id,
            eos_id,
            stop_tokens,
        })
    }

    /// Encode ordinary text into token IDs, optionally adding BOS/EOS tokens.
    pub fn encode(&self, s: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens = self.bpe.encode_ordinary(s);
        if bos {
            tokens.insert(0, self.bos_id);
        }
        if eos {
            tokens.push(self.eos_id);
        }
        tokens
    }
}

impl Tokenizer for LlamaTokenizer {
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self
            .bpe
            .decode_bytes(tokens)
            .map_err(|e| Error::Msg(format!("failed to decode tokens: {e}")))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.stop_tokens.contains(&token)
    }
}

/// A single chat message with a `role` (`"system"`, `"user"`, or `"assistant"`) and `content`.
#[derive(Debug, Clone)]
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

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Llama 3 Instruct chat prompt formatter (direct port of `ChatFormat` in Meta's `tokenizer.py`).
pub struct ChatFormat<'a> {
    tokenizer: &'a LlamaTokenizer,
}

impl<'a> ChatFormat<'a> {
    pub fn new(tokenizer: &'a LlamaTokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn encode_header(&self, role: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        tokens.push(self.tokenizer.special_tokens["<|start_header_id|>"]);
        tokens.extend(self.tokenizer.encode(role, false, false));
        tokens.push(self.tokenizer.special_tokens["<|end_header_id|>"]);
        tokens.extend(self.tokenizer.encode("\n\n", false, false));
        tokens
    }

    pub fn encode_message(&self, message: &Message) -> Vec<u32> {
        let mut tokens = self.encode_header(&message.role);
        tokens.extend(self.tokenizer.encode(message.content.trim(), false, false));
        tokens.push(self.tokenizer.special_tokens["<|eot_id|>"]);
        tokens
    }

    pub fn encode_dialog_prompt(&self, dialog: &[Message]) -> Vec<u32> {
        let mut tokens = Vec::new();
        tokens.push(self.tokenizer.special_tokens["<|begin_of_text|>"]);
        for message in dialog {
            tokens.extend(self.encode_message(message));
        }
        tokens.extend(self.encode_header("assistant"));
        tokens
    }
}
