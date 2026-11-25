//! Provided GSM8K types, reasoning-extended GPT-2 tokenizer, safe arithmetic evaluator,
//! HuggingFace checkpoint/dataset downloaders, and `.pth`/`.safetensors` weight I/O helpers
//! for Homework 7.
//!
//! **Do not modify this file.**
//!
//! Provides:
//! - [`Gsm8kExample`]: `{ "question", "answer" }` JSON record from GSM8K.
//! - [`Tokenizer`]: trait implemented by [`ReasoningTokenizer`], [`Gpt2Tokenizer`], and mock
//!   tokenizers in the test suites.
//! - [`Gpt2Tokenizer`]: standard GPT-2 (`r50k_base`) tokenizer for base-model generation.
//! - [`ReasoningTokenizer`]: GPT-2 (`r50k_base`) extended with the exact Homework 7 special
//!   tokens (`50257..=50266`):
//!   - `"<|endoftext|>"` $\to$ `50256`
//!   - `"<QUESTION>"` $\to$ `50257`, `"</QUESTION>"` $\to$ `50258`
//!   - `"<THINK>"` $\to$ `50259`, `"</THINK>"` $\to$ `50260`
//!   - `"<TOOL>"` $\to$ `50261`, `"</TOOL>"` $\to$ `50262`
//!   - `"<RESPONSE>"` $\to$ `50263`, `"</RESPONSE>"` $\to$ `50264`
//!   - `"<ANSWER>"` $\to$ `50265`, `"</ANSWER>"` $\to$ `50266`
//! - [`eval_arithmetic_expr`]: safe recursive-descent arithmetic expression evaluator
//!   (Rust counterpart to Python's `eval()` for numeric expressions in [`crate::eval_tool`]).
//! - [`ModelConfig`]: hyperparameter struct matching `params.json` in `zkolter/RL-Homework`.
//! - [`download_hw7_models`], [`download_gsm8k_datasets`], and [`download_hw7_files`]:
//!   locate or download `model_base.pth`, `model_sft.pth`, `params.json`, `gsm8k_train.json`,
//!   and `gsm8k_test.json` (honoring `HW7_DIR` and `.` before hitting the network).
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

/// HuggingFace model repository hosting the Homework 7 pretrained checkpoints and `params.json`.
pub const HW7_REPO_ID: &str = "zkolter/RL-Homework";

/// Canonical raw JSONL URLs for the GSM8K `main` train and test splits.
const GSM8K_TRAIN_JSONL_URL: &str =
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl";
const GSM8K_TEST_JSONL_URL: &str =
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl";

/// Reasoning tag strings matching `homework/hw7/hw7.py`.
pub const QUESTION_OPEN: &str = "<QUESTION>";
pub const QUESTION_CLOSE: &str = "</QUESTION>";
pub const THINK_OPEN: &str = "<THINK>";
pub const THINK_CLOSE: &str = "</THINK>";
pub const TOOL_OPEN: &str = "<TOOL>";
pub const TOOL_CLOSE: &str = "</TOOL>";
pub const RESPONSE_OPEN: &str = "<RESPONSE>";
pub const RESPONSE_CLOSE: &str = "</RESPONSE>";
pub const ANSWER_OPEN: &str = "<ANSWER>";
pub const ANSWER_CLOSE: &str = "</ANSWER>";

/// Exact special token IDs matching `homework/hw7/hw7.py` and `model_sft.pth`.
pub const EOT_ID: u32 = 50_256;
pub const QUESTION_OPEN_ID: u32 = 50_257;
pub const QUESTION_CLOSE_ID: u32 = 50_258;
pub const THINK_OPEN_ID: u32 = 50_259;
pub const THINK_CLOSE_ID: u32 = 50_260;
pub const TOOL_OPEN_ID: u32 = 50_261;
pub const TOOL_CLOSE_ID: u32 = 50_262;
pub const RESPONSE_OPEN_ID: u32 = 50_263;
pub const RESPONSE_CLOSE_ID: u32 = 50_264;
pub const ANSWER_OPEN_ID: u32 = 50_265;
pub const ANSWER_CLOSE_ID: u32 = 50_266;

/// GPT-2 (`r50k_base`) pre-tokenization regex pattern (exact match to `tiktoken_rs::r50k_base`).
const R50K_PAT_STR: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

/// A single GSM8K math word problem (`{"question": "...", "answer": "..."}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gsm8kExample {
    pub question: String,
    pub answer: String,
}

impl Gsm8kExample {
    pub fn new(question: impl Into<String>, answer: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            answer: answer.into(),
        }
    }
}

/// Trait for tokenizers used across Homework 7 (`pretokenize_gsm8k`, `get_loss_mask`,
/// `DataLoader`, `generate_parallel_tool`, `extract_answer`, `grade_responses`,
/// `eval_model`, `rl_loss`, and `train_llm_rl`).
pub trait Tokenizer {
    /// Encode `text` into token IDs, allowing all special tokens
    /// (counterpart to `tokenizer.encode(text, allowed_special="all")`).
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode a slice of token IDs back into text.
    fn decode(&self, tokens: &[u32]) -> Result<String>;

    /// Look up a special token's ID by name (e.g., `"<THINK>"`, `"</TOOL>"`, `"</ANSWER>"`).
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

/// GPT-2 (`r50k_base`) tokenizer augmented with Homework 7's 10 reasoning & tool special tokens
/// (`<QUESTION>`..`</ANSWER>` mapped to `50257..=50266`).
///
/// Exact Rust counterpart to the custom `tiktoken.Encoding` built in `homework/hw7/hw7.py`.
pub struct ReasoningTokenizer {
    bpe: CoreBPE,
    special_tokens: HashMap<String, u32>,
}

impl ReasoningTokenizer {
    /// Construct the Homework 7 reasoning tokenizer.
    pub fn new() -> Result<Self> {
        let base = tiktoken_rs::r50k_base()
            .map_err(|e| Error::Msg(format!("failed to load GPT-2 tiktoken encoding: {e}")))?;

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
            (QUESTION_OPEN, QUESTION_OPEN_ID),
            (QUESTION_CLOSE, QUESTION_CLOSE_ID),
            (THINK_OPEN, THINK_OPEN_ID),
            (THINK_CLOSE, THINK_CLOSE_ID),
            (TOOL_OPEN, TOOL_OPEN_ID),
            (TOOL_CLOSE, TOOL_CLOSE_ID),
            (RESPONSE_OPEN, RESPONSE_OPEN_ID),
            (RESPONSE_CLOSE, RESPONSE_CLOSE_ID),
            (ANSWER_OPEN, ANSWER_OPEN_ID),
            (ANSWER_CLOSE, ANSWER_CLOSE_ID),
        ];

        let mut special_fx: FxHashMap<String, Rank> = FxHashMap::default();
        let mut special_tokens = HashMap::with_capacity(specials.len());
        for (name, id) in specials {
            special_fx.insert(name.to_string(), id);
            special_tokens.insert(name.to_string(), id);
        }

        let bpe = CoreBPE::new(encoder, special_fx, R50K_PAT_STR)
            .map_err(|e| Error::Msg(format!("failed to construct reasoning CoreBPE: {e}")))?;

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

impl Tokenizer for ReasoningTokenizer {
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

// ---------------- Safe Arithmetic Evaluator (for Question 8: `eval_tool`) ----------------

struct ArithParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ArithParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn starts_with(&mut self, pat: &[u8]) -> bool {
        self.skip_ws();
        self.bytes[self.pos..].starts_with(pat)
    }

    fn parse_expr(&mut self) -> std::result::Result<f64, String> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs += rhs;
                }
                Some(b'-') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs -= rhs;
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> std::result::Result<f64, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            if self.starts_with(b"//") {
                self.pos += 2;
                let rhs = self.parse_unary()?;
                if rhs == 0.0 {
                    return Err("division by zero".to_string());
                }
                lhs = (lhs / rhs).floor();
            } else if self.starts_with(b"*") && !self.starts_with(b"**") {
                self.pos += 1;
                let rhs = self.parse_unary()?;
                lhs *= rhs;
            } else if self.starts_with(b"/") {
                self.pos += 1;
                let rhs = self.parse_unary()?;
                if rhs == 0.0 {
                    return Err("division by zero".to_string());
                }
                lhs /= rhs;
            } else if self.starts_with(b"%") {
                self.pos += 1;
                let rhs = self.parse_unary()?;
                if rhs == 0.0 {
                    return Err("modulo by zero".to_string());
                }
                lhs = ((lhs % rhs) + rhs) % rhs;
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> std::result::Result<f64, String> {
        match self.peek() {
            Some(b'+') => {
                self.pos += 1;
                self.parse_unary()
            }
            Some(b'-') => {
                self.pos += 1;
                Ok(-self.parse_unary()?)
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> std::result::Result<f64, String> {
        let base = self.parse_primary()?;
        if self.starts_with(b"**") {
            self.pos += 2;
            let exp = self.parse_unary()?;
            Ok(base.powf(exp))
        } else {
            Ok(base)
        }
    }

    fn parse_primary(&mut self) -> std::result::Result<f64, String> {
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let val = self.parse_expr()?;
                match self.peek() {
                    Some(b')') => {
                        self.pos += 1;
                        Ok(val)
                    }
                    _ => Err("missing closing parenthesis".to_string()),
                }
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.parse_number(),
            Some(c) => Err(format!("unexpected character '{}'", c as char)),
            None => Err("unexpected end of expression".to_string()),
        }
    }

    fn parse_number(&mut self) -> std::result::Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        let mut saw_digit = false;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            saw_digit = true;
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                saw_digit = true;
                self.pos += 1;
            }
        }
        if !saw_digit {
            return Err("invalid number literal".to_string());
        }
        if self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b'e' | b'E') {
            self.pos += 1;
            if self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b'+' | b'-') {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err("invalid exponent in number literal".to_string());
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid utf-8 in number".to_string())?;
        slice
            .parse::<f64>()
            .map_err(|e| format!("failed to parse number '{slice}': {e}"))
    }
}

/// Safely evaluate a basic arithmetic expression (`+`, `-`, `*`, `/`, `//`, `%`, `**`,
/// unary `+`/`-`, and parentheses) into a finite `f64`.
///
/// Provided Rust counterpart to Python's `eval()` for numeric expressions in
/// [`crate::eval_tool`]. Rejects invalid syntax, trailing tokens, division by zero, and
/// non-finite (`NaN` / `inf`) results.
pub fn eval_arithmetic_expr(expr: &str) -> std::result::Result<f64, String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Err("empty expression".to_string());
    }
    let mut parser = ArithParser::new(trimmed);
    let val = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!(
            "unexpected trailing token at byte {}",
            parser.pos
        ));
    }
    if !val.is_finite() {
        return Err("non-finite result".to_string());
    }
    Ok(val)
}

// ---------------- Model Config, Downloads & Checkpoint I/O ----------------

/// Model hyperparameters parsed from `params.json` in `zkolter/RL-Homework`.
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

/// Local paths to all Homework 7 checkpoints, model config, and GSM8K datasets.
#[derive(Debug, Clone)]
pub struct Hw7Files {
    pub model_base: PathBuf,
    pub model_sft: PathBuf,
    pub params: PathBuf,
    pub gsm8k_train: PathBuf,
    pub gsm8k_test: PathBuf,
}

fn cache_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("modernaicourse").join("hw7");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home)
            .join(".cache")
            .join("modernaicourse")
            .join("hw7");
    }
    std::env::temp_dir().join("modernaicourse").join("hw7")
}

fn find_local_file(filename: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HW7_DIR") {
        let p = Path::new(&dir).join(filename);
        if p.is_file() {
            return Some(p);
        }
    }
    let cwd = Path::new(".").join(filename);
    if cwd.is_file() {
        return Some(cwd);
    }
    None
}

/// Locate or download `model_base.pth`, `model_sft.pth`, and `params.json` from
/// `zkolter/RL-Homework` via `hf-hub`.
///
/// Resolution order per file:
/// 1. `$HW7_DIR/<file>` if `HW7_DIR` is set and the file exists.
/// 2. `./<file>` if present in the current working directory.
/// 3. Download/cache via `hf-hub` from `zkolter/RL-Homework`.
pub fn download_hw7_models() -> Result<(PathBuf, PathBuf, PathBuf)> {
    let mut api_repo: Option<hf_hub::api::sync::ApiRepo> = None;
    let mut resolve = |name: &str| -> Result<PathBuf> {
        if let Some(local) = find_local_file(name) {
            return Ok(local);
        }
        if api_repo.is_none() {
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| Error::Msg(format!("failed to initialize hf-hub API: {e}")))?;
            api_repo = Some(api.model(HW7_REPO_ID.to_string()));
        }
        api_repo
            .as_ref()
            .unwrap()
            .get(name)
            .map_err(|e| Error::Msg(format!("failed to download {name} from {HW7_REPO_ID}: {e}")))
    };

    Ok((
        resolve("model_base.pth")?,
        resolve("model_sft.pth")?,
        resolve("params.json")?,
    ))
}

fn ensure_gsm8k_split(filename: &str, url: &str) -> Result<PathBuf> {
    if let Some(local) = find_local_file(filename) {
        return Ok(local);
    }
    let dir = cache_dir();
    let cached = dir.join(filename);
    if cached.is_file() {
        return Ok(cached);
    }
    fs::create_dir_all(&dir).map_err(Error::wrap)?;

    let body = ureq::get(url)
        .call()
        .map_err(|e| Error::Msg(format!("failed to download {url}: {e}")))?
        .body_mut()
        .read_to_string()
        .map_err(|e| Error::Msg(format!("failed to read {url}: {e}")))?;

    let mut examples = Vec::new();
    for (idx, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ex: Gsm8kExample = serde_json::from_str(trimmed).map_err(|e| {
            Error::Msg(format!("failed to parse GSM8K JSONL line {}: {e}", idx + 1))
        })?;
        examples.push(ex);
    }

    let json = serde_json::to_string_pretty(&examples)
        .map_err(|e| Error::Msg(format!("failed to serialize {filename}: {e}")))?;
    fs::write(&cached, json).map_err(Error::wrap)?;
    Ok(cached)
}

/// Locate or download & cache `gsm8k_train.json` and `gsm8k_test.json`.
///
/// Resolution order per file:
/// 1. `$HW7_DIR/gsm8k_{train,test}.json`
/// 2. `./gsm8k_{train,test}.json`
/// 3. `~/.cache/modernaicourse/hw7/gsm8k_{train,test}.json`
/// 4. Download canonical JSONL from `openai/grade-school-math` and save as JSON array.
pub fn download_gsm8k_datasets() -> Result<(PathBuf, PathBuf)> {
    let train = ensure_gsm8k_split("gsm8k_train.json", GSM8K_TRAIN_JSONL_URL)?;
    let test = ensure_gsm8k_split("gsm8k_test.json", GSM8K_TEST_JSONL_URL)?;
    Ok((train, test))
}

/// Locate or download all Homework 7 model checkpoints, `params.json`, and GSM8K splits.
pub fn download_hw7_files() -> Result<Hw7Files> {
    let (model_base, model_sft, params) = download_hw7_models()?;
    let (gsm8k_train, gsm8k_test) = download_gsm8k_datasets()?;
    Ok(Hw7Files {
        model_base,
        model_sft,
        params,
        gsm8k_train,
        gsm8k_test,
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
