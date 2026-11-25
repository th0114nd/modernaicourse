//! Problem 2: Pretokenizing a JSON chat dataset into a JSON list of token ID sequences.

#[allow(unused_imports)]
use std::fs::{self, File};
#[allow(unused_imports)]
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[allow(unused_imports)]
use candle_core::{Error, Result};

#[allow(unused_imports)]
use crate::chat::{Message, Tokenizer};
#[allow(unused_imports)]
use crate::convert_to_chat_format;

/// Convert a JSON chat dataset (`Vec<Vec<Message>>`) at `in_filename` into a JSON file
/// of token ID lists (`Vec<Vec<u32>>`) written to `out_filename`.
///
/// # Algorithm
/// 1. Deserialize `Vec<Vec<Message>>` from `in_filename` via `serde_json`.
/// 2. For each conversation `messages`, format it with [`convert_to_chat_format`] and
///    encode it with `tokenizer.encode(&text)` (which allows all special tokens,
///    matching `tokenizer.encode(text, allowed_special="all")` in `hw6.py`).
/// 3. Serialize the resulting `Vec<Vec<u32>>` as JSON to `out_filename`.
#[allow(unused_variables)]
pub fn pretokenize_chat<T: Tokenizer, P1: AsRef<Path>, P2: AsRef<Path>>(
    tokenizer: &T,
    in_filename: P1,
    out_filename: P2,
) -> Result<()> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
