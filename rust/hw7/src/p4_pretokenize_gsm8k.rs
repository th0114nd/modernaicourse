//! Problem 4: Pretokenizing a GSM8K JSON dataset into tokenized sequences (`pretokenize_gsm8k`).

#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[allow(unused_imports)]
use candle_core::{Error, Result};

#[allow(unused_imports)]
use crate::{Gsm8kExample, Tokenizer, gsm8k_to_text};

/// Read GSM8K examples from `in_filename` (JSON array of [`Gsm8kExample`]), convert each example
/// via [`gsm8k_to_text`], encode it with `tokenizer.encode(&text)` (allowing all special tokens),
/// and write the resulting `Vec<Vec<u32>>` as JSON to `out_filename`.
#[allow(unused_variables)]
pub fn pretokenize_gsm8k<T, P1, P2>(
    tokenizer: &T,
    in_filename: P1,
    out_filename: P2,
) -> Result<()>
where
    T: Tokenizer,
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
