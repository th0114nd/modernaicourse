//! Part 0l: Autoregressive text generation with KV caching and configurable stop token.
//!
//! Updated from HW5 to accept an arbitrary `eot_token: Option<u32>` so chat generation
//! can stop on `</ASSISTANT>` (`Some(50300)`) while base-model generation can pass `None`
//! or `Some(tokenizer.eot_token())`.

use std::io::{self, Write};

#[allow(unused_imports)]
use candle_core::{DType, Error, Result, Tensor};

use crate::chat::Tokenizer;
use crate::nn::CausalLM;
#[allow(unused_imports)]
use crate::nn::multinomial;

/// Generate tokens autoregressively using KV caching, optionally stopping at `eot_token`
/// and optionally streaming decoded tokens to `writer`.
///
/// # Algorithm (difference from HW5)
/// Identical to HW5's `generate_with_writer`, except instead of hardcoding
/// `tokenizer.eot_token()`, stop when `eot_token == Some(next_token)` (after appending
/// `next_token` to `generated` and writing it to `writer`). If `eot_token` is `None`,
/// generate all `max_tokens` tokens.
#[allow(unused_variables, unused_mut)]
pub fn generate_with_writer<M: CausalLM, T: Tokenizer>(
    model: &mut M,
    prompt_tokens: &[u32],
    tokenizer: &T,
    eot_token: Option<u32>,
    temp: f64,
    max_tokens: usize,
    mut writer: Option<&mut dyn Write>,
) -> Result<Vec<u32>> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}

/// Provided wrapper (do not modify): autoregressive generation matching `hw6.py`'s signature.
///
/// When `verbose` is `true`, streams decoded tokens directly to stdout.
pub fn generate<M: CausalLM, T: Tokenizer>(
    model: &mut M,
    prompt_tokens: &[u32],
    tokenizer: &T,
    eot_token: Option<u32>,
    temp: f64,
    max_tokens: usize,
    verbose: bool,
) -> Result<Vec<u32>> {
    if verbose {
        let mut stdout = io::stdout();
        generate_with_writer(
            model,
            prompt_tokens,
            tokenizer,
            eot_token,
            temp,
            max_tokens,
            Some(&mut stdout),
        )
    } else {
        generate_with_writer(
            model,
            prompt_tokens,
            tokenizer,
            eot_token,
            temp,
            max_tokens,
            None,
        )
    }
}
