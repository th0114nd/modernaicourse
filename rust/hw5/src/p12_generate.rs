//! Problem 12: Autoregressive text generation with KV caching and EOT stopping.

use std::io::{self, Write};

#[allow(unused_imports)]
use candle_core::{DType, Error, Result, Tensor};

use crate::nn::CausalLM;
#[allow(unused_imports)]
use crate::nn::multinomial;
use crate::tinystories::Tokenizer;

/// Generate tokens autoregressively using KV caching, optionally streaming decoded
/// tokens to `writer`.
///
/// # Algorithm (differences from HW4)
/// 1. If `max_tokens == 0`, return `Ok(Vec::new())` immediately.
/// 2. Determine the model's device via `let device = model.device().clone();` and build
///    input token tensors on `&device` (so CPU and CUDA models both work seamlessly).
/// 3. **Prefill**: create a `[1, prompt_tokens.len()]` `u32` tensor on `&device` and call
///    `model.forward(&prompt_tensor, 0, true)?`. Track `seq_pos = prompt_tokens.len()`.
/// 4. Repeat up to `max_tokens` times:
///    - Extract the logits of the **last** sequence position as a 1D `DType::F32` tensor
///      of shape `[vocab_size]`.
///    - Compute temperature-scaled probabilities: divide logits by `temp`, apply numerically
///      stable softmax (`max`-shifted `exp` divided by sum), and sample `next_token` via
///      [`multinomial`].
///    - Append `next_token` to `generated`.
///    - If `writer` is `Some(w)`, write `tokenizer.decode(&[next_token])?` to `w` and flush `w`.
///      *(Matches `hw5_tests.py`: the token is written before checking `eot_token`.)*
///    - If `next_token == tokenizer.eot_token()` or `generated.len() == max_tokens`, break.
///    - **Decode step**: wrap `[next_token]` as a `[1, 1]` `u32` tensor on `&device`, call
///      `model.forward(&next_tensor, seq_pos, true)?`, and advance `seq_pos += 1`.
/// 5. Return `generated` (excluding `prompt_tokens`, including `eot_token` if sampled).
#[allow(unused_variables, unused_mut)]
pub fn generate_with_writer<M: CausalLM, T: Tokenizer>(
    model: &mut M,
    prompt_tokens: &[u32],
    tokenizer: &T,
    temp: f64,
    max_tokens: usize,
    mut writer: Option<&mut dyn Write>,
) -> Result<Vec<u32>> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}

/// Provided wrapper (do not modify): autoregressive generation matching `hw5.py`'s signature.
///
/// When `verbose` is `true`, streams decoded tokens directly to stdout.
pub fn generate<M: CausalLM, T: Tokenizer>(
    model: &mut M,
    prompt_tokens: &[u32],
    tokenizer: &T,
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
            temp,
            max_tokens,
            Some(&mut stdout),
        )
    } else {
        generate_with_writer(model, prompt_tokens, tokenizer, temp, max_tokens, None)
    }
}
