//! Provided neural-network traits, sampling, and text/IO helpers for Homework 5.
//!
//! **Do not modify this file.**
//!
//! Provides:
//! - [`CausalLM`]: trait implemented by [`crate::LLM`] (and lightweight mock models in tests)
//!   so [`crate::train_llm`] and [`crate::generate`] work generically.
//! - [`Optimizer`]: trait implemented by [`crate::Adam`] (and test optimizers like SGD)
//!   for updating parameters from a Candle [`GradStore`].
//! - [`multinomial`]: samples a single token index from a 1D probability tensor
//!   (Candle counterpart to `torch.multinomial(probs, num_samples=1)`).
//! - [`split_keep_whitespace`]: splits text before each whitespace character while keeping
//!   the whitespace attached to the following word (Rust counterpart to Python's
//!   `re.split(r'(?=\s)', text)` with empty strings omitted).
//! - [`read_char_chunk`]: reads up to `max_chars` UTF-8 characters from a buffered reader
//!   without splitting multi-byte UTF-8 sequences (Rust counterpart to Python's `f.read(chunk_size)`).

use std::io::{self, BufRead};

use candle_core::{DType, Device, Result, Tensor, backprop::GradStore};
use rand::Rng;

/// Trait for causal language models supporting both training and KV-cached inference.
///
/// Implemented by [`crate::LLM`] and by lightweight mock models in the test suites.
pub trait CausalLM {
    /// Compute next-token logits of shape `[batch, seq_len, vocab_size]` from token IDs
    /// of shape `[batch, seq_len]` (`DType::U32`).
    fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor>;

    /// Return the Candle [`Device`] on which this model's parameters reside.
    ///
    /// Used by [`crate::generate_with_writer`] to place input token tensors on the same
    /// device as the model (counterpart to `next(model.parameters()).device` in PyTorch).
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

/// Split `text` right before every whitespace character, keeping each whitespace
/// character at the start of the following segment (and dropping empty segments).
///
/// This is the exact Rust counterpart to Python's `re.split(r'(?=\s)', text)` (with
/// empty leading segments omitted), because Rust's standard `regex` crate does not
/// support lookahead assertions.
///
/// # Examples
/// - `" test target test"` $\to$ `[" test", " target", " test"]`
/// - `"a b b"` $\to$ `["a", " b", " b"]`
/// - `"hi there\nthere"` $\to$ `["hi", " there", "\nthere"]`
pub fn split_keep_whitespace(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() && idx > start {
            parts.push(&text[start..idx]);
            start = idx;
        }
    }
    if start < text.len() {
        parts.push(&text[start..]);
    }
    parts
}

/// Read up to `max_chars` UTF-8 scalar values from `reader` without splitting multi-byte
/// UTF-8 sequences. Returns an empty `String` at EOF.
///
/// Rust counterpart to Python's `f.read(chunk_size)` on a text file opened in UTF-8 mode.
/// Non-UTF-8 bytes (if any) are decoded lossily via `String::from_utf8_lossy`.
pub fn read_char_chunk<R: BufRead>(reader: &mut R, max_chars: usize) -> io::Result<String> {
    if max_chars == 0 {
        return Ok(String::new());
    }
    let mut out = String::new();
    let mut chars_read = 0usize;
    let mut pending: Vec<u8> = Vec::new();

    while chars_read < max_chars {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            if !pending.is_empty() {
                let s = String::from_utf8_lossy(&pending);
                for ch in s.chars() {
                    if chars_read >= max_chars {
                        break;
                    }
                    out.push(ch);
                    chars_read += 1;
                }
            }
            break;
        }

        let mut consumed = 0usize;
        if pending.is_empty() {
            match std::str::from_utf8(buf) {
                Ok(valid) => {
                    for (idx, ch) in valid.char_indices() {
                        if chars_read >= max_chars {
                            consumed = idx;
                            break;
                        }
                        out.push(ch);
                        chars_read += 1;
                        consumed = idx + ch.len_utf8();
                    }
                }
                Err(err) => {
                    let valid_len = err.valid_up_to();
                    if valid_len > 0 {
                        let valid = std::str::from_utf8(&buf[..valid_len]).unwrap();
                        for (idx, ch) in valid.char_indices() {
                            if chars_read >= max_chars {
                                consumed = idx;
                                break;
                            }
                            out.push(ch);
                            chars_read += 1;
                            consumed = idx + ch.len_utf8();
                        }
                    } else if let Some(bad_len) = err.error_len() {
                        out.push(char::REPLACEMENT_CHARACTER);
                        chars_read += 1;
                        consumed = bad_len;
                    } else {
                        // Incomplete multi-byte sequence at buffer boundary.
                        pending.extend_from_slice(buf);
                        consumed = buf.len();
                    }
                }
            }
        } else {
            // Complete the pending multi-byte UTF-8 character byte-by-byte.
            pending.push(buf[0]);
            consumed = 1;
            match std::str::from_utf8(&pending) {
                Ok(s) => {
                    for ch in s.chars() {
                        out.push(ch);
                        chars_read += 1;
                    }
                    pending.clear();
                }
                Err(err) => {
                    if let Some(bad_len) = err.error_len() {
                        out.push(char::REPLACEMENT_CHARACTER);
                        chars_read += 1;
                        pending.drain(..bad_len);
                    }
                }
            }
        }
        reader.consume(consumed);
    }

    Ok(out)
}
