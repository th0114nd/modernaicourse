//! Problem 8: Chunked pretokenization of a text file to a binary `u16` token file.

#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

#[allow(unused_imports)]
use candle_core::{Error, Result};

#[allow(unused_imports)]
use crate::{Tokenizer, read_char_chunk};

/// Tokenize a text file in chunks of `chunk_size` characters and write the resulting token IDs
/// as packed little-endian `u16` integers to `out_filename`.
///
/// # Algorithm
/// 1. Open `in_filename` wrapped in a [`BufReader`] and create/truncate `out_filename`
///    wrapped in a [`BufWriter`].
/// 2. Loop up to `max_chunks` times (or until EOF if `max_chunks` is `None`):
///    - Read up to `chunk_size` characters using [`read_char_chunk`].
///    - Stop if the returned chunk is empty (EOF).
///    - Encode the chunk via `tokenizer.encode(&chunk)`.
///    - Write each token ID as 2 little-endian bytes (`(tok as u16).to_le_bytes()`) to the output file.
/// 3. Flush the output writer.
///
/// # Inputs
/// - `tokenizer`: tokenizer implementing [`Tokenizer`]
/// - `in_filename`: input `.txt` file path
/// - `out_filename`: output binary `.bin` file path (packed little-endian `u16`)
/// - `chunk_size`: maximum number of Unicode characters read per chunk (e.g. `1 << 20`)
/// - `max_chunks`: optional cap on the number of chunks to process
#[allow(unused_variables)]
pub fn pretokenize_data<T, P1, P2>(
    tokenizer: &T,
    in_filename: P1,
    out_filename: P2,
    chunk_size: usize,
    max_chunks: Option<usize>,
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
