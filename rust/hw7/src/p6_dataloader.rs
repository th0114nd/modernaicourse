//! Problem 6: Dynamically-padded GSM8K `DataLoader` yielding `(x, y, mask)` minibatches.

#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::io::BufReader;
use std::path::Path;

#[allow(unused_imports)]
use candle_core::{Device, Error, Result, Tensor};

#[allow(unused_imports)]
use crate::{Tokenizer, get_loss_mask};

/// GSM8K data loader backed by a pretokenized JSON file (`Vec<Vec<u32>>`).
///
/// Unlike Homework 6's fixed-length `DataLoaderChat`, this loader computes
/// `max_len = max(example.len())` dynamically over each minibatch and pads every row in that
/// batch to `seq_len = max_len - 1`:
/// - `x`: `[batch_size, seq_len]` (`DType::U32`) — `tokens[0..seq_len]`, zero-padded with `0`
/// - `y`: `[batch_size, seq_len]` (`DType::U32`) — `tokens[1..seq_len + 1]`, zero-padded with `0`
/// - `mask`: `[batch_size, seq_len]` (`DType::U8`, `1` for `true`, `0` for `false`) —
///   `get_loss_mask(tokens)[1..seq_len + 1]`, padded with `0` (`false`)
///
/// Incomplete trailing batches (fewer than `batch_size` examples) are dropped.
/// Iterating via `loader.iter()` or `for batch in &loader` always starts fresh at batch `0`.
#[derive(Debug, Clone)]
pub struct DataLoader {
    #[allow(dead_code)]
    examples: Vec<Vec<u32>>,
    #[allow(dead_code)]
    masks: Vec<Vec<bool>>,
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    num_batches: usize,
}

/// Iterator over `(x, y, mask)` minibatches yielded by [`DataLoader::iter`].
#[derive(Debug)]
pub struct DataLoaderIter<'a> {
    #[allow(dead_code)]
    loader: &'a DataLoader,
    #[allow(dead_code)]
    current_batch: usize,
}

impl DataLoader {
    /// Load a pretokenized JSON file (`Vec<Vec<u32>>`) and precompute reasoning loss masks
    /// via [`get_loss_mask`].
    ///
    /// # Requirements
    /// - Return an error if `batch_size == 0` or if `filename` cannot be read/parsed.
    /// - Compute `num_batches = examples.len() / batch_size` (dropping any trailing partial batch).
    #[allow(unused_variables)]
    pub fn new<P: AsRef<Path>, T: Tokenizer>(
        filename: P,
        batch_size: usize,
        tokenizer: &T,
        device: Device,
    ) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Number of complete minibatches in this dataset.
    pub fn len(&self) -> usize {
        self.num_batches
    }

    /// Returns `true` if the dataset contains zero complete minibatches.
    pub fn is_empty(&self) -> bool {
        self.num_batches == 0
    }

    /// Configured minibatch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Create a fresh iterator starting at batch `0`.
    pub fn iter(&self) -> DataLoaderIter<'_> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Result<(Tensor, Tensor, Tensor)>;

    /// Build and return the next dynamically-padded `(x, y, mask)` minibatch of shape
    /// `[batch_size, max_len - 1]`.
    fn next(&mut self) -> Option<Self::Item> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

/// Provided `IntoIterator` impl (do not modify): enables `for batch in &loader`.
impl<'a> IntoIterator for &'a DataLoader {
    type Item = Result<(Tensor, Tensor, Tensor)>;
    type IntoIter = DataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
