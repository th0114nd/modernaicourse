//! Problem 9: Disk-backed sequential token `DataLoader`.

#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[allow(unused_imports)]
use candle_core::{Device, Error, Result, Tensor};

/// Sequential token data loader backed by a packed little-endian `u16` binary file.
///
/// Instead of loading the entire tokenized dataset into RAM, each step seeks to byte offset
/// `batch_idx * seq_len * batch_size * 2`, reads `(seq_len + 1) * batch_size` 16-bit tokens,
/// reshapes to `[batch_size, seq_len + 1]` (`DType::U32` on `device`), and yields:
/// - `x`: columns `0 .. seq_len` (`[batch_size, seq_len]`)
/// - `y`: columns `1 .. seq_len + 1` (`[batch_size, seq_len]`)
///
/// Iterating over `&loader` (via `loader.iter()` or `for batch in &loader`) always starts
/// fresh at batch `0`.
#[derive(Debug, Clone)]
pub struct DataLoader {
    #[allow(dead_code)]
    filename: PathBuf,
    #[allow(dead_code)]
    seq_len: usize,
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    num_batches: usize,
}

/// Iterator over `(x, y)` token minibatches yielded by [`DataLoader::iter`].
#[derive(Debug)]
pub struct DataLoaderIter<'a> {
    #[allow(dead_code)]
    loader: &'a DataLoader,
    #[allow(dead_code)]
    current_batch: usize,
    #[allow(dead_code)]
    file: Option<File>,
}

impl DataLoader {
    /// Initialize a disk-backed `DataLoader` for a packed little-endian `u16` token file.
    ///
    /// # Requirements
    /// - Return an error if `seq_len == 0` or `batch_size == 0`, or if `filename` cannot be stat'd.
    /// - Compute `num_batches`: batch `b` starts at byte offset `b * seq_len * batch_size * 2`
    ///   and requires `(seq_len + 1) * batch_size * 2` bytes, so `num_batches` is the number of
    ///   batches `b >= 0` for which `b * seq_len * batch_size * 2 + (seq_len + 1) * batch_size * 2 <= file_size`.
    #[allow(unused_variables)]
    pub fn new<P: AsRef<Path>>(
        filename: P,
        seq_len: usize,
        batch_size: usize,
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

    /// Create a fresh iterator starting at batch `0`.
    pub fn iter(&self) -> DataLoaderIter<'_> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Result<(Tensor, Tensor)>;

    /// Seek to the current batch, read `(seq_len + 1) * batch_size` `u16` tokens,
    /// and return `(x, y)` of shape `[batch_size, seq_len]` (`DType::U32`).
    fn next(&mut self) -> Option<Self::Item> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

/// Provided `IntoIterator` impl (do not modify): enables `for batch in &loader`.
impl<'a> IntoIterator for &'a DataLoader {
    type Item = Result<(Tensor, Tensor)>;
    type IntoIter = DataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
