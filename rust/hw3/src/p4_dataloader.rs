//! Problem 4: Sequential minibatch `DataLoader`.

#[allow(unused_imports)]
use candle_core::{Result, Tensor};

/// Sequential minibatch iterator over a paired `(x, y)` dataset.
///
/// Iterating over `&loader` (or calling `loader.iter()`) yields sequential contiguous
/// slices `(x_batch, y_batch)` of size `batch_size` (with a smaller final batch when
/// `N` is not divisible by `batch_size`). Iterating again starts fresh from index `0`.
#[derive(Debug, Clone)]
pub struct DataLoader {
    #[allow(dead_code)]
    x: Tensor,
    #[allow(dead_code)]
    y: Tensor,
    #[allow(dead_code)]
    batch_size: usize,
}

/// Iterator returned by [`DataLoader::iter`] and `(&loader).into_iter()`.
#[derive(Debug)]
pub struct DataLoaderIter<'a> {
    #[allow(dead_code)]
    loader: &'a DataLoader,
    #[allow(dead_code)]
    offset: usize,
}

impl DataLoader {
    /// Initialize a sequential minibatch data loader.
    ///
    /// Returns `Err` if `batch_size == 0` or if `x` and `y` have different lengths along
    /// dimension `0`.
    ///
    /// # Inputs
    /// - `x`: input dataset tensor of shape `[N, ...]`
    /// - `y`: target tensor of shape `[N]`
    /// - `batch_size`: number of examples per minibatch (`>= 1`)
    #[allow(unused_variables)]
    pub fn new(x: Tensor, y: Tensor, batch_size: usize) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Create a fresh iterator starting at minibatch `0`.
    pub fn iter(&self) -> DataLoaderIter<'_> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Result<(Tensor, Tensor)>;

    /// Return the next `(x_batch, y_batch)` minibatch, or `None` when exhausted.
    ///
    /// Hint: use [`Tensor::narrow`] along dimension `0`.
    fn next(&mut self) -> Option<Self::Item> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl<'a> IntoIterator for &'a DataLoader {
    type Item = Result<(Tensor, Tensor)>;
    type IntoIter = DataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
