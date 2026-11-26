//! Problem 4: Sequential minibatch `DataLoader`.

use candle_core::{bail, Result, Tensor};

/// Sequential minibatch iterator over a paired `(x, y)` dataset.
///
/// Iterating over `&loader` (or calling `loader.iter()`) yields sequential contiguous
/// slices `(x_batch, y_batch)` of size `batch_size` (with a smaller final batch when
/// `N` is not divisible by `batch_size`). Iterating again starts fresh from index `0`.
#[derive(Debug, Clone)]
pub struct DataLoader {
    x: Tensor,
    y: Tensor,
    batch_size: usize,
}

/// Iterator returned by [`DataLoader::iter`] and `(&loader).into_iter()`.
#[derive(Debug)]
pub struct DataLoaderIter<'a> {
    loader: &'a DataLoader,
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
    pub fn new(x: Tensor, y: Tensor, batch_size: usize) -> Result<Self> {
        // BEGIN YOUR CODE
        let ns = x.dims();
        if ns.is_empty() {
            bail!("empty x dims");
        }
        let n = y.dims1()?;
        if ns[0] != n {
            bail!("batch size mismatch");
        }
        if batch_size == 0 {
            bail!("empty batch size")
        }
        Ok(Self { x, y, batch_size })
        // END YOUR CODE
    }

    /// Create a fresh iterator starting at minibatch `0`.
    pub fn iter(&self) -> DataLoaderIter<'_> {
        // BEGIN YOUR CODE
        DataLoaderIter {
            loader: self,
            offset: 0,
        }
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
        || -> Result<Option<(Tensor, Tensor)>> {
            let start = self.offset * self.loader.batch_size;
            self.offset += 1;
            let n = self.loader.y.dims1()?;
            if start >= n {
                return Ok(None);
            }
            let len = self.loader.batch_size.min(n - start);
            let x_b = self.loader.x.narrow(0, start, len)?;
            let y_b = self.loader.y.narrow(0, start, len)?;
            Ok(Some((x_b, y_b)))
        }()
        .transpose()
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
