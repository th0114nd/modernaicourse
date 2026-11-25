//! Problem 6: Train and return a linear MNIST classifier (`eval_linear_model`).

#[allow(unused_imports)]
use candle_core::{Device, Result};

#[allow(unused_imports)]
use crate::{Linear, load_mnist_dataloaders, train_model};

/// Train a [`Linear`] classifier (`784 -> 10`) on MNIST and return the trained model.
///
/// The returned model is evaluated on the full 10,000-example MNIST test set and must
/// achieve test classification error `< 0.10` (10%).
///
/// Hint: use [`load_mnist_dataloaders`] with `batch_size = 100` and [`train_model`]
/// with `lr = 0.2` (5–20 epochs comfortably reach ~8% test error).
pub fn eval_linear_model() -> Result<Linear> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
