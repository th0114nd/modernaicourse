//! Problem 6: Train and return a linear MNIST classifier (`eval_linear_model`).

#[allow(unused_imports)]
use candle_core::{Device, Result};

#[allow(unused_imports)]
use crate::{load_mnist_dataloaders, train_model, Linear};

/// Train a [`Linear`] classifier (`784 -> 10`) on MNIST and return the trained model.
///
/// The returned model is evaluated on the full 10,000-example MNIST test set and must
/// achieve test classification error `< 0.10` (10%).
///
/// Hint: use [`load_mnist_dataloaders`] with `batch_size = 100` and [`train_model`]
/// with `lr = 0.2` (5–20 epochs comfortably reach ~8% test error).
pub fn eval_linear_model() -> Result<Linear> {
    // BEGIN YOUR CODE
    let device = Device::Cpu;
    let (train_loader, test_loader) = load_mnist_dataloaders(100, &device)?;
    let model = Linear::new(28 * 28, 10, &device)?;
    train_model(model, &train_loader, &test_loader, 0.2, 10)
    // END YOUR CODE
}
