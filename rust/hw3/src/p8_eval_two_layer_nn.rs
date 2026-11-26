//! Problem 8: Train and return a two-layer ReLU neural network on MNIST (`eval_two_layer_nn`).

use candle_core::{Device, Result};

use crate::{load_mnist_dataloaders, train_model, TwoLayerNN};

/// Train a [`TwoLayerNN`] on MNIST and return the trained model.
///
/// The returned model is evaluated on the MNIST test set and must achieve test
/// classification error `< 0.03` (3%) on `test[:2000]` (and `< 0.04` on `test[:1000]`).
///
/// Hint: `TwoLayerNN::new(784, 300, 10, &device)` with `lr = 0.3` and `batch_size = 100`
/// reaches `< 2%` test error in 10–20 epochs (and `< 2.5%` in ~6–8 epochs).
/// Run tests with `--release` for fast CPU training!
pub fn eval_two_layer_nn() -> Result<TwoLayerNN> {
    // BEGIN YOUR CODE
    let device = Device::Cpu;
    let (train_loader, test_loader) = load_mnist_dataloaders(100, &device)?;
    let model = TwoLayerNN::new(784, 300, 10, &device)?;
    train_model(model, &train_loader, &test_loader, 0.3, 20)
    // END YOUR CODE
}
