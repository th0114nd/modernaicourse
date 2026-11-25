//! Provided neural-network scaffolding and training helpers for Homework 3.
//!
//! **Do not modify this file.**
//!
//! Defines the [`Module`], [`Loss`], and [`Optimizer`] traits (the Rust/Candle
//! counterparts to PyTorch's `nn.Module` and `optim.Optimizer`), plus the
//! provided [`load_mnist_dataloaders`] and [`train_model`] helpers ported from
//! `homework/hw3/hw3.py`.

use candle_core::{Device, Result, Tensor, Var, backprop::GradStore};
use common::mnist;

use crate::{CrossEntropyLoss, DataLoader, SGD, epoch};

/// A trainable neural network layer or model (Candle counterpart to `torch.nn.Module`).
///
/// Implementing `Module` lets layers be composed into deeper models and passed
/// generically to [`epoch`] and [`train_model`].
pub trait Module {
    /// Apply the module's forward pass to an input tensor.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Return all trainable [`Var`] parameters owned by this module (including
    /// any nested submodules).
    fn parameters(&self) -> Vec<Var>;
}

/// A loss function mapping predicted logits and integer targets to a scalar loss tensor.
pub trait Loss {
    /// Compute the scalar loss between `logits` (`[N, k]`, `f32`) and `y` (`[N]`, `u32`).
    fn forward(&self, logits: &Tensor, y: &Tensor) -> Result<Tensor>;
}

/// A parameter optimizer updated from a Candle [`GradStore`].
pub trait Optimizer {
    /// Apply one parameter update using gradients returned by `loss.backward()?`.
    fn step(&mut self, grads: &GradStore) -> Result<()>;

    /// No-op in Candle: unlike PyTorch's persistent `.grad` buffers,
    /// `Tensor::backward` returns a fresh [`GradStore`] on every step.
    fn zero_grad(&mut self) {}
}

/// Load MNIST train and test splits as flattened `[N, 784]` `f32` inputs in `[0, 1]`
/// and `[N]` `u32` labels, wrapped in sequential [`DataLoader`]s.
pub fn load_mnist_dataloaders(
    batch_size: usize,
    device: &Device,
) -> Result<(DataLoader, DataLoader)> {
    let train = mnist::load_train(device)?;
    let test = mnist::load_test(device)?;

    let x_train = train.images.reshape((train.len(), 784))?;
    let y_train_u32: Vec<u32> = train.labels.iter().map(|&l| l as u32).collect();
    let y_train = Tensor::from_vec(y_train_u32, (train.len(),), device)?;

    let x_test = test.images.reshape((test.len(), 784))?;
    let y_test_u32: Vec<u32> = test.labels.iter().map(|&l| l as u32).collect();
    let y_test = Tensor::from_vec(y_test_u32, (test.len(),), device)?;

    let train_loader = DataLoader::new(x_train, y_train, batch_size)?;
    let test_loader = DataLoader::new(x_test, y_test, batch_size)?;
    Ok((train_loader, test_loader))
}

/// Train a [`Module`] using [`SGD`] and [`CrossEntropyLoss`] for `n_epochs` epochs,
/// printing train and test metrics after each epoch.
///
/// Direct port of `train_model` from `homework/hw3/hw3.py`.
pub fn train_model<M: Module>(
    model: M,
    train_dataloader: &DataLoader,
    test_dataloader: &DataLoader,
    lr: f64,
    n_epochs: usize,
) -> Result<M> {
    let mut opt = SGD::new(model.parameters(), lr);
    let loss_fn = CrossEntropyLoss::new();
    for i in 0..n_epochs {
        let (train_loss, train_err) = epoch(&model, train_dataloader, &loss_fn, Some(&mut opt))?;
        let (test_loss, test_err) = epoch(&model, test_dataloader, &loss_fn, None)?;
        println!(
            "Epoch {:2}/{n_epochs} | Train Loss: {train_loss:.4}, Train Error: {train_err:.4}, \
             Test Loss: {test_loss:.4}, Test Error: {test_err:.4}",
            i + 1
        );
    }
    Ok(model)
}
