//! Train a linear classifier on MNIST using `hw2::train_sgd`.
//!
//! Replaces the interactive slider cell at the end of `homework/hw2/hw2.py`.
//!
//! Usage:
//!   cargo run --release -p hw2 --example train_mnist -- [step_size] [epochs] [batch_size]
//!
//! Defaults: `step_size = 0.1`, `epochs = 5`, `batch_size = 100`.
//! Try tuning hyperparameters to achieve test error under 8% (best linear model is < 7.5%).

use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use common::mnist;
use hw2::{cross_entropy_loss, error, train_sgd};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let step_size: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let batch_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);

    let device = Device::Cpu;
    println!("Loading MNIST...");
    let train = mnist::load_train(&device)?;
    let test = mnist::load_test(&device)?;

    let x_train = train.images.reshape((train.len(), 784))?;
    let y_train_u32: Vec<u32> = train.labels.iter().map(|&l| l as u32).collect();
    let y_train = Tensor::from_vec(y_train_u32, (train.len(),), &device)?;

    let x_test = test.images.reshape((test.len(), 784))?;
    let y_test_u32: Vec<u32> = test.labels.iter().map(|&l| l as u32).collect();
    let y_test = Tensor::from_vec(y_test_u32, (test.len(),), &device)?;

    println!(
        "Training linear classifier (step_size={step_size}, epochs={epochs}, batch_size={batch_size})..."
    );
    let start = Instant::now();
    let w = train_sgd(&x_train, &y_train, epochs, step_size, batch_size)?;
    let elapsed = start.elapsed();

    let train_logits = x_train.matmul(&w.t()?)?;
    let test_logits = x_test.matmul(&w.t()?)?;

    let train_loss = cross_entropy_loss(&train_logits, &y_train)?.to_scalar::<f32>()?;
    let train_err = error(&train_logits, &y_train)?.to_scalar::<f32>()?;
    let test_loss = cross_entropy_loss(&test_logits, &y_test)?.to_scalar::<f32>()?;
    let test_err = error(&test_logits, &y_test)?.to_scalar::<f32>()?;

    println!("Done in {:.2}s", elapsed.as_secs_f64());
    println!(
        "Train loss: {train_loss:.4} | Train error: {:.2}%",
        train_err * 100.0
    );
    println!(
        "Test  loss: {test_loss:.4} | Test  error: {:.2}%",
        test_err * 100.0
    );
    Ok(())
}
