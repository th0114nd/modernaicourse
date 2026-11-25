//! Train a linear model, two-layer neural network, or multi-layer neural network on MNIST.
//!
//! Replaces the interactive training buttons and multi-layer experimentation cells in
//! `homework/hw3/hw3.py`.
//!
//! Usage:
//!   cargo run --release -p hw3 --example train_mnist -- [model] [lr] [epochs] [batch_size] [hidden_dims]
//!
//! Models:
//!   linear     - Linear(784, 10)               (default lr=0.2, epochs=20)
//!   two-layer  - TwoLayerNN(784, 300, 10)      (default lr=0.3, epochs=20)
//!   multi      - MultiLayerNN(784, 10, hidden) (default lr=0.3, epochs=20, hidden="300,100")
//!
//! Examples:
//!   cargo run --release -p hw3 --example train_mnist -- linear
//!   cargo run --release -p hw3 --example train_mnist -- two-layer 0.3 20 100
//!   cargo run --release -p hw3 --example train_mnist -- multi 0.3 20 100 300,100

use std::time::Instant;

use candle_core::{Device, Result};
use hw3::{
    Linear, MultiLayerNN, TwoLayerNN, load_mnist_dataloaders, train_model,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("two-layer");
    let default_lr = if kind == "linear" { 0.2 } else { 0.3 };
    let lr: f64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_lr);
    let epochs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let batch_size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);
    let hidden_dims: Vec<usize> = args
        .get(5)
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.parse::<usize>().expect("valid integer hidden dim"))
                .collect()
        })
        .unwrap_or_else(|| vec![300, 100]);

    let device = Device::Cpu;
    println!("Loading MNIST (batch_size={batch_size})...");
    let (train_loader, test_loader) = load_mnist_dataloaders(batch_size, &device)?;

    let start = Instant::now();
    match kind {
        "linear" => {
            println!("Training Linear(784 -> 10) for {epochs} epochs (lr={lr})...");
            let model = Linear::new(784, 10, &device)?;
            let _trained = train_model(model, &train_loader, &test_loader, lr, epochs)?;
        }
        "two-layer" => {
            let hidden = hidden_dims.first().copied().unwrap_or(300);
            println!("Training TwoLayerNN(784 -> {hidden} -> 10) for {epochs} epochs (lr={lr})...");
            let model = TwoLayerNN::new(784, hidden, 10, &device)?;
            let _trained = train_model(model, &train_loader, &test_loader, lr, epochs)?;
        }
        "multi" => {
            println!(
                "Training MultiLayerNN(784 -> {:?} -> 10) for {epochs} epochs (lr={lr})...",
                hidden_dims
            );
            let model = MultiLayerNN::new(784, 10, &hidden_dims, &device)?;
            let _trained = train_model(model, &train_loader, &test_loader, lr, epochs)?;
        }
        other => {
            eprintln!("Unknown model '{other}'. Expected one of: linear, two-layer, multi.");
            std::process::exit(1);
        }
    }

    println!("Finished in {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
