//! Homework 3: Training Neural Networks in Candle.
//!
//! See `README.md` for full problem descriptions.

pub mod nn;
pub mod p1_linear;
pub mod p2_cross_entropy_loss;
pub mod p3_sgd;
pub mod p4_dataloader;
pub mod p5_epoch;
pub mod p6_eval_linear_model;
pub mod p7_two_layer_nn;
pub mod p8_eval_two_layer_nn;
pub mod p9_multi_layer_nn;

pub use nn::{Loss, Module, Optimizer, load_mnist_dataloaders, train_model};
pub use p1_linear::Linear;
pub use p2_cross_entropy_loss::CrossEntropyLoss;
pub use p3_sgd::SGD;
pub use p4_dataloader::{DataLoader, DataLoaderIter};
pub use p5_epoch::epoch;
pub use p6_eval_linear_model::eval_linear_model;
pub use p7_two_layer_nn::TwoLayerNN;
pub use p8_eval_two_layer_nn::eval_two_layer_nn;
pub use p9_multi_layer_nn::MultiLayerNN;
