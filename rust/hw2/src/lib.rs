//! Homework 2: Automatic Differentiation and Linear Models.
//!
//! See `README.md` for full problem descriptions.

pub mod autograd;
pub mod p1a_add;
pub mod p1b_subtract;
pub mod p1c_divide;
pub mod p1d_power;
pub mod p1e_log;
pub mod p1f_exp;
pub mod p2_compute_gradients;
pub mod p3a_cross_entropy_loss;
pub mod p3b_error;
pub mod p4_train_sgd;

pub use autograd::{Function, Multiply, Negate, Variable};
pub use p1a_add::Add;
pub use p1b_subtract::Subtract;
pub use p1c_divide::Divide;
pub use p1d_power::Power;
pub use p1e_log::Log;
pub use p1f_exp::Exp;
pub use p2_compute_gradients::compute_gradients;
pub use p3a_cross_entropy_loss::cross_entropy_loss;
pub use p3b_error::error;
pub use p4_train_sgd::train_sgd;
