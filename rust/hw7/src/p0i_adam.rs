//! Part 0i: `Adam` optimizer with per-parameter moment buffers keyed by `TensorId`.
//!
//! Identical to HW6 (`Adam`); copy your implementation from `hw6`.

use std::collections::HashMap;

#[allow(unused_imports)]
use candle_core::{Result, Tensor, TensorId, Var, backprop::GradStore};

use crate::Optimizer;

/// Adam optimizer maintaining bias-corrected first (`m`) and second (`v`) moment estimates.
#[derive(Debug, Clone)]
pub struct Adam {
    pub params: Vec<Var>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    /// First-moment buffers, keyed by `var.id()`.
    pub m: HashMap<TensorId, Tensor>,
    /// Second-moment buffers, keyed by `var.id()`.
    pub v: HashMap<TensorId, Tensor>,
    /// Step counter `t` (starts at `0`, incremented to `1` on the first `step`).
    pub t: usize,
}

impl Adam {
    /// Initialize `Adam` with zero-filled `m` and `v` buffers for every parameter in `params`.
    #[allow(unused_variables)]
    pub fn new(params: Vec<Var>, lr: f64, betas: (f64, f64), eps: f64) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Provided convenience method (do not modify): delegates to [`Optimizer::step`].
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        <Self as Optimizer>::step(self, grads)
    }

    /// Provided no-op (do not modify): Candle produces a fresh [`GradStore`] on every `backward()`.
    pub fn zero_grad(&mut self) {
        <Self as Optimizer>::zero_grad(self);
    }
}

impl Optimizer for Adam {
    /// Apply one bias-corrected Adam update using gradients from `grads`.
    #[allow(unused_variables)]
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
