//! Problem 10: `Adam` optimizer with per-parameter moment buffers keyed by `TensorId`.

use std::collections::HashMap;

#[allow(unused_imports)]
use candle_core::{Result, Tensor, TensorId, Var, backprop::GradStore};

use crate::Optimizer;

/// Adam optimizer maintaining bias-corrected first (`m`) and second (`v`) moment estimates:
///
/// $$
/// \begin{aligned}
/// m &\leftarrow \beta_1 m + (1 - \beta_1) g \\
/// v &\leftarrow \beta_2 v + (1 - \beta_2) g^2 \\
/// \hat{m} &\leftarrow \frac{m}{1 - \beta_1^t}, \quad \hat{v} \leftarrow \frac{v}{1 - \beta_2^t} \\
/// w &\leftarrow w - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
/// \end{aligned}
/// $$
///
/// # Candle Design Note
/// Each [`Var`] has a stable [`TensorId`] (`var.id()`) that persists across in-place
/// `var.set(&new_tensor)?` updates. Store the per-parameter first- and second-moment
/// tensors in `self.m` and `self.v` keyed by `var.id()`, pre-initialized to zeros in
/// [`Adam::new`]. Parameters without a gradient in `grads` should be skipped gracefully.
#[derive(Debug, Clone)]
pub struct Adam {
    pub params: Vec<Var>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    /// First-moment buffers (`u` in `hw5.py`), keyed by `var.id()`.
    pub m: HashMap<TensorId, Tensor>,
    /// Second-moment buffers (`v` in `hw5.py`), keyed by `var.id()`.
    pub v: HashMap<TensorId, Tensor>,
    /// Step counter `t` (starts at `0`, incremented to `1` on the first `step`).
    pub t: usize,
}

impl Adam {
    /// Initialize `Adam` with zero-filled `m` and `v` buffers for every parameter in `params`.
    ///
    /// # Inputs
    /// - `params`: trainable parameters (`model.parameters()`)
    /// - `lr`: learning rate $\eta$ (e.g. `1e-3`)
    /// - `betas`: `(\beta_1, \beta_2)` decay rates (e.g. `(0.9, 0.999)` or `(0.9, 0.95)`)
    /// - `eps`: numerical stability constant $\epsilon$ (e.g. `1e-8`)
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
