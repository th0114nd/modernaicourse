//! Problem 3: Stochastic Gradient Descent (`SGD`) optimizer.

#[allow(unused_imports)]
use candle_core::{backprop::GradStore, Result, Var};

use crate::nn::Optimizer;

/// Stochastic Gradient Descent optimizer updating a stored `Vec<Var>` of parameters:
///
/// $$\theta \leftarrow \theta - \eta \nabla_\theta L$$
///
/// Unlike PyTorch, Candle's `loss.backward()?` returns a fresh [`GradStore`] each step
/// and [`Var::set`] replaces a parameter's underlying tensor without building an autograd
/// graph — so no `torch.no_grad()` block or gradient zeroing is required.
#[derive(Debug, Clone)]
pub struct SGD {
    #[allow(dead_code)]
    params: Vec<Var>,
    #[allow(dead_code)]
    learning_rate: f64,
}

impl SGD {
    /// Initialize an SGD optimizer over a list of model parameters.
    ///
    /// # Inputs
    /// - `parameters`: `Vec<Var>` — trainable parameters to optimize (e.g. `model.parameters()`)
    /// - `learning_rate`: `f64` — gradient descent step size $\eta$
    pub fn new(parameters: Vec<Var>, learning_rate: f64) -> Self {
        // BEGIN YOUR CODE
        Self {
            params: parameters,
            learning_rate,
        }
        // END YOUR CODE
    }
}

impl Optimizer for SGD {
    /// Apply one SGD update $\theta \leftarrow \theta - \eta \nabla_\theta L$ to every
    /// stored parameter that has a gradient in `grads`.
    ///
    /// Parameters absent from `grads` (i.e. `grads.get(p.as_tensor())` is `None`) should
    /// be skipped gracefully.
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        // BEGIN YOUR CODE
        for p in self.params.iter_mut() {
            if let Some(g) = grads.get(p.as_tensor()) {
                let new_t = (p.as_tensor() - self.learning_rate * g)?;
                p.set(&new_t)?;
            }
        }
        Ok(())
        // END YOUR CODE
    }
}
