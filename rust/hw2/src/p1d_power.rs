//! Problem 1d: `Power` function `f(x) = x^d`.

use crate::autograd::Function;

/// Implements the power function `f(x) = x^d` for a fixed constant exponent `d`.
///
/// We do not differentiate with respect to `d`; instead `d` is stored on the struct
/// via [`Power::new`]. Be sure to handle `d == 0.0` cleanly (where `f(x) = 1.0`
/// and `df/dx = 0.0`, even when `x == 0.0`).
#[derive(Debug, Clone, Copy)]
pub struct Power {
    /// Constant exponent `d`.
    pub d: f64,
}

impl Power {
    /// Create a `Power` function for exponent `d`.
    pub fn new(d: f64) -> Self {
        Self { d }
    }
}

impl Function for Power {
    #[allow(unused_variables)]
    fn forward(&self, args: &[f64]) -> f64 {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    #[allow(unused_variables)]
    fn backward(&self, grad: f64, args: &[f64]) -> Vec<f64> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
