//! Problem 1c: `Divide` function `f(x, y) = x / y`.

use crate::autograd::Function;

/// Implements division between two variables: `f(x, y) = x / y`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Divide;

impl Function for Divide {
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
