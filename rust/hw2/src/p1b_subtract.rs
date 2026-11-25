//! Problem 1b: `Subtract` function `f(x, y) = x - y`.

use crate::autograd::Function;

/// Implements subtraction between two variables: `f(x, y) = x - y`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Subtract;

impl Function for Subtract {
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
