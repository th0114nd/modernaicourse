//! Problem 1f: `Exp` function `f(x) = e^x`.

use crate::autograd::Function;

/// Implements the natural exponential `f(x) = e^x`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Exp;

impl Function for Exp {
    #[allow(unused_variables)]
    fn forward(&self, args: &[f64]) -> f64 {
        // BEGIN YOUR CODE
        assert_eq!(1, args.len());
        args[0].exp()
        // END YOUR CODE
    }

    #[allow(unused_variables)]
    fn backward(&self, grad: f64, args: &[f64]) -> Vec<f64> {
        // BEGIN YOUR CODE
        assert_eq!(1, args.len());
        vec![grad * args[0].exp()]
        // END YOUR CODE
    }
}
