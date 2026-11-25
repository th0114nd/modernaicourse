//! Problem 1e: `Log` function `f(x) = ln(x)`.

use crate::autograd::Function;

/// Implements the natural logarithm `f(x) = ln(x)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Log;

impl Function for Log {
    #[allow(unused_variables)]
    fn forward(&self, args: &[f64]) -> f64 {
        // BEGIN YOUR CODE
        assert_eq!(1, args.len());
        args[0].ln()
        // END YOUR CODE
    }

    #[allow(unused_variables)]
    fn backward(&self, grad: f64, args: &[f64]) -> Vec<f64> {
        // BEGIN YOUR CODE
        assert_eq!(1, args.len());
        assert!(0.0 < args[0]);
        vec![grad / args[0]]
        // END YOUR CODE
    }
}
