//! Problem 3c: Polynomial differentiation.

use crate::Polynomial;

/// Compute the derivative of a polynomial with respect to `x`.
///
/// For instance, if `p = 4x^3 + 3x + 3`, then `poly_derivative(&p) = 12x^2 + 3`.
///
/// # Input
/// - `p`: [`Polynomial`] to differentiate
///
/// # Output
/// - [`Polynomial`] representing `p'(x)`.
#[allow(unused_variables)]
pub fn poly_derivative(p: &Polynomial) -> Polynomial {
    // BEGIN YOUR CODE
    Polynomial::new(
        p.coefficients
            .iter()
            .copied()
            .enumerate()
            .skip(1)
            .map(|(i, a)| (i as f64) * a)
            .collect(),
    )
    // END YOUR CODE
}
