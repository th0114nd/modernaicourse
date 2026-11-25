//! Problem 3b: Polynomial multiplication.

use crate::Polynomial;

/// Multiply two polynomials together and return the result as a new [`Polynomial`].
///
/// For instance, if:
/// - `p1 = 3x^3 + 2x + 3`
/// - `p2 = 2x^2 + 5`
///
/// then `poly_mul(&p1, &p2) = 6x^5 + 19x^3 + 6x^2 + 10x + 15`.
///
/// # Inputs
/// - `p1`: first [`Polynomial`]
/// - `p2`: second [`Polynomial`]
///
/// # Output
/// - [`Polynomial`] representing `p1 * p2`.
#[allow(unused_variables)]
pub fn poly_mul(p1: &Polynomial, p2: &Polynomial) -> Polynomial {
    // BEGIN YOUR CODE
    let mut out = vec![0.0; p1.degree() + p2.degree() + 1];
    for (i, a) in p1.coefficients.iter().copied().enumerate() {
        for (j, b) in p2.coefficients.iter().copied().enumerate() {
            out[i + j] += a * b;
        }
    }
    Polynomial::new(out)
    // END YOUR CODE
}
