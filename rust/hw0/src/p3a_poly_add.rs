//! Problem 3a: Polynomial addition.

use crate::Polynomial;

/// Add two polynomials together.
///
/// For instance, if:
/// - `p1 = 3x^3 + 4x^2 + 3`
/// - `p2 = x^2 + 5x + 5`
///
/// then `poly_add(&p1, &p2) = 3x^3 + 5x^2 + 5x + 8`.
///
/// # Inputs
/// - `p1`: first [`Polynomial`]
/// - `p2`: second [`Polynomial`]
///
/// # Output
/// - [`Polynomial`] representing `p1 + p2`.
#[allow(unused_variables)]
pub fn poly_add(p1: &Polynomial, p2: &Polynomial) -> Polynomial {
    // BEGIN YOUR CODE
    let (p1, p2) = if p1.degree() > p2.degree() {
        (p2, p1)
    } else {
        (p1, p2)
    };
    let mut out = p2.coefficients.clone();
    for (i, v) in p1.coefficients.iter().enumerate() {
        out[i] += v;
    }
    Polynomial::new(out)
    // END YOUR CODE
}
