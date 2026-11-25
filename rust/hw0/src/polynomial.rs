//! Provided `Polynomial` type for Problem 3.
//!
//! Do **not** modify this file.

use std::fmt;

/// Represents a polynomial as a vector of coefficients.
///
/// Entry `coefficients[i]` (zero-indexed) is the coefficient of the `x^i` term.
/// For instance, `vec![1.0, 0.0, 4.0, 3.0]` represents `3x^3 + 4x^2 + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub coefficients: Vec<f64>,
}

impl Polynomial {
    /// Construct a `Polynomial` from coefficients `[c_0, c_1, ..., c_d]`,
    /// trimming any redundant trailing zeros so the highest-degree coefficient
    /// is non-zero (unless this is the zero polynomial `[0.0]`).
    pub fn new(mut coefficients: Vec<f64>) -> Self {
        if coefficients.is_empty() {
            coefficients.push(0.0);
        }
        while coefficients.len() > 1 && *coefficients.last().unwrap() == 0.0 {
            coefficients.pop();
        }
        Self { coefficients }
    }

    /// Return the degree of the polynomial (`coefficients.len() - 1`).
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Check approximate equality of two polynomials within `tol` per coefficient.
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        if self.coefficients.len() != other.coefficients.len() {
            return false;
        }
        self.coefficients
            .iter()
            .zip(&other.coefficients)
            .all(|(a, b)| (a - b).abs() <= tol)
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficients.is_empty() {
            return write!(f, "0");
        }
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c != 0.0 {
                match i {
                    0 => terms.push(format!("{c}")),
                    1 => terms.push(format!("{c}x")),
                    _ => terms.push(format!("{c}x^{i}")),
                }
            }
        }
        if terms.is_empty() {
            terms.push("0".to_string());
        }
        terms.reverse();
        write!(f, "{}", terms.join(" + "))
    }
}
