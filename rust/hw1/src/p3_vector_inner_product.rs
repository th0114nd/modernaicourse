//! Problem 3: Vector inner product.

use candle_core::bail;
use candle_core::{Result, Tensor};

/// Compute the inner product `<x, y> = sum_i x_i * y_i` without using built-in matmul/dot.
///
/// Return an `Err` if either input is not 1D or if their lengths do not match.
///
/// # Inputs
/// - `x`: 1D `f32` tensor of shape `[n]`
/// - `y`: 1D `f32` tensor of shape `[n]`
///
/// # Output
/// - `f32` scalar equal to `x^T y`
pub fn vector_inner_product(x: &Tensor, y: &Tensor) -> Result<f32> {
    // BEGIN YOUR CODE
    let x_vec: Vec<f32> = x.to_vec1()?;
    let y_vec: Vec<f32> = y.to_vec1()?;
    if x_vec.len() != y_vec.len() {
        bail!("mismatched lengths");
    }
    Ok(x_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum())
    // END YOUR CODE
}
