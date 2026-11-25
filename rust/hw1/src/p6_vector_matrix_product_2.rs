//! Problem 6: Vector-matrix product approach #2 (row linear combination).

use candle_core::{Result, Tensor};

#[allow(unused_imports)]
use crate::vector_add;

/// Compute the vector-matrix product `x^T A` as a linear combination of the rows of `A`
/// weighted by entries of `x`, using `vector_add` (scalar-vector multiplication is allowed).
///
/// Return an `Err` if `x` is not 1D, `a` is not 2D, or `x.dim(0) != a.dim(0)`.
///
/// # Inputs
/// - `x`: 1D `f32` tensor of shape `[m]`
/// - `a`: 2D `f32` tensor of shape `[m, n]`
///
/// # Output
/// - 1D `f32` tensor of shape `[n]` equal to `x^T A`
#[allow(unused_variables)]
pub fn vector_matrix_product_2(x: &Tensor, a: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
