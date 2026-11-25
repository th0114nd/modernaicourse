//! Problem 5: Matrix-vector product approach #2 (column linear combination).

use candle_core::{Result, Tensor};

#[allow(unused_imports)]
use crate::vector_add;

/// Compute the matrix-vector product `A x` as a linear combination of the columns of `A`
/// weighted by entries of `x`, using `vector_add` (scalar-vector multiplication is allowed).
///
/// Return an `Err` if `a` is not 2D, `x` is not 1D, or `a.dim(1) != x.dim(0)`.
///
/// # Inputs
/// - `a`: 2D `f32` tensor of shape `[m, n]`
/// - `x`: 1D `f32` tensor of shape `[n]`
///
/// # Output
/// - 1D `f32` tensor of shape `[m]` equal to `A x`
#[allow(unused_variables)]
pub fn matrix_vector_product_2(a: &Tensor, x: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
