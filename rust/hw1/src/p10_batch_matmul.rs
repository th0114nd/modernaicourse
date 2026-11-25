//! Problem 10: Batch matrix multiplication.

use candle_core::{bail, IndexOp, Result, Tensor};

#[allow(unused_imports)]
use crate::{matmul_1, matmul_2, matmul_3};

/// Compute batch matrix multiplication for arbitrary N-D tensors (`rank >= 2`) with matching
/// leading batch dimensions, using one of `matmul_1`, `matmul_2`, or `matmul_3`.
///
/// Return an `Err` if:
/// - either tensor has rank `< 2`,
/// - `a.rank() != b.rank()`,
/// - the leading `rank - 2` batch dimensions do not match, or
/// - the inner matrix dimensions (`a.dims()[rank - 1]` and `b.dims()[rank - 2]`) do not match.
///
/// # Inputs
/// - `a`: N-D `f32` tensor of shape `[..., m, n]`
/// - `b`: N-D `f32` tensor of shape `[..., n, p]`
///
/// # Output
/// - N-D `f32` tensor of shape `[..., m, p]`
#[allow(unused_variables)]
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    let a_dims = a.dims();
    let b_dims = b.dims();
    let Some((batch_dims, a_end)) = a_dims.split_last_chunk::<2>() else {
        bail!("rank error")
    };
    let Some((batch_dims_b, b_end)) = b_dims.split_last_chunk::<2>() else {
        bail!("rank error")
    };
    if batch_dims != batch_dims_b {
        bail!("mismatched front dims");
    }
    let (m, n, p) = (a_end[0], a_end[1], b_end[1]);
    if n != b_end[0] {
        bail!("mismatched back dims");
    }
    let batch_size: usize = batch_dims.iter().product();
    let a = a.reshape((batch_size, m, n))?;
    let b = b.reshape((batch_size, n, p))?;
    let mut out = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let a_i = a.i(i)?;
        let b_i = b.i(i)?;
        let c_i = a_i.matmul(&b_i)?;
        out.push(c_i)
    }
    let c_dims = &[batch_dims, &[m, p]].concat();
    Tensor::stack(&out, 0)?.reshape(c_dims.as_slice())
    // END YOUR CODE
}
