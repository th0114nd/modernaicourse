//! Problem 11: Block matrix multiplication (4x4 subblocks).

use candle_core::{bail, IndexOp, Result, Tensor};

/// Provided helper (do not modify): simulate a fast `4 x 4` fused multiply-add `Z + A B`.
///
/// Because Candle tensors are immutable, this returns the updated `4 x 4` tensor rather than
/// mutating `z` in place.
pub fn add_matmul_44(z: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if z.dims() != [4, 4] || a.dims() != [4, 4] || b.dims() != [4, 4] {
        candle_core::bail!(
            "add_matmul_44 requires (4, 4) tensors, got z={:?}, a={:?}, b={:?}",
            z.dims(),
            a.dims(),
            b.dims()
        );
    }
    let zv = z.to_vec2::<f32>()?;
    let av = a.to_vec2::<f32>()?;
    let bv = b.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(16);
    for i in 0..4 {
        for j in 0..4 {
            let sum = zv[i][j]
                + av[i][0] * bv[0][j]
                + av[i][1] * bv[1][j]
                + av[i][2] * bv[2][j]
                + av[i][3] * bv[3][j];
            out.push(sum);
        }
    }
    Tensor::from_vec(out, (4, 4), z.device())
}

/// Compute `A B` via `4 x 4` block matrix multiplication using **only** [`add_matmul_44`].
///
/// Return an `Err` if either input is not 2D, `a.dim(1) != b.dim(0)`, or any dimension
/// (`m`, `n`, `p`) is not a positive multiple of 4.
///
/// # Inputs
/// - `a`: 2D `f32` tensor of shape `[4m, 4n]`
/// - `b`: 2D `f32` tensor of shape `[4n, 4p]`
///
/// # Output
/// - 2D `f32` tensor of shape `[4m, 4p]` equal to `A B`
#[allow(unused_variables)]
pub fn block_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    let (m, n) = a.dims2()?;
    let (q, p) = b.dims2()?;
    if q != n {
        bail!("dim mismatch")
    };
    if m * n * p == 0 {
        bail!("bad dim")
    }
    if !m.is_multiple_of(4) || !n.is_multiple_of(4) || !p.is_multiple_of(4) {
        bail!("bad align")
    }
    let mut strips = Vec::with_capacity(m >> 2);
    for i in (0..m).step_by(4) {
        let mut row_blocks = Vec::with_capacity(p >> 2);
        for k in (0..p).step_by(4) {
            let mut z = Tensor::zeros((4, 4), a.dtype(), a.device())?;
            for j in (0..n).step_by(4) {
                let a = a.i(((i..i + 4), (j..j + 4)))?;
                let b = b.i(((j..j + 4), (k..k + 4)))?;
                z = add_matmul_44(&z, &a, &b)?;
            }
            row_blocks.push(z)
        }
        strips.push(Tensor::cat(&row_blocks, 1)?);
    }
    Tensor::cat(&strips, 0)
    // END YOUR CODE
}
