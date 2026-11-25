//! Deterministic tensor helpers and `torch.allclose`-style assertions.

use candle_core::{DType, Device, Result, Tensor};

/// Create a 1D `f32` tensor of `steps` evenly spaced values from `start` to `end` (inclusive),
/// matching `torch.linspace(start, end, steps)`.
pub fn linspace(start: f32, end: f32, steps: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = match steps {
        0 => Vec::new(),
        1 => vec![start],
        n => {
            let denom = (n - 1) as f64;
            (0..n)
                .map(|i| (start as f64 + (end as f64 - start as f64) * (i as f64 / denom)) as f32)
                .collect()
        }
    };
    Tensor::from_vec(data, (steps,), device)
}

/// Deterministic pseudo-random standard normal `f32` tensor using SplitMix64 + Box-Muller.
///
/// Avoids external RNG version drift and guarantees 100% reproducible tests across platforms.
pub fn seeded_randn(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut next_f64 = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        // Map to (0, 1] so ln(u1) is finite.
        ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    };

    let mut data = Vec::with_capacity(n);
    while data.len() < n {
        let u1 = next_f64();
        let u2 = next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push((r * theta.cos()) as f32);
        if data.len() < n {
            data.push((r * theta.sin()) as f32);
        }
    }
    Tensor::from_vec(data, shape, device)
}

/// Assert two tensors have identical shapes and elementwise `|a - b| <= atol + rtol * |b|`
/// (matching `torch.allclose` semantics).
pub fn assert_allclose(actual: &Tensor, expected: &Tensor, rtol: f32, atol: f32) {
    assert_eq!(
        actual.dims(),
        expected.dims(),
        "tensor shape mismatch: actual {:?} vs expected {:?}",
        actual.dims(),
        expected.dims()
    );
    let a = actual
        .to_dtype(DType::F32)
        .expect("cast actual to f32")
        .flatten_all()
        .expect("flatten actual")
        .to_vec1::<f32>()
        .expect("extract actual vec");
    let b = expected
        .to_dtype(DType::F32)
        .expect("cast expected to f32")
        .flatten_all()
        .expect("flatten expected")
        .to_vec1::<f32>()
        .expect("extract expected vec");

    let mut max_diff = 0.0f32;
    let mut worst_idx = None;
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let tol = atol + rtol * bv.abs();
        if (!av.is_finite() || diff > tol) && (diff > max_diff || worst_idx.is_none()) {
            max_diff = diff;
            worst_idx = Some((i, av, bv, tol));
        }
    }
    if let Some((i, av, bv, tol)) = worst_idx {
        panic!(
            "tensors not close (rtol={rtol}, atol={atol}, shape={:?}): worst mismatch at flat index {i}: actual={av}, expected={bv}, diff={} > tol={tol}",
            actual.dims(),
            (av - bv).abs(),
        );
    }
}
