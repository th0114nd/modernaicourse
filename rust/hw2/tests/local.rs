//! Local sanity-check tests for Homework 2.
//!
//! Ported directly from `homework/hw2/hw2_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw2 --test local`

mod local {
    use candle_core::{Device, Tensor};
    use common::assert_allclose;
    use hw2::{
        Add, Divide, Exp, Function, Log, Power, Subtract, Variable, cross_entropy_loss, error,
        train_sgd,
    };

    fn assert_close_f64(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "values not close: actual={actual}, expected={expected}, diff={}, tol={tol}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn test_add() {
        let f = Add;
        assert_close_f64(f.forward(&[2.5, -0.5]), 2.0, 1e-12);
        let g = f.backward(3.0, &[2.5, -0.5]);
        assert_eq!(g.len(), 2);
        assert_close_f64(g[0], 3.0, 1e-12);
        assert_close_f64(g[1], 3.0, 1e-12);
    }

    #[test]
    fn test_subtract() {
        let f = Subtract;
        assert_close_f64(f.forward(&[2.5, -0.5]), 3.0, 1e-12);
        let g = f.backward(3.0, &[2.5, -0.5]);
        assert_eq!(g.len(), 2);
        assert_close_f64(g[0], 3.0, 1e-12);
        assert_close_f64(g[1], -3.0, 1e-12);
    }

    #[test]
    fn test_divide() {
        let f = Divide;
        assert_close_f64(f.forward(&[9.0, 3.0]), 3.0, 1e-12);
        let g = f.backward(4.0, &[9.0, 3.0]);
        assert_eq!(g.len(), 2);
        assert_close_f64(g[0], 4.0 / 3.0, 1e-9);
        assert_close_f64(g[1], -4.0, 1e-9);
    }

    #[test]
    fn test_power() {
        let f = Power::new(3.0);
        assert_close_f64(f.forward(&[2.0]), 8.0, 1e-12);
        let g = f.backward(2.0, &[2.0]);
        assert_eq!(g.len(), 1);
        assert_close_f64(g[0], 24.0, 1e-12);

        let f0 = Power::new(0.0);
        assert_close_f64(f0.forward(&[5.0]), 1.0, 1e-12);
        let g0 = f0.backward(7.0, &[5.0]);
        assert_eq!(g0.len(), 1);
        assert_close_f64(g0[0], 0.0, 1e-12);
    }

    #[test]
    fn test_log() {
        let f = Log;
        assert_close_f64(f.forward(&[std::f64::consts::E]), 1.0, 1e-9);
        let g = f.backward(2.0, &[4.0]);
        assert_eq!(g.len(), 1);
        assert_close_f64(g[0], 0.5, 1e-9);
    }

    #[test]
    fn test_exp() {
        let f = Exp;
        assert_close_f64(f.forward(&[1.0]), std::f64::consts::E, 1e-9);
        let g = f.backward(2.0, &[1.0]);
        assert_eq!(g.len(), 1);
        assert_close_f64(g[0], 2.0 * std::f64::consts::E, 1e-9);
    }

    #[test]
    fn test_compute_gradients() {
        let x = Variable::new(3.0);
        let y = Variable::new(4.0);
        let z = (-(&x * &y) * &x * &x) * (-&y);
        z.compute_gradients();
        assert_close_f64(z.grad().expect("z.grad"), 1.0, 1e-12);
        assert_close_f64(x.grad().expect("x.grad"), 432.0, 1e-9);
        assert_close_f64(y.grad().expect("y.grad"), 216.0, 1e-9);

        let w = Variable::new(-2.0);
        w.compute_gradients();
        assert_close_f64(w.grad().expect("w.grad"), 1.0, 1e-12);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let y_pred = Tensor::new(&[[2.0_f32, 1.0, 0.0], [0.0, 2.0, 1.0]], &Device::Cpu).unwrap();
        let y = Tensor::new(&[0_u32, 2], &Device::Cpu).unwrap();
        let loss = cross_entropy_loss(&y_pred, &y).unwrap();
        assert_eq!(loss.rank(), 0, "expected scalar (0D) tensor");
        let expected = Tensor::new(0.907606_f32, &Device::Cpu).unwrap();
        assert_allclose(&loss, &expected, 1e-5, 1e-6);
    }

    #[test]
    fn test_error() {
        let y_pred = Tensor::new(
            &[[3.0_f32, 1.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]],
            &Device::Cpu,
        )
        .unwrap();
        let y = Tensor::new(&[0_u32, 1, 1, 1], &Device::Cpu).unwrap();
        let err = error(&y_pred, &y).unwrap();
        assert_eq!(err.rank(), 0, "expected scalar (0D) tensor");
        let expected = Tensor::new(0.25_f32, &Device::Cpu).unwrap();
        assert_allclose(&err, &expected, 1e-5, 1e-5);
    }

    #[test]
    fn test_train_sgd() {
        let x = Tensor::new(
            &[[2.0_f32, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]],
            &Device::Cpu,
        )
        .unwrap();
        let y = Tensor::new(&[1_u32, 1, 0, 0], &Device::Cpu).unwrap();

        let w_small = train_sgd(&x, &y, 1, 0.1, 2).unwrap();
        let w_expected = Tensor::new(
            &[[-0.13340412_f32, -0.13340412], [0.13340412, 0.13340412]],
            &Device::Cpu,
        )
        .unwrap();
        assert_allclose(&w_small, &w_expected, 1e-5, 1e-6);

        let w = train_sgd(&x, &y, 20, 0.1, 2).unwrap();
        assert_eq!(w.dims(), &[2, 2]);
        let preds = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .argmax(1)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(preds, vec![1, 1, 0, 0]);
        let norm_sq: f32 = w
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(norm_sq > 1e-4, "expected non-zero trained weights");
    }
}
