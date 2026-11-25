//! Grader-equivalent ("submit") tests for Homework 2.
//!
//! Ported from the exact deterministic `submit_*` inputs in `hw2_tests.py` with verified
//! analytical ground-truth assertions, plus additional edge cases and gradient checks.
//! Run with: `cargo test -p hw2 --test submit`

mod submit {
    use candle_core::{Device, Tensor, Var};
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

    /// Reference multinomial logistic regression cross-entropy loss in f64.
    fn ref_cross_entropy(logits: &[Vec<f64>], targets: &[usize]) -> f64 {
        let mut total = 0.0;
        for (row, &t) in logits.iter().zip(targets.iter()) {
            let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = max_v + row.iter().map(|&v| (v - max_v).exp()).sum::<f64>().ln();
            total += lse - row[t];
        }
        total / logits.len() as f64
    }

    /// Reference minibatch SGD in f64 using exact analytical cross-entropy gradients.
    fn ref_train_sgd(
        x: &[Vec<f64>],
        y: &[usize],
        epochs: usize,
        step_size: f64,
        batch_size: usize,
    ) -> Vec<Vec<f64>> {
        let n_examples = x.len();
        let n_features = x[0].len();
        let n_classes = *y.iter().max().unwrap() + 1;
        let mut w = vec![vec![0.0_f64; n_features]; n_classes];

        for _ in 0..epochs {
            let mut start = 0;
            while start < n_examples {
                let end = (start + batch_size).min(n_examples);
                let b_len = (end - start) as f64;
                let mut grad = vec![vec![0.0_f64; n_features]; n_classes];

                for i in start..end {
                    let xi = &x[i];
                    let yi = y[i];
                    let logits: Vec<f64> = (0..n_classes)
                        .map(|c| (0..n_features).map(|j| xi[j] * w[c][j]).sum::<f64>())
                        .collect();
                    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = logits.iter().map(|&v| (v - max_l).exp()).collect();
                    let sum_e: f64 = exps.iter().sum();

                    for c in 0..n_classes {
                        let prob = exps[c] / sum_e;
                        let coeff = (prob - if c == yi { 1.0 } else { 0.0 }) / b_len;
                        for j in 0..n_features {
                            grad[c][j] += coeff * xi[j];
                        }
                    }
                }

                for c in 0..n_classes {
                    for j in 0..n_features {
                        w[c][j] -= step_size * grad[c][j];
                    }
                }
                start = end;
            }
        }
        w
    }

    #[test]
    fn submit_add() {
        let f = Add;
        assert_close_f64(f.forward(&[-1.5, 3.0]), 1.5, 1e-12);
        let g = f.backward(2.5, &[-1.5, 3.0]);
        assert_eq!(g.len(), 2);
        assert_close_f64(g[0], 2.5, 1e-12);
        assert_close_f64(g[1], 2.5, 1e-12);

        // Edge case: zero incoming grad and negative values.
        assert_close_f64(f.forward(&[-4.25, -0.75]), -5.0, 1e-12);
        let g0 = f.backward(0.0, &[-4.25, -0.75]);
        assert_eq!(g0, vec![0.0, 0.0]);
    }

    #[test]
    fn submit_subtract() {
        let f = Subtract;
        assert_close_f64(f.forward(&[-1.5, 3.0]), -4.5, 1e-12);
        let g = f.backward(2.5, &[-1.5, 3.0]);
        assert_eq!(g.len(), 2);
        assert_close_f64(g[0], 2.5, 1e-12);
        assert_close_f64(g[1], -2.5, 1e-12);

        // Negative incoming grad.
        let gn = f.backward(-1.25, &[10.0, 4.0]);
        assert_eq!(gn.len(), 2);
        assert_close_f64(gn[0], -1.25, 1e-12);
        assert_close_f64(gn[1], 1.25, 1e-12);
    }

    #[test]
    fn submit_divide() {
        let f = Divide;
        assert_close_f64(f.forward(&[-8.0, 2.0]), -4.0, 1e-12);
        let g = f.backward(1.5, &[-8.0, 2.0]);
        assert_eq!(g.len(), 2);
        assert_close_f64(g[0], 0.75, 1e-12);
        assert_close_f64(g[1], 3.0, 1e-12);

        // Finite-difference cross-check at non-trivial point.
        let (x, y, grad, h) = (3.7, -2.3, -1.8, 1e-6);
        let g_fd = f.backward(grad, &[x, y]);
        let fd_x = grad * (f.forward(&[x + h, y]) - f.forward(&[x - h, y])) / (2.0 * h);
        let fd_y = grad * (f.forward(&[x, y + h]) - f.forward(&[x, y - h])) / (2.0 * h);
        assert_close_f64(g_fd[0], fd_x, 1e-7);
        assert_close_f64(g_fd[1], fd_y, 1e-7);
    }

    #[test]
    fn submit_power() {
        let f = Power::new(3.0);
        let f0 = Power::new(0.0);
        assert_close_f64(f.forward(&[-2.0]), -8.0, 1e-12);
        let g = f.backward(1.5, &[-2.0]);
        assert_eq!(g.len(), 1);
        assert_close_f64(g[0], 18.0, 1e-12);

        let g0 = f0.backward(7.0, &[5.0]);
        assert_eq!(g0.len(), 1);
        assert_close_f64(g0[0], 0.0, 1e-12);

        // Edge case: d = 0 at x = 0 must not produce NaN or panic.
        assert_close_f64(f0.forward(&[0.0]), 1.0, 1e-12);
        let g0_zero = f0.backward(7.0, &[0.0]);
        assert_eq!(g0_zero.len(), 1);
        assert_close_f64(g0_zero[0], 0.0, 1e-12);

        // Fractional and negative exponents.
        let f_half = Power::new(0.5);
        assert_close_f64(f_half.forward(&[4.0]), 2.0, 1e-12);
        assert_close_f64(f_half.backward(3.0, &[4.0])[0], 0.75, 1e-12);

        let f_neg = Power::new(-2.0);
        assert_close_f64(f_neg.forward(&[2.0]), 0.25, 1e-12);
        assert_close_f64(f_neg.backward(4.0, &[2.0])[0], -1.0, 1e-12);
    }

    #[test]
    fn submit_log() {
        let f = Log;
        assert_close_f64(f.forward(&[3.5]), 3.5_f64.ln(), 1e-12);
        let g = f.backward(2.0, &[3.5]);
        assert_eq!(g.len(), 1);
        assert_close_f64(g[0], 2.0 / 3.5, 1e-12);

        // Edge case: x = 1.0.
        assert_close_f64(f.forward(&[1.0]), 0.0, 1e-12);
        assert_close_f64(f.backward(-3.5, &[1.0])[0], -3.5, 1e-12);
    }

    #[test]
    fn submit_exp() {
        let f = Exp;
        assert_close_f64(f.forward(&[-0.5]), (-0.5_f64).exp(), 1e-12);
        let g = f.backward(2.0, &[-0.5]);
        assert_eq!(g.len(), 1);
        assert_close_f64(g[0], 2.0 * (-0.5_f64).exp(), 1e-12);

        // Edge case: x = 0.0.
        assert_close_f64(f.forward(&[0.0]), 1.0, 1e-12);
        assert_close_f64(f.backward(-2.25, &[0.0])[0], -2.25, 1e-12);
    }

    #[test]
    fn submit_compute_gradients() {
        // Exact submit_compute_gradients test from hw2_tests.py (uses only Multiply and Negate).
        let x = Variable::new(1.5);
        let y = Variable::new(-2.0);
        let a = &x * &y * (-&x) * (-&y);
        let z = -(&a * &a);
        z.compute_gradients();

        assert_close_f64(z.value(), -81.0, 1e-12);
        assert_close_f64(z.grad().expect("z.grad"), 1.0, 1e-12);
        assert_close_f64(a.grad().expect("a.grad"), -18.0, 1e-9);
        assert_close_f64(x.grad().expect("x.grad"), -216.0, 1e-9);
        assert_close_f64(y.grad().expect("y.grad"), 162.0, 1e-9);
    }

    #[test]
    fn submit_compute_gradients_full_expressions() {
        // Notebook example 1: d = (x * y + x^2) / y = x + x^2 / y
        let x = Variable::new(3.0);
        let y = Variable::new(4.0);
        let d = (&x * &y + x.pow(2.0)) / &y;
        d.compute_gradients();
        assert_close_f64(d.value(), 5.25, 1e-12);
        assert_close_f64(x.grad().expect("dx"), 2.5, 1e-9);
        assert_close_f64(y.grad().expect("dy"), -9.0 / 16.0, 1e-9);

        // Notebook example 2: ((x * y + x^2 + z).log() / (z^3).exp())
        let eval_expr = |xv: f64, yv: f64, zv: f64| -> (f64, f64, f64, f64) {
            let x = Variable::new(xv);
            let y = Variable::new(yv);
            let z = Variable::new(zv);
            let out = (&x * &y + x.pow(2.0) + &z).log() / z.pow(3.0).exp();
            out.compute_gradients();
            (
                out.value(),
                x.grad().unwrap(),
                y.grad().unwrap(),
                z.grad().unwrap(),
            )
        };

        let (xv, yv, zv) = (3.0, 4.0, 1.2);
        let (val, gx, gy, gz) = eval_expr(xv, yv, zv);
        let u = xv * yv + xv * xv + zv;
        let v = (zv.powi(3)).exp();
        assert_close_f64(val, u.ln() / v, 1e-12);
        assert_close_f64(gx, (yv + 2.0 * xv) / (u * v), 1e-10);
        assert_close_f64(gy, xv / (u * v), 1e-10);
        assert_close_f64(gz, (1.0 / u - 3.0 * zv * zv * u.ln()) / v, 1e-10);

        // Central finite-difference verification.
        let h = 1e-6;
        let fd_x = (eval_expr(xv + h, yv, zv).0 - eval_expr(xv - h, yv, zv).0) / (2.0 * h);
        let fd_y = (eval_expr(xv, yv + h, zv).0 - eval_expr(xv, yv - h, zv).0) / (2.0 * h);
        let fd_z = (eval_expr(xv, yv, zv + h).0 - eval_expr(xv, yv, zv - h).0) / (2.0 * h);
        assert_close_f64(gx, fd_x, 1e-7);
        assert_close_f64(gy, fd_y, 1e-7);
        assert_close_f64(gz, fd_z, 1e-7);

        // Diamond graph using Subtract and Add: (x - y) * (x + y) = x^2 - y^2.
        let x2 = Variable::new(2.5);
        let y2 = Variable::new(-1.5);
        let out2 = (&x2 - &y2) * (&x2 + &y2);
        out2.compute_gradients();
        assert_close_f64(out2.value(), 4.0, 1e-12);
        assert_close_f64(x2.grad().unwrap(), 5.0, 1e-12);
        assert_close_f64(y2.grad().unwrap(), 3.0, 1e-12);
    }

    #[test]
    fn submit_cross_entropy_loss() {
        // Exact submit_cross_entropy_loss input from hw2_tests.py.
        let logits_vec = vec![
            vec![1.0_f64, 0.0, -1.0],
            vec![2.0, 1.0, 0.0],
            vec![-1.0, 2.0, 1.0],
        ];
        let targets = vec![0usize, 2, 1];
        let expected_val = ref_cross_entropy(&logits_vec, &targets) as f32;

        let y_pred = Tensor::new(
            &[[1.0_f32, 0.0, -1.0], [2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]],
            &Device::Cpu,
        )
        .unwrap();
        let y = Tensor::new(&[0_u32, 2, 1], &Device::Cpu).unwrap();
        let loss = cross_entropy_loss(&y_pred, &y).unwrap();
        assert_eq!(loss.rank(), 0);
        assert_allclose(&loss, &Tensor::new(expected_val, &Device::Cpu).unwrap(), 1e-5, 1e-5);

        // Numerical stability check with large shifted logits.
        let shifted = Tensor::new(
            &[[1001.0_f32, 1000.0, 999.0], [-998.0, -999.0, -1000.0], [499.0, 502.0, 501.0]],
            &Device::Cpu,
        )
        .unwrap();
        let loss_shifted = cross_entropy_loss(&shifted, &y).unwrap();
        assert_allclose(&loss_shifted, &loss, 1e-4, 1e-4);

        // Differentiability check: dL/dy_pred == (softmax(y_pred) - one_hot(y)) / N.
        let y_pred_var = Var::from_tensor(&y_pred).unwrap();
        let loss_var = cross_entropy_loss(y_pred_var.as_tensor(), &y).unwrap();
        let grads = loss_var.backward().unwrap();
        let grad_logits = grads
            .get(y_pred_var.as_tensor())
            .expect("cross_entropy_loss must be differentiable w.r.t. y_pred");

        let mut expected_grad = Vec::new();
        let n = logits_vec.len() as f64;
        for (row, &t) in logits_vec.iter().zip(targets.iter()) {
            let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&v| (v - max_v).exp()).collect();
            let sum_e: f64 = exps.iter().sum();
            for (j, &e) in exps.iter().enumerate() {
                let p = e / sum_e;
                expected_grad.push(((p - if j == t { 1.0 } else { 0.0 }) / n) as f32);
            }
        }
        let expected_grad_t = Tensor::from_vec(expected_grad, (3, 3), &Device::Cpu).unwrap();
        assert_allclose(grad_logits, &expected_grad_t, 1e-5, 1e-5);

        // Shape mismatch returns Err.
        let bad_y = Tensor::new(&[0_u32, 1], &Device::Cpu).unwrap();
        assert!(cross_entropy_loss(&y_pred, &bad_y).is_err());
    }

    #[test]
    fn submit_error() {
        // Exact submit_error input from hw2_tests.py.
        let y_pred = Tensor::new(
            &[[3.0_f32, 1.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]],
            &Device::Cpu,
        )
        .unwrap();
        let y = Tensor::new(&[0_u32, 1, 1, 1], &Device::Cpu).unwrap();
        let err = error(&y_pred, &y).unwrap();
        assert_eq!(err.rank(), 0);
        assert_allclose(&err, &Tensor::new(0.25_f32, &Device::Cpu).unwrap(), 1e-5, 1e-5);

        // 0% error and 100% error multi-class cases.
        let logits_mc = Tensor::new(
            &[
                [5.0_f32, 1.0, 0.0, -1.0],
                [-2.0, 0.5, 4.0, 1.0],
                [0.0, 0.1, 0.2, 0.3],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let y_perfect = Tensor::new(&[0_u32, 2, 3], &Device::Cpu).unwrap();
        let y_wrong = Tensor::new(&[1_u32, 0, 2], &Device::Cpu).unwrap();
        assert_allclose(
            &error(&logits_mc, &y_perfect).unwrap(),
            &Tensor::new(0.0_f32, &Device::Cpu).unwrap(),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &error(&logits_mc, &y_wrong).unwrap(),
            &Tensor::new(1.0_f32, &Device::Cpu).unwrap(),
            1e-5,
            1e-5,
        );

        // Shape mismatch returns Err.
        let bad_y = Tensor::new(&[0_u32, 1], &Device::Cpu).unwrap();
        assert!(error(&logits_mc, &bad_y).is_err());
    }

    #[test]
    fn submit_train_sgd() {
        // Exact submit_train_sgd input from hw2_tests.py.
        let x_rows = vec![
            vec![2.0_f64, 1.0],
            vec![1.0, 2.0],
            vec![-2.0, -1.0],
            vec![-1.0, -2.0],
        ];
        let y_vec = vec![1usize, 1, 0, 0];
        let w_ref_rows = ref_train_sgd(&x_rows, &y_vec, 15, 0.1, 2);
        let w_ref_flat: Vec<f32> = w_ref_rows
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();
        let w_ref = Tensor::from_vec(w_ref_flat, (2, 2), &Device::Cpu).unwrap();

        let x = Tensor::new(
            &[[2.0_f32, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]],
            &Device::Cpu,
        )
        .unwrap();
        let y = Tensor::new(&[1_u32, 1, 0, 0], &Device::Cpu).unwrap();

        let w = train_sgd(&x, &y, 15, 0.1, 2).unwrap();
        assert_allclose(&w, &w_ref, 1e-4, 1e-5);

        let preds = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .argmax(1)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(preds, vec![1, 1, 0, 0]);

        let norm = w
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        let expected_norm = w_ref
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert_close_f64(norm as f64, expected_norm as f64, 1e-4);

        // Extra edge case: multi-class (k = 3) with non-divisible batch_size (N = 5, batch_size = 2).
        let x_mc_rows = vec![
            vec![2.0_f64, 0.5, -1.0],
            vec![-1.0, 2.0, 0.5],
            vec![0.5, -1.0, 2.0],
            vec![1.5, 0.2, -0.8],
            vec![-0.8, 1.8, 0.3],
        ];
        let y_mc_vec = vec![0usize, 1, 2, 0, 1];
        let w_mc_ref_rows = ref_train_sgd(&x_mc_rows, &y_mc_vec, 10, 0.2, 2);
        let w_mc_ref_flat: Vec<f32> = w_mc_ref_rows
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();
        let w_mc_ref = Tensor::from_vec(w_mc_ref_flat, (3, 3), &Device::Cpu).unwrap();

        let x_mc_flat: Vec<f32> = x_mc_rows
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();
        let x_mc = Tensor::from_vec(x_mc_flat, (5, 3), &Device::Cpu).unwrap();
        let y_mc = Tensor::new(&[0_u32, 1, 2, 0, 1], &Device::Cpu).unwrap();

        let w_mc = train_sgd(&x_mc, &y_mc, 10, 0.2, 2).unwrap();
        assert_allclose(&w_mc, &w_mc_ref, 1e-4, 1e-4);
        let err_mc = error(&x_mc.matmul(&w_mc.t().unwrap()).unwrap(), &y_mc)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(err_mc, 0.0, "multi-class separable set should reach 0 error");

        // Invalid inputs return Err.
        assert!(train_sgd(&x, &y, 1, 0.1, 0).is_err());
        let bad_y = Tensor::new(&[0_u32, 1], &Device::Cpu).unwrap();
        assert!(train_sgd(&x, &bad_y, 1, 0.1, 2).is_err());
    }
}
