//! Grader-equivalent ("submit") tests for Homework 3.
//!
//! Ported from the exact deterministic `submit_*` inputs in `hw3_tests.py` with verified
//! analytical ground-truth assertions, plus additional edge cases and gradient checks.
//! Run with: `cargo test --release -p hw3 --test submit`

mod submit {
    use candle_core::{D, DType, Device, Result, Tensor, Var, backprop::GradStore};
    use common::{assert_allclose, mnist, seeded_randn};
    use hw3::{
        CrossEntropyLoss, DataLoader, Linear, Loss, Module, MultiLayerNN, Optimizer, SGD,
        TwoLayerNN, epoch, eval_linear_model, eval_two_layer_nn,
    };

    fn assert_close_f64(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "values not close: actual={actual}, expected={expected}, diff={}, tol={tol}",
            (actual - expected).abs()
        );
    }

    fn ref_cross_entropy(logits: &[Vec<f64>], targets: &[usize]) -> f64 {
        let mut total = 0.0;
        for (row, &t) in logits.iter().zip(targets.iter()) {
            let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = max_v + row.iter().map(|&v| (v - max_v).exp()).sum::<f64>().ln();
            total += lse - row[t];
        }
        total / logits.len() as f64
    }

    fn matmul_wt(x: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                w.iter()
                    .map(|w_row| row.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect()
    }

    fn relu_2d(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| row.iter().map(|&v| v.max(0.0)).collect())
            .collect()
    }

    fn to_tensor_2d(rows: &[Vec<f64>], device: &Device) -> Tensor {
        let r = rows.len();
        let c = rows[0].len();
        let flat: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();
        Tensor::from_vec(flat, (r, c), device).unwrap()
    }

    /// Reference linear layer for decoupling `submit_epoch`.
    struct RefLinear {
        weight: Var,
    }

    impl Module for RefLinear {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            x.broadcast_matmul(&self.weight.as_tensor().t()?)
        }

        fn parameters(&self) -> Vec<Var> {
            vec![self.weight.clone()]
        }
    }

    struct RefCrossEntropy;

    impl Loss for RefCrossEntropy {
        fn forward(&self, logits: &Tensor, y: &Tensor) -> Result<Tensor> {
            let lse = logits.log_sum_exp(1)?;
            let target = logits.gather(&y.unsqueeze(1)?, 1)?.squeeze(1)?;
            (lse - target)?.mean_all()
        }
    }

    struct RefSgd {
        params: Vec<Var>,
        lr: f64,
    }

    impl Optimizer for RefSgd {
        fn step(&mut self, grads: &GradStore) -> Result<()> {
            for p in &self.params {
                if let Some(g) = grads.get(p.as_tensor()) {
                    p.set(&(p.as_tensor() - (g * self.lr)?)?)?;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn submit_linear() {
        let device = Device::Cpu;
        let layer = Linear::new(4, 3, &device).unwrap();
        let w = Tensor::new(
            &[
                [1.0_f32, -1.0, 0.5, 2.0],
                [-0.5, 0.0, 1.0, -1.5],
                [2.0, 1.5, -0.5, 0.25],
            ],
            &device,
        )
        .unwrap();
        layer.weight.set(&w).unwrap();

        let x = Tensor::new(
            &[[1.0_f32, 2.0, -1.0, 0.5], [0.0, -1.0, 2.0, 3.0]],
            &device,
        )
        .unwrap();

        let expected_2d = Tensor::new(
            &[[-0.5_f32, -2.25, 5.625], [8.0, -2.5, -1.75]],
            &device,
        )
        .unwrap();
        assert_allclose(&layer.forward(&x).unwrap(), &expected_2d, 1e-5, 1e-5);

        // 3D batched input: stack([X, -X])
        let x3d = Tensor::stack(&[&x, &x.neg().unwrap()], 0).unwrap();
        let expected_3d =
            Tensor::stack(&[&expected_2d, &expected_2d.neg().unwrap()], 0).unwrap();
        assert_allclose(&layer.forward(&x3d).unwrap(), &expected_3d, 1e-5, 1e-5);

        // Parameter tracking & autograd gradient check: d/dW sum(X @ W^T) = [1, 1, 1]^T @ (sum_b X_b).
        assert_eq!(layer.parameters().len(), 1);
        let loss = layer.forward(&x).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let grad_w = grads
            .get(layer.weight.as_tensor())
            .expect("Linear.weight must receive gradients");
        let expected_grad_w = Tensor::new(
            &[
                [1.0_f32, 1.0, 1.0, 3.5],
                [1.0, 1.0, 1.0, 3.5],
                [1.0, 1.0, 1.0, 3.5],
            ],
            &device,
        )
        .unwrap();
        assert_allclose(grad_w, &expected_grad_w, 1e-5, 1e-5);

        // Kaiming/He init distribution check (mean ~ 0, std ~ sqrt(2 / in_dim)).
        let big = Linear::new(200, 1000, &device).unwrap();
        let vals = big.weight.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let n = vals.len() as f64;
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let std = (vals
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0))
            .sqrt();
        assert!(mean.abs() < 5e-3, "expected mean near 0, got {mean}");
        assert_close_f64(std, (2.0_f64 / 200.0).sqrt(), 5e-3);
    }

    #[test]
    fn submit_cross_entropy_loss() {
        let device = Device::Cpu;
        let loss_fn = CrossEntropyLoss::new();

        // Exact submit_CrossEntropyLoss input 1 from hw3_tests.py.
        let logits_rows = vec![
            vec![2.0_f64, 1.0, 0.0],
            vec![0.0, 2.0, 1.0],
            vec![1.5, -0.5, 0.25],
        ];
        let targets = vec![0usize, 2, 1];
        let expected1 = ref_cross_entropy(&logits_rows, &targets) as f32;

        let logits = to_tensor_2d(&logits_rows, &device);
        let y = Tensor::new(&[0_u32, 2, 1], &device).unwrap();
        let loss1 = loss_fn.forward(&logits, &y).unwrap();
        assert_eq!(loss1.rank(), 0);
        assert_allclose(&loss1, &Tensor::new(expected1, &device).unwrap(), 1e-5, 1e-5);

        // Exact submit_CrossEntropyLoss input 2 (large logits stability).
        let big_rows = vec![vec![1000.0_f64, 1002.0], vec![999.0, 998.0]];
        let big_targets = vec![1usize, 0];
        let expected2 = ref_cross_entropy(&big_rows, &big_targets) as f32;
        let logits2 = to_tensor_2d(&big_rows, &device);
        let y2 = Tensor::new(&[1_u32, 0], &device).unwrap();
        let loss2 = loss_fn.forward(&logits2, &y2).unwrap();
        assert_allclose(&loss2, &Tensor::new(expected2, &device).unwrap(), 1e-5, 1e-5);

        // Analytical gradient check: dL/dlogits = (softmax(logits) - one_hot(y)) / N.
        let logits_var = Var::from_tensor(&logits).unwrap();
        let loss_var = loss_fn.forward(logits_var.as_tensor(), &y).unwrap();
        let grads = loss_var.backward().unwrap();
        let grad_logits = grads
            .get(logits_var.as_tensor())
            .expect("CrossEntropyLoss must be differentiable w.r.t. logits");

        let n = logits_rows.len() as f64;
        let mut expected_grad = Vec::new();
        for (row, &t) in logits_rows.iter().zip(targets.iter()) {
            let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&v| (v - max_v).exp()).collect();
            let sum_e: f64 = exps.iter().sum();
            for (j, &e) in exps.iter().enumerate() {
                expected_grad.push(((e / sum_e - if j == t { 1.0 } else { 0.0 }) / n) as f32);
            }
        }
        let expected_grad_t = Tensor::from_vec(expected_grad, (3, 3), &device).unwrap();
        assert_allclose(grad_logits, &expected_grad_t, 1e-5, 1e-5);

        // Shape mismatch returns Err.
        let bad_y = Tensor::new(&[0_u32, 1], &device).unwrap();
        assert!(loss_fn.forward(&logits, &bad_y).is_err());
    }

    #[test]
    fn submit_sgd() {
        let device = Device::Cpu;
        // Exact submit_SGD weights, grads, and learning_rate from hw3_tests.py.
        let w = Var::new(&[[1.0_f32, -2.0], [0.5, 3.0]], &device).unwrap();
        let b = Var::new(&[0.25_f32, -0.75], &device).unwrap();
        let unused = Var::new(&[42.0_f32, -7.0], &device).unwrap();

        let mut opt = SGD::new(vec![w.clone(), b.clone(), unused.clone()], 0.2);

        // Construct a linear loss whose exact gradients w.r.t. w and b equal g_w and g_b.
        let g_w = Tensor::new(&[[0.1_f32, -0.4], [-0.2, 0.3]], &device).unwrap();
        let g_b = Tensor::new(&[0.5_f32, -0.25], &device).unwrap();
        let loss = ((w.as_tensor() * &g_w).unwrap().sum_all().unwrap()
            + (b.as_tensor() * &g_b).unwrap().sum_all().unwrap())
        .unwrap();
        let grads = loss.backward().unwrap();

        opt.step(&grads).unwrap();

        let expected_w = Tensor::new(&[[0.98_f32, -1.92], [0.54, 2.94]], &device).unwrap();
        let expected_b = Tensor::new(&[0.15_f32, -0.70], &device).unwrap();
        let expected_unused = Tensor::new(&[42.0_f32, -7.0], &device).unwrap();

        assert_allclose(w.as_tensor(), &expected_w, 1e-5, 1e-5);
        assert_allclose(b.as_tensor(), &expected_b, 1e-5, 1e-5);
        // Unused parameter (absent from GradStore) must remain untouched.
        assert_allclose(unused.as_tensor(), &expected_unused, 0.0, 0.0);

        // Calling zero_grad + a second step continues updating cleanly.
        opt.zero_grad();
        let loss2 = ((w.as_tensor() * &g_w).unwrap().sum_all().unwrap()
            + (b.as_tensor() * &g_b).unwrap().sum_all().unwrap())
        .unwrap();
        opt.step(&loss2.backward().unwrap()).unwrap();

        let expected_w2 = Tensor::new(&[[0.96_f32, -1.84], [0.58, 2.88]], &device).unwrap();
        let expected_b2 = Tensor::new(&[0.05_f32, -0.65], &device).unwrap();
        assert_allclose(w.as_tensor(), &expected_w2, 1e-5, 1e-5);
        assert_allclose(b.as_tensor(), &expected_b2, 1e-5, 1e-5);
    }

    #[test]
    fn submit_dataloader() {
        let device = Device::Cpu;
        // Exact submit_DataLoader input from hw3_tests.py.
        let x_data: Vec<f32> = (0..14).map(|i| i as f32).collect();
        let x = Tensor::from_vec(x_data, (7, 2), &device).unwrap();
        let y = Tensor::new(&[3_u32, 0, 4, 1, 5, 2, 6], &device).unwrap();
        let loader = DataLoader::new(x.clone(), y.clone(), 3).unwrap();

        let batches: Vec<(Tensor, Tensor)> =
            (&loader).into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(batches.len(), 3);

        assert_allclose(
            &batches[0].0,
            &Tensor::new(&[[0.0_f32, 1.0], [2.0, 3.0], [4.0, 5.0]], &device).unwrap(),
            0.0,
            0.0,
        );
        assert_eq!(batches[0].1.to_vec1::<u32>().unwrap(), vec![3, 0, 4]);

        assert_allclose(
            &batches[1].0,
            &Tensor::new(&[[6.0_f32, 7.0], [8.0, 9.0], [10.0, 11.0]], &device).unwrap(),
            0.0,
            0.0,
        );
        assert_eq!(batches[1].1.to_vec1::<u32>().unwrap(), vec![1, 5, 2]);

        assert_allclose(
            &batches[2].0,
            &Tensor::new(&[[12.0_f32, 13.0]], &device).unwrap(),
            0.0,
            0.0,
        );
        assert_eq!(batches[2].1.to_vec1::<u32>().unwrap(), vec![6]);

        // Edge case: batch_size larger than dataset yields a single full batch.
        let loader_big = DataLoader::new(x.clone(), y.clone(), 100).unwrap();
        let big_batches: Vec<_> = (&loader_big).into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(big_batches.len(), 1);
        assert_allclose(&big_batches[0].0, &x, 0.0, 0.0);

        // Edge case: exact divisibility (N = 6, batch_size = 2 -> 3 full batches).
        let x6 = x.narrow(0, 0, 6).unwrap();
        let y6 = y.narrow(0, 0, 6).unwrap();
        let loader_even = DataLoader::new(x6, y6, 2).unwrap();
        let even_batches: Vec<_> = (&loader_even)
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(even_batches.len(), 3);
        assert!(even_batches.iter().all(|(xb, _)| xb.dim(0).unwrap() == 2));

        // Edge case: empty dataset yields 0 batches.
        let x_empty = Tensor::zeros((0, 2), DType::F32, &device).unwrap();
        let y_empty = Tensor::zeros((0,), DType::U32, &device).unwrap();
        let loader_empty = DataLoader::new(x_empty, y_empty, 4).unwrap();
        assert_eq!(loader_empty.iter().count(), 0);

        // Invalid inputs return Err.
        assert!(DataLoader::new(x.clone(), y.clone(), 0).is_err());
        let bad_y = Tensor::new(&[0_u32, 1], &device).unwrap();
        assert!(DataLoader::new(x, bad_y, 2).is_err());
    }

    #[test]
    fn submit_epoch() {
        let device = Device::Cpu;
        // Exact submit_epoch input from hw3_tests.py.
        let x_rows = vec![
            vec![1.0_f64, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
        ];
        let y_vec = vec![0usize, 1, 1, 0];
        let w_init_rows = vec![vec![1.0_f64, -1.0], vec![-0.5, 0.25]];

        let x = to_tensor_2d(&x_rows, &device);
        let y = Tensor::new(&[0_u32, 1, 1, 0], &device).unwrap();
        let w_init = to_tensor_2d(&w_init_rows, &device);

        let model = RefLinear {
            weight: Var::from_tensor(&w_init).unwrap(),
        };
        let loss_fn = RefCrossEntropy;

        let make_batches = |bs: usize| -> Vec<Result<(Tensor, Tensor)>> {
            let n = x.dim(0).unwrap();
            let mut out = Vec::new();
            let mut start = 0;
            while start < n {
                let len = bs.min(n - start);
                out.push(Ok((
                    x.narrow(0, start, len).unwrap(),
                    y.narrow(0, start, len).unwrap(),
                )));
                start += len;
            }
            out
        };

        // 1. Analytical evaluation metrics on W_init.
        let logits_eval = matmul_wt(&x_rows, &w_init_rows);
        let expected_eval_loss = ref_cross_entropy(&logits_eval, &y_vec);
        let expected_eval_err = logits_eval
            .iter()
            .zip(y_vec.iter())
            .filter(|(row, t)| {
                let pred = if row[1] > row[0] { 1 } else { 0 };
                pred != **t
            })
            .count() as f64
            / y_vec.len() as f64;

        let (eval_loss, eval_err) = epoch(&model, make_batches(2), &loss_fn, None).unwrap();
        assert_close_f64(eval_loss, expected_eval_loss, 1e-5);
        assert_close_f64(eval_err, expected_eval_err, 1e-5);
        assert_allclose(model.weight.as_tensor(), &w_init, 0.0, 0.0);

        // 2. Analytical 1-epoch SGD training pass with lr = 0.2, batch_size = 2.
        let mut w_ref = w_init_rows.clone();
        let mut ref_total_loss = 0.0;
        let mut ref_total_err = 0.0;
        for chunk_start in (0..4).step_by(2) {
            let bx = &x_rows[chunk_start..chunk_start + 2];
            let by = &y_vec[chunk_start..chunk_start + 2];
            let b_logits = matmul_wt(bx, &w_ref);
            ref_total_loss += ref_cross_entropy(&b_logits, by) * bx.len() as f64;
            for (row, &t) in b_logits.iter().zip(by.iter()) {
                let pred = if row[1] > row[0] { 1 } else { 0 };
                if pred != t {
                    ref_total_err += 1.0;
                }
            }
            // Exact gradient of mean cross-entropy w.r.t. W.
            let mut grad = vec![vec![0.0_f64; 2]; 2];
            for (xi, (row, &t)) in bx.iter().zip(b_logits.iter().zip(by.iter())) {
                let max_v = row[0].max(row[1]);
                let e0 = (row[0] - max_v).exp();
                let e1 = (row[1] - max_v).exp();
                let s = e0 + e1;
                let probs = [e0 / s, e1 / s];
                for c in 0..2 {
                    let coeff = (probs[c] - if c == t { 1.0 } else { 0.0 }) / bx.len() as f64;
                    for j in 0..2 {
                        grad[c][j] += coeff * xi[j];
                    }
                }
            }
            for c in 0..2 {
                for j in 0..2 {
                    w_ref[c][j] -= 0.2 * grad[c][j];
                }
            }
        }
        let expected_train_loss = ref_total_loss / 4.0;
        let expected_train_err = ref_total_err / 4.0;
        let expected_w_after = to_tensor_2d(&w_ref, &device);

        let mut opt = RefSgd {
            params: model.parameters(),
            lr: 0.2,
        };
        let (train_loss, train_err) =
            epoch(&model, make_batches(2), &loss_fn, Some(&mut opt)).unwrap();

        assert_close_f64(train_loss, expected_train_loss, 1e-5);
        assert_close_f64(train_err, expected_train_err, 1e-5);
        assert_allclose(model.weight.as_tensor(), &expected_w_after, 1e-5, 1e-5);

        // 3. Extra edge case: uneven batches (batch_size = 3 on N = 4) must weight by batch size.
        let model_uneven = RefLinear {
            weight: Var::from_tensor(&w_init).unwrap(),
        };
        let (uneven_loss, uneven_err) =
            epoch(&model_uneven, make_batches(3), &loss_fn, None).unwrap();
        assert_close_f64(uneven_loss, expected_eval_loss, 1e-5);
        assert_close_f64(uneven_err, expected_eval_err, 1e-5);
    }

    #[test]
    fn submit_eval_linear_model() {
        let device = Device::Cpu;
        let model = eval_linear_model().unwrap();

        let test = mnist::load_test(&device).unwrap();
        let x_test = test.images.reshape((test.len(), 784)).unwrap();
        let y_test_u32: Vec<u32> = test.labels.iter().map(|&l| l as u32).collect();
        let y_test = Tensor::from_vec(y_test_u32, (test.len(),), &device).unwrap();

        let logits = model.forward(&x_test).unwrap();
        assert_eq!(logits.dims(), &[10_000, 10]);
        assert_eq!(model.weight.dims(), &[10, 784]);

        let err = logits
            .argmax(D::Minus1)
            .unwrap()
            .ne(&y_test)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(err < 0.1, "expected linear MNIST test error < 0.10, got {err:.4}");
    }

    #[test]
    fn submit_two_layer_nn() {
        let device = Device::Cpu;
        // Exact submit_TwoLayerNN weights & inputs from hw3_tests.py.
        let w1_rows = vec![
            vec![1.0_f64, -1.0, 0.5],
            vec![-0.5, 0.0, 1.0],
            vec![0.25, -0.75, 0.5],
            vec![1.5, 0.5, -1.0],
        ];
        let w2_rows = vec![
            vec![1.0_f64, -2.0, 0.5, 0.0],
            vec![-1.0, 0.5, 1.5, -0.5],
        ];
        let x_rows = vec![
            vec![1.0_f64, 2.0, -1.0],
            vec![0.5, -0.5, 0.25],
            vec![-1.0, 0.0, 1.0],
        ];
        let neg_x_rows: Vec<Vec<f64>> = x_rows
            .iter()
            .map(|r| r.iter().map(|&v| -v).collect())
            .collect();

        let model = TwoLayerNN::new(3, 4, 2, &device).unwrap();
        model.linear1.weight.set(&to_tensor_2d(&w1_rows, &device)).unwrap();
        model.linear2.weight.set(&to_tensor_2d(&w2_rows, &device)).unwrap();

        let ref_2d_pos = matmul_wt(&relu_2d(&matmul_wt(&x_rows, &w1_rows)), &w2_rows);
        let ref_2d_neg = matmul_wt(&relu_2d(&matmul_wt(&neg_x_rows, &w1_rows)), &w2_rows);

        let x = to_tensor_2d(&x_rows, &device);
        let expected_2d = to_tensor_2d(&ref_2d_pos, &device);
        assert_allclose(&model.forward(&x).unwrap(), &expected_2d, 1e-5, 1e-5);

        let x3d = Tensor::stack(&[&x, &x.neg().unwrap()], 0).unwrap();
        let expected_3d =
            Tensor::stack(&[&expected_2d, &to_tensor_2d(&ref_2d_neg, &device)], 0).unwrap();
        assert_allclose(&model.forward(&x3d).unwrap(), &expected_3d, 1e-5, 1e-5);

        assert_eq!(model.parameters().len(), 2);

        // Verify autograd gradients flow cleanly through ReLU into both weight matrices
        // using an input whose pre-activations avoid the non-differentiable ReLU kink at 0.
        let x_grad_rows = vec![
            vec![1.1_f64, 2.3, -0.7],
            vec![0.4, -0.9, 0.35],
            vec![-1.2, 0.3, 0.85],
        ];
        let x_grad = to_tensor_2d(&x_grad_rows, &device);
        let loss = model.forward(&x_grad).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g1 = grads
            .get(model.linear1.weight.as_tensor())
            .expect("linear1.weight must receive gradients")
            .to_vec2::<f32>()
            .unwrap();
        let g2 = grads
            .get(model.linear2.weight.as_tensor())
            .expect("linear2.weight must receive gradients")
            .to_vec2::<f32>()
            .unwrap();

        // Central finite-difference check on both W1 and W2.
        let eval_f64 = |w1: &[Vec<f64>], w2: &[Vec<f64>]| -> f64 {
            let out = matmul_wt(&relu_2d(&matmul_wt(&x_grad_rows, w1)), w2);
            out.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum()
        };
        let eps = 1e-5;
        for i in 0..4 {
            for j in 0..3 {
                let mut w1p = w1_rows.clone();
                let mut w1m = w1_rows.clone();
                w1p[i][j] += eps;
                w1m[i][j] -= eps;
                let fd = (eval_f64(&w1p, &w2_rows) - eval_f64(&w1m, &w2_rows)) / (2.0 * eps);
                assert_close_f64(g1[i][j] as f64, fd, 1e-3);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                let mut w2p = w2_rows.clone();
                let mut w2m = w2_rows.clone();
                w2p[i][j] += eps;
                w2m[i][j] -= eps;
                let fd = (eval_f64(&w1_rows, &w2p) - eval_f64(&w1_rows, &w2m)) / (2.0 * eps);
                assert_close_f64(g2[i][j] as f64, fd, 1e-3);
            }
        }
    }

    #[test]
    fn submit_eval_two_layer_nn() {
        let device = Device::Cpu;
        let model = eval_two_layer_nn().unwrap();

        let test = mnist::load_test(&device).unwrap();
        let x_sub = test
            .images
            .reshape((test.len(), 784))
            .unwrap()
            .narrow(0, 0, 1000)
            .unwrap();
        let y_sub_u32: Vec<u32> = test.labels[..1000].iter().map(|&l| l as u32).collect();
        let y_sub = Tensor::from_vec(y_sub_u32, (1000,), &device).unwrap();

        let logits = model.forward(&x_sub).unwrap();
        assert_eq!(logits.dims(), &[1000, 10]);
        assert_eq!(model.parameters().len(), 2);

        let err = logits
            .argmax(D::Minus1)
            .unwrap()
            .ne(&y_sub)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            err < 0.04,
            "expected TwoLayerNN MNIST test[:1000] error < 0.04, got {err:.4}"
        );
    }

    #[test]
    fn submit_multi_layer_nn() {
        let device = Device::Cpu;
        // Exact submit_MultiLayerNN weights & inputs from hw3_tests.py.
        let w0_rows = vec![
            vec![1.0_f64, -1.0],
            vec![0.5, 0.25],
            vec![-0.75, 0.5],
        ];
        let w1_rows = vec![
            vec![1.0_f64, -0.5, 0.25],
            vec![-1.0, 0.5, 1.0],
        ];
        let w2_rows = vec![vec![2.0_f64, -1.5]];
        let x_rows = vec![
            vec![1.0_f64, 2.0],
            vec![-1.0, 0.5],
            vec![0.25, -0.75],
        ];
        let neg_x_rows: Vec<Vec<f64>> = x_rows
            .iter()
            .map(|r| r.iter().map(|&v| -v).collect())
            .collect();

        let model = MultiLayerNN::new(2, 1, &[3, 2], &device).unwrap();
        assert_eq!(model.linears.len(), 3);
        assert_eq!(model.parameters().len(), 3);

        model.linears[0].weight.set(&to_tensor_2d(&w0_rows, &device)).unwrap();
        model.linears[1].weight.set(&to_tensor_2d(&w1_rows, &device)).unwrap();
        model.linears[2].weight.set(&to_tensor_2d(&w2_rows, &device)).unwrap();

        let forward_ref = |xr: &[Vec<f64>]| {
            let h1 = relu_2d(&matmul_wt(xr, &w0_rows));
            let h2 = relu_2d(&matmul_wt(&h1, &w1_rows));
            matmul_wt(&h2, &w2_rows)
        };

        let x = to_tensor_2d(&x_rows, &device);
        let expected_2d = to_tensor_2d(&forward_ref(&x_rows), &device);
        assert_allclose(&model.forward(&x).unwrap(), &expected_2d, 1e-5, 1e-5);

        let x3d = Tensor::stack(&[&x, &x.neg().unwrap()], 0).unwrap();
        let expected_3d = Tensor::stack(
            &[&expected_2d, &to_tensor_2d(&forward_ref(&neg_x_rows), &device)],
            0,
        )
        .unwrap();
        assert_allclose(&model.forward(&x3d).unwrap(), &expected_3d, 1e-5, 1e-5);

        // Edge case: empty hidden_dims => single linear layer with no ReLU (negative outputs preserved).
        let single = MultiLayerNN::new(2, 2, &[], &device).unwrap();
        assert_eq!(single.linears.len(), 1);
        assert_eq!(single.parameters().len(), 1);
        let neg_eye = Tensor::new(&[[-1.0_f32, 0.0], [0.0, -1.0]], &device).unwrap();
        single.linears[0].weight.set(&neg_eye).unwrap();
        let pos_in = Tensor::new(&[[2.0_f32, 3.0]], &device).unwrap();
        assert_allclose(
            &single.forward(&pos_in).unwrap(),
            &Tensor::new(&[[-2.0_f32, -3.0]], &device).unwrap(),
            1e-5,
            1e-5,
        );

        // Deep 5-layer network gradient flow check across all layers.
        let deep = MultiLayerNN::new(5, 3, &[8, 6, 5, 4], &device).unwrap();
        assert_eq!(deep.linears.len(), 5);
        assert_eq!(deep.parameters().len(), 5);
        let x_deep = seeded_randn(&[7, 5], 99, &device).unwrap();
        let out_deep = deep.forward(&x_deep).unwrap();
        assert_eq!(out_deep.dims(), &[7, 3]);
        let grads = out_deep.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        for (i, p) in deep.parameters().iter().enumerate() {
            assert!(
                grads.get(p.as_tensor()).is_some(),
                "layer {i} weight did not receive a gradient"
            );
        }
    }
}
