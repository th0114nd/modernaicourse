//! Local sanity-check tests for Homework 3.
//!
//! Ported directly from `homework/hw3/hw3_tests.py` (`test_*` functions).
//! Run with: `cargo test --release -p hw3 --test local`

mod local {
    use candle_core::{D, DType, Device, Result, Tensor, Var, backprop::GradStore};
    use common::{assert_allclose, mnist, seeded_randn};
    use hw3::{
        CrossEntropyLoss, DataLoader, Linear, Loss, Module, MultiLayerNN, Optimizer, SGD,
        TwoLayerNN, epoch, eval_linear_model, eval_two_layer_nn,
    };

    /// Minimal reference linear layer used so `test_sgd` and `test_epoch` stay
    /// decoupled from the student's `Linear` implementation.
    struct RefLinear {
        weight: Var,
    }

    impl RefLinear {
        fn from_tensor(w: &Tensor) -> Result<Self> {
            Ok(Self {
                weight: Var::from_tensor(w)?,
            })
        }
    }

    impl Module for RefLinear {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            x.broadcast_matmul(&self.weight.as_tensor().t()?)
        }

        fn parameters(&self) -> Vec<Var> {
            vec![self.weight.clone()]
        }
    }

    /// Reference 2-layer ReLU MLP for `test_sgd`.
    struct RefTwoLayer {
        l1: RefLinear,
        l2: RefLinear,
    }

    impl Module for RefTwoLayer {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let h = self.l1.forward(x)?.relu()?;
            self.l2.forward(&h)
        }

        fn parameters(&self) -> Vec<Var> {
            vec![self.l1.weight.clone(), self.l2.weight.clone()]
        }
    }

    /// Reference cross-entropy loss for decoupling `test_sgd` and `test_epoch`.
    struct RefCrossEntropy;

    impl Loss for RefCrossEntropy {
        fn forward(&self, logits: &Tensor, y: &Tensor) -> Result<Tensor> {
            let lse = logits.log_sum_exp(1)?;
            let target = logits.gather(&y.unsqueeze(1)?, 1)?.squeeze(1)?;
            (lse - target)?.mean_all()
        }
    }

    /// Reference SGD optimizer for decoupling `test_epoch`.
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

    fn ref_cross_entropy_scalar(logits: &Tensor, y: &Tensor) -> f32 {
        let rows = logits.to_vec2::<f32>().unwrap();
        let targets = y.to_vec1::<u32>().unwrap();
        let mut total = 0.0_f64;
        for (row, &t) in rows.iter().zip(targets.iter()) {
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = max_v + row.iter().map(|&v| (v as f64 - max_v).exp()).sum::<f64>().ln();
            total += lse - row[t as usize] as f64;
        }
        (total / rows.len() as f64) as f32
    }

    fn make_targets(n: usize, num_classes: u32, seed: u64, device: &Device) -> Tensor {
        let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
        let vals: Vec<u32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as u32) % num_classes
            })
            .collect();
        Tensor::from_vec(vals, (n,), device).unwrap()
    }

    #[test]
    fn test_linear() {
        let device = Device::Cpu;
        let layer = Linear::new(10, 20, &device).unwrap();
        assert_eq!(layer.weight.dims(), &[20, 10]);
        assert_eq!(layer.parameters().len(), 1);

        let w_t = layer.weight.as_tensor().t().unwrap();

        let x2d = seeded_randn(&[50, 10], 0, &device).unwrap();
        let out2d = layer.forward(&x2d).unwrap();
        assert_eq!(out2d.dims(), &[50, 20]);
        let ref2d = x2d.matmul(&w_t).unwrap();
        assert_allclose(&out2d, &ref2d, 1e-5, 1e-5);

        let x3d = seeded_randn(&[7, 9, 10], 1, &device).unwrap();
        let out3d = layer.forward(&x3d).unwrap();
        assert_eq!(out3d.dims(), &[7, 9, 20]);
        let ref3d = x3d
            .reshape((63, 10))
            .unwrap()
            .matmul(&w_t)
            .unwrap()
            .reshape((7, 9, 20))
            .unwrap();
        assert_allclose(&out3d, &ref3d, 1e-5, 1e-5);

        // Kaiming/He init std check: sqrt(2 / in_dim)
        let big = Linear::new(100, 1000, &device).unwrap();
        let vals = big.weight.as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let n = vals.len() as f64;
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        let std = var.sqrt();
        let expected_std = (2.0_f64 / 100.0).sqrt();
        assert!(
            (std - expected_std).abs() < 5e-3,
            "expected weight std near {expected_std:.5}, got {std:.5}"
        );
    }

    #[test]
    fn test_cross_entropy_loss() {
        let device = Device::Cpu;
        let loss_fn = CrossEntropyLoss::new();

        let logits = seeded_randn(&[100, 50], 1, &device).unwrap();
        let y = make_targets(100, 50, 1, &device);
        let mine = loss_fn.forward(&logits, &y).unwrap();
        assert_eq!(mine.rank(), 0, "expected scalar (0D) loss tensor");
        let expected = Tensor::new(ref_cross_entropy_scalar(&logits, &y), &device).unwrap();
        assert_allclose(&mine, &expected, 1e-5, 1e-5);

        // Large-logit numerical stability check.
        let logits_big = Tensor::new(
            &[[1000.0_f32, 1001.0, 999.5], [1.0, -2.0, 0.5]],
            &device,
        )
        .unwrap();
        let y_big = Tensor::new(&[1_u32, 2], &device).unwrap();
        let mine_big = loss_fn.forward(&logits_big, &y_big).unwrap();
        let expected_big =
            Tensor::new(ref_cross_entropy_scalar(&logits_big, &y_big), &device).unwrap();
        assert_allclose(&mine_big, &expected_big, 1e-5, 1e-5);
    }

    #[test]
    fn test_sgd() {
        let device = Device::Cpu;
        let w1_init = seeded_randn(&[4, 6], 20, &device).unwrap();
        let w2_init = seeded_randn(&[3, 4], 21, &device).unwrap();

        let model = RefTwoLayer {
            l1: RefLinear::from_tensor(&w1_init).unwrap(),
            l2: RefLinear::from_tensor(&w2_init).unwrap(),
        };
        let ref_model = RefTwoLayer {
            l1: RefLinear::from_tensor(&w1_init).unwrap(),
            l2: RefLinear::from_tensor(&w2_init).unwrap(),
        };

        let mut opt = SGD::new(model.parameters(), 0.05);
        let mut ref_opt = RefSgd {
            params: ref_model.parameters(),
            lr: 0.05,
        };
        let loss_fn = RefCrossEntropy;

        let x = seeded_randn(&[12, 6], 22, &device).unwrap();
        let y = make_targets(12, 3, 22, &device);

        for _ in 0..3 {
            opt.zero_grad();
            ref_opt.zero_grad();

            let loss = loss_fn.forward(&model.forward(&x).unwrap(), &y).unwrap();
            let ref_loss = loss_fn.forward(&ref_model.forward(&x).unwrap(), &y).unwrap();

            let grads = loss.backward().unwrap();
            let ref_grads = ref_loss.backward().unwrap();

            opt.step(&grads).unwrap();
            ref_opt.step(&ref_grads).unwrap();
        }

        for (p, p_ref) in model.parameters().iter().zip(ref_model.parameters().iter()) {
            assert_allclose(p.as_tensor(), p_ref.as_tensor(), 1e-5, 1e-6);
        }
    }

    #[test]
    fn test_dataloader() {
        let device = Device::Cpu;
        let x_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_vec(x_data, (12, 2), &device).unwrap();
        let y_data: Vec<u32> = (0..12).collect();
        let y = Tensor::from_vec(y_data, (12,), &device).unwrap();

        let loader = DataLoader::new(x.clone(), y.clone(), 5).unwrap();
        let batches: Vec<(Tensor, Tensor)> =
            (&loader).into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(batches.len(), 3);

        for (i, (xb, yb)) in batches.iter().enumerate() {
            let start = 5 * i;
            let len = 5.min(12 - start);
            assert_allclose(xb, &x.narrow(0, start, len).unwrap(), 0.0, 0.0);
            assert_eq!(
                yb.to_vec1::<u32>().unwrap(),
                y.narrow(0, start, len).unwrap().to_vec1::<u32>().unwrap()
            );
        }

        // Re-iterating starts fresh from batch 0.
        let batches2: Vec<(Tensor, Tensor)> =
            loader.iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(batches2.len(), 3);
        for ((xb1, yb1), (xb2, yb2)) in batches.iter().zip(batches2.iter()) {
            assert_allclose(xb1, xb2, 0.0, 0.0);
            assert_eq!(yb1.to_vec1::<u32>().unwrap(), yb2.to_vec1::<u32>().unwrap());
        }
    }

    #[test]
    fn test_epoch() {
        let device = Device::Cpu;
        let x = seeded_randn(&[11, 4], 30, &device).unwrap();
        let y = make_targets(11, 3, 30, &device);
        let loss_fn = RefCrossEntropy;

        // Evaluation pass (opt = None) must match full-batch loss/error and leave weights untouched.
        let w_init = seeded_randn(&[3, 4], 31, &device).unwrap();
        let model = RefLinear::from_tensor(&w_init).unwrap();
        let split4 = |x: &Tensor, y: &Tensor, bs: usize| -> Vec<Result<(Tensor, Tensor)>> {
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

        let (eval_loss, eval_err) = epoch(&model, split4(&x, &y, 4), &loss_fn, None).unwrap();

        let logits = model.forward(&x).unwrap();
        let ref_loss = loss_fn.forward(&logits, &y).unwrap().to_scalar::<f32>().unwrap() as f64;
        let ref_err = logits
            .argmax(D::Minus1)
            .unwrap()
            .ne(&y)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;

        assert!(
            (eval_loss - ref_loss).abs() < 1e-5,
            "eval_loss={eval_loss}, ref_loss={ref_loss}"
        );
        assert!(
            (eval_err - ref_err).abs() < 1e-5,
            "eval_err={eval_err}, ref_err={ref_err}"
        );
        assert_allclose(model.weight.as_tensor(), &w_init, 0.0, 0.0);

        // Training pass (opt = Some) must match reference minibatch updates and weighted metrics.
        let model = RefLinear::from_tensor(&w_init).unwrap();
        let ref_model = RefLinear::from_tensor(&w_init).unwrap();
        let mut opt = RefSgd {
            params: model.parameters(),
            lr: 0.1,
        };
        let mut ref_opt = RefSgd {
            params: ref_model.parameters(),
            lr: 0.1,
        };

        let (train_loss, train_err) =
            epoch(&model, split4(&x, &y, 3), &loss_fn, Some(&mut opt)).unwrap();

        let mut total_loss = 0.0_f64;
        let mut total_err = 0.0_f64;
        let mut total_n = 0usize;
        for batch in split4(&x, &y, 3) {
            let (xb, yb) = batch.unwrap();
            let b = yb.dim(0).unwrap();
            let logits = ref_model.forward(&xb).unwrap();
            let l = loss_fn.forward(&logits, &yb).unwrap();
            total_loss += l.to_scalar::<f32>().unwrap() as f64 * b as f64;
            total_err += logits
                .argmax(D::Minus1)
                .unwrap()
                .ne(&yb)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            total_n += b;
            let grads = l.backward().unwrap();
            ref_opt.step(&grads).unwrap();
        }
        let ref_train_loss = total_loss / total_n as f64;
        let ref_train_err = total_err / total_n as f64;

        assert!(
            (train_loss - ref_train_loss).abs() < 1e-5,
            "train_loss={train_loss}, ref={ref_train_loss}"
        );
        assert!(
            (train_err - ref_train_err).abs() < 1e-5,
            "train_err={train_err}, ref={ref_train_err}"
        );
        assert_allclose(model.weight.as_tensor(), ref_model.weight.as_tensor(), 1e-5, 1e-6);
    }

    #[test]
    fn test_eval_linear_model() {
        let device = Device::Cpu;
        let model = eval_linear_model().unwrap();

        let test = mnist::load_test(&device).unwrap();
        let x_test = test.images.reshape((test.len(), 784)).unwrap();
        let y_test_u32: Vec<u32> = test.labels.iter().map(|&l| l as u32).collect();
        let y_test = Tensor::from_vec(y_test_u32, (test.len(),), &device).unwrap();

        let logits = model.forward(&x_test).unwrap();
        assert_eq!(logits.dims(), &[10_000, 10]);
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
    fn test_two_layer_nn() {
        let device = Device::Cpu;
        let model = TwoLayerNN::new(8, 16, 5, &device).unwrap();
        assert_eq!(model.linear1.weight.dims(), &[16, 8]);
        assert_eq!(model.linear2.weight.dims(), &[5, 16]);
        assert_eq!(model.parameters().len(), 2);

        let w1_t = model.linear1.weight.as_tensor().t().unwrap();
        let w2_t = model.linear2.weight.as_tensor().t().unwrap();

        let x2d = seeded_randn(&[13, 8], 40, &device).unwrap();
        let ref2d = x2d.matmul(&w1_t).unwrap().relu().unwrap().matmul(&w2_t).unwrap();
        assert_allclose(&model.forward(&x2d).unwrap(), &ref2d, 1e-5, 1e-5);

        let x3d = seeded_randn(&[2, 7, 8], 41, &device).unwrap();
        let ref3d = x3d
            .reshape((14, 8))
            .unwrap()
            .matmul(&w1_t)
            .unwrap()
            .relu()
            .unwrap()
            .matmul(&w2_t)
            .unwrap()
            .reshape((2, 7, 5))
            .unwrap();
        assert_allclose(&model.forward(&x3d).unwrap(), &ref3d, 1e-5, 1e-5);
    }

    #[test]
    fn test_eval_two_layer_nn() {
        let device = Device::Cpu;
        let model = eval_two_layer_nn().unwrap();

        let test = mnist::load_test(&device).unwrap();
        let x_sub = test
            .images
            .reshape((test.len(), 784))
            .unwrap()
            .narrow(0, 0, 2000)
            .unwrap();
        let y_sub_u32: Vec<u32> = test.labels[..2000].iter().map(|&l| l as u32).collect();
        let y_sub = Tensor::from_vec(y_sub_u32, (2000,), &device).unwrap();

        let logits = model.forward(&x_sub).unwrap();
        assert_eq!(logits.dims(), &[2000, 10]);
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
            err < 0.03,
            "expected TwoLayerNN MNIST test[:2000] error < 0.03, got {err:.4}"
        );
    }

    #[test]
    fn test_multi_layer_nn() {
        let device = Device::Cpu;
        let hidden_dims = [7, 5, 4];
        let model = MultiLayerNN::new(6, 3, &hidden_dims, &device).unwrap();
        assert_eq!(model.linears.len(), hidden_dims.len() + 1);
        assert_eq!(model.parameters().len(), hidden_dims.len() + 1);

        let ref_forward = |x: &Tensor| -> Tensor {
            let mut h = x.clone();
            for (i, layer) in model.linears.iter().enumerate() {
                h = h.broadcast_matmul(&layer.weight.as_tensor().t().unwrap()).unwrap();
                if i + 1 < model.linears.len() {
                    h = h.relu().unwrap();
                }
            }
            h
        };

        let x2d = seeded_randn(&[9, 6], 50, &device).unwrap();
        assert_allclose(&model.forward(&x2d).unwrap(), &ref_forward(&x2d), 1e-5, 1e-5);

        let x3d = seeded_randn(&[2, 4, 6], 51, &device).unwrap();
        assert_allclose(&model.forward(&x3d).unwrap(), &ref_forward(&x3d), 1e-5, 1e-5);
    }
}
