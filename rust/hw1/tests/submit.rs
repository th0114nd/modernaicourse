//! Grader-equivalent ("submit") tests for Homework 1.
//!
//! Ports the exact deterministic `submit_*` inputs from `homework/hw1/hw1_tests.py`
//! with verified ground-truth assertions, plus additional edge cases.
//! Run with: `cargo test -p hw1 --test submit`

mod submit {
    use candle_core::{Device, Tensor};
    use common::{assert_allclose, linspace, mnist, seeded_randn};
    use hw1::{
        batch_matmul, block_matmul, classify_zero_one, matmul_1, matmul_2, matmul_3,
        matrix_vector_product_1, matrix_vector_product_2, vector_add, vector_inner_product,
        vector_matrix_product_2,
    };

    const RTOL: f32 = 1e-4;
    const ATOL: f32 = 1e-4;

    #[test]
    fn submit_classify_zero_one() {
        let data = mnist::load_train_zero_one(&Device::Cpu).expect("load MNIST 0/1");

        // Exact mugrade submit slice: images 100..120
        let mut correct_20 = 0usize;
        for i in 100..120 {
            let pred = classify_zero_one(&data.image(i).unwrap()).unwrap();
            assert!(pred == 0 || pred == 1, "prediction must be 0 or 1, got {pred}");
            if pred == data.labels[i] {
                correct_20 += 1;
            }
        }
        assert!(
            correct_20 >= 18,
            "expected at least 18/20 (90%) on submit slice 100..120, got {correct_20}/20"
        );

        // Extra held-out check on images 120..320 to catch overfitting to the first 100
        let mut correct_200 = 0usize;
        for i in 120..320 {
            let pred = classify_zero_one(&data.image(i).unwrap()).unwrap();
            if pred == data.labels[i] {
                correct_200 += 1;
            }
        }
        let acc = correct_200 as f64 / 200.0;
        assert!(
            acc > 0.90,
            "expected > 90% accuracy on held-out slice 120..320, got {:.2}%",
            acc * 100.0
        );
    }

    #[test]
    fn submit_vector_add() {
        // Exact mugrade submit inputs
        let a = linspace(0.0, 1.0, 7, &Device::Cpu).unwrap();
        let b = linspace(2.0, 3.0, 7, &Device::Cpu).unwrap();
        let z = vector_add(&a, &b).unwrap();
        assert_allclose(&z, &(&a + &b).unwrap(), RTOL, ATOL);
        assert!(vector_add(&a, &seeded_randn(&[5], 1, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: length 1, negative cancellation, non-1D rejection
        let s1 = Tensor::new(&[3.5f32], &Device::Cpu).unwrap();
        let s2 = Tensor::new(&[-3.5f32], &Device::Cpu).unwrap();
        assert_allclose(
            &vector_add(&s1, &s2).unwrap(),
            &Tensor::new(&[0.0f32], &Device::Cpu).unwrap(),
            RTOL,
            ATOL,
        );
        let m2d = seeded_randn(&[2, 3], 2, &Device::Cpu).unwrap();
        assert!(vector_add(&m2d, &m2d).is_err());
    }

    #[test]
    fn submit_vector_inner_product() {
        // Exact mugrade submit inputs: sum_{k=0}^6 (k/6)*(2 + k/6) = 343/36
        let a = linspace(0.0, 1.0, 7, &Device::Cpu).unwrap();
        let b = linspace(2.0, 3.0, 7, &Device::Cpu).unwrap();
        let z = vector_inner_product(&a, &b).unwrap();
        let expected = 343.0f32 / 36.0;
        assert!(
            (z - expected).abs() <= ATOL + RTOL * expected.abs(),
            "inner product mismatch: actual={z}, expected={expected}"
        );
        assert!(vector_inner_product(&a, &seeded_randn(&[5], 3, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: orthogonal vectors, length 1, non-1D rejection
        let e1 = Tensor::new(&[1.0f32, 0.0, -2.0], &Device::Cpu).unwrap();
        let e2 = Tensor::new(&[2.0f32, 9.0, 1.0], &Device::Cpu).unwrap();
        assert!(vector_inner_product(&e1, &e2).unwrap().abs() < ATOL);
        let m2d = seeded_randn(&[2, 2], 4, &Device::Cpu).unwrap();
        assert!(vector_inner_product(&m2d, &m2d).is_err());
    }

    #[test]
    fn submit_matrix_vector_product_1() {
        let a = linspace(0.0, 1.0, 8 * 7, &Device::Cpu)
            .unwrap()
            .reshape((8, 7))
            .unwrap();
        let b = linspace(2.0, 3.0, 7, &Device::Cpu).unwrap();
        let z = matrix_vector_product_1(&a, &b).unwrap();
        let expected = a.matmul(&b.unsqueeze(1).unwrap()).unwrap().squeeze(1).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);
        assert!(matrix_vector_product_1(&a, &seeded_randn(&[5], 5, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: 1x1, rank mismatch
        let a1 = Tensor::new(&[[2.5f32]], &Device::Cpu).unwrap();
        let x1 = Tensor::new(&[-4.0f32], &Device::Cpu).unwrap();
        assert_allclose(
            &matrix_vector_product_1(&a1, &x1).unwrap(),
            &Tensor::new(&[-10.0f32], &Device::Cpu).unwrap(),
            RTOL,
            ATOL,
        );
        assert!(matrix_vector_product_1(&b, &b).is_err());
        assert!(matrix_vector_product_1(&a, &a).is_err());
    }

    #[test]
    fn submit_matrix_vector_product_2() {
        let a = linspace(0.0, 1.0, 8 * 7, &Device::Cpu)
            .unwrap()
            .reshape((8, 7))
            .unwrap();
        let b = linspace(2.0, 3.0, 7, &Device::Cpu).unwrap();
        let z = matrix_vector_product_2(&a, &b).unwrap();
        let expected = a.matmul(&b.unsqueeze(1).unwrap()).unwrap().squeeze(1).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);
        assert!(matrix_vector_product_2(&a, &seeded_randn(&[5], 6, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: 1x1, rank mismatch
        let a1 = Tensor::new(&[[2.5f32]], &Device::Cpu).unwrap();
        let x1 = Tensor::new(&[-4.0f32], &Device::Cpu).unwrap();
        assert_allclose(
            &matrix_vector_product_2(&a1, &x1).unwrap(),
            &Tensor::new(&[-10.0f32], &Device::Cpu).unwrap(),
            RTOL,
            ATOL,
        );
        assert!(matrix_vector_product_2(&b, &b).is_err());
    }

    #[test]
    fn submit_vector_matrix_product_2() {
        let a = linspace(0.0, 1.0, 7 * 8, &Device::Cpu)
            .unwrap()
            .reshape((7, 8))
            .unwrap();
        let b = linspace(2.0, 3.0, 7, &Device::Cpu).unwrap();
        let z = vector_matrix_product_2(&b, &a).unwrap();
        let expected = b.unsqueeze(0).unwrap().matmul(&a).unwrap().squeeze(0).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);
        assert!(vector_matrix_product_2(&seeded_randn(&[5], 7, &Device::Cpu).unwrap(), &a).is_err());

        // Extra edge cases: 1x1, rank mismatch
        let a1 = Tensor::new(&[[2.5f32]], &Device::Cpu).unwrap();
        let x1 = Tensor::new(&[-4.0f32], &Device::Cpu).unwrap();
        assert_allclose(
            &vector_matrix_product_2(&x1, &a1).unwrap(),
            &Tensor::new(&[-10.0f32], &Device::Cpu).unwrap(),
            RTOL,
            ATOL,
        );
        assert!(vector_matrix_product_2(&a, &a).is_err());
    }

    #[test]
    fn submit_matmul_1() {
        let a = linspace(0.0, 1.0, 3 * 8, &Device::Cpu)
            .unwrap()
            .reshape((3, 8))
            .unwrap();
        let b = linspace(2.0, 3.0, 8 * 5, &Device::Cpu)
            .unwrap()
            .reshape((8, 5))
            .unwrap();
        let z = matmul_1(&a, &b).unwrap();
        assert_allclose(&z, &a.matmul(&b).unwrap(), RTOL, ATOL);
        assert!(matmul_1(&a, &seeded_randn(&[4, 5], 8, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: identity matrix, 1x1, 1D rejection
        let eye = Tensor::eye(3, candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_allclose(&matmul_1(&eye, &a).unwrap(), &a, RTOL, ATOL);
        assert!(matmul_1(&linspace(0.0, 1.0, 3, &Device::Cpu).unwrap(), &a).is_err());
    }

    #[test]
    fn submit_matmul_2() {
        let a = linspace(0.0, 1.0, 3 * 8, &Device::Cpu)
            .unwrap()
            .reshape((3, 8))
            .unwrap();
        let b = linspace(2.0, 3.0, 8 * 5, &Device::Cpu)
            .unwrap()
            .reshape((8, 5))
            .unwrap();
        let z = matmul_2(&a, &b).unwrap();
        assert_allclose(&z, &a.matmul(&b).unwrap(), RTOL, ATOL);
        assert!(matmul_2(&a, &seeded_randn(&[4, 5], 9, &Device::Cpu).unwrap()).is_err());

        let eye = Tensor::eye(3, candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_allclose(&matmul_2(&eye, &a).unwrap(), &a, RTOL, ATOL);
    }

    #[test]
    fn submit_matmul_3() {
        let a = linspace(0.0, 1.0, 3 * 8, &Device::Cpu)
            .unwrap()
            .reshape((3, 8))
            .unwrap();
        let b = linspace(2.0, 3.0, 8 * 5, &Device::Cpu)
            .unwrap()
            .reshape((8, 5))
            .unwrap();
        let z = matmul_3(&a, &b).unwrap();
        assert_allclose(&z, &a.matmul(&b).unwrap(), RTOL, ATOL);
        assert!(matmul_3(&a, &seeded_randn(&[4, 5], 10, &Device::Cpu).unwrap()).is_err());

        let eye = Tensor::eye(3, candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_allclose(&matmul_3(&eye, &a).unwrap(), &a, RTOL, ATOL);
    }

    #[test]
    fn submit_batch_matmul() {
        // Exact mugrade submit inputs
        let a = linspace(0.0, 1.0, 4 * 2 * 3 * 8, &Device::Cpu)
            .unwrap()
            .reshape((4, 2, 3, 8))
            .unwrap();
        let b = linspace(2.0, 3.0, 4 * 2 * 8 * 5, &Device::Cpu)
            .unwrap()
            .reshape((4, 2, 8, 5))
            .unwrap();
        let z = batch_matmul(&a, &b).unwrap();
        assert_allclose(&z, &a.matmul(&b).unwrap(), RTOL, ATOL);

        assert!(batch_matmul(&a, &seeded_randn(&[4, 2, 3, 8], 11, &Device::Cpu).unwrap()).is_err());
        assert!(batch_matmul(&a, &seeded_randn(&[2, 2, 3, 8], 12, &Device::Cpu).unwrap()).is_err());
        assert!(batch_matmul(&a, &seeded_randn(&[2, 8, 5], 13, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: 2D (0 batch dims), 3D (1 batch dim), 1D rejection
        let a2 = seeded_randn(&[3, 4], 14, &Device::Cpu).unwrap();
        let b2 = seeded_randn(&[4, 2], 15, &Device::Cpu).unwrap();
        assert_allclose(&batch_matmul(&a2, &b2).unwrap(), &a2.matmul(&b2).unwrap(), RTOL, ATOL);

        let a3 = seeded_randn(&[3, 2, 4], 16, &Device::Cpu).unwrap();
        let b3 = seeded_randn(&[3, 4, 5], 17, &Device::Cpu).unwrap();
        assert_allclose(&batch_matmul(&a3, &b3).unwrap(), &a3.matmul(&b3).unwrap(), RTOL, ATOL);

        let v1 = seeded_randn(&[4], 18, &Device::Cpu).unwrap();
        assert!(batch_matmul(&v1, &v1).is_err());
    }

    #[test]
    fn submit_block_matmul() {
        // Exact mugrade submit inputs
        let a = linspace(0.0, 1.0, 8 * 16, &Device::Cpu)
            .unwrap()
            .reshape((8, 16))
            .unwrap();
        let b = linspace(2.0, 3.0, 16 * 4, &Device::Cpu)
            .unwrap()
            .reshape((16, 4))
            .unwrap();
        let z = block_matmul(&a, &b).unwrap();
        assert_allclose(&z, &a.matmul(&b).unwrap(), RTOL, ATOL);

        assert!(block_matmul(&a, &seeded_randn(&[8, 4], 19, &Device::Cpu).unwrap()).is_err());
        assert!(block_matmul(&a, &seeded_randn(&[16, 6], 20, &Device::Cpu).unwrap()).is_err());

        // Extra edge cases: single 4x4 block, non-multiple-of-4 first matrix, 3D rejection
        let a4 = seeded_randn(&[4, 4], 21, &Device::Cpu).unwrap();
        let b4 = seeded_randn(&[4, 4], 22, &Device::Cpu).unwrap();
        assert_allclose(&block_matmul(&a4, &b4).unwrap(), &a4.matmul(&b4).unwrap(), RTOL, ATOL);

        let bad_a = seeded_randn(&[6, 8], 23, &Device::Cpu).unwrap();
        let ok_b = seeded_randn(&[8, 4], 24, &Device::Cpu).unwrap();
        assert!(block_matmul(&bad_a, &ok_b).is_err());

        let t3 = seeded_randn(&[2, 4, 4], 25, &Device::Cpu).unwrap();
        assert!(block_matmul(&t3, &t3).is_err());
    }
}
