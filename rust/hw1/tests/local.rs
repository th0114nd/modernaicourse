//! Local sanity-check tests for Homework 1.
//!
//! Ported directly from `homework/hw1/hw1_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw1 --test local`

mod local {
    use candle_core::Device;
    use common::{assert_allclose, mnist, seeded_randn};
    use hw1::{
        batch_matmul, block_matmul, classify_zero_one, matmul_1, matmul_2, matmul_3,
        matrix_vector_product_1, matrix_vector_product_2, vector_add, vector_inner_product,
        vector_matrix_product_2,
    };

    const RTOL: f32 = 1e-4;
    const ATOL: f32 = 1e-4;

    #[test]
    fn test_classify_zero_one() {
        let data = mnist::load_train_zero_one(&Device::Cpu).expect("load MNIST 0/1");
        assert_eq!(
            classify_zero_one(&data.image(0).unwrap()).unwrap(),
            0,
            "first 0/1 MNIST image should be classified as 0"
        );
        assert_eq!(
            classify_zero_one(&data.image(1).unwrap()).unwrap(),
            1,
            "second 0/1 MNIST image should be classified as 1"
        );

        let n = 100;
        let mut correct = 0usize;
        for i in 0..n {
            let pred = classify_zero_one(&data.image(i).unwrap()).unwrap();
            if pred == data.labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / n as f64;
        assert!(
            accuracy > 0.9,
            "expected accuracy > 0.90 on first 100 images, got {:.2}% ({correct}/{n})",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_vector_add() {
        let a = seeded_randn(&[5], 101, &Device::Cpu).unwrap();
        let b = seeded_randn(&[5], 102, &Device::Cpu).unwrap();
        let z = vector_add(&a, &b).unwrap();
        let expected = (&a + &b).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[6], 103, &Device::Cpu).unwrap();
        assert!(vector_add(&a, &bad).is_err());
    }

    #[test]
    fn test_vector_inner_product() {
        let a = seeded_randn(&[5], 201, &Device::Cpu).unwrap();
        let b = seeded_randn(&[5], 202, &Device::Cpu).unwrap();
        let z = vector_inner_product(&a, &b).unwrap();
        let expected = (&a * &b)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (z - expected).abs() <= ATOL + RTOL * expected.abs(),
            "inner product mismatch: actual={z}, expected={expected}"
        );

        let bad = seeded_randn(&[6], 203, &Device::Cpu).unwrap();
        assert!(vector_inner_product(&a, &bad).is_err());
    }

    #[test]
    fn test_matrix_vector_product_1() {
        let a = seeded_randn(&[5, 4], 301, &Device::Cpu).unwrap();
        let b = seeded_randn(&[4], 302, &Device::Cpu).unwrap();
        let z = matrix_vector_product_1(&a, &b).unwrap();
        let expected = a
            .matmul(&b.unsqueeze(1).unwrap())
            .unwrap()
            .squeeze(1)
            .unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[6], 303, &Device::Cpu).unwrap();
        assert!(matrix_vector_product_1(&a, &bad).is_err());
    }

    #[test]
    fn test_matrix_vector_product_2() {
        let a = seeded_randn(&[5, 4], 401, &Device::Cpu).unwrap();
        let b = seeded_randn(&[4], 402, &Device::Cpu).unwrap();
        let z = matrix_vector_product_2(&a, &b).unwrap();
        let expected = a
            .matmul(&b.unsqueeze(1).unwrap())
            .unwrap()
            .squeeze(1)
            .unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[6], 403, &Device::Cpu).unwrap();
        assert!(matrix_vector_product_2(&a, &bad).is_err());
    }

    #[test]
    fn test_vector_matrix_product_2() {
        let a = seeded_randn(&[4, 5], 501, &Device::Cpu).unwrap();
        let b = seeded_randn(&[4], 502, &Device::Cpu).unwrap();
        let z = vector_matrix_product_2(&b, &a).unwrap();
        let expected = b
            .unsqueeze(0)
            .unwrap()
            .matmul(&a)
            .unwrap()
            .squeeze(0)
            .unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[6], 503, &Device::Cpu).unwrap();
        assert!(vector_matrix_product_2(&bad, &a).is_err());
    }

    #[test]
    fn test_matmul_1() {
        let a = seeded_randn(&[4, 5], 601, &Device::Cpu).unwrap();
        let b = seeded_randn(&[5, 6], 602, &Device::Cpu).unwrap();
        let z = matmul_1(&a, &b).unwrap();
        let expected = a.matmul(&b).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[4, 5], 603, &Device::Cpu).unwrap();
        assert!(matmul_1(&a, &bad).is_err());
    }

    #[test]
    fn test_matmul_2() {
        let a = seeded_randn(&[4, 5], 701, &Device::Cpu).unwrap();
        let b = seeded_randn(&[5, 6], 702, &Device::Cpu).unwrap();
        let z = matmul_2(&a, &b).unwrap();
        let expected = a.matmul(&b).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[4, 5], 703, &Device::Cpu).unwrap();
        assert!(matmul_2(&a, &bad).is_err());
    }

    #[test]
    fn test_matmul_3() {
        let a = seeded_randn(&[4, 5], 801, &Device::Cpu).unwrap();
        let b = seeded_randn(&[5, 6], 802, &Device::Cpu).unwrap();
        let z = matmul_3(&a, &b).unwrap();
        let expected = a.matmul(&b).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        let bad = seeded_randn(&[4, 5], 803, &Device::Cpu).unwrap();
        assert!(matmul_3(&a, &bad).is_err());
    }

    #[test]
    fn test_batch_matmul() {
        let a = seeded_randn(&[2, 3, 4, 5], 901, &Device::Cpu).unwrap();
        let b = seeded_randn(&[2, 3, 5, 6], 902, &Device::Cpu).unwrap();
        let z = batch_matmul(&a, &b).unwrap();
        let expected = a.matmul(&b).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        assert!(
            batch_matmul(&a, &seeded_randn(&[2, 3, 4, 5], 903, &Device::Cpu).unwrap()).is_err()
        );
        assert!(
            batch_matmul(&a, &seeded_randn(&[4, 3, 5, 6], 904, &Device::Cpu).unwrap()).is_err()
        );
        assert!(batch_matmul(&a, &seeded_randn(&[3, 5, 6], 905, &Device::Cpu).unwrap()).is_err());
    }

    #[test]
    fn test_block_matmul() {
        let a = seeded_randn(&[16, 12], 1001, &Device::Cpu).unwrap();
        let b = seeded_randn(&[12, 8], 1002, &Device::Cpu).unwrap();
        let z = block_matmul(&a, &b).unwrap();
        let expected = a.matmul(&b).unwrap();
        assert_allclose(&z, &expected, RTOL, ATOL);

        assert!(block_matmul(&a, &seeded_randn(&[16, 12], 1003, &Device::Cpu).unwrap()).is_err());
        assert!(block_matmul(&a, &seeded_randn(&[12, 7], 1004, &Device::Cpu).unwrap()).is_err());
    }
}
