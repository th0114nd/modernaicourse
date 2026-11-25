//! Grader-equivalent ("submit") tests for Homework 0.
//!
//! Ports the exact test inputs from `homework/hw0/hw0_tests.py` (`submit_*` functions)
//! with verified ground-truth assertions, plus additional edge cases.
//! Run with: `cargo test -p hw0 --test submit`

mod submit {
    use hw0::{Polynomial, add, poly_add, poly_derivative, poly_mul, primes};

    const TOL: f64 = 1e-12;

    fn assert_poly_close(actual: &Polynomial, expected: &Polynomial) {
        assert!(
            actual.approx_eq(expected, TOL),
            "polynomial mismatch:\n  actual:   {:?} ({})\n  expected: {:?} ({})",
            actual.coefficients,
            actual,
            expected.coefficients,
            expected,
        );
    }

    #[test]
    fn submit_add() {
        // Exact mugrade submit cases
        assert_eq!(add(-1i64, 5i64), 4i64);
        assert!((add(1.0f64, 2.5f64) - 3.5f64).abs() < TOL);
        let int_sum: i64 = add(1i64, 3i64);
        assert_eq!(int_sum, 4i64);

        // Extra edge cases
        assert_eq!(add(0i32, 0i32), 0i32);
        assert_eq!(add(-42i64, 42i64), 0i64);
        assert!((add(-3.25f64, -1.75f64) - (-5.0f64)).abs() < TOL);
    }

    #[test]
    fn submit_primes() {
        // Exact mugrade submit cases
        assert_eq!(
            primes(100),
            vec![
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97,
            ]
        );
        let p10k = primes(10_000);
        assert_eq!(p10k.len(), 1229);
        assert_eq!(p10k.last().copied(), Some(9973));

        // Extra edge cases: small bounds, prime exclusive upper bound, prime-square bounds
        assert_eq!(primes(0), Vec::<usize>::new());
        assert_eq!(primes(1), Vec::<usize>::new());
        assert_eq!(primes(2), Vec::<usize>::new());
        assert_eq!(primes(3), vec![2]);
        assert_eq!(primes(7), vec![2, 3, 5]);
        assert_eq!(primes(9), vec![2, 3, 5, 7]);
        assert_eq!(primes(25), vec![2, 3, 5, 7, 11, 13, 17, 19, 23]);
    }

    #[test]
    fn submit_poly_add() {
        // Exact mugrade submit cases
        let p1 = Polynomial::new(vec![2.0, 4.0, -1.0, 3.0]);
        let p2 = Polynomial::new(vec![1.0, 8.0, -2.0]);
        let p3 = Polynomial::new(vec![-5.0, 2.0, 0.0, 4.0]);
        let p4 = Polynomial::new(vec![0.9, 0.4, -1.7, 2.5]);

        assert_poly_close(
            &poly_add(&p1, &p2),
            &Polynomial::new(vec![3.0, 12.0, -3.0, 3.0]),
        );
        assert_poly_close(
            &poly_add(&p1, &p3),
            &Polynomial::new(vec![-3.0, 6.0, -1.0, 7.0]),
        );
        assert_poly_close(
            &poly_add(&p1, &Polynomial::new(vec![0.0])),
            &Polynomial::new(vec![2.0, 4.0, -1.0, 3.0]),
        );
        assert_poly_close(
            &poly_add(&p2, &p4),
            &Polynomial::new(vec![1.9, 8.4, -3.7, 2.5]),
        );

        // Extra edge cases: commutativity, full cancellation, partial leading cancellation
        assert_poly_close(&poly_add(&p2, &p1), &poly_add(&p1, &p2));
        let neg_p1 = Polynomial::new(vec![-2.0, -4.0, 1.0, -3.0]);
        assert_poly_close(&poly_add(&p1, &neg_p1), &Polynomial::new(vec![0.0]));
        assert_poly_close(
            &poly_add(
                &Polynomial::new(vec![1.0, 2.0, 3.0]),
                &Polynomial::new(vec![4.0, -2.0, -3.0]),
            ),
            &Polynomial::new(vec![5.0]),
        );
    }

    #[test]
    fn submit_poly_mul() {
        // Exact mugrade submit cases
        let p1 = Polynomial::new(vec![2.0, 4.0, -1.0, 3.0]);
        let p2 = Polynomial::new(vec![1.0, 8.0, -2.0]);
        let p3 = Polynomial::new(vec![-5.0, 2.0, 0.0, 4.0]);
        let p4 = Polynomial::new(vec![0.9, 0.4, -1.7, 2.5]);

        assert_poly_close(
            &poly_mul(&p1, &p2),
            &Polynomial::new(vec![2.0, 20.0, 27.0, -13.0, 26.0, -6.0]),
        );
        assert_poly_close(
            &poly_mul(&p1, &p3),
            &Polynomial::new(vec![-10.0, -16.0, 13.0, -9.0, 22.0, -4.0, 12.0]),
        );
        assert_poly_close(
            &poly_mul(&p1, &Polynomial::new(vec![0.0])),
            &Polynomial::new(vec![0.0]),
        );
        assert_poly_close(
            &poly_mul(&p2, &p4),
            &Polynomial::new(vec![0.9, 7.6, -0.3, -11.9, 23.4, -5.0]),
        );

        // Extra edge cases: commutativity, zero on left, multiplicative identity, constants
        assert_poly_close(&poly_mul(&p2, &p1), &poly_mul(&p1, &p2));
        assert_poly_close(
            &poly_mul(&Polynomial::new(vec![0.0]), &p4),
            &Polynomial::new(vec![0.0]),
        );
        assert_poly_close(&poly_mul(&Polynomial::new(vec![1.0]), &p4), &p4);
        assert_poly_close(
            &poly_mul(&Polynomial::new(vec![3.0]), &Polynomial::new(vec![-4.0])),
            &Polynomial::new(vec![-12.0]),
        );
    }

    #[test]
    fn submit_poly_derivative() {
        // Exact mugrade submit cases
        let p1 = Polynomial::new(vec![2.0, 4.0, -1.0, 3.0]);
        let p2 = Polynomial::new(vec![1.0, 8.0, -2.0]);
        let p3 = Polynomial::new(vec![0.9, 0.4, -1.7, 2.5]);

        assert_poly_close(&poly_derivative(&p1), &Polynomial::new(vec![4.0, -2.0, 9.0]));
        assert_poly_close(&poly_derivative(&p2), &Polynomial::new(vec![8.0, -4.0]));
        assert_poly_close(
            &poly_derivative(&p3),
            &Polynomial::new(vec![0.4, -3.4, 7.5]),
        );
        assert_poly_close(
            &poly_derivative(&poly_derivative(&p1)),
            &Polynomial::new(vec![-2.0, 18.0]),
        );

        // Extra edge cases: nonzero constant, repeated differentiation to zero, sparse higher-degree
        assert_poly_close(
            &poly_derivative(&Polynomial::new(vec![42.0])),
            &Polynomial::new(vec![0.0]),
        );
        let d3 = poly_derivative(&poly_derivative(&poly_derivative(&p1)));
        assert_poly_close(&d3, &Polynomial::new(vec![18.0]));
        assert_poly_close(&poly_derivative(&d3), &Polynomial::new(vec![0.0]));
        assert_poly_close(
            &poly_derivative(&Polynomial::new(vec![0.0, 0.0, 0.0, 0.0, 5.0])),
            &Polynomial::new(vec![0.0, 0.0, 0.0, 20.0]),
        );
    }
}
