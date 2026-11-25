//! Local sanity-check tests for Homework 0.
//!
//! Ported directly from `homework/hw0/hw0_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw0 --test local`

mod local {
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
    fn test_add() {
        assert_eq!(add(5i64, 6i64), 11i64);
        assert!((add(2.1f64, 2.3f64) - 4.4f64).abs() < TOL);
        let f: f64 = add(4.0, 2.1);
        assert!((f - 6.1).abs() < TOL);
    }

    #[test]
    fn test_primes() {
        let p = primes(10);
        assert_eq!(p, vec![2, 3, 5, 7]);
    }

    #[test]
    fn test_poly_add() {
        let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
        let p2 = Polynomial::new(vec![0.0, 2.0]);
        let p3 = Polynomial::new(vec![-1.0, 6.0, 7.0, -5.0]);
        let p4 = Polynomial::new(vec![0.3, 0.4, 1.6, 1.9]);

        assert_poly_close(&poly_add(&p1, &p2), &Polynomial::new(vec![1.0, 7.0, 0.0, 5.0]));
        assert_poly_close(&poly_add(&p1, &p3), &Polynomial::new(vec![0.0, 11.0, 7.0]));
        assert_poly_close(&poly_add(&p1, &Polynomial::new(vec![0.0])), &p1);
        assert_poly_close(&poly_add(&p2, &p4), &Polynomial::new(vec![0.3, 2.4, 1.6, 1.9]));
    }

    #[test]
    fn test_poly_mul() {
        let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
        let p2 = Polynomial::new(vec![0.0, 2.0]);
        let p3 = Polynomial::new(vec![-1.0, 6.0, 7.0, -5.0]);
        let p4 = Polynomial::new(vec![0.3, 0.4, 1.6, 1.9]);

        assert_poly_close(
            &poly_mul(&p1, &p2),
            &Polynomial::new(vec![0.0, 2.0, 10.0, 0.0, 10.0]),
        );
        assert_poly_close(
            &poly_mul(&p1, &p3),
            &Polynomial::new(vec![-1.0, 1.0, 37.0, 25.0, 5.0, 35.0, -25.0]),
        );
        assert_poly_close(&poly_mul(&p1, &Polynomial::new(vec![1.0])), &p1);
        assert_poly_close(
            &poly_mul(&p1, &p4),
            &Polynomial::new(vec![0.3, 1.9, 3.6, 11.4, 11.5, 8.0, 9.5]),
        );
    }

    #[test]
    fn test_poly_derivative() {
        let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
        let p2 = Polynomial::new(vec![0.3, 0.4, 1.6]);

        assert_poly_close(&poly_derivative(&p1), &Polynomial::new(vec![5.0, 0.0, 15.0]));
        assert_poly_close(&poly_derivative(&p2), &Polynomial::new(vec![0.4, 3.2]));
        assert_poly_close(
            &poly_derivative(&Polynomial::new(vec![0.0])),
            &Polynomial::new(vec![0.0]),
        );
    }
}
