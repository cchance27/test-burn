#![cfg(test)]

use crate::context::{MatmulDims, q8_should_use_mlx_for_m1, set_q8_m1_mlx_min_n_override_for_tests};

#[test]
fn q8_m1_large_n_prefers_mlx_by_default_threshold() {
    set_q8_m1_mlx_min_n_override_for_tests(None);

    let dims = MatmulDims {
        batch: 1,
        m: 1,
        n: 151_936,
        k: 896,
    };
    assert!(q8_should_use_mlx_for_m1(&dims, false, false));

    let small_dims = MatmulDims {
        batch: 1,
        m: 1,
        n: 896,
        k: 896,
    };
    assert!(!q8_should_use_mlx_for_m1(&small_dims, false, false));
}

#[test]
fn q8_m1_threshold_override_controls_decision() {
    // Force always-on for m==1.
    set_q8_m1_mlx_min_n_override_for_tests(Some(0));
    let dims = MatmulDims {
        batch: 1,
        m: 1,
        n: 896,
        k: 896,
    };
    assert!(q8_should_use_mlx_for_m1(&dims, false, false));

    // Force always-off for typical shapes.
    set_q8_m1_mlx_min_n_override_for_tests(Some(1_000_000));
    let big_dims = MatmulDims {
        batch: 1,
        m: 1,
        n: 151_936,
        k: 896,
    };
    assert!(!q8_should_use_mlx_for_m1(&big_dims, false, false));

    // Reset override for other tests.
    set_q8_m1_mlx_min_n_override_for_tests(None);
}

#[test]
fn q8_m1_requires_non_transposed_operands() {
    set_q8_m1_mlx_min_n_override_for_tests(Some(0));
    let dims = MatmulDims {
        batch: 1,
        m: 1,
        n: 151_936,
        k: 896,
    };

    assert!(!q8_should_use_mlx_for_m1(&dims, true, false));
    assert!(!q8_should_use_mlx_for_m1(&dims, false, true));
    assert!(q8_should_use_mlx_for_m1(&dims, false, false));

    set_q8_m1_mlx_min_n_override_for_tests(None);
}
