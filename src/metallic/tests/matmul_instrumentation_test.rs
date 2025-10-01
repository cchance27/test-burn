use std::time::Duration;

use crate::metallic::context::MatMulInstrumentation;
use crate::metallic::kernels::matmul::MatMulBackendKind;
use crate::metallic::metrics::MatMulBackendStats;

#[test]
fn matmul_samples_populate_backend_stats() {
    let instrumentation = MatMulInstrumentation::default();

    // Simulate the command buffer completion handler invoking the context sampler.
    let duration = Duration::from_micros(42);
    instrumentation.record(MatMulBackendKind::Mlx, duration);

    let samples = instrumentation.take_samples();
    assert_eq!(samples.len(), 1, "expected a single recorded sample");

    let sample = &samples[0];
    assert_eq!(sample.backend, MatMulBackendKind::Mlx);
    assert!(sample.duration > Duration::ZERO, "duration should be non-zero");

    let mut stats = MatMulBackendStats::default();
    stats.record(sample.backend, sample.duration);
    assert!(stats.mlx().has_samples(), "stat accumulator should register the sample");
    assert!(stats.mlx().last_ms() > 0.0, "last recorded duration should be non-zero");
}
