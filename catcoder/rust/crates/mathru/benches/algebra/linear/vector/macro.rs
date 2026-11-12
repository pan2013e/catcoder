use criterion::Criterion;
use mathru::algebra::linear::vector::Vector;
use mathru::vector;

criterion_group!(bench_macro, bench_vector_macro, bench_vec_macro);

fn bench_vector_macro(bench: &mut Criterion) {
    bench.bench_function("vector macro", move |bh| {
        bh.iter(|| {
            for _ in 0..1000000 {
                let _ = vector![1.0f64, 2.0];
            }
        });
    });
}

fn bench_vec_macro(bench: &mut Criterion) {
    bench.bench_function("vec macro", move |bh| {
        bh.iter(|| {
            for _ in 0..1000000 {
                let _ = vec![1.0f64, 2.0];
            }
        });
    });
}
