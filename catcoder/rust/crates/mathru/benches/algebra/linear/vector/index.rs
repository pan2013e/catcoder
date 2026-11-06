use criterion::Criterion;
use mathru::algebra::abstr::cast::FromPrimitive;
use mathru::algebra::linear::vector::Vector;
use mathru::vector;

criterion_group!(bench_index, bench_vector_index, bench_vec_index);

fn bench_vector_index(bench: &mut Criterion) {
    let mut v = vector![1.0f64, 2.0];
    bench.bench_function("vector index", move |b| {
        b.iter(|| {
            for i in 0..1000000u64 {
                v[(i % 2) as usize] = f64::from_u64(i);
            }
        });
    });
}

fn bench_vec_index(bench: &mut Criterion) {
    let mut v = vec![1.0f64, 2.0];
    bench.bench_function("vec index", move |b| {
        b.iter(|| {
            for i in 0..1000000u64 {
                v[(i % 2) as usize] = i as f64;
            }
        });
    });
}
