use criterion::Criterion;
use mathru::algebra::linear::vector::Vector;

criterion_group!(
    bench_vector_sub_own_vector,
    vector_sub_own_vector,
    vec_sub_own_vec
);

fn vector_sub_own_vector(bench: &mut Criterion) {
    bench.bench_function("vector sub own vector", move |bh| {
        bh.iter(|| {
            let vec1: Vector<f64> = Vector::new_column(vec![3.0; 100000]);
            let vec2: Vector<f64> = Vector::new_column(vec![3.0; 100000]);

            let _: Vector<f64> = vec1 - vec2;
        });
    });
}

fn vec_sub_own_vec(bench: &mut Criterion) {
    bench.bench_function("vec sub own vec", move |bh| {
        bh.iter(|| {
            let vec1: Vec<f64> = vec![3.0; 100000];
            let vec2: Vec<f64> = vec![3.0; 100000];

            let _ = vec1
                .into_iter()
                .zip(&vec2)
                .map(|(a, b)| a - b)
                .collect::<Vec<f64>>();
        });
    });
}
