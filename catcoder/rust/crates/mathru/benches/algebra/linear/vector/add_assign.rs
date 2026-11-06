use criterion::Criterion;
use mathru::algebra::linear::vector::Vector;

criterion_group!(
    bench_vector_add_assign_vector,
    vector_add_assign_vector,
    vec_add_assign_vec,
);

fn vector_add_assign_vector(bench: &mut Criterion) {
    bench.bench_function("vector add_assign vector", move |bh| {
        bh.iter(|| {
            let mut vec: Vector<f64> = Vector::new_column(vec![3.0; 100000]);
            let b: Vector<f64> = Vector::new_column(vec![3.0; 100000]);
            vec += b;
        });
    });
}

fn vec_add_assign_vec(bench: &mut Criterion) {
    bench.bench_function("vec add_assign vec", move |bh| {
        bh.iter(|| {
            let mut vec = vec![3.0; 100000];
            let b = vec![3.0; 100000];
            vec.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b)
        });
    });
}
