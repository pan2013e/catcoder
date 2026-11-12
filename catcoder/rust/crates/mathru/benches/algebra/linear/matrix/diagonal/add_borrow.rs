use criterion::Criterion;
use mathru::algebra::linear::vector::Vector;

criterion_group!(
    bench_vector_add_borrow_vector,
    vector_add_borrow_vector,
    vec_add_borrow_vec
);

fn vector_add_borrow_vector(bench: &mut Criterion) {
    bench.bench_function("vector add borrow vector", move |bh| {
        let vec1: Vector<f64> = Vector::new_column(vec![3.0; 100000]);
        let vec2: Vector<f64> = Vector::new_column(vec![3.0; 100000]);
        bh.iter(|| {
            let _: Vector<f64> = &vec1 + &vec2;
        });
    });
}

fn vec_add_borrow_vec(bench: &mut Criterion) {
    bench.bench_function("vec add borrow vec", move |bh| {
        let vec1: Vec<f64> = vec![3.0; 100000];
        let vec2: Vec<f64> = vec![3.0; 100000];
        bh.iter(|| {
            let _ = vec1
                .iter()
                .zip(&vec2)
                .map(|(a, b)| a + b)
                .collect::<Vec<f64>>();
        });
    });
}
