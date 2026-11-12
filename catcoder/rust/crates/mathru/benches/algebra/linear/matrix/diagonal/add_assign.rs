use criterion::Criterion;
use mathru::algebra::linear::matrix::Diagonal;

criterion_group!(
    bench_diagonal_add_assign_diagonal,
    diagonal_add_assign_diagonal,
);

fn diagonal_add_assign_diagonal(bench: &mut Criterion) {
    bench.bench_function("diagonal add_assign diagonal", move |bh| {
        bh.iter(|| {
            let diag_value = [1.0f64; 128];
            let mut d1 = Diagonal::new(&diag_value);
            let d2 = Diagonal::new(&diag_value);
            d1 += d2;
        });
    });
}
