use criterion::Criterion;
use mathru::algebra::linear::matrix::General;

criterion_group!(bench_general_sub_assign_general, general_sub_assign_general,);

fn general_sub_assign_general(bench: &mut Criterion) {
    bench.bench_function("general sub_assign general", move |bh| {
        bh.iter(|| {
            let diag1 = vec![1.0f64; 100 * 100];
            let mut d1 = General::new(100, 100, diag1.clone());
            let d2 = General::new(100, 100, diag1);
            d1 -= d2;
            assert!(d1[[0, 0]] != 1.0)
        });
    });
}
