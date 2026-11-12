use criterion::Criterion;
use mathru::algebra::linear::matrix::General;

criterion_group!(
    bench_general_mul,
    mul_matrix_own,
    bench_mul_matrix_borrow,
    bench_mul_matrix_mut_borrow,
    bench_mul_scalar_own,
    bench_mul_scalar_mut_borrow,
    bench_mul_scalar_borrow
);

fn mul_matrix_own(bench: &mut Criterion) {
    bench.bench_function("mul matrix own", move |bh| {
        bh.iter(|| {
            let a: General<f64> = General::new(100, 100, vec![3.0; 10000]);
            let b: General<f64> = General::new(100, 100, vec![3.0; 10000]);
            let _ = a * b;
        });
    });
}

fn mul_dynmatrix(bench: &mut Criterion) {
    bench.bench_function("mul matrix own", move |bh| {
        bh.iter(|| {});
    });
}

fn bench_mul_matrix_borrow(bench: &mut Criterion) {
    let a: General<f64> = General::new(100, 100, vec![3.0; 10000]);
    let b: General<f64> = General::new(100, 100, vec![3.0; 10000]);

    bench.bench_function("mul matrix borrow", move |bh| {
        bh.iter(|| {
            let _ = &a * &b;
        });
    });
}

fn bench_mul_matrix_mut_borrow(bench: &mut Criterion) {
    let mut a: General<f64> = General::new(100, 100, vec![3.0; 10000]);
    let b: General<f64> = General::new(100, 100, vec![3.0; 10000]);

    bench.bench_function("mul matrix mut borrow", move |bh| {
        bh.iter(|| {
            let _ = &mut a * &b;
        });
    });
}

fn bench_mul_scalar_own(bench: &mut Criterion) {
    bench.bench_function("mul scalar own", move |bh| {
        bh.iter(|| {
            let matrix: General<f64> = General::new(100, 100, vec![3.0; 10000]);
            let _ = matrix * 3.0f64;
        });
    });
}

fn bench_mul_scalar_borrow(bench: &mut Criterion) {
    let matrix: General<f64> = General::new(100, 100, vec![3.0; 10000]);
    bench.bench_function("mul scalar borrow", move |bh| {
        bh.iter(|| {
            let _ = &matrix * &3.0f64;
        });
    });
}

fn bench_mul_scalar_mut_borrow(bench: &mut Criterion) {
    let mut matrix: General<f64> = General::new(100, 100, vec![3.0; 10000]);
    bench.bench_function("mul scalar mut borrow", move |bh| {
        bh.iter(|| {
            let _ = (&mut matrix) * &3.0f64;
        });
    });
}
