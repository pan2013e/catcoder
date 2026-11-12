use criterion::Criterion;
use mathru::algebra::abstr::cast::FromPrimitive;

criterion_group!(bench_from, from, cast,);

fn from(bench: &mut Criterion) {
    bench.bench_function("T::from", move |bh| {
        bh.iter(|| {
            let mut sum = 0.0;
            for _ in 0..1000000 {
                let a = f64::from(1.0 / 3.0);
                sum += a;
            }
            assert!(sum > 0.0);
        });
    });
}

fn cast(bench: &mut Criterion) {
    bench.bench_function("cast", move |bh| {
        bh.iter(|| {
            let mut sum = 0.0;
            for _ in 0..1000000 {
                let a = 1.0 / 3.0 as f64;
                sum += a;
            }

            assert!(sum > 0.0);
        });
    });
}
