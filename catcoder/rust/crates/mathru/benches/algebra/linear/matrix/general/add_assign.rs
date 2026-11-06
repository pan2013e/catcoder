use criterion::Criterion;
use mathru::algebra::linear::matrix::General;

criterion_group!(
    bench_general_add_assign_general,
    general_add_assign_general,
    //vec_add_assign_vec
);

fn general_add_assign_general(bench: &mut Criterion) {
    bench.bench_function("general add_assign general", move |bh| {
        bh.iter(|| {
            let data1 = vec![1.0f64; 100 * 100];
            let mut g1 = General::new(100, 100, data1.clone());
            let g2 = General::new(100, 100, data1);
            g1 += g2;
            assert!(g1[[0, 0]] != 1.0)
        });
    });
}

// fn vec_add_assign_vec(bench: &mut Criterion) {
//     bench.bench_function("vec add_assign vec", move |bh| {
//         bh.iter(|| {
//             let g1 = vec![1.0f64; 100 * 100];
//             let g2 = g1.clone();

//             let _ = vec1
//                 .iter()
//                 .zip(&vec2)
//                 .map(|(a, b)| a + b)
//                 .collect::<Vec<f64>>();
//         });
//     });
// }
