#[macro_use]
extern crate criterion;

mod algebra;
mod analysis;

criterion_main!(
    // algebra::linear::vector::add_own::bench_vector_add_own_vector,
    // algebra::linear::vector::add_borrow::bench_vector_add_borrow_vector,
    // algebra::linear::vector::add_assign::bench_vector_add_assign_vector,
    // algebra::linear::vector::sub_assign::bench_vector_sub_assign_vector,
    // algebra::linear::vector::sub_borrow::bench_vector_sub_borrow_vector,
    // algebra::linear::vector::sub_own::bench_vector_sub_own_vector,
    // algebra::linear::matrix::general::add::add,
    // algebra::linear::matrix::general::sub::sub,
    // algebra::linear::matrix::general::mul::mul,
    // //analysis::vector_bench::euler,
    // algebra::linear::matrix::general::matrix,
    // analysis::fast_ode::ode,
    // algebra::linear::vector::r#macro::bench_macro,
    // algebra::linear::vector::index::bench_index,
    // algebra::abstr::from::bench_from,
    algebra::linear::matrix::general::add_borrow::bench_general_add_borrow_general,
    algebra::linear::matrix::general::add_assign::bench_general_add_assign_general,
    algebra::linear::matrix::general::add_own::bench_general_add_own_general,
    algebra::linear::matrix::general::sub_borrow::bench_general_sub_borrow_general,
    algebra::linear::matrix::general::sub_assign::bench_general_sub_assign_general,
    algebra::linear::matrix::general::sub_own::bench_general_sub_own_general,
    algebra::linear::matrix::diagonal::add_assign::bench_diagonal_add_assign_diagonal
);
