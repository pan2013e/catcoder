use crate::{
    algebra::{
        abstr::Real,
        linear::{Matrix, vector::Vector},
    },
    analysis::differential_equation::ordinary::{
        ImplicitInitialValueProblem, ImplicitInitialValueProblemBuilder,
    },
};

/// Van der Pol oscillator
/// ```math
/// x_{1}^{'}(t) = x_{2}(t) \\
/// x_{2}^{'}(t) = \epsilon((1 - x_{1}(t)^{2})x_{2}(t) - x_{1}(t)) \\
/// ```
///
/// ```math
/// x_{1}(0) = 1 \\
/// x_{2}(0) = 0 \\
/// \epsilon = 0.1 \\
/// ```
pub fn implicit_van_der_pol_osc<T>() -> ImplicitInitialValueProblem<'static, T>
where
    T: Real,
{
    ImplicitInitialValueProblemBuilder::new(
        &|_t, x| {
            let epsilon = T::from_f64(0.1);
            let x_1 = x[0];
            let x_2 = x[1];
            vector![x_2; epsilon * (T::one() - (x_1 * x_1)) * x_2 - x_1]
        },
        &|_t, x| {
            let epsilon = T::from_f64(0.1);
            let x_1 = x[0];
            let x_2 = x[1];
            matrix![T::zero(), T::one(); -T::from_f64(2.0) * epsilon * x_1 * x_2  - T::one(), (T::one() - x_1 *
		    x_1) * epsilon]
        },
        T::zero(),
        vector![T::from_f64(1.0); T::from_f64(0.0)]
    )
    .t_end(T::from_f64(30.0))
    .build()
}
