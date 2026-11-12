//! Solves an implicit ODE equation using backward Euler.
use super::ImplicitFixedStepSizeMethod;
use crate::analysis::differential_equation::ordinary::ImplicitODE;

use crate::{
    algebra::{
        abstr::Real,
        linear::{matrix::General, vector::vector::Vector},
    },
    analysis::{Function, Jacobian, NewtonRaphson},
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

/// Solves an ODE using backward Euler
///
/// <https://en.wikipedia.org/wiki/Backward_Euler_method>
///
/// # Example
///
/// For this example, we want to solve the following stiff initial value problem:
/// ```math
/// 0 = -4(y(t) -2) - y(t)^{'} = f(t, y, y^{'})
/// ```
/// The initial condition is $y(0) = 1.0$ and we solve it in the interval
/// $\lbrack 0, 2\rbrack$.\ The following equation is the closed solution for
/// this ODE:
/// ```math
/// y(t) = 2 - e^{-t}
/// ```
///
/// ```
/// use mathru::{
///     algebra::linear::{matrix::General, vector::Vector},
///     analysis::differential_equation::ordinary::{solver::implicit::runge_kutta::{ImplicitEuler, ImplicitFixedStepper}, ImplicitODE,
///     ImplicitInitialValueProblemBuilder},
///     matrix, vector
/// };
///
/// pub struct ImplicitODEExample
/// {
/// }
///
/// impl ImplicitODE<f64> for ImplicitODEExample
/// {
///     fn ode(&self, _t: &f64, x: &Vector<f64>) -> Vector<f64>
///     {
///         let result = (x * &-4.0) + 8.0;
///         return result;
///     }
///
///     fn jacobian(&self, _t: &f64, _input: &Vector<f64>) -> General<f64>
///     {
///         let jacobian = matrix![-4.0];
///         return jacobian;
///     }
/// }
///
/// let ode = ImplicitODEExample{};
/// let x_start: f64 = 0.0;
/// let x_end: f64 = 2.0;
///
/// let problem = ImplicitInitialValueProblemBuilder::<f64, ImplicitODEExample>::new(&ode, x_start, vector![1.0]).t_end(x_end).build();
///
/// let step_size: f64 = 0.0001;
/// let solver: ImplicitFixedStepper<f64> = ImplicitFixedStepper::new(0.0001);
///
/// // Solve the ODE
/// let (t, x): (Vec<f64>, Vec<Vector<f64>>) = solver.solve(&problem, &ImplicitEuler::default()).unwrap();
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct ImplicitEuler<T> {
    root_finder: NewtonRaphson<T>,
}

impl<T> Default for ImplicitEuler<T>
where
    T: Real,
{
    /// Creates a backward Euler instance
    fn default() -> ImplicitEuler<T> {
        ImplicitEuler {
            root_finder: NewtonRaphson::new(100, T::from_f64(0.00000001)),
        }
    }
}

impl<T> ImplicitFixedStepSizeMethod<T> for ImplicitEuler<T>
where
    T: Real,
{
    fn do_step<F>(&self, prob: &F, t_n: &T, x_n: &Vector<T>, h: &T) -> Vector<T>
    where
        F: ImplicitODE<T>,
    {
        let t: T = *t_n + *h;
        let ie_helper = ImplicitEulerHelper::new(prob, &t, x_n, h);
        let x_n = self.root_finder.find_root(&ie_helper, x_n).unwrap();

        x_n
    }

    /// Euler's method is a first order method
    fn order(&self) -> u8 {
        1
    }
}

/// this structure is implemented, therewith it is possible to implement the
/// traits needed by NewtonRaphson without exposing this traits.
struct ImplicitEulerHelper<'a, T, F>
where
    T: Real,
    F: ImplicitODE<T>,
{
    function: &'a F,
    t: &'a T,
    x: &'a Vector<T>,
    h: &'a T,
}

impl<'a, T, F> ImplicitEulerHelper<'a, T, F>
where
    T: Real,
    F: ImplicitODE<T>,
{
    pub fn new(
        function: &'a F,
        t: &'a T,
        x: &'a Vector<T>,
        h: &'a T,
    ) -> ImplicitEulerHelper<'a, T, F> {
        ImplicitEulerHelper { function, t, x, h }
    }
}

impl<'a, T, F> Function<Vector<T>> for ImplicitEulerHelper<'a, T, F>
where
    T: Real,
    F: ImplicitODE<T>,
{
    type Codomain = Vector<T>;

    ///$x(t_{n+1}) = y(t_n) + hf(t_{n+1}, x(t_{n+1})$
    ///
    /// g(z) = y(t_n) + hf(t_{n+1}, z) - z)$
    fn eval(&self, z: &Vector<T>) -> Vector<T> {
        &(self.x + &(&self.function.ode(self.t, z) * self.h)) - z
    }
}

impl<'a, T, F> Jacobian<T> for ImplicitEulerHelper<'a, T, F>
where
    T: Real,
    F: ImplicitODE<T>,
{
    /// $ \frac{\partial g(z)}{\partial z} = h \frac{\partial f(t_{n+1},
    /// z)}{\partial z} - I$
    fn jacobian(&self, z: &Vector<T>) -> General<T> {
        let (m, _n): (usize, usize) = z.dim();
        self.function.jacobian(self.t, z) * *self.h - General::one(m)
    }
}
