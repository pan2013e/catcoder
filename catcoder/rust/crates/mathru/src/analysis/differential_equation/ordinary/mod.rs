//! Ordinary differential equation
//!
//! Fore more information:<br>
//! <https://en.wikipedia.org/wiki/Ordinary_differential_equation>
//!
//! Because ODE higher order can always be reduced to a system of first order
//! ODEs,  the implemented algorithms only support to solve first order ODEs.
//!
//! ```math
//! \frac{dy}{dt}=f(t, y)
//! ```
//!
//!
//! Solves an ODE using the Runge-Kutta-Dormand-Prince algorithm.
//!
//!<https://en.wikipedia.org/wiki/Dormand-Prince_method>
//!
//! # Example
//!
//! For this example, we want to solve the following ordinary differential
//! equation:
//! ```math
//! \frac{dy}{dt} = ay = f(t, y)
//! ```
//! The initial condition is $y(0) = 0.5$ and we solve it in the interval
//! $\lbrack 0, 2\rbrack$ The following equation is the closed solution for
//! this ODE:
//! ```math
//! y(t) = C a e^{at}
//! ```
//! $C$ is a parameter and depends on the initial condition $y(t_{0})$
//! ```math
//! C = \frac{y(t_{0})}{ae^{at_{0}}}
//! ```
//!
//! In this example, we set $a=2$
//! ```
//! # #[macro_use]
//! # extern crate mathru;
//! # fn main()
//! # {
//! use mathru::{
//!     algebra::linear::vector::Vector,
//!     analysis::differential_equation::ordinary::{ExplicitODE, ExplicitInitialValueProblem, ExplicitInitialValueProblemBuilder, solver::explicit::runge_kutta::adaptive::{DormandPrince54, ProportionalControl}},
//! };
//!
//! pub struct Ode
//! {
//! }
//!
//! impl ExplicitODE<f64> for Ode
//! {
//!    fn ode(&self, _x: &f64, y: &Vector<f64>) -> Vector<f64>
//!    {
//!        y * &2.0f64
//!    }
//! }
//!
//! let ode = Ode{};
//!
//! let problem = ExplicitInitialValueProblemBuilder::new(
//!     &ode,
//!     0.0,
//!     vector![0.5]
//! ).t_end(2.0)
//! .build();
//!
//! let h_0: f64 = 0.1;
//! let fac: f64 = 0.9;
//! let fac_min: f64 = 0.01;
//! let fac_max: f64 = 2.0;
//! let n_max: u32 = 500;
//! let abs_tol: f64 = 10e-8;
//! let rel_tol: f64 = 10e-6;
//!
//! let solver: ProportionalControl<f64> = ProportionalControl::new(n_max, h_0, fac, fac_min, fac_max, abs_tol, rel_tol);
//!
//! // Solve the ODE
//! let (t, y): (Vec<f64>, Vec<Vector<f64>>) = solver.solve(&problem, &DormandPrince54::default()).unwrap();
//!
//! # }
//! ```

pub mod problem;
pub mod solver;

mod implicit_ode;
pub use implicit_ode::ImplicitODE;

mod explicit_ode;
pub use explicit_ode::ExplicitODE;
mod explicit_initial_value_problem;
pub use explicit_initial_value_problem::{
    ExplicitInitialValueProblem, ExplicitInitialValueProblemBuilder,
};

mod implicit_initial_value_problem;
pub use implicit_initial_value_problem::{
    ImplicitInitialValueProblem, ImplicitInitialValueProblemBuilder,
};
