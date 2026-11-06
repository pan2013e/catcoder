//! Solves an ODE using Adams-Bashforth method.
use crate::{
    algebra::{abstr::Real, linear::vector::vector::Vector},
    analysis::differential_equation::ordinary::{ExplicitInitialValueProblem, ExplicitODE},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

/// Adams-Bashforth method
///
/// # Example
///
/// For this example, we want to solve the following ordinary differential
/// equation:
/// ```math
/// \frac{dy}{dt} = ay = f(t, y)
/// ```
/// The initial condition is $y(0) = 0.5$ and we solve it in the interval
/// $\lbrack 0, 2\rbrack$ The following equation is the closed solution for
/// this ODE:
/// ```math
/// y(t) = C a e^{at}
/// ```
/// $C$ is a parameter and depends on the initial condition $y(t_{0})$
/// ```math
/// C = \frac{y(t_{0})}{ae^{at_{0}}}
/// ```
///
/// In this example, we set $a=2$
/// ```
/// # #[macro_use]
/// # extern crate mathru;
/// # fn main()
/// # {
/// use mathru::{
///     algebra::linear::vector::Vector,
///     analysis::differential_equation::ordinary::{ExplicitODE, ExplicitInitialValueProblemBuilder, solver::explicit::AdamsBashforth},
/// };
///
/// pub struct Ode
/// {
/// }
///
/// impl ExplicitODE<f64> for Ode
/// {
///    fn ode(&self, _x: &f64, y: &Vector<f64>) -> Vector<f64>
///    {
///        y * &2.0f64
///    }
/// }
///
/// let ode = Ode{};
///
/// let problem = ExplicitInitialValueProblemBuilder::new(
///     &ode,
///     0.0,
///     vector![0.5]
/// )
/// .t_end(2.0)
/// .build();
///
/// let step_size: f64 = 0.001;
/// let solver: AdamsBashforth<f64> = AdamsBashforth::new(1, step_size);
///
/// // Solve the ODE
/// let (t, y): (Vec<f64>, Vec<Vector<f64>>) = solver.solve(&problem).unwrap();
///
/// # }
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct AdamsBashforth<T> {
    k: u8,
    step_size: T,
}

impl<T> AdamsBashforth<T>
where
    T: Real,
{
    pub fn new(k: u8, step_size: T) -> AdamsBashforth<T> {
        if k == 0 || k > 5 {
            panic!();
        }
        if step_size <= T::zero() {
            panic!();
        }

        AdamsBashforth { k, step_size }
    }
}

impl<T> AdamsBashforth<T>
where
    T: Real,
{
    /// Solves `prob` using Adams' method.
    ///
    /// # Arguments
    ///
    /// # Return
    ///
    /// The solver returns a vector and a matrix, containing the times used in
    /// each step of the algorithm and the respectful values for that time.
    ///
    /// # Panic
    ///
    pub fn solve<O>(
        &self,
        prob: &ExplicitInitialValueProblem<T, O>,
    ) -> Result<(Vec<T>, Vec<Vector<T>>), String>
    where
        O: ExplicitODE<T>,
    {
        let t_start: T = prob.t_start();
        let t_stop: T = prob.t_end().unwrap();

        if t_start > t_stop {
            panic!();
        }

        let mut x_n: Vector<T> = prob.init_cond();
        let mut t_n: T = t_start;

        let limit = ((t_stop - t_start) / self.step_size).ceil() + T::one();
        let steps: usize = limit.to_u64() as usize;
        let mut t_vec: Vec<T> = Vec::with_capacity(steps);
        let mut res_vec: Vec<Vector<T>> = Vec::with_capacity(steps);

        t_vec.push(t_n);
        res_vec.push(x_n.clone());

        //calculate initial steps
        if self.k >= 2 {
            let h: T = self.step_size.min(t_stop - t_n);
            let x_n = AdamsBashforth::step_s1(prob, &t_vec, &res_vec, h);
            t_n += h;

            t_vec.push(t_n);
            res_vec.push(x_n);
        }

        if self.k >= 3 {
            let h: T = self.step_size.min(t_stop - t_n);
            let x_n = AdamsBashforth::step_s2(prob, &t_vec, &res_vec, h);
            t_n += h;

            t_vec.push(t_n);
            res_vec.push(x_n);
        }

        if self.k >= 4 {
            let h: T = self.step_size.min(t_stop - t_n);
            let x_n = AdamsBashforth::step_s3(prob, &t_vec, &res_vec, h);
            t_n += h;

            t_vec.push(t_n);
            res_vec.push(x_n);
        }

        if self.k >= 5 {
            let h: T = self.step_size.min(t_stop - t_n);
            let x_n = AdamsBashforth::step_s4(prob, &t_vec, &res_vec, h);
            t_n += h;

            t_vec.push(t_n);
            res_vec.push(x_n);
        }

        let step = match self.k {
            1 => AdamsBashforth::step_s1,
            2 => AdamsBashforth::step_s2,
            3 => AdamsBashforth::step_s3,
            4 => AdamsBashforth::step_s4,
            5 => AdamsBashforth::step_s5,
            _ => panic!(),
        };

        while (t_n - t_stop).abs() > T::from_f64(0.0000000001) {
            //Step size
            let h: T = self.step_size.min(t_stop - t_n);

            x_n = step(prob, &t_vec, &res_vec, h);
            t_n += h;

            t_vec.push(t_n);
            res_vec.push(x_n.clone());
        }

        Ok((t_vec, res_vec))
    }
}

impl<T> AdamsBashforth<T>
where
    T: Real,
{
    fn step_s1<O>(
        prob: &ExplicitInitialValueProblem<T, O>,
        t: &[T],
        x: &Vec<Vector<T>>,
        h: T,
    ) -> Vector<T>
    where
        O: ExplicitODE<T>,
    {
        let n: usize = x.len() - 1;
        let x_n: &Vector<T> = &x[n];
        let t_n: &T = &t[n];
        let ode = prob.ode();
        x_n + &(&ode.ode(t_n, x_n) * &h)
    }

    fn step_s2<O>(
        prob: &ExplicitInitialValueProblem<T, O>,
        t: &[T],
        x: &Vec<Vector<T>>,
        h: T,
    ) -> Vector<T>
    where
        O: ExplicitODE<T>,
    {
        let n: usize = x.len() - 1;
        let x_n: &Vector<T> = &x[n];
        let t_n: &T = &t[n];
        let x_n1: &Vector<T> = &x[n - 1];
        let t_n1: &T = &t[n - 1];
        let ode = prob.ode();
        x_n + &((ode.ode(t_n, x_n) * T::from_f64(3.0 / 2.0)
            + ode.ode(t_n1, x_n1) * T::from_f64(-0.5))
            * h)
    }

    fn step_s3<O>(
        prob: &ExplicitInitialValueProblem<T, O>,
        t: &[T],
        x: &Vec<Vector<T>>,
        h: T,
    ) -> Vector<T>
    where
        O: ExplicitODE<T>,
    {
        let n: usize = x.len() - 1;
        let x_n: &Vector<T> = &x[n];
        let t_n: &T = &t[n];
        let x_n1: &Vector<T> = &x[n - 1];
        let t_n1: &T = &t[n - 1];
        let x_n2: &Vector<T> = &x[n - 2];
        let t_n2: &T = &t[n - 2];
        let ode = prob.ode();

        x_n + &((ode.ode(t_n, x_n) * T::from_f64(23.0 / 12.0)
            + ode.ode(t_n1, x_n1) * T::from_f64(-16.0 / 12.0)
            + ode.ode(t_n2, x_n2) * T::from_f64(5.0 / 12.0))
            * h)
    }

    fn step_s4<O>(
        prob: &ExplicitInitialValueProblem<T, O>,
        t: &[T],
        x: &Vec<Vector<T>>,
        h: T,
    ) -> Vector<T>
    where
        O: ExplicitODE<T>,
    {
        let n: usize = x.len() - 1;
        let x_n: &Vector<T> = &x[n];
        let t_n: &T = &t[n];
        let x_n1: &Vector<T> = &x[n - 1];
        let t_n1: &T = &t[n - 1];
        let x_n2: &Vector<T> = &x[n - 2];
        let t_n2: &T = &t[n - 2];
        let x_n3: &Vector<T> = &x[n - 3];
        let t_n3: &T = &t[n - 3];
        let ode = prob.ode();

        x_n + &((ode.ode(t_n, x_n) * T::from_f64(55.0 / 24.0)
            + ode.ode(t_n1, x_n1) * T::from_f64(-59.0 / 24.0)
            + ode.ode(t_n2, x_n2) * T::from_f64(37.0 / 24.0)
            + ode.ode(t_n3, x_n3) * T::from_f64(-9.0 / 24.0))
            * h)
    }

    fn step_s5<O>(
        prob: &ExplicitInitialValueProblem<T, O>,
        t: &[T],
        x: &Vec<Vector<T>>,
        h: T,
    ) -> Vector<T>
    where
        O: ExplicitODE<T>,
    {
        let n: usize = x.len() - 1;
        let x_n: &Vector<T> = &x[n];
        let t_n: &T = &t[n];
        let x_n1: &Vector<T> = &x[n - 1];
        let t_n1: &T = &t[n - 1];
        let x_n2: &Vector<T> = &x[n - 2];
        let t_n2: &T = &t[n - 2];
        let x_n3: &Vector<T> = &x[n - 3];
        let t_n3: &T = &t[n - 3];
        let x_n4: &Vector<T> = &x[n - 4];
        let t_n4: &T = &t[n - 4];
        let ode = prob.ode();

        x_n + &((ode.ode(t_n, x_n) * T::from_f64(1901.0 / 720.0)
            + ode.ode(t_n1, x_n1) * T::from_f64(-2774.0 / 720.0)
            + ode.ode(t_n2, x_n2) * T::from_f64(2616.0 / 720.0)
            + ode.ode(t_n3, x_n3) * T::from_f64(-1274.0 / 720.0)
            + ode.ode(t_n4, x_n4) * T::from_f64(251.0 / 720.0))
            * h)
    }
}
