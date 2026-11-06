//! Adaptive step size stepper
use crate::algebra::{abstr::Real, linear::vector::vector::Vector};
use crate::analysis::differential_equation::ordinary::solver::explicit::runge_kutta::adaptive::ExplicitRKEmbeddedMethod;
use crate::analysis::differential_equation::ordinary::ExplicitInitialValueProblem;
use crate::analysis::differential_equation::ordinary::ExplicitODE;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;
use std::default::Default;

/// Proportional Control
///
/// Solving Ordinary Differential Equations I
/// Nonstiff Problems
/// E. Hairer, S. P. Nørsett, G. Wanner
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub struct ProportionalControl<T> {
    /// Step size
    n_max: u32,
    h_0: Option<T>,
    fac: T,
    fac_min: T,
    fac_max: T,
    /// abs_tol: Absolute tolerance. This is the tolerance on local error
    /// estimates, not necessarily the global error. Defaults to 1e-6.
    abs_tol: T,
    /// reltol: Relative tolerance. This is the tolerance on local error
    /// estimates, not necessarily the global error. Defaults to 1e-3.
    rel_tol: T,
}

impl<T> Default for ProportionalControl<T>
where
    T: Real,
{
    fn default() -> ProportionalControl<T> {
        ProportionalControl::new(
            1000,
            T::from_f64(0.02),
            T::from_f64(0.8),
            T::from_f64(0.001),
            T::from_f64(3.0),
            T::from_f64(1.0e-6),
            T::from_f64(1.0e-3),
        )
    }
}

impl<T> ProportionalControl<T>
where
    T: Real,
{
    /// Creates an instance with the given step size
    ///
    /// # Param
    ///
    /// *'fac_min':
    /// *'fac_max': 1.5 <= fac_max <= 5.0
    pub fn new(
        n_max: u32,
        h_0: T,
        fac: T,
        fac_min: T,
        fac_max: T,
        abs_tol: T,
        rel_tol: T,
    ) -> ProportionalControl<T> {
        ProportionalControl {
            n_max,
            h_0: Some(h_0),
            fac,
            fac_min,
            fac_max,
            abs_tol,
            rel_tol,
        }
    }

    /// Returns the absolute tolerance
    pub fn get_abs_tol(&self) -> &T {
        &self.abs_tol
    }

    /// Returns the relative tolerance
    pub fn get_rel_tol(&self) -> &T {
        &self.rel_tol
    }

    /// Sets the absolute tolerance
    ///
    /// # Parameters
    ///
    /// * 'abs_tol': abs_tol >= 0.0
    /// # Panics
    ///
    /// if 'abs_tol' < 0.0
    pub fn set_abs_tol(&mut self, abs_tol: T) {
        if abs_tol < T::zero() {
            panic!();
        }
        self.abs_tol = abs_tol;
    }

    /// Sets the relative tolerance
    ///
    /// # Parameters
    ///
    /// * 'rel_tol': rel_tol >= 0.0
    /// # Panics
    ///
    /// if 'rel_tol' < 0.0
    pub fn set_rel_tol(&mut self, rel_tol: T) {
        if rel_tol < T::zero() {
            panic!();
        }
        self.rel_tol = rel_tol;
    }

    /// # Arguments
    ///
    /// * 'prob' is an explicit ordinary differential equation
    /// * 'method' is the method to solve the ivp
    ///
    /// # Return
    ///
    /// Ok:
    ///

    pub fn solve<M, O>(
        &self,
        prob: &ExplicitInitialValueProblem<T, O>,
        method: &M,
    ) -> Result<(Vec<T>, Vec<Vector<T>>), &'static str>
    where
        M: ExplicitRKEmbeddedMethod<T>,
        O: ExplicitODE<T>,
    {
        let t_start: T = prob.t_start();

        let t_stop: Option<T> = prob.t_end();

        let tableau = method.tableau();

        let (p, p_s): (u8, u8) = tableau.order();

        let q: T = T::from_u8(p.min(p_s));
        let l: T = T::one() / (q + T::one());
        let mut x_n: Vector<T> = prob.init_cond();
        let mut t_n: T = t_start;
        let mut h: T = self.h_0.unwrap_or_else(|| self.calc_initial_step(prob, p));

        let mut t_vec: Vec<T> = vec![t_n];

        let mut res_vec: Vec<Vector<T>> = vec![x_n.clone()];

        let mut n: u32 = 0;

        let mut t_smaller_t_stop = t_stop.map_or(true, |t_e| t_n < t_e);
        let mut callback_condition = prob.callback().map_or(true, |func| func(&t_n, &x_n));

        while n < self.n_max && t_smaller_t_stop && callback_condition {
            h = t_stop.map_or(h, |t_e| h.min(t_e - t_n));

            let (y_n, y_n_s): (Vector<T>, Vector<T>) = tableau.do_step(prob.ode(), &t_n, &x_n, &h);
            let err: T = self.calc_error(&y_n, &y_n_s, &x_n);

            if err <= T::one() {
                t_n += h;

                x_n = y_n;

                t_vec.push(t_n);
                res_vec.push(x_n.clone());
                n += 1;

                t_smaller_t_stop = t_stop.map_or(true, |t_e| t_n < t_e);
                callback_condition = prob.callback().map_or(true, |func| func(&t_n, &x_n));
            }

            if err != T::zero() {
                let mut s: T = self.fac * (T::one() / err).pow(l);

                if s < self.fac_min {
                    s = self.fac_min;
                }

                if s > self.fac_max {
                    s = self.fac_max;
                }

                h = s * h;
            }
        }

        match t_stop {
            Some(t_e) => {
                if t_n < t_e && callback_condition {
                    Err("Maximum number of iterations reached")
                } else {
                    Ok((t_vec, res_vec))
                }
            }
            None => Ok((t_vec, res_vec)),
        }
    }

    fn calc_error(&self, y: &Vector<T>, y_h: &Vector<T>, y_p: &Vector<T>) -> T {
        let (m, _n) = y.dim();

        let sum =
            y.iter()
                .zip(y_h.iter().zip(y_p.iter()))
                .fold(T::zero(), |s, (y_i, (y_h_i, y_p_i))| {
                    let y_max_i = y_i.abs().max(y_p_i.abs());
                    let sc_i: T = self.abs_tol + y_max_i * self.rel_tol;

                    let k: T = (*y_i - *y_h_i) / sc_i;
                    s + k * k
                });

        (sum / T::from_f64(m as f64)).sqrt()
    }

    fn calc_initial_step<O>(&self, prob: &ExplicitInitialValueProblem<T, O>, p: u8) -> T
    where
        O: ExplicitODE<T>,
    {
        let ode = prob.ode();
        let y_0 = prob.init_cond();
        let d_0: T = self.norm(&y_0);

        let x_0 = prob.t_start();

        let f_y_0 = ode.ode(&x_0, &y_0);
        let d_1: T = self.norm(&f_y_0);

        let h_0_limit = T::from_f64(1.0e-5);
        let h_0: T = if d_0 < h_0_limit || d_1 < h_0_limit {
            T::from_f64(1.0e-6)
        } else {
            T::from_f32(0.01) * (d_0 / d_1)
        };

        let y_1 = y_0 + &f_y_0 * &h_0;

        let f_y_1 = ode.ode(&(x_0 + h_0), &y_1);

        let d_2 = self.norm(&(f_y_1 - f_y_0)) / h_0;

        let h_1 = if d_0.max(d_1) <= T::from_f64(1.0e-15) {
            T::from_f64(1.0e-15).max(T::from_f64(1.0e-3) * h_0)
        } else {
            (T::from_f64(0.01) / d_1.max(d_2)).pow(T::one() / (T::from_u8(p + 1)))
        };

        h_1.min(T::from_f64(100.0) * h_0)
    }

    fn norm(&self, x: &Vector<T>) -> T {
        let (m, _n) = x.dim();

        let sum = x.iter().fold(T::zero(), |s, x_i| {
            let sc_i: T = self.abs_tol + x_i.abs() * self.rel_tol;
            let k: T = *x_i / sc_i;
            s + k * k
        });

        (sum / T::from_f64(m as f64)).sqrt()
    }
}

pub struct ProportionalControlBuilder<T> {
    n_max: u32,
    fac: T,
    fac_min: T,
    fac_max: T,
    abs_tol: T,
    rel_tol: T,
    h_0: Option<T>,
}

impl<T> ProportionalControlBuilder<T>
where
    T: Real,
{
    /// # Parameters
    ///
    /// * 'abs_tol': Absolute tolerance. This is the tolerance on local error
    /// estimates, not necessarily the global error.
    /// * 'rel_tol: Relative tolerance. This is the tolerance on local error
    /// estimates, not necessarily the global error.
    pub fn new(
        n_max: u32,
        fac: T,
        fac_min: T,
        fac_max: T,
        abs_tol: T,
        rel_tol: T,
    ) -> ProportionalControlBuilder<T> {
        ProportionalControlBuilder {
            n_max,
            fac,
            fac_min,
            fac_max,
            abs_tol,
            rel_tol,
            h_0: None,
        }
    }

    pub fn init_step_size(&mut self, h_0: T) -> &mut Self {
        self.h_0 = Some(h_0);
        self
    }

    pub fn build(&self) -> ProportionalControl<T> {
        ProportionalControl {
            n_max: self.n_max,
            h_0: self.h_0,
            fac: self.fac,
            fac_min: self.fac_min,
            fac_max: self.fac_max,
            abs_tol: self.abs_tol,
            rel_tol: self.rel_tol,
        }
    }
}
