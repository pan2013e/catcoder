/// Fixed step size Stepper
use crate::algebra::{abstr::Real, linear::vector::vector::Vector};
use crate::analysis::differential_equation::ordinary::solver::implicit::runge_kutta::ImplicitFixedStepSizeMethod;
use crate::analysis::differential_equation::ordinary::{ImplicitInitialValueProblem, ImplicitODE};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct ImplicitFixedStepper<T> {
    /// Step size
    step_size: T,
}

impl<T> ImplicitFixedStepper<T>
where
    T: Real,
{
    /// Creates an instance with the given step size
    ///
    /// # Arguments
    ///
    /// * 'step_size'
    ///
    /// # Panics
    ///
    /// if 'step_size' <= 0.0
    pub fn new(step_size: T) -> ImplicitFixedStepper<T> {
        if step_size <= T::zero() {
            panic!();
        }
        ImplicitFixedStepper { step_size }
    }

    pub fn solve<O, M>(
        &self,
        prob: &ImplicitInitialValueProblem<T, O>,
        method: &M,
    ) -> Result<(Vec<T>, Vec<Vector<T>>), String>
    where
        O: ImplicitODE<T>,
        M: ImplicitFixedStepSizeMethod<T>,
    {
        let init = prob.init_cond();
        let t_start = prob.t_start();
        let t_stop = prob.t_end();

        let mut x_n: Vector<T> = init;

        let mut t_n: T = t_start;
        let mut h: T = self.step_size;

        let mut t_vec: Vec<T> = vec![t_n];
        let mut res_vec: Vec<Vector<T>> = vec![x_n.clone()];

        let mut t_smaller_t_stop = t_stop.map_or(true, |t_e| t_n < t_e);
        let mut callback_condition = prob.callback().map_or(true, |func| func(&t_n, &x_n));

        while t_smaller_t_stop && callback_condition {
            h = t_stop.map_or(h, |t_e| h.min(t_e - t_n));

            x_n = method.do_step(prob.ode(), &t_n, &x_n, &h);
            t_n += h;

            t_vec.push(t_n);
            res_vec.push(x_n.clone());

            t_smaller_t_stop = t_stop.map_or(true, |t_e| t_n < t_e);
            callback_condition = prob.callback().map_or(true, |func| func(&t_n, &x_n));
        }

        Ok((t_vec, res_vec))
    }

    pub fn get_step_size(&self) -> &T {
        &self.step_size
    }

    pub fn set_step_size(&mut self, step_size: T) {
        self.step_size = step_size;
    }
}
