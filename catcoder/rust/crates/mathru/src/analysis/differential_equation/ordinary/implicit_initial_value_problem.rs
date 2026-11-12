use crate::algebra::abstr::Real;
use crate::algebra::linear::vector::Vector;
use crate::analysis::differential_equation::ordinary::ImplicitODE;

#[derive(Clone)]
pub struct ImplicitInitialValueProblem<'a, T, O>
where
    T: Real,
    O: ImplicitODE<T>,
{
    ode: &'a O,
    t_start: T,
    init_cond: Vector<T>,
    t_end: Option<T>,
    callback: Option<&'a (dyn Fn(&T, &Vector<T>) -> bool + 'a)>,
    //dense_output: Option<Vec<T>>
}

impl<'a, T, O> ImplicitInitialValueProblem<'a, T, O>
where
    T: Real,
    O: ImplicitODE<T>,
{
    pub fn ode(&self) -> &'a O {
        self.ode
    }

    pub fn t_start(&self) -> T {
        self.t_start
    }

    pub fn t_end(&self) -> Option<T> {
        self.t_end
    }

    pub fn callback(&self) -> Option<&'a dyn Fn(&T, &Vector<T>) -> bool> {
        self.callback
    }

    pub fn init_cond(&self) -> Vector<T> {
        self.init_cond.clone()
    }
}

#[derive(Clone)]
pub struct ImplicitInitialValueProblemBuilder<'a, T, O>
where
    T: Real,
    O: ImplicitODE<T>,
{
    ode: &'a O,
    t_start: T,
    init_cond: Vector<T>,
    t_end: Option<T>,
    callback: Option<&'a (dyn Fn(&T, &Vector<T>) -> bool + 'a)>,
    //dense_output: Option<Vec<T>>
}

impl<'a, T, O> ImplicitInitialValueProblemBuilder<'a, T, O>
where
    T: Real,
    O: ImplicitODE<T>,
{
    ///
    pub fn new(
        ode: &'a O,
        t_start: T,
        init_cond: Vector<T>,
    ) -> ImplicitInitialValueProblemBuilder<'a, T, O> {
        ImplicitInitialValueProblemBuilder {
            ode,
            t_start,
            init_cond,
            callback: None,
            t_end: None,
            //dense_output: None,
        }
    }

    pub fn t_end(&mut self, t: T) -> &mut Self {
        self.t_end = Some(t);
        self
    }

    pub fn callback(&mut self, callback: &'a (dyn Fn(&T, &Vector<T>) -> bool + 'a)) -> &mut Self {
        self.callback = Some(callback);
        self
    }

    ///
    /// # Panics
    ///
    pub fn build(&self) -> ImplicitInitialValueProblem<'a, T, O> {
        if self.callback.is_none() && self.t_end.is_none() {
            panic!("Either callback or t_end has to be set")
        }

        ImplicitInitialValueProblem {
            ode: self.ode,
            t_start: self.t_start,
            init_cond: self.init_cond.clone(),
            t_end: self.t_end,
            callback: self.callback,
            //dense_output: self.dense_output.clone()
        }
    }
}
