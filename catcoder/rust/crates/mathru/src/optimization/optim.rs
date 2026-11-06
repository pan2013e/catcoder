use crate::algebra::{
    abstr::Real,
    linear::{matrix::General, vector::vector::Vector},
};

pub trait Optim<T>
where
    T: Real,
{
    fn eval(&self, _x: &Vector<T>) -> Vector<T>;
    // Computes the Jacobian at the given x
    fn jacobian(&self, _x: &Vector<T>) -> General<T>;
    /// Computes the Hessian at the given value x
    fn hessian(&self, _x: &Vector<T>) -> General<T>;
}
