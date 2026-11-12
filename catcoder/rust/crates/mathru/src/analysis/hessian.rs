use crate::algebra::{
    abstr::Real,
    linear::{matrix::General, vector::vector::Vector},
};

pub trait Hessian<T>
where
    T: Real,
{
    fn hessian(&self, input: &Vector<T>) -> General<T>;
}
