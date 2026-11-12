use crate::algebra::{
    abstr::Real,
    linear::{matrix::General, vector::vector::Vector},
};

pub trait Jacobian<T>
where
    T: Real,
{
    fn jacobian(&self, input: &Vector<T>) -> General<T>;
}
