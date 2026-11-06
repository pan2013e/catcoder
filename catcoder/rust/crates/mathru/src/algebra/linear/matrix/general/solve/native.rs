use crate::algebra::{
    abstr::{AbsDiffEq, Field, RelativeEq, Scalar},
    linear::{matrix::General, vector::vector::Vector},
};

use crate::algebra::linear::matrix::Solve;

impl<T> Solve<Vector<T>> for General<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    /// Solves $Ax = y$
    /// where $A \in \mathbb{R}^{m * n}, x \in \mathbb{R}^n, y \in \mathbb{R}^m$
    fn solve(&self, rhs: &Vector<T>) -> Result<Vector<T>, ()> {
        self.dec_lu()?.solve(rhs)
    }
}

impl<T> Solve<General<T>> for General<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    fn solve(&self, rhs: &General<T>) -> Result<General<T>, ()> {
        self.dec_lu()?.solve(rhs)
    }
}
