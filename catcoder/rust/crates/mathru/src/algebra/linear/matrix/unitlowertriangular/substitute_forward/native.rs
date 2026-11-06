use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::substitute::SubstituteForward;
use crate::algebra::linear::matrix::{General, UnitLowerTriangular};
use crate::algebra::linear::vector::Vector;

impl<T> SubstituteForward<Vector<T>> for UnitLowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    fn substitute_forward(&self, a: Vector<T>) -> Result<Vector<T>, ()> {
        let mut b: Vector<T> = a;
        let rows = self.matrix.nrows();

        for k in 0..rows {
            for l in 0..k {
                b[k] = b[k] - self[[k, l]] * b[l];
            }
        }

        Ok(b)
    }
}

impl<T> SubstituteForward<General<T>> for UnitLowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    fn substitute_forward(&self, a: General<T>) -> Result<General<T>, ()> {
        let mut b: General<T> = a;
        let rows = self.matrix.nrows();

        for k in 0..rows {
            for l in 0..k {
                b.set_row(&(b.get_row(k) - (b.get_row(l) * self[[k, l]])), k);
            }
        }

        Ok(b)
    }
}
