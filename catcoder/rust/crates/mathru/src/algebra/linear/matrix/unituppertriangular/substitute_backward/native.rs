use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::substitute::SubstituteBackward;
use crate::algebra::linear::{
    matrix::{General, UnitUpperTriangular},
    vector::Vector,
};

impl<T> SubstituteBackward<Vector<T>> for UnitUpperTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_backward(&self, c: Vector<T>) -> Result<Vector<T>, ()> {
        let mut b: Vector<T> = c;
        let rows = self.matrix.nrows();

        for k in (0..rows).rev() {
            for l in (k + 1..rows).rev() {
                b[k] = b[k] - self[[k, l]] * b[l];
            }
        }

        Ok(b)
    }
}

impl<T> SubstituteBackward<General<T>> for UnitUpperTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_backward(&self, a: General<T>) -> Result<General<T>, ()> {
        let mut b: General<T> = a;
        let rows = self.matrix.nrows();

        for k in (0..rows).rev() {
            for l in (k + 1..rows).rev() {
                b.set_row(&(b.get_row(k) - (b.get_row(l) * self[[k, l]])), k);
            }
        }
        Ok(b)
    }
}
