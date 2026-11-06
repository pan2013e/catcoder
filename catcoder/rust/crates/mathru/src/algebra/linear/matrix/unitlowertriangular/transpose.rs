use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::{Transpose, UnitLowerTriangular, UnitUpperTriangular};

impl<T> Transpose for UnitLowerTriangular<T>
where
    T: Field + Scalar,
{
    type Output = UnitUpperTriangular<T>;

    fn transpose(self) -> Self::Output {
        UnitUpperTriangular::new(self.matrix.transpose())
    }
}
