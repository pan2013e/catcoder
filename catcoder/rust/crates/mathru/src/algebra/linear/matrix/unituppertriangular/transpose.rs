use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::{Transpose, UnitLowerTriangular, UnitUpperTriangular};

impl<T> Transpose for UnitUpperTriangular<T>
where
    T: Field + Scalar,
{
    type Output = UnitLowerTriangular<T>;

    fn transpose(self) -> Self::Output {
        UnitLowerTriangular::new(self.matrix.transpose())
    }
}
