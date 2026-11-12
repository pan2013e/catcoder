use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::{LowerTriangular, Transpose, UpperTriangular};

impl<T> Transpose for LowerTriangular<T>
where
    T: Field + Scalar,
{
    type Output = UpperTriangular<T>;

    fn transpose(self) -> Self::Output {
        UpperTriangular::new(self.matrix.transpose())
    }
}
