use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::{LowerTriangular, Transpose, UpperTriangular};

impl<T> Transpose for UpperTriangular<T>
where
    T: Field + Scalar,
{
    type Output = LowerTriangular<T>;

    fn transpose(self) -> Self::Output {
        LowerTriangular::new(self.matrix.transpose())
    }
}
