use crate::algebra::linear::matrix::{General, LowerTriangular, UnitLowerTriangular};

impl<T> From<General<T>> for LowerTriangular<T> {
    fn from(matrix: General<T>) -> Self {
        LowerTriangular { matrix }
    }
}

impl<T> From<UnitLowerTriangular<T>> for LowerTriangular<T> {
    fn from(ut: UnitLowerTriangular<T>) -> Self {
        LowerTriangular { matrix: ut.matrix }
    }
}
