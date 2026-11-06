use crate::algebra::linear::matrix::{General, UnitLowerTriangular};

impl<T> From<General<T>> for UnitLowerTriangular<T> {
    fn from(matrix: General<T>) -> Self {
        UnitLowerTriangular { matrix }
    }
}
