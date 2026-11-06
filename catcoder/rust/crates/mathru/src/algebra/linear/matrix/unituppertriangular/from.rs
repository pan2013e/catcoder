use crate::algebra::linear::matrix::{General, UnitUpperTriangular};

impl<T> From<General<T>> for UnitUpperTriangular<T> {
    fn from(matrix: General<T>) -> Self {
        UnitUpperTriangular { matrix }
    }
}
