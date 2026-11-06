use crate::algebra::linear::matrix::{General, UpperTriangular};

impl<T> From<General<T>> for UpperTriangular<T> {
    fn from(matrix: General<T>) -> Self {
        UpperTriangular { matrix }
    }
}
