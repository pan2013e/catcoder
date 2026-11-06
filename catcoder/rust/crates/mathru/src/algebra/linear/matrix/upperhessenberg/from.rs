use crate::algebra::linear::matrix::{General, UpperHessenberg};

impl<T> From<General<T>> for UpperHessenberg<T> {
    fn from(matrix: General<T>) -> Self {
        UpperHessenberg { matrix }
    }
}
