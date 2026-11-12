use crate::algebra::linear::matrix::{Diagonal, General};

impl<T> From<General<T>> for Diagonal<T> {
    fn from(matrix: General<T>) -> Self {
        Diagonal { matrix }
    }
}
