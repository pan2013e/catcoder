use crate::algebra::{
    abstr::{Field, Scalar},
    linear::{
        matrix::{
            Diagonal, LowerTriangular, UnitLowerTriangular, UnitUpperTriangular, UpperHessenberg,
            UpperTriangular,
        },
        vector::Vector,
    },
};

use super::General;

impl<T> From<Vector<T>> for General<T>
where
    T: Field + Scalar,
{
    fn from(v: Vector<T>) -> Self {
        let (v_m, v_n): (usize, usize) = v.dim();

        General::new(v_m, v_n, v.convert_to_vec())
    }
}

impl<T> From<LowerTriangular<T>> for General<T> {
    fn from(ut: LowerTriangular<T>) -> Self {
        ut.matrix
    }
}

impl<T> From<UnitLowerTriangular<T>> for General<T> {
    fn from(ut: UnitLowerTriangular<T>) -> Self {
        ut.matrix
    }
}

impl<T> From<UpperTriangular<T>> for General<T> {
    fn from(ut: UpperTriangular<T>) -> Self {
        ut.matrix
    }
}

impl<T> From<UnitUpperTriangular<T>> for General<T> {
    fn from(ut: UnitUpperTriangular<T>) -> Self {
        ut.matrix
    }
}

impl<T> From<UpperHessenberg<T>> for General<T> {
    fn from(ut: UpperHessenberg<T>) -> Self {
        ut.matrix
    }
}

impl<T> From<Diagonal<T>> for General<T> {
    fn from(ut: Diagonal<T>) -> Self {
        ut.matrix
    }
}
