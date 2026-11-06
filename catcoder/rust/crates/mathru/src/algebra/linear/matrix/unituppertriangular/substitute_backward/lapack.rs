use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::substitute::SubstituteBackward;
use crate::algebra::linear::{
    matrix::{General, UnitUpperTriangular},
    vector::Vector,
};

impl<T> SubstituteBackward<Vector<T>> for UnitUpperTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_backward(&self, b: Vector<T>) -> Result<Vector<T>, ()> {
        let (b_m, b_n): (usize, usize) = b.dim();

        let mut b_data = b.convert_to_vec();
        T::xtrsm(
            'L',
            'U',
            'N',
            'U',
            b_m as i32,
            b_n as i32,
            T::one(),
            self.matrix.data.as_slice(),
            b_m as i32,
            b_data.as_mut_slice(),
            b_m as i32,
        );

        Ok(Vector::new_column(b_data))
    }
}

impl<T> SubstituteBackward<General<T>> for UnitUpperTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_backward(&self, b: General<T>) -> Result<General<T>, ()> {
        let (m, n) = self.matrix.dim();

        let mut c: General<T> = b;
        T::xtrsm(
            'L',
            'U',
            'N',
            'U',
            m as i32,
            n as i32,
            T::one(),
            self.matrix.data.as_slice(),
            m as i32,
            c.data.as_mut_slice(),
            m as i32,
        );

        Ok(c)
    }
}
