use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::substitute::SubstituteForward;
use crate::algebra::linear::matrix::{General, LowerTriangular};
use crate::algebra::linear::vector::Vector;

impl<T> SubstituteForward<Vector<T>> for LowerTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_forward(&self, b: Vector<T>) -> Result<Vector<T>, ()> {
        let (b_m, b_n): (usize, usize) = b.dim();
        let mut b_data = b.convert_to_vec();
        let (m, _n) = self.matrix.dim();

        T::xtrsm(
            'L',
            'L',
            'N',
            'N',
            b_m as i32,
            b_n as i32,
            T::one(),
            self.matrix.data.as_slice(),
            m as i32,
            b_data.as_mut_slice(),
            b_m as i32,
        );

        Ok(Vector::new_column(b_data))
    }
}

impl<T> SubstituteForward<General<T>> for LowerTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_forward(&self, b: General<T>) -> Result<General<T>, ()> {
        let mut c: General<T> = b;
        let (m, _n) = self.matrix.dim();

        T::xtrsm(
            'L',
            'L',
            'N',
            'N',
            c.m as i32,
            c.n as i32,
            T::one(),
            self.matrix.data.as_slice(),
            m as i32,
            c.data.as_mut_slice(),
            c.m as i32,
        );

        Ok(c)
    }
}
