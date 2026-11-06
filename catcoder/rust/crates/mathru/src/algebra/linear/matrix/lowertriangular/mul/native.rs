use std::ops::Mul;

use crate::algebra::{
    abstr::{Field, Scalar},
    linear::{matrix::LowerTriangular, vector::Vector},
};

impl<'a, 'b, T> Mul<&'b LowerTriangular<T>> for &'a LowerTriangular<T>
where
    T: Field + Scalar,
{
    type Output = LowerTriangular<T>;

    fn mul(self, rhs: &'b LowerTriangular<T>) -> Self::Output {
        // Implement optimized version
        LowerTriangular {
            matrix: (&self.matrix).mul(&rhs.matrix),
        }
    }
}

impl<'a, 'b, T> Mul<&'b LowerTriangular<T>> for &'a mut LowerTriangular<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut LowerTriangular<T>;

    fn mul(self, rhs: &'b LowerTriangular<T>) -> Self::Output {
        // Implement optimized version

        let _ = (&mut self.matrix).mul(&rhs.matrix);
        self
    }
}

impl<T> Mul<LowerTriangular<T>> for LowerTriangular<T>
where
    T: Field + Scalar,
{
    type Output = LowerTriangular<T>;

    fn mul(self, rhs: LowerTriangular<T>) -> Self::Output {
        // Implement optimized version
        LowerTriangular {
            matrix: self.matrix.mul(rhs.matrix),
        }
    }
}

/// Multiplies matrix by vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a LowerTriangular<T>
where
    T: Field + Scalar,
{
    type Output = Vector<T>;

    fn mul(self, v: &'b Vector<T>) -> Vector<T> {
        // Implement optimized version
        (&self.matrix).mul(v)
    }
}
