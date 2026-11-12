use std::ops::Mul;

use crate::algebra::{
    abstr::{Field, Scalar},
    linear::{matrix::UpperTriangular, vector::Vector},
};

impl<'a, 'b, T> Mul<&'b UpperTriangular<T>> for &'a UpperTriangular<T>
where
    T: Field + Scalar,
{
    type Output = UpperTriangular<T>;

    fn mul(self, rhs: &'b UpperTriangular<T>) -> Self::Output {
        // Implement optimized version
        UpperTriangular {
            matrix: (&self.matrix).mul(&rhs.matrix),
        }
    }
}

impl<'a, 'b, T> Mul<&'b UpperTriangular<T>> for &'a mut UpperTriangular<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut UpperTriangular<T>;

    fn mul(self, rhs: &'b UpperTriangular<T>) -> Self::Output {
        // Implement optimized version

        let _ = (&mut self.matrix).mul(&rhs.matrix);
        self
    }
}

impl<T> Mul<UpperTriangular<T>> for UpperTriangular<T>
where
    T: Field + Scalar,
{
    type Output = UpperTriangular<T>;

    fn mul(self, rhs: UpperTriangular<T>) -> Self::Output {
        // Implement optimized version
        UpperTriangular {
            matrix: self.matrix.mul(rhs.matrix),
        }
    }
}

/// Multiplies matrix by vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a UpperTriangular<T>
where
    T: Field + Scalar,
{
    type Output = Vector<T>;

    fn mul(self, v: &'b Vector<T>) -> Vector<T> {
        // Implement optimized version
        (&self.matrix).mul(v)
    }
}
