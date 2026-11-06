use crate::algebra::linear::matrix::LowerTriangular;
use core::fmt::{self, Display};

impl<T> Display for LowerTriangular<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.matrix.fmt(f)
    }
}
