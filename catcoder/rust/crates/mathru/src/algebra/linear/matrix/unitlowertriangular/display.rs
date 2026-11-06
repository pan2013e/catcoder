use crate::algebra::linear::matrix::UnitLowerTriangular;
use core::fmt::{self, Display};

impl<T> Display for UnitLowerTriangular<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.matrix.fmt(f)
    }
}
