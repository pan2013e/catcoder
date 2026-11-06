use crate::algebra::linear::matrix::UpperTriangular;
use core::fmt::{self, Display};

impl<T> Display for UpperTriangular<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.matrix.fmt(f)
    }
}
