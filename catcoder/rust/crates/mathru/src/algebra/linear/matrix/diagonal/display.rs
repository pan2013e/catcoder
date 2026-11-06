use core::fmt;
use std::fmt::Display;

use super::Diagonal;

impl<T> Display for Diagonal<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.matrix.fmt(f)
    }
}
