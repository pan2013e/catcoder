pub trait Determinant<T> {
    /// Calculates the determinant
    ///
    fn det(&self) -> T;
}
