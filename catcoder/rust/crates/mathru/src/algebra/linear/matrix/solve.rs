pub trait Solve<T> {
    /// A * x = b
    fn solve(&self, rhs: &T) -> Result<T, ()>;
}
