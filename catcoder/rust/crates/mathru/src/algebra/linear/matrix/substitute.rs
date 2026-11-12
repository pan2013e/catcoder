pub trait SubstituteForward<T> {
    fn substitute_forward(&self, b: T) -> Result<T, ()>;
}

pub trait SubstituteBackward<T> {
    fn substitute_backward(&self, b: T) -> Result<T, ()>;
}
