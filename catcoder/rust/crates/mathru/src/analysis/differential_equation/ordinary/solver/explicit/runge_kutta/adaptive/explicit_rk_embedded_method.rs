use crate::analysis::differential_equation::ordinary::solver::explicit::runge_kutta::adaptive::ExplicitRKEmbedded;

pub trait ExplicitRKEmbeddedMethod<T> {
    fn tableau(&self) -> &ExplicitRKEmbedded<T>;
}
