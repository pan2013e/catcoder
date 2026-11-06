//! Traits for abstract algebraic structures and implementations.

#[cfg(feature = "native")]
pub use self::scalar::MatrixMultiply;
#[cfg(feature = "lapack")]
pub use self::scalar::{Blas, Lapack};

pub use self::{
    abeliangroup::{AbelianGroup, AbelianGroupAdd, AbelianGroupMul},
    abs_diff_eq::{AbsDiff, AbsDiffEq},
    complex::Complex,
    field::Field,
    group::{Group, GroupAdd, GroupMul},
    identity::Identity,
    integer::Integer,
    loop_::Loop,
    magma::{Magma, MagmaAdd, MagmaMul},
    monoid::{Monoid, MonoidAdd, MonoidMul, One, Zero},
    natural::Natural,
    operator::{Addition, Multiplication, Operator},
    polynomial::Polynomial,
    quasigroup::Quasigroup,
    real::Real,
    relative_eq::{Relative, RelativeEq},
    ring::{CommutativeRing, Ring},
    scalar::Scalar,
    semigroup::{Semigroup, SemigroupAdd, SemigroupMul},
    sign::Sign,
};

mod abeliangroup;
mod abs_diff_eq;
mod bound;
pub mod cast;
mod field;
mod group;
mod identity;
mod integer;
mod loop_;
mod magma;
mod monoid;
mod natural;
mod operator;
mod quasigroup;
mod relative_eq;
mod ring;
mod scalar;
mod semigroup;
mod semiring;
mod sign;

mod polynomial;
#[macro_use]
//pub mod integer;
//pub mod natural;
mod real;
mod complex;
