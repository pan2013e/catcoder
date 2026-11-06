//! Distributions
//!
//! Each univariate distribution implements the trait Continuous or Discrete for
//! discrete distributions.
//!
//! Fore more information:
//! <https://en.wikipedia.org/wiki/Probability_distribution>
//!
mod bernoulli;
mod beta;
mod binomial;
mod chisquare;
mod distrib_impl;
mod exponential;
mod gamma;
mod log_normal;
mod normal;
mod poisson;
mod raisedcosine;
mod t;
mod uniform;

pub use self::{
    bernoulli::Bernoulli,
    beta::Beta,
    binomial::Binomial,
    chisquare::ChiSquare,
    distrib_impl::{Continuous, Discrete, Distribution},
    exponential::Exponential,
    gamma::Gamma,
    log_normal::LogNormal,
    normal::Normal,
    poisson::Poisson,
    raisedcosine::RaisedCosine,
    t::T,
    uniform::Uniform,
};
