//! Vectors, matrices and their operations.

//! The linear algebra module supports BLAS/LAPACK if it is enabled via
//! features.
//! The interface is identical, but the BLAS/LAPACK backend may be somewhat
//! more efficient.
//! BLAS/LAPACK support can be enable in the `Cargo.toml` file like so:
//! ```text
//! [dependencies.mathru]
//! version = "^0.15"
//! default-features = false
//! features = "openblas"
//! ```
//! One of the following implementations for linear algebra can be activated
//! as a feature:
//!
//!   - native: Native Rust implementation(activated by default)
//!   - [openblas](https://www.openblas.net/): Optimized BLAS library
//!   - [netlib](https://www.netlib.org/): Collection of mathematical software,
//!     papers, and databases
//!   - [intel-mkl](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html):
//!     Intel Math Kernel Library
//!   - [accelerate](https://developer.apple.com/documentation/accelerate) Make
//!     large-scale mathematical computations and image calculations, optimized
//!     for high performance and low-energy consumption. (macOS only)

#[macro_use]
pub mod vector;
#[macro_use]
pub mod matrix;
#[cfg(feature = "lapack")]
pub mod blas;
#[cfg(feature = "lapack")]
pub mod lapack;
#[cfg(feature = "native")]
pub mod matrixmultiply;
