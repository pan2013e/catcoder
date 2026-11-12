//! This Rust library provides a wide range of mathematical routines such as
//! linear algebra, differential equations, integration, interpolation, statistics and numerical optimization.
//!
//! # Getting Started
//!
//! Add a new dependency to your `Cargo.toml` file.
//! ```text
//! [dependencies]
//! mathru = "0.15"
//! ```
//! You can check, if it works with a simple program like this:
//!```rust
//! use mathru::vector;
//! use mathru::algebra::linear::{vector::Vector, matrix::General};
//! use mathru::algebra::linear::matrix::{Solve};
//!
//! fn main() {
//!     // set inputs:
//!     //
//!     // a = [1.0  -3.0]    b = [1.0]
//!     //     [2.0  -7.0]        [3.0]
//!     let a: General<f64> = General::new(2, 2, vec![1.0, 2.0, -3.0, -7.0]);
//!     let b: Vector<f64> = vector![1.0; 3.0];
//!
//!     // Solve a * x = b
//!     let x: Vector<f64> = a.solve(&b).unwrap();
//!     assert_eq!(&a * &x, b);
//!
//!     // Print result
//!     println!("x = [{} {}]", x[0], x[1]);
//! }
//! ```
//!
//! ## BLAS/LAPACK Support
//!
//! Mathru has a native Rust implementation of all its functions.
//! However, linear algebra functions are also implemented with a BLAS/LAPACK
//! backend.
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

#[cfg(feature = "lapack")]
extern crate blas;
#[cfg(feature = "lapack")]
extern crate blas_src;
#[cfg(feature = "lapack")]
extern crate lapack;

#[macro_use]
pub mod algebra;
pub mod analysis;
pub mod elementary;
pub mod optimization;
pub mod special;
pub mod statistics;
