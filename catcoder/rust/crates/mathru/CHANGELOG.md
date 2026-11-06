# Change Log
All notable changes starting with the version 0.6.9 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com) and this project adheres to [Semantic Versioning](https://semver.org).

## [0.15.2]
### Changed
- Move project from https://gitlab.com/matthiaseiholzer/mathru to  https://gitlab.com/rustmath/mathru

### Added
- Implement CI task to collect code coverage information    
- Implement unit tests

### Fixed
- Bug in 'gamma_ur_inv' and 'error' function

## [0.15.1]
### Changed
- Implement benchmarks for basic vector and matrix operations
- Improve performance of basic vector and matrix operations
- Improve differential equation solver speed
- Update Criterion dependency to version ^0.5

## [0.15.0]
### Added
- Bessel polynomials
- Implement special matrix types like diagonal, lower triangular etc., which allow for fast computation with specialized routines for particular matrix types

### Changed
- Upgrade blas-src to version 0.9.0
- Upgrade lapack-src to version 0.9.0
- Update KaTeX dependency to version 0.16.7

### Fixed
- Bug in calculating the determinant of a permuation matrix [Issue #15](https://gitlab.com/rustmath/mathru/-/issues/15)

## [0.14.0] - 2022-10-02
### Added
- ExplicitInitialValueProblem and corresponding ExplicitInitialValueProblemBuilder as structs for explicit IVPs
- Cubic splines

### Fixed
- Some clippy warnings

### Changed
- Update KaTeX dependency

## [0.13.0] - 2022-06-19
### Fixed
- Fix bug as reported in [Issue #8](https://gitlab.com/rustmath/mathru/-/issues/8) and [Issue #12](https://gitlab.com/rustmath/mathru/-/issues/12)
### Changed
- Code refactoring

## [0.12.0] - 2022-03-13.
### Added
- Implement convert-mint feature that can be enabled to convert from and to types of the mint crate.

## [0.11.3] - 2022-03-05

### Fixed
- Fix failing docs.rs build

## [0.11.2]

### Added
- Legendre polynomials
- Chebyshev polynomials first & second kind

## [0.11.1]

## [0.11.0]

### Added
- Implement Newton-Cotes and Gauss-Legendre as integration methods

### Changed
- Code refactoring
- Performance improvements


## [0.10.1]
- Fixed Bug in LU decomposition [Issue #7](https://gitlab.com/rustmath/mathru/-/issues/7)

## [0.10.0]
- Code refactoring
- Bug fix in QR decomposition algorithm

## [0.9.1]
- Update README.md and lib.rs to newest version

## [0.9.0]
- Implement additional ODE solvers
- Update dependencies
- Implement explicit ODE solvers with Butcher tableaus

## [0.8.4]
- Replace the out-of-place transpose algorithm with an in-place algorithm, therewith it matches to the documentation

## [0.8.3]
- Make serde dependency optional

## [0.8.2]
- Update dependencies
- Implement inverse of lower/upper regularized incomplete gamma function
- Improve accuracy of of the quantile function of the Chi-square distribution

## [0.8.1]
- README corrections

## [0.8.0]
- Implement polynomial

## [0.7.4]
- Improve documentation
- Update dependencies

## [0.7.3]
- Column/row iterators implemented
- Serde support

## [0.7.2]
- assert_eq! for floats replaced with assert_relative_eq! and assert_diff_abs_eq!
- CI pipeline for blas/lapack backends fixed
- Dev dependencies updated

## [0.7.1]
- Invalid URLs in README.md fixed

## [0.7.0]
- Using SemVer for release versioning
- log-normal distribution
- Different changes on beta, gamma and error functions

## [0.6.10]
- Native Rust code, Openblas, Netlib, Intel-Mkl and Accelerate are now usable as linear algebra libraries

## [0.6.9]
- Eigen decomposition is implemented
- Implicit Euler
- Backward differentiation formula
