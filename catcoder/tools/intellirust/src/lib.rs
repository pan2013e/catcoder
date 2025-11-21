mod analyzer;
mod binding;

pub(crate) mod __private;

pub mod visitor;

pub use analyzer::Workspace;

use pyo3::prelude::*;

#[pymodule]
fn intellirust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    binding::add_class(m)?;
    Ok(())
}
