pub mod graph;
pub mod loader;
pub mod base_models;
pub mod features;
pub mod decision_table;
mod python;

use pyo3::prelude::*;

#[pymodule]
fn edge_models_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
