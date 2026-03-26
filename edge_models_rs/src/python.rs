use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use crate::base_models;
use crate::decision_table::{self, BaseModel};
use crate::graph::Graph;
use crate::loader;

#[pyclass]
pub struct PyGraph {
    pub inner: Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new(num_nodes: usize, edges: Vec<(u32, u32)>) -> Self {
        PyGraph {
            inner: Graph::from_edges(num_nodes, &edges),
        }
    }

    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    fn degree(&self, node: usize) -> u32 {
        self.inner.degree(node)
    }

    fn degrees(&self) -> Vec<u32> {
        self.inner.degrees()
    }

    fn density(&self) -> f64 {
        self.inner.density()
    }

    fn has_edge(&self, u: u32, v: u32) -> bool {
        self.inner.has_edge(u, v)
    }
}

/// Load a TU-format dataset directory. Returns a list of PyGraph.
#[pyfunction]
fn load_dataset(dir: &str, name: &str) -> PyResult<Vec<PyGraph>> {
    let graphs = loader::load_tu_dataset(Path::new(dir), name)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    Ok(graphs.into_iter().map(|g| PyGraph { inner: g }).collect())
}

#[pyfunction]
fn er_bpe(n: usize, m: usize, density: f64) -> f64 {
    base_models::er_bits_per_edge(n, m, density)
}

#[pyfunction]
fn pa_bpe(degrees: Vec<u32>, n: usize, m: usize) -> f64 {
    base_models::pa_bits_per_edge(&degrees, n, m)
}

#[pyfunction]
fn config_bpe(degrees: Vec<u32>, n: usize, m: usize) -> f64 {
    base_models::configuration_model_bits_per_edge(&degrees, n, m)
}

/// Run the decision table model. Returns a dict with results.
#[pyfunction]
#[pyo3(signature = (graph, base="er", min_frequency=10, n_samples=500000, seed=42))]
fn run_decision_table(
    graph: &PyGraph,
    base: &str,
    min_frequency: usize,
    n_samples: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let base_model = match base {
        "er" => BaseModel::ER,
        "pa" => BaseModel::PA,
        "config" => BaseModel::Config,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            "base must be 'er', 'pa', or 'config'"
        )),
    };

    let result = decision_table::decision_table_model(
        &graph.inner, base_model, min_frequency, n_samples, seed,
    );

    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("bits_per_edge", result.bits_per_edge)?;
        dict.set_item("table_entries", result.table_entries)?;
        dict.set_item("feature_bits", result.feature_bits)?;
        dict.set_item("table_cost_bits", result.table_cost_bits)?;
        dict.set_item("encoding_cost_bits", result.encoding_cost_bits)?;
        dict.set_item("base_only_bpe", result.base_only_bpe)?;
        dict.set_item("improvement", result.improvement)?;
        Ok(dict.into())
    })
}

/// Run all models on a graph. Returns a dict with all results.
#[pyfunction]
#[pyo3(signature = (graph, min_frequency=10, n_samples=500000, seed=42))]
fn compare_all(
    graph: &PyGraph,
    min_frequency: usize,
    n_samples: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let g = &graph.inner;
    let n = g.num_nodes();
    let m = g.num_edges();
    let degrees = g.degrees();
    let density = g.density();

    let er = base_models::er_bits_per_edge(n, m, density);
    let pa = base_models::pa_bits_per_edge(&degrees, n, m);
    let config = base_models::configuration_model_bits_per_edge(&degrees, n, m);

    let dt_er = decision_table::decision_table_model(
        g, BaseModel::ER, min_frequency, n_samples, seed);
    let dt_pa = decision_table::decision_table_model(
        g, BaseModel::PA, min_frequency, n_samples, seed);
    let dt_config = decision_table::decision_table_model(
        g, BaseModel::Config, min_frequency, n_samples, seed);

    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("n", n)?;
        dict.set_item("m", m)?;
        dict.set_item("density", density)?;
        dict.set_item("er_bpe", er)?;
        dict.set_item("pa_bpe", pa)?;
        dict.set_item("config_bpe", config)?;
        dict.set_item("dt_er_bpe", dt_er.bits_per_edge)?;
        dict.set_item("dt_pa_bpe", dt_pa.bits_per_edge)?;
        dict.set_item("dt_config_bpe", dt_config.bits_per_edge)?;
        dict.set_item("dt_er_table", dt_er.table_entries)?;
        dict.set_item("dt_pa_improvement", dt_pa.improvement)?;
        dict.set_item("dt_er_improvement", dt_er.improvement)?;
        Ok(dict.into())
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(er_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(pa_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(config_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(run_decision_table, m)?)?;
    m.add_function(wrap_pyfunction!(compare_all, m)?)?;
    Ok(())
}
