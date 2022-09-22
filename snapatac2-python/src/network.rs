use crate::utils::open_file;

use pyo3::{
    prelude::*,
    PyResult, Python,
    basic::CompareOp,
    types::{PyDict, PyInt, PyFloat},
};

use snapatac2_core::{
    network::link_region_to_promoter,
    utils::gene::{Promoters, read_transcripts},
};
use bed_utils::bed::GenomicRange;
use std::{
    str::FromStr,
    default::Default,
    collections::hash_map::DefaultHasher,
    collections::HashMap,
    io::BufReader,
    hash::{Hash, Hasher},
};

#[pyclass(mapping, module = "snapatac2", subclass)]
#[derive(Debug, Default, PartialEq)]
pub(crate) struct LinkData {
    #[pyo3(get, set)]
    distance: u64,

    #[pyo3(get, set)]
    regr_score: Option<f64>,

    #[pyo3(get, set)]
    correlation_score: Option<f64>,
}

#[pymethods]
impl LinkData {
    #[new]
    #[args(
        distance = "0",
        regr_score = "None",
        correlation_score = "None",
    )]
    fn new(distance: u64, regr_score: Option<f64>, correlation_score: Option<f64>) -> Self {
        LinkData { distance, regr_score, correlation_score }
    }
 
    fn __repr__(&self) -> String { format!("{:?}", self) }

    fn __str__(&self) -> String { self.__repr__() }

    fn __richcmp__(&self, other: PyRef<LinkData>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        match op {
            CompareOp::Eq => (PartialEq::eq(self, &other)).into_py(py),
            CompareOp::Ne => (PartialEq::ne(self, &other)).into_py(py),
            _ => py.NotImplemented(),
        }
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let dict_state = state.cast_as::<PyDict>(py)?;
        self.distance = dict_state.get_item("distance").unwrap().extract()?;
        self.regr_score = dict_state.get_item("regr_score").unwrap().extract()?;
        self.correlation_score = dict_state.get_item("correlation_score")
            .unwrap().extract()?;
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let out_dict = PyDict::new(py);
        out_dict.set_item("distance", self.distance)?;
        out_dict.set_item("regr_score", self.regr_score)?;
        out_dict.set_item("correlation_score", self.correlation_score)?;
        Ok(out_dict.into())
    }
}

#[pyclass(mapping, module = "snapatac2", subclass)]
#[derive(Hash, Eq, PartialEq, Debug, Default)]
pub(crate) struct NodeData {
    #[pyo3(get, set)]
    id: String,

    #[pyo3(get, set)]
    r#type: String,
}

#[pymethods]
impl NodeData {
    #[new]
    #[args(
        id = "\"\"",
        ty = "\"\"",
    )]
    fn new(id: &str, ty: &str) -> Self {
        NodeData { id: id.to_string(), r#type: ty.to_string() }
    }
 
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __richcmp__(&self, other: PyRef<NodeData>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        match op {
            CompareOp::Eq => (PartialEq::eq(self, &other)).into_py(py),
            CompareOp::Ne => (PartialEq::ne(self, &other)).into_py(py),
            _ => py.NotImplemented(),
        }
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let dict_state = state.cast_as::<PyDict>(py)?;
        self.id = dict_state.get_item("id").unwrap().extract()?;
        self.r#type = dict_state.get_item("type").unwrap().extract()?;
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let out_dict = PyDict::new(py);
        out_dict.set_item("id", &self.id)?;
        out_dict.set_item("type", &self.r#type)?;
        Ok(out_dict.into())
    }

    fn __repr__(&self) -> String { format!("{:?}", self) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyfunction]
pub(crate) fn link_region_to_gene(
    regions: Vec<String>,
    annot_fl: &str,
    upstream: u64,
    downstream: u64,
    id_type: &str,
    coding_gene_only: bool,
) -> HashMap<NodeData, Vec<(NodeData, LinkData)>>
{
    let promoters = Promoters::new(
        read_transcripts(BufReader::new(open_file(annot_fl))).into_iter()
            .filter(|x| if coding_gene_only { x.is_coding.unwrap_or(true) } else { true })
            .collect(),
        upstream,
        downstream,
        false,
    );
    let regions_: Vec<GenomicRange> = regions.into_iter().map(|x| GenomicRange::from_str(&x).unwrap()).collect();
    link_region_to_promoter(&regions_, &promoters,).get_linkages(id_type)
        .into_iter().map(|(k, links)| {
            let data = links.into_iter().map(|(i, x)| {
                let mut link_data = LinkData::default();
                link_data.distance = x;
                (NodeData { id: i, r#type: "region".to_string()}, link_data)
            }).collect();
            (NodeData { id: k.to_string(), r#type: "gene".to_string() }, data)
        }).collect()
}
