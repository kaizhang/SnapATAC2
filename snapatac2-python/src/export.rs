use pyo3::{
    prelude::*, PyTypeInfo,
    PyResult, Python,
    exceptions::PyTypeError,
};
use std::collections::{HashSet, HashMap};
use std::path::PathBuf;

use pyanndata::{AnnData, AnnDataSet};

use snapatac2_core::export::Exporter;

#[pyfunction]
pub fn export_bed<'py>(
    py: Python<'py>,
    data: &PyAny,
    barcodes: Vec<&str>,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    dir: &str,
    prefix: &str,
    suffix: &str,
) -> PyResult<HashMap<String, PathBuf>> {
    if data.is_instance(AnnData::type_object(py))? {
        let anndata: AnnData = data.extract()?;
        let x = anndata.0.inner().export_bed(
            Some(&barcodes), &group_by, selections, dir, prefix, suffix,
        ).unwrap();
        Ok(x)
    } else if data.is_instance(AnnDataSet::type_object(py))? {
        let anndata: AnnDataSet = data.extract()?;
        let x = anndata.0.inner().export_bed(
            Some(&barcodes), &group_by, selections, dir, prefix, suffix,
        ).unwrap();
        Ok(x)
    } else {
        return Err(PyTypeError::new_err("expecting an AnnData or AnnDataSet object"));
    }
}

#[pyfunction]
pub fn export_bigwig<'py>(
    py: Python<'py>,
    data: &PyAny,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    resolution: usize,
    dir: &str,
    prefix: &str,
    suffix: &str,
) -> PyResult<HashMap<String, PathBuf>> {
    if data.is_instance(AnnData::type_object(py))? {
        let anndata: AnnData = data.extract()?;
        let x = anndata.0.inner().export_bigwig(
            &group_by, selections, resolution, dir, prefix, suffix,
        ).unwrap();
        Ok(x)
    } else if data.is_instance(AnnDataSet::type_object(py))? {
        let anndata: AnnDataSet = data.extract()?;
        let x = anndata.0.inner().export_bigwig(
            &group_by, selections, resolution, dir, prefix, suffix,
        ).unwrap();
        Ok(x)
    } else {
        return Err(PyTypeError::new_err("expecting an AnnData or AnnDataSet object"));
    }
}