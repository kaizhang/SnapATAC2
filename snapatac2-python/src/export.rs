use crate::utils::{AnnDataObj, extract_anndata};
use snapatac2_core::export::Exporter;

use pyo3::{prelude::*, Python};
use std::{collections::{HashSet, HashMap}, path::PathBuf};
use anyhow::Result;

#[pyfunction]
pub fn export_bed<'py>(
    py: Python<'py>,
    anndata: &PyAny,
    barcodes: Vec<&str>,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
) -> Result<HashMap<String, PathBuf>> {
    match extract_anndata(py, anndata)? {
        AnnDataObj::AnnData(data) => data.inner().export_bed(
            Some(&barcodes), &group_by, selections, dir, prefix, suffix,
        ),
        AnnDataObj::AnnDataSet(data) => data.inner().export_bed(
            Some(&barcodes), &group_by, selections, dir, prefix, suffix,
        ),
        AnnDataObj::PyAnnData(data) => data.export_bed(
            Some(&barcodes), &group_by, selections, dir, prefix, suffix,
        ),
    }
}

#[pyfunction]
pub fn export_bigwig<'py>(
    py: Python<'py>,
    anndata: &PyAny,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    resolution: usize,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
) -> Result<HashMap<String, PathBuf>> {
    match extract_anndata(py, anndata)? {
        AnnDataObj::AnnData(data) => data.inner()
            .export_bigwig(&group_by, selections, resolution, dir, prefix, suffix),
        AnnDataObj::AnnDataSet(data) => data.inner()
            .export_bigwig(&group_by, selections, resolution, dir, prefix, suffix),
        AnnDataObj::PyAnnData(data) => data
            .export_bigwig(&group_by, selections, resolution, dir, prefix, suffix),
    }
}