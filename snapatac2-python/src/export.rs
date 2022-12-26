use crate::utils::AnnDataLike;
use snapatac2_core::export::Exporter;

use std::ops::Deref;
use anndata::Backend;
use anndata_hdf5::H5;
use pyo3::prelude::*;
use std::{collections::{HashSet, HashMap}, path::PathBuf};
use anyhow::Result;

#[pyfunction]
pub fn export_bed(
    anndata: AnnDataLike,
    barcodes: Vec<&str>,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
) -> Result<HashMap<String, PathBuf>> {
    macro_rules! run {
        ($data:expr) => {
            $data.export_bed(
                Some(&barcodes), &group_by, selections, dir, prefix, suffix,
            )
        }
    }
    crate::with_anndata!(&anndata, run)
}

#[pyfunction]
pub fn export_bigwig(
    anndata: AnnDataLike,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    resolution: usize,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
) -> Result<HashMap<String, PathBuf>> {
    macro_rules! run {
        ($data:expr) => {
            $data.export_bigwig(&group_by, selections, resolution, dir, prefix, suffix)
        }
    }
    crate::with_anndata!(&anndata, run)
}