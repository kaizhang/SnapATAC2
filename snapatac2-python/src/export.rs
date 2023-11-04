use crate::utils::AnnDataLike;
use snapatac2_core::{export::{Exporter, Normalization}, utils};

use std::ops::Deref;
use anndata::Backend;
use anndata_hdf5::H5;
use pyo3::prelude::*;
use std::{collections::{HashSet, HashMap}, path::PathBuf};
use anyhow::Result;
use bed_utils::bed::{GenomicRange, io::Reader, tree::BedTree};
use std::str::FromStr;

#[pyfunction]
pub fn export_fragments(
    anndata: AnnDataLike,
    barcodes: Vec<&str>,
    group_by: Vec<&str>,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
    selections: Option<HashSet<&str>>,
    compression: Option<&str>,
    compression_level: Option<u32>,
) -> Result<HashMap<String, PathBuf>> {
    macro_rules! run {
        ($data:expr) => {
            $data.export_fragments(
                Some(&barcodes), &group_by, selections, dir, prefix, suffix,
                compression.map(|x| utils::Compression::from_str(x).unwrap()), compression_level
            )
        }
    }
    crate::with_anndata!(&anndata, run)
}

#[pyfunction]
pub fn export_bedgraph(
    anndata: AnnDataLike,
    group_by: Vec<&str>,
    resolution: usize,
    dir: PathBuf,
    prefix: &str,
    suffix:&str,
    selections: Option<HashSet<&str>>,
    smooth_length: Option<u64>,
    blacklist: Option<PathBuf>,
    normalization: Option<&str>,
    ignore_for_norm: Option<HashSet<&str>>,
    min_frag_length: Option<u64>,
    max_frag_length: Option<u64>,
    compression: Option<&str>,
    compression_level: Option<u32>,
    temp_dir: Option<PathBuf>,
) -> Result<HashMap<String, PathBuf>> {
    let black: Option<BedTree<()>> = blacklist.map(|black| {
        Reader::new(utils::open_file_for_read(black), None)
            .into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ()))
            .collect()
    });

    let normalization = normalization.map(|x| Normalization::from_str(x).unwrap());

    macro_rules! run {
        ($data:expr) => {
            $data.export_bedgraph(
                &group_by, selections, resolution, smooth_length, black.as_ref(), normalization,
                ignore_for_norm.as_ref(), min_frag_length, max_frag_length, dir, prefix,
                suffix, compression.map(|x| utils::Compression::from_str(x).unwrap()), compression_level, temp_dir
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