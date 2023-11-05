use crate::utils::AnnDataLike;
use snapatac2_core::{export::{Exporter, Normalization, CoverageOutputFormat}, utils};

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
pub fn export_coverage(
    anndata: AnnDataLike,
    group_by: Vec<&str>,
    resolution: usize,
    dir: PathBuf,
    prefix: &str,
    suffix:&str,
    output_format: &str,
    selections: Option<HashSet<&str>>,
    blacklist: Option<PathBuf>,
    normalization: Option<&str>,
    ignore_for_norm: Option<HashSet<&str>>,
    min_frag_length: Option<u64>,
    max_frag_length: Option<u64>,
    compression: Option<&str>,
    compression_level: Option<u32>,
    temp_dir: Option<PathBuf>,
    num_threads: Option<usize>,
) -> Result<HashMap<String, PathBuf>> {
    let black: Option<BedTree<()>> = blacklist.map(|black| {
        Reader::new(utils::open_file_for_read(black), None)
            .into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ()))
            .collect()
    });

    let normalization = normalization.map(|x| Normalization::from_str(x).unwrap());
    let output_format = CoverageOutputFormat::from_str(output_format).unwrap();

    macro_rules! run {
        ($data:expr) => {
            $data.export_coverage(
                &group_by, selections, resolution, black.as_ref(), normalization,
                ignore_for_norm.as_ref(), min_frag_length, max_frag_length, dir, prefix,
                suffix, output_format, compression.map(|x| utils::Compression::from_str(x).unwrap()),
                compression_level, temp_dir, num_threads,
            )
        }
    }
    crate::with_anndata!(&anndata, run)
}