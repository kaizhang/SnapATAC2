use crate::utils::{read_genomic_ranges, AnnDataLike};
use snapatac2_core::{
    export::{CoverageOutputFormat, Exporter, Normalization},
    utils,
};

use anndata::Backend;
use anndata_hdf5::H5;
use anyhow::Result;
use bed_utils::bed::{io::Reader, map::GIntervalMap, GenomicRange};
use pyo3::{prelude::*, pybacked::PyBackedStr};
use std::ops::Deref;
use std::str::FromStr;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

#[pyfunction]
#[pyo3(signature = (anndata, barcodes, group_by, dir, prefix, suffix, selections=None,
       min_frag_length=None, max_frag_length=None, compression=None, compression_level=None))]
pub fn export_fragments(
    anndata: AnnDataLike,
    barcodes: Vec<PyBackedStr>,
    group_by: Vec<PyBackedStr>,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
    selections: Option<HashSet<PyBackedStr>>,
    min_frag_length: Option<u64>,
    max_frag_length: Option<u64>,
    compression: Option<&str>,
    compression_level: Option<u32>,
) -> Result<HashMap<String, PathBuf>> {
    let barcodes = barcodes.iter().map(|x| x.as_ref()).collect();
    let group_by = group_by.iter().map(|x| x.as_ref()).collect();
    let selections = selections
        .as_ref()
        .map(|s| s.iter().map(|x| x.as_ref()).collect());
    macro_rules! run {
        ($data:expr) => {
            $data.export_fragments(
                Some(&barcodes),
                &group_by,
                selections,
                min_frag_length,
                max_frag_length,
                dir,
                prefix,
                suffix,
                compression.map(|x| utils::Compression::from_str(x).unwrap()),
                compression_level,
            )
        };
    }
    crate::with_anndata!(&anndata, run)
}

#[pyfunction]
#[pyo3(signature = (anndata, group_by, resolution, dir, prefix, suffix, output_format,
       strategy, selections=None, blacklist=None, normalization=None, include_for_norm=None,
       exclude_for_norm=None, min_frag_length=None, max_frag_length=None, smooth_base=None,
       compression=None, compression_level=None, temp_dir=None, num_threads=None))]
pub fn export_coverage(
    anndata: AnnDataLike,
    group_by: Vec<PyBackedStr>,
    resolution: usize,
    dir: PathBuf,
    prefix: &str,
    suffix: &str,
    output_format: &str,
    strategy: &str,
    selections: Option<HashSet<PyBackedStr>>,
    blacklist: Option<PathBuf>,
    normalization: Option<&str>,
    include_for_norm: Option<&Bound<'_, PyAny>>,
    exclude_for_norm: Option<&Bound<'_, PyAny>>,
    min_frag_length: Option<u64>,
    max_frag_length: Option<u64>,
    smooth_base: Option<u64>,
    compression: Option<&str>,
    compression_level: Option<u32>,
    temp_dir: Option<PathBuf>,
    num_threads: Option<usize>,
) -> Result<HashMap<String, PathBuf>> {
    let group_by = group_by.iter().map(|x| x.as_ref()).collect();
    let selections = selections
        .as_ref()
        .map(|s| s.iter().map(|x| x.as_ref()).collect());
    let include_for_norm = include_for_norm.as_ref().map(|s| {
        read_genomic_ranges(s)
            .unwrap()
            .into_iter()
            .map(|x| (x, ()))
            .collect()
    });
    let exclude_for_norm = exclude_for_norm.as_ref().map(|s| {
        read_genomic_ranges(s)
            .unwrap()
            .into_iter()
            .map(|x| (x, ()))
            .collect()
    });

    let black: Option<GIntervalMap<()>> = blacklist.map(|black| {
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
                &group_by,
                selections,
                resolution,
                black.as_ref(),
                normalization,
                include_for_norm.as_ref(),
                exclude_for_norm.as_ref(),
                min_frag_length,
                max_frag_length,
                strategy.try_into()?,
                smooth_base,
                dir,
                prefix,
                suffix,
                output_format,
                compression.map(|x| utils::Compression::from_str(x).unwrap()),
                compression_level,
                temp_dir,
                num_threads,
            )
        };
    }
    crate::with_anndata!(&anndata, run)
}
