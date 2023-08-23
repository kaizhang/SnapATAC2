use crate::utils::{AnnDataLike, open_file};
use snapatac2_core::utils::clip_peak;
use snapatac2_core::{export::Exporter, utils::merge_peaks};
use snapatac2_core::preprocessing::SnapData;

use std::{ops::Deref, path::PathBuf};
use anndata::Backend;
use anndata_hdf5::H5;
use pyanndata::data::PyDataFrame;
use pyo3::prelude::*;
use bed_utils::bed::{BEDLike, GenomicRange, io::Reader, tree::BedTree};
use flate2::read::MultiGzDecoder;
use tempfile::Builder;
use log::info;
use std::fs::File;
use polars::{prelude::{DataFrame, NamedFrom}, series::Series};
use std::collections::HashSet;
use anyhow::Result;

#[pyfunction]
pub fn call_peaks(
    anndata: AnnDataLike,
    group_by: Vec<&str>,
    q_value: f64,
    nolambda: bool,
    shift: i64,
    extension_size: i64,
    selections: Option<HashSet<&str>>,
    blacklist: Option<PathBuf>,
    out_dir: Option<&str>,
) -> Result<PyDataFrame> {
    let dir = Builder::new().tempdir_in("./").unwrap();
    let temp_dir = out_dir.unwrap_or(dir.path().to_str().unwrap());

    macro_rules! run {
        ($data:expr) => {{
            let peaks = $data.call_peaks(
                q_value, &group_by, selections, temp_dir, "", ".NarrowPeak.gz",
                nolambda, shift, extension_size,
            )?;
            (peaks, $data.read_chrom_sizes()?)
        }}
    }
    let (peak_files, chrom_sizes) = crate::with_anndata!(&anndata, run);
    let peak_iter = peak_files.values().flat_map(|fl|
        Reader::new(MultiGzDecoder::new(File::open(fl).unwrap()), None)
        .into_records().map(Result::unwrap)
    );

    info!("Merging peaks...");
    let peaks:Vec<_> = if let Some(black) = blacklist {
        let black: BedTree<_> = Reader::new(open_file(black), None).into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ())).collect();
        merge_peaks(peak_iter.filter(|x| !black.is_overlapped(x)), 250)
            .flatten()
            .map(|x| clip_peak(x, &chrom_sizes))
            .collect()
    } else {
        merge_peaks(peak_iter, 250)
            .flatten()
            .map(|x| clip_peak(x, &chrom_sizes))
            .collect()
    };

    let n = peaks.len();

    let peaks_str = Series::new("Peaks",
        peaks.iter().map(|x| x.to_genomic_range().pretty_show()).collect::<Vec<_>>());
    let peaks_index: BedTree<usize> = peaks.into_iter().enumerate().map(|(i, x)| (x, i)).collect();
    let iter = peak_files.into_iter().map(|(key, fl)| {
        let mut values = vec![false; n];
        Reader::new(MultiGzDecoder::new(File::open(fl).unwrap()), None)
        .into_records().for_each(|x| {
            let bed: GenomicRange = x.unwrap();
            peaks_index.find(&bed).for_each(|(_, i)| values[*i] = true);
        });
        Series::new(key.as_str(), values)
    });

    let df = DataFrame::new(std::iter::once(peaks_str).chain(iter).collect())?;
    dir.close()?;
    Ok(df.into())
}