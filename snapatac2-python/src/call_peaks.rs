use crate::utils::{AnnDataLike, open_file};
use itertools::Itertools;
use snapatac2_core::{export::Exporter, utils::merge_peaks, preprocessing::SnapData};

use std::{ops::Deref, path::PathBuf, collections::HashMap};
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
use anyhow::{Result, ensure};

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
        ($data:expr) => {
            $data.call_peaks(
                q_value, &group_by, selections, temp_dir, "", ".NarrowPeak.gz",
                nolambda, shift, extension_size,
            )?
        }
    }
    let peak_files = crate::with_anndata!(&anndata, run);
    let peak_iter = peak_files.values().flat_map(|fl|
        Reader::new(MultiGzDecoder::new(File::open(fl).unwrap()), None)
        .into_records().map(Result::unwrap)
    );

    info!("Merging peaks...");
    let peaks:Vec<_> = if let Some(black) = blacklist {
        let black: BedTree<_> = Reader::new(open_file(black), None).into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ())).collect();
        merge_peaks(peak_iter.filter(|x| !black.is_overlapped(x)), 250).flatten().collect()
    } else {
        merge_peaks(peak_iter, 250).flatten().collect()
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

/*
fn create_fwtrack_obj<'py, D: SnapData>(
    py: Python<'py>,
    data: &D,
    barcodes: Option<&Vec<&str>>,
    group_by: &Vec<&str>,
    selections: Option<HashSet<&str>>,
) -> Result<HashMap<String, &'py PyAny>> {
    let macs = py.import("MACS2.IO.FixWidthTrack")?;
    ensure!(data.n_obs() == group_by.len(), "lengths differ");
    let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
    if let Some(select) = selections { groups.retain(|x| select.contains(x)); }
    let mut fw_tracks = groups.into_iter().map(|x| {
        let obj = macs.getattr("FWTrack")?.call1((1000000,))?;
        Ok((x.to_owned(), obj))
    }).collect::<Result<HashMap<_, _>>>()?;

    let style = ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})")?;
    self.raw_count_iter(500)?.into_chrom_values::<usize>().progress_with_style(style).try_for_each(|(vals, start, _)|
        vals.into_iter().enumerate().try_for_each::<_, Result<_>>(|(i, ins)| {
            if let Some((_, fwt)) = fw_tracks.get_mut(group_by[start + i]) {
                ins.into_iter().try_for_each(|x| {
                    let chr = x.chrom();
                    let 
                    let bed = match barcodes {
                        None => format!("{}\t{}\t{}", x.chrom(), x.start(), x.end()),
                        Some(barcodes_) => format!("{}\t{}\t{}\t{}", x.chrom(), x.start(), x.end(), barcodes_[start + i]),
                    };
                    (0..x.value).try_for_each(|_| writeln!(fl, "{}", bed))
                })?;
            }
            Ok(())
        })
    )?;
    Ok(fw_tracks)
}
*/