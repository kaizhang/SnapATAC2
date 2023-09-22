use crate::utils::{AnnDataLike, open_file};
use bed_utils::bed::{Strand, NarrowPeak};
use indicatif::{ProgressStyle, ProgressIterator};
use itertools::Itertools;
use polars::prelude::TakeRandom;
use snapatac2_core::utils::clip_peak;
use snapatac2_core::utils::merge_peaks;
use snapatac2_core::preprocessing::SnapData;

use std::{ops::Deref, path::PathBuf, collections::HashMap};
use anndata::Backend;
use anndata_hdf5::H5;
use pyanndata::data::PyDataFrame;
use pyo3::prelude::*;
use bed_utils::bed::{BEDLike, GenomicRange, io::Reader, tree::BedTree};
use polars::{prelude::{DataFrame, NamedFrom}, series::Series};
use std::collections::HashSet;
use anyhow::{Result, ensure};

#[pyfunction]
pub fn py_merge_peaks<'py>(
    peaks: HashMap<String, PyDataFrame>,
    chrom_sizes: HashMap<String, u64>,
) -> Result<PyDataFrame> {
    let peak_list: Vec<_> = peaks.into_iter().map(|(key, peaks)| {
        let ps = dataframe_to_narrow_peaks(&peaks.into())?;
        Ok((key, ps))
    }).collect::<Result<_>>()?;

    let chrom_sizes = chrom_sizes.into_iter().collect();
    let peaks: Vec<_> = merge_peaks(peak_list.iter().flat_map(|x| x.1.clone()), 250)
        .flatten()
        .map(|x| clip_peak(x, &chrom_sizes))
        .collect();

    let n = peaks.len();
    let peaks_str = Series::new("Peaks",
        peaks.iter().map(|x| x.to_genomic_range().pretty_show()).collect::<Vec<_>>());
    let peaks_index: BedTree<usize> = peaks.into_iter().enumerate().map(|(i, x)| (x, i)).collect();
    let iter = peak_list.iter().map(|(key, ps)| {
        let mut values = vec![false; n];
        ps.into_iter().for_each(|bed| {
            peaks_index.find(bed).for_each(|(_, i)| values[*i] = true);
        });
        Series::new(key.as_str(), values)
    });
    Ok(DataFrame::new(std::iter::once(peaks_str).chain(iter).collect())?.into())
}

#[pyfunction]
pub fn fetch_peaks<'py>(
    peaks: HashMap<String, &'py PyAny>,
    blacklist: Option<PathBuf>,
) -> Result<HashMap<String, PyDataFrame>> {
    let black: BedTree<_> = if let Some(black) = blacklist {
        Reader::new(open_file(black), None)
            .into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ())).collect()
    } else {
        Default::default()
    };
    peaks.into_iter().map(|(key, peaks)| {
        let ps = get_peaks(peaks)?.into_iter().filter(|x| !black.is_overlapped(x)).collect::<Vec<_>>();
        Ok((key, narrow_peak_to_dataframe(ps).unwrap().into()))
    }).collect::<Result<_>>()
}

/// Convert dataframe to narrowpeak
fn dataframe_to_narrow_peaks(df: &DataFrame) -> Result<Vec<NarrowPeak>> {
    let chroms = df.column("chrom").unwrap().utf8()?;
    let starts = df.column("start").unwrap().u64()?;
    let ends = df.column("end").unwrap().u64()?;
    let names = df.column("name").unwrap().utf8()?;
    let scores = df.column("score").unwrap().u16()?;
    let strands = df.column("strand").unwrap().utf8()?;
    let signal_values = df.column("signal_value").unwrap().f64()?;
    let p_values = df.column("p_value").unwrap().f64()?;
    let q_values = df.column("q_value").unwrap().f64()?;
    let peaks = df.column("peak").unwrap().u64()?;

    let mut narrow_peaks = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        narrow_peaks.push(NarrowPeak {
            chrom: chroms.get(i).map(|x| x.to_string()).unwrap(),
            start: starts.get(i).unwrap(),
            end: ends.get(i).unwrap(),
            name: names.get(i).map(|x| x.to_string()),
            score: scores.get(i).map(|x| x.try_into().unwrap()),
            strand: strands.get(i).map(|x| x.to_string().parse().unwrap()),
            signal_value: signal_values.get(i).unwrap(),
            p_value: p_values.get(i).unwrap(),
            q_value: q_values.get(i).unwrap(),
            peak: peaks.get(i).unwrap(),
        });
    }

    Ok(narrow_peaks)
}

fn narrow_peak_to_dataframe<I: IntoIterator<Item = NarrowPeak>>(narrow_peaks: I) -> Result<DataFrame> {
    // Separate Vec collections for each column
    let mut chroms = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut names = Vec::new();
    let mut scores = Vec::new();
    let mut strands = Vec::new();
    let mut signal_values = Vec::new();
    let mut p_values = Vec::new();
    let mut q_values = Vec::new();
    let mut peaks = Vec::new();

    for narrow_peak in narrow_peaks {
        chroms.push(narrow_peak.chrom);
        starts.push(narrow_peak.start);
        ends.push(narrow_peak.end);
        names.push(narrow_peak.name);
        scores.push(narrow_peak.score.map(|x| u16::from(x)));
        strands.push(narrow_peak.strand.map(|x| x.to_string()));
        signal_values.push(narrow_peak.signal_value);
        p_values.push(narrow_peak.p_value);
        q_values.push(narrow_peak.q_value);
        peaks.push(narrow_peak.peak);
    }

    // Create a DataFrame from the collected Vecs
    let df = DataFrame::new(vec![
        Series::new("chrom", chroms),
        Series::new("start", starts),
        Series::new("end", ends),
        Series::new("name", names),
        Series::new("score", scores),
        Series::new("strand", strands),
        Series::new("signal_value", signal_values),
        Series::new("p_value", p_values),
        Series::new("q_value", q_values),
        Series::new("peak", peaks),
    ])?;

    Ok(df)
}

fn get_peaks<'py>(peak_io_obj: &'py PyAny) -> Result<Vec<NarrowPeak>> {
    peak_io_obj.getattr("peaks")?.downcast::<pyo3::types::PyDict>().unwrap().iter().flat_map(|(chr, peaks)| {
        let chrom = String::from_utf8(chr.extract().unwrap()).unwrap();
        peaks.downcast::<pyo3::types::PyList>().unwrap().iter().map(|peak| {
            let start = peak.get_item("start")?.extract::<u64>()?;
            let end = peak.get_item("end")?.extract::<u64>()?;
            let fc = peak.get_item("fc")?.extract::<f64>()?;
            let score = peak.get_item("score")?.extract::<f64>()? * 10.0;
            let p_value= peak.get_item("pscore")?.extract::<f64>()?;
            let q_value= peak.get_item("qscore")?.extract::<f64>()?;
            let peak = peak.get_item("summit")?.extract::<u64>()? - start;
            Ok(NarrowPeak {
                chrom: chrom.clone(),
                start,
                end,
                name: None,
                score: Some((score as u16).min(1000).try_into().unwrap()),
                strand: None,
                signal_value: fc,
                p_value,
                q_value,
                peak,
            })
        }).collect::<Vec<_>>()
    }).collect()
}

#[pyfunction]
pub fn create_fwtrack_obj<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    group_by: Vec<&str>,
    max_frag_size: Option<u64>,
    selections: Option<HashSet<&str>>,
) -> Result<HashMap<String, &'py PyAny>> {
    macro_rules! run {
        ($data:expr) => {{
            _create_fwtrack_obj(py, $data, &group_by, max_frag_size, selections)
        }}
    }
    crate::with_anndata!(&anndata, run)
}
 
fn _create_fwtrack_obj<'py, D: SnapData>(
    py: Python<'py>,
    data: &D,
    group_by: &Vec<&str>,
    max_frag_size: Option<u64>,
    selections: Option<HashSet<&str>>,
) -> Result<HashMap<String, &'py PyAny>> {
    let macs = py.import("MACS3.Signal.FixWidthTrack")?;
    ensure!(data.n_obs() == group_by.len(), "lengths differ");
    let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
    if let Some(select) = selections { groups.retain(|x| select.contains(x)); }
    let mut fw_tracks = groups.into_iter().map(|x| {
        let obj = macs.getattr("FWTrack")?.call1((1000000,))?;
        Ok((x.to_owned(), obj))
    }).collect::<Result<HashMap<_, _>>>()?;

    let style = ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})")?;
    data.get_count_iter(500)?.into_raw().progress_with_style(style).try_for_each(|(vals, start, _)|
        vals.into_iter().enumerate().try_for_each::<_, Result<_>>(|(i, beds)| {
            if let Some(fwt) = fw_tracks.get_mut(group_by[start + i]) {
                beds.into_iter().for_each(|x| {
                    let chr = x.chrom().as_bytes();
                    match x.strand() {
                        None => if max_frag_size.map_or(true, |s| s >= x.len()) {
                            fwt.call_method1("add_loc", (chr, x.start(), 0)).unwrap();
                            fwt.call_method1("add_loc", (chr, x.end() - 1, 1)).unwrap();
                        },
                        Some(Strand::Forward) => {
                            fwt.call_method1("add_loc", (chr, x.start(), 0)).unwrap();
                        },
                        Some(Strand::Reverse) => {
                            fwt.call_method1("add_loc", (chr, x.end() - 1, 1)).unwrap();
                        },
                    }
                })
            }
            Ok(())
        })
    )?;
    for (_, fwt) in fw_tracks.iter_mut() {
        fwt.call_method0("finalize")?;
    }
    Ok(fw_tracks)
}