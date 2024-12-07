use crate::utils::AnnDataLike;

use anyhow::{bail, Context};
use bed_utils::bed::{BroadPeak, NarrowPeak, Strand};
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use snapatac2_core::utils::{self, Compression};
use snapatac2_core::{
    SnapData,
    preprocessing::Fragment,
    utils::{clip_peak, merge_peaks, open_file_for_write},
};

use anndata::Backend;
use anndata_hdf5::H5;
use anyhow::{ensure, Result};
use bed_utils::bed::{
    io::Reader,
    map::{GIntervalIndexSet, GIntervalMap},
    BEDLike, GenomicRange,
};
use polars::{
    prelude::{DataFrame, NamedFrom},
    series::Series,
};
use pyo3::{prelude::*, pybacked::PyBackedStr};
use pyo3_polars::PyDataFrame;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashSet;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::{collections::HashMap, ops::Deref, path::PathBuf};

#[pyfunction]
pub fn py_merge_peaks<'py>(
    peaks: HashMap<String, PyDataFrame>,
    chrom_sizes: HashMap<String, u64>,
    half_width: u64,
) -> Result<PyDataFrame> {
    let peak_list: Vec<_> = peaks
        .into_iter()
        .map(|(key, peaks)| {
            let ps = dataframe_to_narrow_peaks(&peaks.into())?;
            Ok((key, ps))
        })
        .collect::<Result<_>>()?;

    let chrom_sizes = chrom_sizes.into_iter().collect();
    let peaks: Vec<_> = merge_peaks(peak_list.iter().flat_map(|x| x.1.clone()), half_width)
        .flatten()
        .map(|x| clip_peak(x, &chrom_sizes))
        .collect();

    let n = peaks.len();
    let peaks_str = Series::new(
        "Peaks",
        peaks
            .iter()
            .map(|x| x.to_genomic_range().pretty_show())
            .collect::<Vec<_>>(),
    );
    let peaks_index: GIntervalIndexSet = peaks.into_iter().collect();
    let iter = peak_list.iter().map(|(key, ps)| {
        let mut values = vec![false; n];
        ps.into_iter().for_each(|bed| {
            peaks_index
                .find_index_of(bed)
                .for_each(|i| values[i] = true);
        });
        Series::new(key.as_str(), values)
    });
    Ok(PyDataFrame(DataFrame::new(
        std::iter::once(peaks_str).chain(iter).collect(),
    )?))
}

#[pyfunction]
pub fn find_reproducible_peaks<'py>(
    peaks: &Bound<'py, PyAny>,
    replicates: Vec<Bound<'py, PyAny>>,
    blacklist: Option<PathBuf>,
) -> Result<PyDataFrame> {
    let black: GIntervalMap<_> = if let Some(black) = blacklist {
        Reader::new(utils::open_file_for_read(black), None)
            .into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ()))
            .collect()
    } else {
        GIntervalMap::new()
    };
    let replicates = replicates
        .into_iter()
        .map(|x| {
            GIntervalMap::from_iter(get_genomic_ranges(&x).unwrap().into_iter().map(|x| (x, ())))
        })
        .collect::<Vec<_>>();

    if let Ok(peaks) = get_narrow_peaks(peaks) {
        Ok(PyDataFrame(narrow_peak_to_dataframe(
            peaks
                .into_iter()
                .filter(|x| {
                    !black.is_overlapped(x) && replicates.iter().all(|y| y.is_overlapped(x))
                })
                .collect::<Vec<_>>(),
        )?))
    } else {
        let peaks = get_broad_peaks(peaks)?;
        Ok(PyDataFrame(broad_peak_to_dataframe(
            peaks
                .into_iter()
                .filter(|x| {
                    !black.is_overlapped(x) && replicates.iter().all(|y| y.is_overlapped(x))
                })
                .collect::<Vec<_>>(),
        )?))
    }
}

#[pyfunction]
pub fn fetch_peaks<'py>(
    peaks: HashMap<String, Bound<'py, PyAny>>,
    blacklist: Option<PathBuf>,
) -> Result<HashMap<String, PyDataFrame>> {
    let black: GIntervalMap<_> = if let Some(black) = blacklist {
        Reader::new(utils::open_file_for_read(black), None)
            .into_records::<GenomicRange>()
            .map(|x| (x.unwrap(), ()))
            .collect()
    } else {
        GIntervalMap::new()
    };
    peaks
        .into_iter()
        .map(|(key, peaks)| {
            let ps = get_narrow_peaks(&peaks)?
                .into_iter()
                .filter(|x| !black.is_overlapped(x))
                .collect::<Vec<_>>();
            Ok((key, PyDataFrame(narrow_peak_to_dataframe(ps).unwrap())))
        })
        .collect::<Result<_>>()
}

/// Convert dataframe to narrowpeak
fn dataframe_to_narrow_peaks(df: &DataFrame) -> Result<Vec<NarrowPeak>> {
    let chroms = df.column("chrom").unwrap().str()?;
    let starts = df.column("start").unwrap().u64()?;
    let ends = df.column("end").unwrap().u64()?;
    let names = df.column("name").unwrap().str()?;
    let scores = df.column("score").unwrap().u16()?;
    let strands = df.column("strand").unwrap().str()?;
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
            name: names
                .get(i)
                .and_then(|x| if x == "." { None } else { Some(x.to_string()) }),
            score: scores.get(i).map(|x| (x as u16).try_into().unwrap()),
            strand: strands.get(i).and_then(|x| {
                if x == "." {
                    None
                } else {
                    Some(x.parse().unwrap())
                }
            }),
            signal_value: signal_values.get(i).unwrap(),
            p_value: p_values.get(i),
            q_value: q_values.get(i),
            peak: peaks.get(i).unwrap(),
        });
    }

    Ok(narrow_peaks)
}

fn narrow_peak_to_dataframe<I: IntoIterator<Item = NarrowPeak>>(
    narrow_peaks: I,
) -> Result<DataFrame> {
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
        names.push(narrow_peak.name.unwrap_or(".".to_string()));
        scores.push(narrow_peak.score.map(|x| u16::from(x)));
        strands.push(
            narrow_peak
                .strand
                .map_or(".".to_string(), |x| x.to_string()),
        );
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

fn broad_peak_to_dataframe<I: IntoIterator<Item = BroadPeak>>(
    peaks: I,
) -> Result<DataFrame> {
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

    for peak in peaks {
        chroms.push(peak.chrom);
        starts.push(peak.start);
        ends.push(peak.end);
        names.push(peak.name.unwrap_or(".".to_string()));
        scores.push(peak.score.map(|x| u16::from(x)));
        strands.push(
            peak
                .strand
                .map_or(".".to_string(), |x| x.to_string()),
        );
        signal_values.push(peak.signal_value);
        p_values.push(peak.p_value);
        q_values.push(peak.q_value);
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
    ])?;

    Ok(df)
}



fn get_genomic_ranges<'py>(peak_io_obj: &Bound<'py, PyAny>) -> Result<Vec<GenomicRange>> {
    peak_io_obj
        .getattr("peaks")?
        .downcast::<pyo3::types::PyDict>()
        .unwrap()
        .iter()
        .flat_map(|(chr, peaks)| {
            let chrom = String::from_utf8(chr.extract().unwrap()).unwrap();
            peaks
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .iter()
                .map(|peak| {
                    let start = peak.get_item("start")?.extract::<u64>()?;
                    let end = peak.get_item("end")?.extract::<u64>()?;
                    Ok(GenomicRange::new(chrom.clone(), start, end))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn get_narrow_peaks<'py>(peak_io_obj: &Bound<'py, PyAny>) -> Result<Vec<NarrowPeak>> {
    peak_io_obj
        .getattr("peaks")?
        .downcast::<pyo3::types::PyDict>()
        .unwrap()
        .iter()
        .flat_map(|(chr, peaks)| {
            let chrom = String::from_utf8(chr.extract().unwrap()).unwrap();
            peaks
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .iter()
                .map(|peak| {
                    let start = peak.get_item("start")?.extract::<u64>()?;
                    let end = peak.get_item("end")?.extract::<u64>()?;
                    let fc = peak.get_item("fc")?.extract::<f64>()?;
                    let score = peak.get_item("score")?.extract::<f64>()? * 10.0;
                    let p_value = peak.get_item("pscore")?.extract::<f64>()?;
                    let q_value = peak.get_item("qscore")?.extract::<f64>()?;
                    let peak = peak.get_item("summit")?.extract::<u64>()? - start;
                    Ok(NarrowPeak {
                        chrom: chrom.clone(),
                        start,
                        end,
                        name: None,
                        score: Some((score as u16).min(1000).try_into().unwrap()),
                        strand: None,
                        signal_value: fc,
                        p_value: if p_value < 0.0 { None } else { Some(p_value) },
                        q_value: if q_value < 0.0 { None } else { Some(q_value) },
                        peak,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn get_broad_peaks<'py>(peak_io_obj: &Bound<'py, PyAny>) -> Result<Vec<BroadPeak>> {
    peak_io_obj
        .getattr("peaks")?
        .downcast::<pyo3::types::PyDict>()
        .unwrap()
        .iter()
        .flat_map(|(chr, peaks)| {
            let chrom = String::from_utf8(chr.extract().unwrap()).unwrap();
            peaks
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .iter()
                .map(|peak| {
                    let start = peak.get_item("start")?.extract::<u64>()?;
                    let end = peak.get_item("end")?.extract::<u64>()?;
                    let fc = peak.get_item("fc")?.extract::<f64>()?;
                    let score = peak.get_item("score")?.extract::<f64>()? * 10.0;
                    let p_value = peak.get_item("pscore")?.extract::<f64>()?;
                    let q_value = peak.get_item("qscore")?.extract::<f64>()?;
                    Ok(BroadPeak {
                        chrom: chrom.clone(),
                        start,
                        end,
                        name: None,
                        score: Some((score as u16).min(1000).try_into().unwrap()),
                        strand: None,
                        signal_value: fc,
                        p_value: if p_value < 0.0 { None } else { Some(p_value) },
                        q_value: if q_value < 0.0 { None } else { Some(q_value) },
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[pyfunction]
pub fn create_fwtrack_obj<'py>(
    py: Python<'py>,
    files: Vec<PathBuf>,
) -> Result<(Bound<'py, PyAny>, Vec<Bound<'py, PyAny>>)> {
    let macs = py.import_bound("MACS3.Signal.FixWidthTrack")?;
    let merged = macs.getattr("FWTrack")?.call1((1000000,))?;
    let has_replicate = files.len() > 1;
    let replicates = files
        .into_iter()
        .map(|fl| {
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("buffer_size", 100000)?;
            let fwt = macs.getattr("FWTrack")?.call((), Some(&kwargs))?;
            let reader = utils::open_file_for_read(&fl);
            bed_utils::bed::io::Reader::new(reader, None)
                .into_records::<Fragment>()
                .try_for_each(|x| {
                    let x = x?;
                    let chr = x.chrom().as_bytes();
                    match x.strand() {
                        None => {
                            fwt.call_method1("add_loc", (chr, x.start(), 0))?;
                            fwt.call_method1("add_loc", (chr, x.end() - 1, 1))?;
                            if has_replicate {
                                merged.call_method1("add_loc", (chr, x.start(), 0))?;
                                merged.call_method1("add_loc", (chr, x.end() - 1, 1))?;
                            }
                        }
                        Some(Strand::Forward) => {
                            fwt.call_method1("add_loc", (chr, x.start(), 0))?;
                            if has_replicate {
                                merged.call_method1("add_loc", (chr, x.start(), 0))?;
                            }
                        }
                        Some(Strand::Reverse) => {
                            fwt.call_method1("add_loc", (chr, x.end() - 1, 1))?;
                            if has_replicate {
                                merged.call_method1("add_loc", (chr, x.end() - 1, 1))?;
                            }
                        }
                    }
                    anyhow::Ok(())
                })?;
            fwt.call_method0("finalize")?;
            Ok(fwt)
        })
        .collect::<Result<Vec<_>>>()?;
    if has_replicate {
        merged.call_method0("finalize")?;
        Ok((merged, replicates))
    } else {
        Ok((replicates.into_iter().next().unwrap(), Vec::new()))
    }
}

#[pyfunction]
pub fn export_tags(
    anndata: AnnDataLike,
    dir: PathBuf,
    group_by: Vec<PyBackedStr>,
    replicates: Option<Vec<PyBackedStr>>,
    max_frag_size: Option<u64>,
    selections: Option<HashSet<String>>,
) -> Result<HashMap<String, Vec<PathBuf>>> {
    macro_rules! run {
        ($data:expr) => {
            _export_tags(
                $data,
                dir,
                &group_by,
                replicates.as_ref(),
                max_frag_size,
                selections,
            )
        };
    }

    crate::with_anndata!(&anndata, run)
}

fn _export_tags<D: SnapData, P: AsRef<std::path::Path>>(
    data: &D,
    dir: P,
    group_by: &Vec<PyBackedStr>,
    replicates: Option<&Vec<PyBackedStr>>,
    max_frag_size: Option<u64>,
    selections: Option<HashSet<String>>,
) -> Result<HashMap<String, Vec<PathBuf>>> {
    // Get keys
    ensure!(data.n_obs() == group_by.len(), "lengths differ");
    let keys: Vec<(&str, &str)> = match replicates {
        Some(rep) => group_by
            .iter()
            .zip(rep.iter())
            .map(|(x, y)| (x.as_ref(), y.as_ref()))
            .collect(),
        None => group_by.iter().map(|x| (x.as_ref(), "")).collect(),
    };
    let mut unique_keys: HashSet<(&str, &str)> = keys.iter().cloned().unique().collect();

    // Create output files
    if let Some(select) = selections {
        unique_keys.retain(|x| select.contains(x.0));
    }
    std::fs::create_dir_all(dir.as_ref())
        .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;
    let files = unique_keys
        .into_iter()
        .map(|(a, b)| {
            let filename = format!("{}_{}.zst", a, b);
            if !sanitize_filename::is_sanitized(&filename) {
                bail!("invalid filename: {}", filename);
            }
            let filename = dir.as_ref().join(&filename);
            let writer = open_file_for_write(&filename, Some(Compression::Zstd), Some(1))?;
            let val = (filename, Arc::new(Mutex::new(writer)));
            Ok(((a, b), val))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    // Export
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
    )?;
    let mut fragments = data.get_fragment_iter(1000)?;
    if let Some(max_size) = max_frag_size {
        fragments = fragments.max_fragment_size(max_size);
    }
    fragments
        .into_fragment_groups(|x| keys[x])
        .progress_with_style(style)
        .for_each(|vals| {
            vals.into_par_iter().for_each(|(i, beds)| {
                if let Some((_, fl)) = files.get(&i) {
                    let mut fl = fl.lock().unwrap();
                    beds.into_iter()
                        .for_each(|bed| writeln!(fl, "{}", bed.1).unwrap());
                }
            })
        });
    let mut result = HashMap::new();
    files.into_iter().for_each(|((a, _), (filename, _))| {
        result
            .entry(a.to_owned())
            .or_insert_with(Vec::new)
            .push(filename);
    });
    Ok(result)
}

#[pyfunction]
pub fn call_peaks_bulk<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    macs3_options: &Bound<'_, PyAny>,
    max_frag_size: Option<u64>,
) -> Result<PyDataFrame> {
    macro_rules! run {
        ($data:expr) => {
            _call_peaks_bulk(py, $data, macs3_options, max_frag_size)
        };
    }
    let peaks = crate::with_anndata!(&anndata, run)?;

    Ok(PyDataFrame(narrow_peak_to_dataframe(peaks)?))
}

fn _call_peaks_bulk<'py, D: SnapData>(
    py: Python<'py>,
    data: &D,
    macs3_options: &Bound<'_, PyAny>,
    max_frag_size: Option<u64>,
) -> Result<Vec<NarrowPeak>> {
    let macs = py.import_bound("MACS3.Signal.FixWidthTrack")?;
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("buffer_size", 100000)?;
    let fwt = macs.getattr("FWTrack")?.call((), Some(&kwargs))?;

    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
    )?;
    let mut fragments = data.get_fragment_iter(1000)?;
    if let Some(max_size) = max_frag_size {
        fragments = fragments.max_fragment_size(max_size);
    }
    fragments
        .into_fragments()
        .progress_with_style(style)
        .try_for_each(|vals| {
            vals.0.into_iter().flatten().try_for_each(|x| {
                let chr = x.chrom().as_bytes();
                match x.strand() {
                    None => {
                        fwt.call_method1("add_loc", (chr, x.start(), 0))?;
                        fwt.call_method1("add_loc", (chr, x.end() - 1, 1))?;
                    }
                    Some(Strand::Forward) => {
                        fwt.call_method1("add_loc", (chr, x.start(), 0))?;
                    }
                    Some(Strand::Reverse) => {
                        fwt.call_method1("add_loc", (chr, x.end() - 1, 1))?;
                    }
                }
                anyhow::Ok(())
            })
        })?;
    fwt.call_method0("finalize")?;

    let outputs = pyo3::types::PyDict::new_bound(py);
    let inputs = pyo3::types::PyDict::new_bound(py);
    inputs.set_item("fwt", fwt)?;
    inputs.set_item("options", macs3_options)?;
    py.run_bound(
        r#"
from MACS3.Signal.PeakDetect import PeakDetect
peakdetect = PeakDetect(treat=fwt, opt=options)
peakdetect.call_peaks()
peakdetect.peaks.filter_fc(fc_low=options.fecutoff)
peaks = peakdetect.peaks
"#,
        Some(&inputs),
        Some(&outputs),
    )?;
    get_narrow_peaks(&outputs.get_item("peaks")?.unwrap())
}
