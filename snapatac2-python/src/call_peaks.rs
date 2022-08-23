use snapatac2_core::{export::Exporter, utils::merge_peaks};

use anndata_rs::anndata_trait::DataIO;
use pyo3::{
    prelude::*,
    PyResult, Python,
    type_object::PyTypeObject,
};
use bed_utils::bed::{BEDLike, GenomicRange, io::Reader, tree::BedTree};
use flate2::read::MultiGzDecoder;
use tempfile::Builder;
use std::fs::File;
use polars::prelude::DataFrame;
use polars::series::Series;
use polars::prelude::NamedFrom;
use pyanndata::{AnnData, AnnDataSet};
use std::collections::HashSet;

#[pyfunction]
pub fn call_peaks<'py>(
    py: Python<'py>,
    data: &PyAny,
    group_by: Vec<&str>,
    selections: Option<HashSet<&str>>,
    q_value: f64,
    out_dir: Option<&str>,
    key_added: &str,
) -> PyResult<()> {
    let dir = Builder::new().tempdir_in("./").unwrap();

    let peak_files = if data.is_instance(AnnData::type_object(py))? {
        let anndata: AnnData = data.extract()?;
        let x = anndata.0.inner().call_peaks(
            q_value, &group_by, selections, out_dir.unwrap_or(dir.path().to_str().unwrap()), "", ".bed.gz",
        ).unwrap();
        x
    } else if data.is_instance(AnnDataSet::type_object(py))? {
        let anndata: AnnDataSet = data.extract()?;
        let x = anndata.0.inner().call_peaks(
            q_value, &group_by, selections, out_dir.unwrap_or(dir.path().to_str().unwrap()), "", ".bed.gz",
        ).unwrap();
        x
    } else {
        panic!("expecting an AnnData or AnnDataSet object");
    };

    let peak_iter = peak_files.values().flat_map(|fl|
        Reader::new(
            MultiGzDecoder::new(File::open(fl).unwrap()),
            None,
        ).into_records().map(Result::unwrap)
    );
    let peaks: Vec<_> = merge_peaks(peak_iter, 250).flatten().collect();
    let n = peaks.len();
    let peaks_str = Series::new(
        "Peaks",
        peaks.iter().map(|x| BEDLike::pretty_show(x)).collect::<Vec<_>>(),
    );
    let peaks_index: BedTree<usize> = peaks.into_iter().enumerate().map(|(i, x)| (x, i)).collect();
    let iter = peak_files.into_iter().map(|(key, fl)| {
        let mut values = vec![false; n];
        Reader::new(
            MultiGzDecoder::new(File::open(fl).unwrap()),
            None,
        ).into_records().for_each(|x| {
            let bed: GenomicRange = x.unwrap();
            peaks_index.find(&bed).for_each(|(_, i)| values[*i] = true);
        });
        Series::new(key.as_str(), values)
    });

    let df: Box<dyn DataIO> = Box::new(DataFrame::new(
        std::iter::once(peaks_str).chain(iter).collect()
    ).unwrap());

    if data.is_instance(AnnData::type_object(py))? {
        let anndata: AnnData = data.extract()?;
        anndata.0.inner().get_uns().inner().add_data(key_added, &df).unwrap();
    } else if data.is_instance(AnnDataSet::type_object(py))? {
        let anndata: AnnDataSet = data.extract()?;
        anndata.0.inner().get_uns().inner().add_data(key_added, &df).unwrap();
    }

    dir.close().unwrap();
    Ok(())
}