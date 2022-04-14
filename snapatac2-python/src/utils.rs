use pyo3::{
    prelude::*,
    types::PyIterator,
    PyResult, Python,
};
use numpy::{PyReadonlyArrayDyn, PyReadonlyArray, Ix1, Ix2, PyArray, IntoPyArray};

use hdf5::H5Type;
use bed_utils::{bed, bed::GenomicRange, bed::BED};
use std::fs::File;
use flate2::read::MultiGzDecoder;
use hdf5;
use linreg::lin_reg_imprecise;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa::traits::{Fit, Predict};
use rand_core::SeedableRng;
use rand_isaac::Isaac64Rng;
use hora::core::ann_index::ANNIndex;

/// Simple linear regression
#[pyfunction]
pub(crate) fn simple_lin_reg(py_iter: &PyIterator) -> PyResult<(f64, f64)> {
    Ok(lin_reg_imprecise(py_iter.map(|x| x.unwrap().extract().unwrap())).unwrap())
}

/// Perform regression
#[pyfunction]
pub(crate) fn jm_regress(
    jm_: PyReadonlyArrayDyn<'_, f64>,
    count_: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<(f64, f64)> {
    let jm = &jm_.as_array();
    let n_row = jm.shape()[0];
    let count = &count_.as_array();
    let iter = (0..n_row).flat_map(|i| (i+1..n_row)
        .map(move |j| (1.0 / (1.0 / count[[i, 0]] + 1.0 / count[[j, 0]] - 1.0), jm[[i, j]]))
    );
    Ok(lin_reg_imprecise(iter).unwrap())
}

#[pyfunction]
pub(crate) fn intersect_bed(regions: Vec<&str>, bed_file: &str) -> PyResult<Vec<bool>> {
    let bed_tree: bed::tree::BedTree<()> = bed::io::Reader::new(open_file(bed_file), None)
        .into_records().map(|x: Result<BED<3>, _>| (x.unwrap(), ())).collect();
    Ok(regions.into_iter()
        .map(|x| bed_tree.is_overlapped(&str_to_genomic_region(x).unwrap()))
        .collect()
    )
}

#[pyfunction]
pub(crate) fn kmeans<'py>(
    py: Python<'py>,
    n_clusters: usize,
    observations_: PyReadonlyArray<'_, f64, Ix2>,
) -> PyResult<&'py PyArray<usize, Ix1>> {
    let seed = 42;
    let rng: Isaac64Rng = SeedableRng::seed_from_u64(seed);
    let observations = DatasetBase::from(observations_.as_array());
    let model = KMeans::params_with_rng(n_clusters, rng)
        .fit(&observations)
        .expect("KMeans fitted");
    Ok(model.predict(observations).targets.into_pyarray(py))
}

// Search for and save nearest neighbors using ANN
#[pyfunction]
pub(crate) fn approximate_nearest_neighbors(
    data_: PyReadonlyArray<'_, f32, Ix2>,
    k: usize,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)>
{
    let data = data_.as_array();
    let dimension = data.shape()[1];
    let mut index = hora::index::hnsw_idx::HNSWIndex::<f32, usize>::new(
        dimension,
        &hora::index::hnsw_params::HNSWParams::<f32>::default(),
    );
    for (i, sample) in data.axis_iter(ndarray::Axis(0)).enumerate() {
        index.add(sample.to_vec().as_slice(), i).unwrap();
    }
    index.build(hora::core::metrics::Metric::Euclidean).unwrap();
    let row_iter = data.axis_iter(ndarray::Axis(0)).map(move |row| {
        index.search_nodes(row.to_vec().as_slice(), k).into_iter()
            .map(|(n, d)| (n.idx().unwrap(), d)).collect::<Vec<_>>()
    });
    Ok(to_csr_matrix(row_iter))
}

pub(crate) fn to_csr_matrix<I, D>(iter: I) -> (Vec<D>, Vec<i32>, Vec<i32>)
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    let mut data: Vec<D> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::new();

    let n = iter.fold(0, |r_idx, mut row| {
        row.sort_by(|a, b| a.0.cmp(&b.0));
        indptr.push(r_idx.try_into().unwrap());
        let new_idx = r_idx + row.len();
        let (mut a, mut b) = row.into_iter().map(|(x, y)| -> (i32, D) {
            (x.try_into().unwrap(), y)
        }).unzip();
        indices.append(&mut a);
        data.append(&mut b);
        new_idx
    });
    indptr.push(n.try_into().unwrap());
    (data, indices, indptr)
}

// Convert string such as "chr1:134-2222" to `GenomicRange`.
pub(crate) fn str_to_genomic_region(txt: &str) -> Option<GenomicRange> {
    let mut iter1 = txt.splitn(2, ":");
    let chr = iter1.next()?;
    let mut iter2 = iter1.next().map(|x| x.splitn(2, "-"))?;
    let start: u64 = iter2.next().map_or(None, |x| x.parse().ok())?;
    let end: u64 = iter2.next().map_or(None, |x| x.parse().ok())?;
    Some(GenomicRange::new(chr, start, end))
}

pub(crate) fn open_file(file: &str) -> Box<dyn std::io::Read> {
    if is_gzipped(file) {
        Box::new(MultiGzDecoder::new(File::open(file).unwrap()))
    } else {
        Box::new(File::open(file).unwrap())
    }
}

/// Determine if a file is gzipped.
pub(crate) fn is_gzipped(file: &str) -> bool {
    MultiGzDecoder::new(File::open(file).unwrap()).header().is_some()
}