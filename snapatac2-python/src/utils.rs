use pyo3::{
    prelude::*,
    types::PyIterator,
    PyResult, Python,
};
use numpy::{Element, PyReadonlyArrayDyn, PyReadonlyArray, Ix1, Ix2, PyArray, IntoPyArray};
use snapatac2_core::utils::similarity;

use hdf5::H5Type;
use bed_utils::{bed, bed::GenomicRange, bed::BED};
use std::{str::FromStr, fs::File};
use flate2::read::MultiGzDecoder;
use hdf5;
use linreg::lin_reg_imprecise;
use linfa::{DatasetBase, traits::{Fit, Predict}};
use linfa_clustering::KMeans;
use rand_core::SeedableRng;
use rand_isaac::Isaac64Rng;
use hora::core::ann_index::ANNIndex;
use nalgebra_sparse::CsrMatrix;

macro_rules! with_sparsity_pattern {
    ($dtype:expr, $indices:expr, $indptr:expr, $n:expr, $fun:ident) => {
        match $dtype {
            "int32" => {
                let indices_ = $indices.extract::<PyReadonlyArray<i32, Ix1>>()?;
                let indptr_ = $indptr.extract::<PyReadonlyArray<i32, Ix1>>()?;
                $fun!(to_sparsity_pattern(&indptr_, &indices_, $n)?)
            },
            "int64" => {
                let indices_ = $indices.extract::<PyReadonlyArray<i64, Ix1>>()?;
                let indptr_ = $indptr.extract::<PyReadonlyArray<i64, Ix1>>()?;
                $fun!(to_sparsity_pattern(&indptr_, &indices_, $n)?)
            },
            ty => panic!("{}", ty),
        }
    }
}
 

#[pyfunction]
pub(crate) fn jaccard_similarity<'py>(
    py: Python<'py>,
    mat: &'py PyAny,
    other: Option<&'py PyAny>,
    weights: Option<PyReadonlyArray<f64, Ix1>>,
) -> PyResult<&'py PyArray<f64, Ix2>> {
    let weights_ = match weights {
        None => None,
        Some(ref ws) => Some(ws.as_slice().unwrap()),
    };

    macro_rules! with_csr {
        ($mat:expr) => {
            match other {
                None => Ok(similarity::jaccard($mat, weights_).into_pyarray(py)),
                Some(mat2) => {
                    macro_rules! xxx {
                        ($m:expr) => { Ok(similarity::jaccard2($mat, $m, weights_).into_pyarray(py)) };
                    }
                    let shape: Vec<usize> = mat2.getattr("shape")?.extract()?;
                    with_sparsity_pattern!(
                        mat2.getattr("indices")?.getattr("dtype")?.getattr("name")?.extract()?,
                        mat2.getattr("indices")?,
                        mat2.getattr("indptr")?,
                        shape[1],
                        xxx
                    )
                },
            }
        };
    }

    let shape: Vec<usize> = mat.getattr("shape")?.extract()?;
    with_sparsity_pattern!(
        mat.getattr("indices")?.getattr("dtype")?.getattr("name")?.extract()?,
        mat.getattr("indices")?,
        mat.getattr("indptr")?,
        shape[1],
        with_csr
    )
}

fn to_sparsity_pattern<'py, I>(
    indptr_: &'py PyReadonlyArray<I, Ix1>,
    indices_: &'py PyReadonlyArray<I, Ix1>,
    n: usize
) -> PyResult<similarity::BorrowedSparsityPattern<'py, I>>
where
    I: Element,
{
    let indptr = indptr_.as_slice().unwrap();
    let indices = indices_.as_slice().unwrap();
    Ok(similarity::BorrowedSparsityPattern::new(indptr, indices, n))
}

#[pyfunction]
pub(crate) fn cosine_similarity<'py>(
    py: Python<'py>,
    mat: &'py PyAny,
    other: Option<&'py PyAny>,
    weights: Option<PyReadonlyArray<f64, Ix1>>,
) -> PyResult<&'py PyArray<f64, Ix2>> {
    let weights_ = match weights {
        None => None,
        Some(ref ws) => Some(ws.as_slice().unwrap()),
    };
    match other {
        None => Ok(similarity::cosine(csr_to_rust(mat)?, weights_).into_pyarray(py)),
        Some(mat2) => Ok(
            similarity::cosine2(
                csr_to_rust(mat)?,
                csr_to_rust(mat2)?,
                weights_,
            ).into_pyarray(py)
        ),
    }
}

fn csr_to_rust<'py>(csr: &'py PyAny) -> PyResult<CsrMatrix<f64>> {
    let shape: Vec<usize> = csr.getattr("shape")?.extract()?;
    let indices = cast_pyarray(csr.getattr("indices")?)?;
    let indptr = cast_pyarray(csr.getattr("indptr")?)?;
    let data = cast_pyarray(csr.getattr("data")?)?;
    Ok(CsrMatrix::try_from_csr_data(
        shape[0], shape[1], indptr, indices, data,
    ).unwrap())
}

fn cast_pyarray<'py, T: Element>(arr: &'py PyAny) -> PyResult<Vec<T>> {
    let vec = match arr.getattr("dtype")?.getattr("name")?.extract()? {
        "uint32" => arr.extract::<PyReadonlyArrayDyn<u32>>()?.cast(false)?.to_vec().unwrap(),
        "int32" => arr.extract::<PyReadonlyArrayDyn<i32>>()?.cast(false)?.to_vec().unwrap(),
        "uint64" => arr.extract::<PyReadonlyArrayDyn<u64>>()?.cast(false)?.to_vec().unwrap(),
        "int64" => arr.extract::<PyReadonlyArrayDyn<i64>>()?.cast(false)?.to_vec().unwrap(),
        "float32" => arr.extract::<PyReadonlyArrayDyn<f32>>()?.cast(false)?.to_vec().unwrap(),
        "float64" => arr.extract::<PyReadonlyArrayDyn<f64>>()?.cast(false)?.to_vec().unwrap(),
        ty => panic!("cannot cast type {}", ty),
    };
    Ok(vec)
}

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
        .map(|x| bed_tree.is_overlapped(&GenomicRange::from_str(x).unwrap()))
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