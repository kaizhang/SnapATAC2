mod anndata;

pub use self::anndata::AnnDataLike;

use anyhow::Result;
use bed_utils::bed::{merge_sorted_bed, BEDLike};
use bed_utils::extsort::ExternalSorterBuilder;
use numpy::{
    Element, IntoPyArray, Ix1, Ix2, PyArray, PyArrayMethods, PyReadonlyArray, PyReadonlyArrayDyn,
};
use pyo3::{prelude::*, types::PyIterator, PyResult, Python};
use snapatac2_core::genome::{
    read_transcripts_from_gff, read_transcripts_from_gtf, Transcript, TranscriptParserOptions,
};
use snapatac2_core::utils;

use bed_utils::{bed, bed::GenomicRange, bed::BED};
use linfa::{
    traits::{Fit, Predict},
    DatasetBase,
};
use linfa_clustering::KMeans;
use linreg::lin_reg_imprecise;
use nalgebra_sparse::CsrMatrix;
use rand_core::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;

macro_rules! with_sparsity_pattern {
    ($dtype:expr, $indices:expr, $indptr:expr, $n:expr, $fun:ident) => {
        match $dtype {
            "int32" => {
                let indices_ = $indices.extract::<PyReadonlyArray<i32, Ix1>>()?;
                let indptr_ = $indptr.extract::<PyReadonlyArray<i32, Ix1>>()?;
                $fun!(to_sparsity_pattern(&indptr_, &indices_, $n)?)
            }
            "int64" => {
                let indices_ = $indices.extract::<PyReadonlyArray<i64, Ix1>>()?;
                let indptr_ = $indptr.extract::<PyReadonlyArray<i64, Ix1>>()?;
                $fun!(to_sparsity_pattern(&indptr_, &indices_, $n)?)
            }
            ty => panic!("{}", ty),
        }
    };
}

#[pyfunction]
pub(crate) fn jaccard_similarity<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyAny>,
    other: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray<f64, Ix1>>,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let weights_ = match weights {
        None => None,
        Some(ref ws) => Some(ws.as_slice().unwrap()),
    };

    macro_rules! with_csr {
        ($mat:expr) => {
            match other {
                None => Ok(utils::similarity::jaccard($mat, weights_).into_pyarray_bound(py)),
                Some(mat2) => {
                    macro_rules! xxx {
                        ($m:expr) => {
                            Ok(utils::similarity::jaccard2($mat, $m, weights_)
                                .into_pyarray_bound(py))
                        };
                    }
                    let shape: Vec<usize> = mat2.getattr("shape")?.extract()?;
                    with_sparsity_pattern!(
                        mat2.getattr("indices")?
                            .getattr("dtype")?
                            .getattr("name")?
                            .extract()?,
                        mat2.getattr("indices")?,
                        mat2.getattr("indptr")?,
                        shape[1],
                        xxx
                    )
                }
            }
        };
    }

    let shape: Vec<usize> = mat.getattr("shape")?.extract()?;
    with_sparsity_pattern!(
        mat.getattr("indices")?
            .getattr("dtype")?
            .getattr("name")?
            .extract()?,
        mat.getattr("indices")?,
        mat.getattr("indptr")?,
        shape[1],
        with_csr
    )
}

fn to_sparsity_pattern<'py, I>(
    indptr_: &'py PyReadonlyArray<I, Ix1>,
    indices_: &'py PyReadonlyArray<I, Ix1>,
    n: usize,
) -> PyResult<utils::similarity::BorrowedSparsityPattern<'py, I>>
where
    I: Element,
{
    let indptr = indptr_.as_slice().unwrap();
    let indices = indices_.as_slice().unwrap();
    Ok(utils::similarity::BorrowedSparsityPattern::new(
        indptr, indices, n,
    ))
}

#[pyfunction]
pub(crate) fn cosine_similarity<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyAny>,
    other: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray<f64, Ix1>>,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let weights_ = match weights {
        None => None,
        Some(ref ws) => Some(ws.as_slice().unwrap()),
    };
    match other {
        None => Ok(utils::similarity::cosine(csr_to_rust(mat)?, weights_).into_pyarray_bound(py)),
        Some(mat2) => {
            Ok(
                utils::similarity::cosine2(csr_to_rust(mat)?, csr_to_rust(mat2)?, weights_)
                    .into_pyarray_bound(py),
            )
        }
    }
}

#[pyfunction]
pub(crate) fn pearson<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    match mat.getattr("dtype")?.getattr("name")?.extract()? {
        "float32" => {
            let mat_ = mat.extract::<PyReadonlyArray<f32, Ix2>>()?.to_owned_array();
            let other_ = other
                .extract::<PyReadonlyArray<f32, Ix2>>()?
                .to_owned_array();
            Ok(utils::similarity::pearson2(mat_, other_)
                .into_pyarray_bound(py)
                .to_object(py))
        }
        "float64" => {
            let mat_ = mat.extract::<PyReadonlyArray<f64, Ix2>>()?.to_owned_array();
            let other_ = other
                .extract::<PyReadonlyArray<f64, Ix2>>()?
                .to_owned_array();
            Ok(utils::similarity::pearson2(mat_, other_)
                .into_pyarray_bound(py)
                .to_object(py))
        }
        ty => panic!("Cannot compute correlation for type {}", ty),
    }
}

#[pyfunction]
pub(crate) fn spearman<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    match mat.getattr("dtype")?.getattr("name")?.extract()? {
        "float32" => {
            let mat_ = mat.extract::<PyReadonlyArray<f32, Ix2>>()?.to_owned_array();
            match other.getattr("dtype")?.getattr("name")?.extract()? {
                "float32" => {
                    let other_ = other
                        .extract::<PyReadonlyArray<f32, Ix2>>()?
                        .to_owned_array();
                    Ok(utils::similarity::spearman2(mat_, other_)
                        .into_pyarray_bound(py)
                        .to_object(py))
                }
                "float64" => {
                    let other_ = other
                        .extract::<PyReadonlyArray<f64, Ix2>>()?
                        .to_owned_array();
                    Ok(utils::similarity::spearman2(mat_, other_)
                        .into_pyarray_bound(py)
                        .to_object(py))
                }
                ty => panic!("Cannot compute correlation for type {}", ty),
            }
        }
        "float64" => {
            let mat_ = mat.extract::<PyReadonlyArray<f64, Ix2>>()?.to_owned_array();
            match other.getattr("dtype")?.getattr("name")?.extract()? {
                "float32" => {
                    let other_ = other
                        .extract::<PyReadonlyArray<f32, Ix2>>()?
                        .to_owned_array();
                    Ok(utils::similarity::spearman2(mat_, other_)
                        .into_pyarray_bound(py)
                        .to_object(py))
                }
                "float64" => {
                    let other_ = other
                        .extract::<PyReadonlyArray<f64, Ix2>>()?
                        .to_owned_array();
                    Ok(utils::similarity::spearman2(mat_, other_)
                        .into_pyarray_bound(py)
                        .to_object(py))
                }
                ty => panic!("Cannot compute correlation for type {}", ty),
            }
        }
        ty => panic!("Cannot compute correlation for type {}", ty),
    }
}

fn csr_to_rust<'py>(csr: &Bound<'py, PyAny>) -> PyResult<CsrMatrix<f64>> {
    let shape: Vec<usize> = csr.getattr("shape")?.extract()?;
    let indices = cast_pyarray(&csr.getattr("indices")?)?;
    let indptr = cast_pyarray(&csr.getattr("indptr")?)?;
    let data = cast_pyarray(&csr.getattr("data")?)?;
    Ok(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
}

fn cast_pyarray<'py, T: Element>(arr: &Bound<'py, PyAny>) -> PyResult<Vec<T>> {
    let vec = match arr.getattr("dtype")?.getattr("name")?.extract()? {
        "uint32" => arr
            .extract::<PyReadonlyArrayDyn<u32>>()?
            .cast(false)?
            .to_vec()
            .unwrap(),
        "int32" => arr
            .extract::<PyReadonlyArrayDyn<i32>>()?
            .cast(false)?
            .to_vec()
            .unwrap(),
        "uint64" => arr
            .extract::<PyReadonlyArrayDyn<u64>>()?
            .cast(false)?
            .to_vec()
            .unwrap(),
        "int64" => arr
            .extract::<PyReadonlyArrayDyn<i64>>()?
            .cast(false)?
            .to_vec()
            .unwrap(),
        "float32" => arr
            .extract::<PyReadonlyArrayDyn<f32>>()?
            .cast(false)?
            .to_vec()
            .unwrap(),
        "float64" => arr
            .extract::<PyReadonlyArrayDyn<f64>>()?
            .cast(false)?
            .to_vec()
            .unwrap(),
        ty => panic!("cannot cast type {}", ty),
    };
    Ok(vec)
}

/// Simple linear regression
#[pyfunction]
pub(crate) fn simple_lin_reg(py_iter: Bound<'_, PyIterator>) -> PyResult<(f64, f64)> {
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
    let iter = (0..n_row).flat_map(|i| {
        (i + 1..n_row).map(move |j| {
            (
                1.0 / (1.0 / count[[i, 0]] + 1.0 / count[[j, 0]] - 1.0),
                jm[[i, j]],
            )
        })
    });
    Ok(lin_reg_imprecise(iter).unwrap())
}

/// Get a list of genomic ranges from a python object. Acceptable types are:
/// - file-like object
/// - list of strings
pub(crate) fn read_genomic_ranges(input: &Bound<'_, PyAny>) -> Result<Vec<GenomicRange>> {
    if let Ok(list) = input.downcast::<pyo3::types::PyList>() {
        list.iter()
            .map(|str| {
                let str: &str = str.extract()?;
                Ok(GenomicRange::from_str(str).unwrap())
            })
            .collect()
    } else {
        let file: PathBuf = input.extract()?;
        let mut reader = bed::io::Reader::new(utils::open_file_for_read(file), None);
        Ok(reader
            .records::<GenomicRange>()
            .map(|x| x.unwrap())
            .collect())
    }
}

/// Read genomic regions from a bed file.
/// Returns a list of strings
#[pyfunction]
pub(crate) fn read_regions(file: PathBuf) -> Vec<String> {
    let mut reader = bed::io::Reader::new(utils::open_file_for_read(file), None);
    reader
        .records::<GenomicRange>()
        .map(|x| x.unwrap().pretty_show())
        .collect()
}

#[pyfunction]
pub(crate) fn intersect_bed<'py>(
    regions: Bound<'py, PyAny>,
    bed_file: &str,
) -> PyResult<Vec<bool>> {
    let bed_tree: bed::map::GIntervalMap<()> =
        bed::io::Reader::new(utils::open_file_for_read(bed_file), None)
            .into_records()
            .map(|x: Result<BED<3>, _>| (x.unwrap(), ()))
            .collect();
    let res = PyIterator::from_bound_object(&regions)?
        .map(|x| {
            bed_tree.is_overlapped(&GenomicRange::from_str(x.unwrap().extract().unwrap()).unwrap())
        })
        .collect();
    Ok(res)
}

#[pyfunction]
pub(crate) fn kmeans<'py>(
    py: Python<'py>,
    n_clusters: usize,
    observations_: PyReadonlyArray<'_, f64, Ix2>,
) -> PyResult<Bound<'py, PyArray<usize, Ix1>>> {
    let seed = 42;
    let rng: Isaac64Rng = SeedableRng::seed_from_u64(seed);
    let observations = DatasetBase::from(observations_.as_array());
    let model = KMeans::params_with_rng(n_clusters, rng)
        .fit(&observations)
        .expect("KMeans fitted");
    Ok(model.predict(observations).targets.into_pyarray_bound(py))
}

/*
#[pyfunction]
pub(crate) fn read_promoters(
    annotation: PathBuf,
) -> Result<PyDataFrame> {
    let transcripts = read_transcripts(annotation, &TranscriptParserOptions::default());
    let (chroms, starts, ends, strands, names) = transcripts.iter().map(|x| {
        let strand = x.strand.as_str();

    }).unzip();
    Ok(DataFrame::new(vec![
        ("chrom", chroms),
        ("start", starts),
        ("end", ends),
        ("strand", strand),
        ("name", name),
    ])?)
}
    */

pub fn read_transcripts<P: AsRef<std::path::Path>>(
    file_path: P,
    options: &TranscriptParserOptions,
) -> Vec<Transcript> {
    let path = if file_path.as_ref().extension().unwrap() == "gz" {
        file_path.as_ref().file_stem().unwrap().as_ref()
    } else {
        file_path.as_ref()
    };
    let file = BufReader::new(utils::open_file_for_read(&file_path));
    if path.extension().unwrap() == "gff" {
        read_transcripts_from_gff(file, options).unwrap()
    } else if path.extension().unwrap() == "gtf" {
        read_transcripts_from_gtf(file, options).unwrap()
    } else {
        read_transcripts_from_gff(file, options).unwrap_or_else(|_| {
            read_transcripts_from_gtf(
                BufReader::new(utils::open_file_for_read(file_path)),
                options,
            )
            .unwrap()
        })
    }
}

#[pyfunction]
pub(crate) fn total_size_of_peaks(peaks: Vec<String>) -> Result<u64> {
    let sorter = ExternalSorterBuilder::new()
        .build()?
        .sort(
            peaks
                .into_iter()
                .map(|x| GenomicRange::from_str(&x).unwrap()),
        )?
        .map(|x| x.unwrap());
    Ok(merge_sorted_bed(sorter).map(|x| x.len()).sum())
}
