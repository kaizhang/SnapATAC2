use crate::utils::AnnDataLike;

use anndata::{
    data::{array::utils::to_csr_data, BoundedSelectInfoElem, SelectInfoElem},
    AnnDataOp, ArrayData, ArrayElemOp, ArrayOp, Backend,
};
use anndata_hdf5::H5;
use anyhow::Result;
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use log::info;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2};
use pyanndata::data::PyArrayData;
use pyo3::prelude::*;
use rand::SeedableRng;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use std::ops::Deref;

#[pyfunction]
pub(crate) fn spectral_embedding<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    selected_features: &Bound<'_, PyAny>,
    n_components: usize,
    random_state: i64,
    feature_weights: Option<Vec<f64>>,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    macro_rules! run {
        ($data:expr) => {{
            let slice = pyanndata::data::to_select_elem(selected_features, $data.n_vars())?;
            let mut mat: CsrMatrix<f64> = $data.x().slice_axis(1, slice)?.unwrap();
            if let Some(weights) = feature_weights {
                normalize(&mut mat, &weights);
            } else {
                let weights = idf(&mat);
                normalize(&mut mat, &weights);
            }

            let (v, u, _) = spectral_mf(mat, n_components, random_state)?;
            anyhow::Ok((v, u))
        }};
    }
    let (evals, evecs) = crate::with_anndata!(&anndata, run)?;

    Ok((
        PyArray1::from_owned_array_bound(py, evals),
        PyArray2::from_owned_array_bound(py, evecs),
    ))
}

#[pyfunction]
pub(crate) fn spectral_embedding_nystrom<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    selected_features: &Bound<'_, PyAny>,
    n_components: usize,
    sample_size: usize,
    weighted_by_degree: bool,
    chunk_size: usize,
    feature_weights: Option<Vec<f64>>,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    macro_rules! run {
        ($data:expr) => {{
            let selected_features =
                pyanndata::data::to_select_elem(selected_features, $data.n_vars())?;
            let weights = if let Some(weights) = feature_weights {
                weights
            } else {
                idf_from_chunks(
                    $data
                        .x()
                        .iter(5000)
                        .map(|x: (CsrMatrix<f64>, _, _)| x.0.select_axis(1, &selected_features)),
                )
            };

            let n_obs = $data.n_obs();
            let mut rng = rand::rngs::StdRng::seed_from_u64(2023);
            let idx = if weighted_by_degree {
                let weights = compute_probs(&compute_degrees($data, &selected_features, &weights));
                rand::seq::index::sample_weighted(&mut rng, n_obs, |i| weights[i], sample_size)?
                    .into_vec()
            } else {
                rand::seq::index::sample(&mut rng, n_obs, sample_size).into_vec()
            };
            let selected_samples = SelectInfoElem::from(idx);
            let mut seed_mat: CsrMatrix<f64> = $data
                .x()
                .slice(&[selected_samples, selected_features.clone()])?
                .unwrap();

            // feature weighting and L2 norm normalization.
            normalize(&mut seed_mat, &weights);

            info!("Compute embeddings for {} landmarks...", sample_size);
            let (v, mut u, d) = spectral_mf(seed_mat.clone(), n_components, 0)?;
            info!("Apply Nystrom to out-of-sample data...");
            nystrom(
                py,
                seed_mat,
                &v,
                &mut u,
                &d,
                $data.x().iter(chunk_size).map(|x: (CsrMatrix<f64>, _, _)| {
                    let mut mat = x.0.select_axis(1, &selected_features);
                    normalize(&mut mat, &weights);
                    mat
                }),
            )
        }};
    }

    let (evals, evecs) = crate::with_anndata!(&anndata, run)?;
    Ok((
        PyArray1::from_owned_array_bound(py, evals),
        PyArray2::from_owned_array_bound(py, evecs),
    ))
}

/// Matrix-free spectral embedding.
/// The input is assumed to be a csr matrix with rows normalized to unit L2 norm.
fn spectral_mf(
    mut input: CsrMatrix<f64>,
    n_components: usize,
    random_state: i64,
) -> Result<(Array1<f64>, Array2<f64>, Array1<f64>)> {
    // Compute degrees.
    let mut col_sum = vec![0.0; input.ncols()];
    input
        .col_indices()
        .iter()
        .zip(input.values().iter())
        .for_each(|(i, x)| col_sum[*i] += x);
    let mut degree_inv: DVector<_> = &input * &DVector::from(col_sum);
    degree_inv.iter_mut().for_each(|x| *x = (*x - 1.0).recip());

    // row-wise normalization using degrees.
    input
        .row_iter_mut()
        .zip(degree_inv.iter())
        .for_each(|(mut row, d)| row.values_mut().iter_mut().for_each(|x| *x *= d.sqrt()));

    // Compute eigenvalues and eigenvectors
    let (v, u) = Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code_bound(
            py,
            "def eigen(X, D, k, seed):
                from scipy.sparse.linalg import LinearOperator, eigsh
                import numpy
                numpy.random.seed(seed)
                def f(v):
                    return X @ (v.T @ X).T - D * v

                n = X.shape[0]
                A = LinearOperator((n, n), matvec=f, dtype=numpy.float64)
                evals, evecs = eigsh(A, k=k, v0=numpy.random.rand(n))
                ix = evals.argsort()[::-1]
                evals = evals[ix]
                evecs = evecs[:, ix]
                return (evals, evecs)",
            "",
            "",
        )?
        .getattr("eigen")?
        .into();
        let args = (
            PyArrayData::from(ArrayData::from(input)),
            PyArray1::from_iter_bound(py, degree_inv.into_iter().copied()),
            n_components,
            random_state,
        );
        let result = fun.call1(py, args)?;
        let (evals, evecs): (&PyArray1<f64>, &PyArray2<f64>) = result.extract(py)?;

        anyhow::Ok((evals.to_owned_array(), evecs.to_owned_array()))
    })?;
    degree_inv.iter_mut().for_each(|x| *x = x.recip());
    Ok((v, u, degree_inv.into_iter().copied().collect()))
}

/// The input is assumed to be a csr matrix with rows normalized to unit L2 norm.
fn nystrom<'py, I>(
    py: Python<'py>,
    seed_matrix: CsrMatrix<f64>,
    evals: &Array1<f64>,
    evecs: &mut Array2<f64>,
    degrees: &Array1<f64>,
    inputs: I,
) -> Result<(Array1<f64>, Array2<f64>)>
where
    I: IntoIterator<Item = CsrMatrix<f64>>,
    I::IntoIter: ExactSizeIterator,
{
    // normalize the eigenvectors by degrees.
    evecs
        .axis_iter_mut(Axis(0))
        .zip(degrees.iter())
        .for_each(|(mut row, d)| row *= d.sqrt().recip());
    // normalize the eigenvectors by eigenvalues.
    evecs
        .axis_iter_mut(Axis(1))
        .zip(evals.iter())
        .for_each(|(mut col, v)| col *= v.recip());
    let evecs_py = PyArray2::from_array_bound(py, evecs);
    let evals_py = PyArray1::from_array_bound(py, evals);
    let seed_matrix_py = PyArrayData::from(ArrayData::from(seed_matrix)).into_py(py);

    let nystrom_py: Py<PyAny> = PyModule::from_code_bound(
        py,
        "def nystrom(seed, sample, evecs, evals):
            import numpy as np
            q = sample @ (seed.T @ evecs)
            t = q.sum(axis=0) * evals
            d = q @ t.reshape((-1, 1))
            d[d<=0] = np.min(d[d>0])
            np.divide(q, np.sqrt(d), out=q)
            return q",
        "",
        "",
    )?
    .getattr("nystrom")?
    .into();

    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
    )
    .unwrap();
    let output: Vec<f64> = inputs
        .into_iter()
        .map(|sample| {
            let args = (
                &seed_matrix_py,
                PyArrayData::from(ArrayData::from(sample)),
                &evecs_py,
                &evals_py,
            );
            nystrom_py
                .call1(py, args)
                .unwrap()
                .extract::<&PyArray2<_>>(py)
                .unwrap()
                .to_vec()
                .unwrap()
        })
        .progress_with_style(style)
        .flatten()
        .collect();

    let ncol = evecs.ncols();
    let nrows = output.len() / ncol;
    Ok((
        evals.clone(),
        Array2::from_shape_vec((nrows, ncol), output).unwrap(),
    ))
}

fn idf(input: &CsrMatrix<f64>) -> Vec<f64> {
    let mut idf = vec![0.0; input.ncols()];
    input.col_indices().iter().for_each(|i| idf[*i] += 1.0);
    let n = input.nrows() as f64;
    if idf.iter().all_equal() {
        vec![1.0; idf.len()]
    } else {
        idf.iter_mut().for_each(|x| {
            if *x == 0.0 {
                *x = 1.0;
            } else if *x == n {
                *x = n - 1.0;
            }
            *x = (n / *x).ln()
        });
        idf
    }
}

fn idf_from_chunks<I>(input: I) -> Vec<f64>
where
    I: IntoIterator<Item = CsrMatrix<f64>>,
{
    let mut iter = input.into_iter().peekable();
    let mut idf = vec![0.0; iter.peek().unwrap().ncols()];
    let mut n = 0.0;
    iter.for_each(|mat| {
        mat.col_indices().iter().for_each(|i| idf[*i] += 1.0);
        n += mat.nrows() as f64;
    });
    if idf.iter().all_equal() {
        vec![1.0; idf.len()]
    } else {
        idf.iter_mut().for_each(|x| {
            if *x == 0.0 {
                *x = 1.0;
            } else if *x == n {
                *x = n - 1.0;
            }
            *x = (n / *x).ln()
        });
        idf
    }
}

/// feature weighting and L2 norm normalization.
fn normalize(input: &mut CsrMatrix<f64>, feature_weights: &[f64]) {
    input.row_iter_mut().par_bridge().for_each(|mut row| {
        let (indices, data) = row.cols_and_values_mut();
        indices
            .iter()
            .zip(data.iter_mut())
            .for_each(|(i, x)| *x *= feature_weights[*i]);

        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        data.iter_mut().for_each(|x| *x /= norm);
    });
}

fn compute_degrees<A: AnnDataOp>(
    adata: &A,
    selected_features: &SelectInfoElem,
    feature_weights: &[f64],
) -> Vec<f64> {
    let n = BoundedSelectInfoElem::new(selected_features, adata.n_vars()).len();
    let mut col_sum = vec![0.0; n];

    // First pass to compute the sum of each column.
    adata.x().iter(5000).for_each(|x: (CsrMatrix<f64>, _, _)| {
        let mut mat = x.0.select_axis(1, selected_features);
        normalize(&mut mat, feature_weights);
        mat.row_iter().for_each(|row| {
            row.col_indices()
                .iter()
                .zip(row.values().iter())
                .for_each(|(i, x)| col_sum[*i] += x);
        });
    });
    let col_sum = DVector::from(col_sum);

    // Second pass to compute the degree.
    adata
        .x()
        .iter(5000)
        .flat_map(|x: (CsrMatrix<f64>, _, _)| {
            let mut mat = x.0.select_axis(1, selected_features);
            normalize(&mut mat, feature_weights);
            let v = &mat * &col_sum;
            v.into_iter().map(|x| *x - 1.0).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn compute_probs(degrees: &[f64]) -> Vec<f64> {
    let s: f64 = degrees.iter().map(|x| x.recip()).sum();
    degrees.iter().map(|x| x.recip() / s).collect()
}

fn hstack(m1: CsrMatrix<f64>, m2: CsrMatrix<f64>) -> CsrMatrix<f64> {
    let c1 = m1.ncols();
    let vec = m1
        .row_iter()
        .zip(m2.row_iter())
        .map(|(r1, r2)| {
            let mut indices = r1.col_indices().to_vec();
            let mut data = r1.values().to_vec();
            indices.extend(r2.col_indices().iter().map(|x| x + c1));
            data.extend(r2.values().iter().map(|x| *x));
            indices
                .into_iter()
                .zip(data.into_iter())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (r, c, offset, ind, data) = to_csr_data(vec, c1 + m2.ncols());
    CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap()
}

/// Multi-view spectral embedding.
#[pyfunction]
pub(crate) fn multi_spectral_embedding<'py>(
    py: Python<'py>,
    anndata: Vec<AnnDataLike>,
    selected_features: Vec<Bound<'_, PyAny>>,
    weights: Vec<f64>,
    n_components: usize,
    random_state: i64,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    info!("Compute normalized views...");
    let mats = anndata
        .into_iter()
        .zip(selected_features.into_iter())
        .map(|(a, s)| {
            macro_rules! get_mat {
                ($data:expr) => {{
                    let slice = pyanndata::data::to_select_elem(&s, $data.n_vars())
                        .expect("Invalid feature selection");
                    let mut mat: CsrMatrix<f64> =
                        $data.x().slice_axis(1, slice).unwrap().expect("X is None");
                    let feature_weights = idf(&mat);

                    // feature weighting and L2 norm normalization.
                    normalize(&mut mat, &feature_weights);
                    let norm = if mat.nrows() <= 2000 {
                        frobenius_norm(&mat)
                    } else {
                        frobenius_norm(&sample_csr(&mat, 2000))
                    };
                    anyhow::Ok((norm, mat))
                }};
            }
            crate::with_anndata!(&a, get_mat).unwrap()
        })
        .collect::<Vec<_>>();
    let ws = mats
        .iter()
        .map(|x| x.0)
        .zip(weights.iter())
        .map(|(n, w)| w / n)
        .collect::<Vec<_>>();
    let w_sum = ws.iter().sum::<f64>();
    let mat = mats
        .into_iter()
        .zip(ws.into_iter())
        .map(|((_, mut mat), w)| {
            let w = (w / w_sum).sqrt();
            mat.values_mut().iter_mut().for_each(|x| *x *= w);
            mat
        })
        .reduce(|a, b| hstack(a, b))
        .unwrap();

    info!("Compute embedding...");
    let (evals, evecs, _) = spectral_mf(mat, n_components, random_state)?;
    Ok((
        PyArray1::from_owned_array_bound(py, evals),
        PyArray2::from_owned_array_bound(py, evecs),
    ))
}

fn frobenius_norm(x: &CsrMatrix<f64>) -> f64 {
    let sum: f64 = Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code_bound(
            py,
            "def f(X):
                import numpy as np
                return np.power(X @ X.T, 2).sum()",
            "",
            "",
        )?
        .getattr("f")?
        .into();
        let args = (PyArrayData::from(ArrayData::from(x)),);
        fun.call1(py, args)?.extract(py)
    })
    .unwrap();
    (sum - x.nrows() as f64).sqrt()
}

fn sample_csr(mat: &CsrMatrix<f64>, n: usize) -> CsrMatrix<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2023);
    let idx = rand::seq::index::sample(&mut rng, mat.nrows(), n).into_vec();
    mat.select_axis(0, SelectInfoElem::from(idx))
}
