use crate::utils::AnnDataLike;

use anndata_hdf5::H5;
use std::ops::Deref;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyanndata::data::PyArrayData;
use pyo3::prelude::*;
use anndata::{ArrayData, AnnDataOp, ArrayElemOp, data::{SelectInfoElem, DynCsrMatrix}, Backend};
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use anyhow::Result;

#[pyfunction]
pub(crate) fn spectral_embedding<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    selected_features: &PyAny,
    n_components: usize,
) -> Result<(&'py PyArray1<f64>, &'py PyArray2<f64>)>
{
    macro_rules! run {
        ($data:expr) => {{
            let slice = pyanndata::data::to_select_elem(selected_features, $data.n_vars())?;
            _spectral_embedding($data, &slice, n_components)
        }}
    }
    let (evals, evecs) = crate::with_anndata!(&anndata, run)?;

    Ok((PyArray1::from_owned_array(py, evals), PyArray2::from_owned_array(py, evecs)))
}

fn _spectral_embedding<A: AnnDataOp>(
    adata: &A,
    features: &SelectInfoElem,
    n_components: usize,
) -> Result<(Array1<f64>, Array2<f64>)>
{
    let mat: CsrMatrix<f64> = adata.x().slice_axis::<DynCsrMatrix, _>(1, features)?.unwrap().try_into()?;
    let idf = idf(&mat);
    spectral_mf(mat, &idf, n_components)
}

fn idf(input: &CsrMatrix<f64>) -> Vec<f64> {
    let mut idf = vec![0.0; input.ncols()];
    input.col_indices().iter().for_each(|i| idf[*i] += 1.0);
    let n = input.nrows() as f64;
    idf.iter_mut().for_each(|x| *x = (n / *x).ln());
    idf
}

/// Matrix-free spectral embedding.
fn spectral_mf(
    mut input: CsrMatrix<f64>,
    feature_weights: &[f64],
    n_components: usize,
) -> Result<(Array1<f64>, Array2<f64>)>
{
    // feature weighting and L2 norm normalization.
    input.row_iter_mut().par_bridge().for_each(|mut row| {
        let (indices, data) = row.cols_and_values_mut();
        indices.iter().zip(data.iter_mut()).for_each(|(i, x)|
            *x *= feature_weights[*i]
        );

        let norm = data.iter().map(|x| x*x).sum::<f64>().sqrt();
        data.iter_mut().for_each(|x| *x /= norm);
    });

    // Compute degrees.
    let mut col_sum = vec![0.0; input.ncols()];
    input.col_indices().iter().zip(input.values().iter()).for_each(|(i, x)| col_sum[*i] += x);
    let mut degree_inv: DVector<_> = &input * &DVector::from(col_sum);
    degree_inv.iter_mut().for_each(|x| *x = (*x - 1.0).recip());

    // row-wise normalization using degrees.
    input.row_iter_mut().zip(degree_inv.iter()).for_each(|(mut row, d)|
        row.values_mut().iter_mut().for_each(|x| *x *= d.sqrt())
    );

    // Compute eigenvalues and eigenvectors
    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            "def eigen(X, D, k):
                from scipy.sparse.linalg import LinearOperator, eigsh
                import numpy
                def f(v):
                    return X @ (v.T @ X).T - D * v

                n = X.shape[0]
                A = LinearOperator((n, n), matvec=f, dtype=numpy.float64)
                evals, evecs = eigsh(A, k=k)
                ix = evals.argsort()[::-1]
                evals = evals[ix]
                evecs = evecs[:, ix]
                return (evals, evecs)",
            "",
            "",
        )?.getattr("eigen")?.into();
        let args = (
            PyArrayData::from(ArrayData::from(input)),
            PyArray1::from_iter(py, degree_inv.into_iter().copied()),
            n_components,
        );
        let result = fun.call1(py, args)?;
        let (evals, evecs): (&PyArray1<f64>, &PyArray2<f64>) = result.extract(py)?;
        Ok((evals.to_owned_array(), evecs.to_owned_array()))
    })
}