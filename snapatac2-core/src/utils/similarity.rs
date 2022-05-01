use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::csr::CsrRowMut;
use ndarray::{Array2, Axis};

pub fn jaccard(mat: &SparsityPattern) -> Array2<f64> {
    let n = mat.major_dim();
    let mut res = Array2::from_diag_elem(n, 1.);

    {
        let mat_t = mat.transpose();
        res.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
            mat.get_lane(i).unwrap().into_iter().for_each(|k|
                mat_t.get_lane(*k).unwrap().into_iter().for_each(|j|
                    if *j > i { row[[*j]] += 1.0; }
                )
            );
        });
    }

    let sizes: Vec<usize> = (0..n).into_par_iter()
        .map(|i| mat.get_lane(i).unwrap().len()).collect();

    (0..n).combinations(2).for_each(|x| {
        let i = x[0];
        let j = x[1];
        let u = sizes[i] + sizes[j];
        let intersect = res[[i, j]];
        let v = if u == 0 { 1.0 } else { intersect / (u as f64 - intersect) };
        res[[i, j]] = v;
        res[[j, i]] = v;
    });
    res
}

pub fn jaccard2(mat1: &SparsityPattern, mat2: &SparsityPattern) -> Array2<f64> {
    let n = mat1.major_dim();
    let m = mat2.major_dim();
    let mut res = Array2::zeros((n, m));

    {
        let mat2_t = mat2.transpose();
        res.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
            mat1.get_lane(i).unwrap().into_iter().for_each(|k|
                mat2_t.get_lane(*k).unwrap().into_iter().for_each(|j| {
                    row[[*j]] += 1.0;
                })
            );
        });
    }

    let sizes1: Vec<usize> = (0..n).into_par_iter()
        .map(|i| mat1.get_lane(i).unwrap().len()).collect();
    let sizes2: Vec<usize> = (0..m).into_par_iter()
        .map(|i| mat2.get_lane(i).unwrap().len()).collect();
    res.indexed_iter_mut().for_each(|((i,j), x)| {
        let u = sizes1[i] + sizes2[j];
        let v = if u == 0 { 1.0 } else { *x / (u as f64 - *x) };
        *x = v;
    });
    res
}

pub fn cosine(mut mat: CsrMatrix<f64>) -> Array2<f64> {
    let n = mat.nrows();
    let mut res = Array2::zeros((n, n));

    {
        let norms: Vec<f64> = mat.row_iter_mut().map(l2_norm).collect();
        norms.into_iter().enumerate().for_each(|(i, x)| if x != 0. {
            res[[i, i]] = 1.;
        });
    }

    {
        let mat_t = mat.transpose();
        res.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
            let csr1 = mat.get_row(i).unwrap();
            csr1.col_indices().into_iter().zip(csr1.values()).for_each(|(k, v1)| {
                let csr2 = mat_t.get_row(*k).unwrap();
                csr2.col_indices().into_iter().zip(csr2.values()).for_each(|(j, v2)|
                    if *j > i { row[[*j]] += v1 * v2; }
                )
            });
        });
    }

    (0..n).combinations(2).for_each(|x| {
        let i = x[0];
        let j = x[1];
        res[[j, i]] = res[[i, j]];
    });
    res
}

pub fn cosine2(mut mat1: CsrMatrix<f64>, mut mat2: CsrMatrix<f64>) -> Array2<f64> {
    let n = mat1.nrows();
    let m = mat2.nrows();
    let mut res = Array2::zeros((n, m));

    mat1.row_iter_mut().for_each(|x| { l2_norm(x); });
    mat2.row_iter_mut().for_each(|x| { l2_norm(x); });

    {
        let mat2_t = mat2.transpose();
        res.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
            let csr1 = mat1.get_row(i).unwrap();
            csr1.col_indices().into_iter().zip(csr1.values()).for_each(|(k, v1)| {
                let csr2 = mat2_t.get_row(*k).unwrap();
                csr2.col_indices().into_iter().zip(csr2.values()).for_each(|(j, v2)| {
                    row[[*j]] += v1 * v2;
                });
            });
        });
    }
    res
}


fn l2_norm(mut row: CsrRowMut<'_, f64>) -> f64 {
    let max = row.values().into_iter().max_by(
        |x, y| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap_or(&0.0).abs();
    if max != 0.0 {
        let mut norm = row.values().into_iter().map(|x| {
            let x_ = x / max;
            x_ * x_
        }).sum::<f64>().sqrt();
        norm *= max;
        row.values_mut().into_iter().for_each(|x| { *x /= norm; });
        norm
    } else {
        0.0
    }
}