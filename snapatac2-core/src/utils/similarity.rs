use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use nalgebra_sparse::pattern::SparsityPattern;
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

/*
pub fn jaccard(mat: &SparsityPattern) -> Vec<f64> {
    let mat_t = mat.transpose();
    let n = mat.major_dim();
    let sizes: Vec<usize> = (0..n).into_par_iter().map(|i| mat.get_lane(i).unwrap().len()).collect();
    (0..n).into_par_iter().flat_map(|i| {
        let mut v: Vec<f64> = vec![0.0; n - i - 1];
        mat.get_lane(i).unwrap().into_iter().for_each(|k| {
            mat_t.get_lane(*k).unwrap().into_iter().for_each(|j| {
                if *j > i {
                    v[*j - i - 1] += 1.0;
                }
            })
        });
        v.iter_mut().enumerate().for_each(|(k, x)| {
            let u = (sizes[i + k + 1] + sizes[i]) as f64;
            *x = *x / (u - *x);
        });
        v
    }).collect()
}
*/

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard() {
        let x = vec![3, 5, 7];
        let y = vec![1, 3, 5, 7, 9, 10, 11];

        assert_eq!(jaccard(x.as_slice(), y.as_slice()), 3.0 / 7.0);
    }

}
*/