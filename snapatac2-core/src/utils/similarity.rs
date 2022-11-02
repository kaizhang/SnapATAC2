use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use nalgebra_sparse::csr::{CsrMatrix, CsrRowMut, CsrRow};
use std::ops::Deref;
use ndarray::{ArrayBase, Array2, Axis};
use statrs::statistics::{Data, Statistics, OrderStatistics, RankTieBreaker};
use num::{One, NumCast, ToPrimitive, Float};

pub type BorrowedSparsityPattern<'a, I> = SparsityPatternBase<&'a [I], &'a [I]>;

impl<'a, I> BorrowedSparsityPattern<'a, I> {
    pub fn new(
        major_offsets: &'a [I],
        minor_indices: &'a [I],
        minor_dim: usize
    ) -> Self {
        Self { major_offsets, minor_indices, minor_dim }
    }
}

pub type OwnedSparsityPattern<I> = SparsityPatternBase<Vec<I>, Vec<I>>;

pub struct SparsityPatternBase<T1, T2> {
    major_offsets: T1,
    minor_indices: T2,
    minor_dim: usize,
}

impl<T1, T2, I> SparsityPatternBase<T1, T2>
where
    T1: Deref<Target = [I]>,
    T2: Deref<Target = [I]>,
{
    pub fn major_dim(&self) -> usize {
        assert!(self.major_offsets.len() > 0);
        self.major_offsets.len() - 1
    }

    pub fn major_offsets(&self) -> &[I] {
        self.major_offsets.deref()
    }

    pub fn minor_indices(&self) -> &[I] {
        self.minor_indices.deref()
    }

    pub fn get_lane(&self, major_index: usize) -> Option<&[I]>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let offset_begin: usize = (*self.major_offsets().get(major_index)?).try_into().unwrap();
        let offset_end: usize = (*self.major_offsets().get(major_index + 1)?).try_into().unwrap();
        Some(&self.minor_indices()[offset_begin..offset_end])
    }

    pub fn transpose(&self) -> OwnedSparsityPattern<I>
    where
        I: TryInto<usize> + Copy + TryFrom<usize> + num::Bounded,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
        <I as TryFrom<usize>>::Error: std::fmt::Debug,
    {
        let n = self.major_dim();
        let (new_offsets, new_minor_indices) = transpose_cs(
            n,
            self.minor_dim,
            self.major_offsets.deref(),
            self.minor_indices.deref(),
        );
        SparsityPatternBase {
            major_offsets: new_offsets,
            minor_indices: new_minor_indices,
            minor_dim: n,
        }
    }
}

/// Compute the Jaccard index between rows of a matrix.
pub fn jaccard<'a, I>(
    mat: BorrowedSparsityPattern<'a, I>,
    weights: Option<&[f64]>,
) -> Array2<f64>
where
    I: TryInto<usize> + Copy + TryFrom<usize> + num::Bounded + std::marker::Sync,
    <I as TryInto<usize>>::Error: std::fmt::Debug,
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let n = mat.major_dim();
    let mut res = Array2::from_diag_elem(n, 1.);

    {
        let mat_t = mat.transpose();
        res.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
            mat.get_lane(i).unwrap().into_iter().for_each(|k| {
                let k_ = (*k).try_into().unwrap();
                mat_t.get_lane(k_).unwrap().into_iter().for_each(|j| {
                    let j_ = (*j).try_into().unwrap();
                    if j_ > i { row[[j_]] += weights.map_or(1.0, |w| w[k_]); }
                })
            });
        });
    }

    let sizes: Vec<f64> = (0..n).into_par_iter().map(|i|
        mat.get_lane(i).unwrap().iter().map(|j|
            weights.map_or(1.0, |w| w[(*j).try_into().unwrap()])
        ).sum()
    ).collect();

    (0..n).combinations(2).for_each(|x| {
        let i = x[0];
        let j = x[1];
        let u = sizes[i] + sizes[j];
        let intersect = res[[i, j]];
        let v = if u == 0.0 { 1.0 } else { intersect / (u - intersect) };
        res[[i, j]] = v;
        res[[j, i]] = v;
    });
    res
}

/// Compute the Jaccard index between rows of two matrices.
pub fn jaccard2<'a, I1, I2>(
    mat1: BorrowedSparsityPattern<'a, I1>,
    mat2: BorrowedSparsityPattern<'a, I2>,
    weights: Option<&[f64]>,
) -> Array2<f64>
where
    I1: TryInto<usize> + Copy + TryFrom<usize> + num::Bounded + std::marker::Sync,
    <I1 as TryInto<usize>>::Error: std::fmt::Debug,
    <I1 as TryFrom<usize>>::Error: std::fmt::Debug,
    I2: TryInto<usize> + Copy + TryFrom<usize> + num::Bounded + std::marker::Sync,
    <I2 as TryInto<usize>>::Error: std::fmt::Debug,
    <I2 as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let n = mat1.major_dim();
    let m = mat2.major_dim();
    let mut res = Array2::zeros((n, m));

    {
        let mat2_t = mat2.transpose();
        res.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {
            mat1.get_lane(i).unwrap().into_iter().for_each(|k| {
                let k_ = (*k).try_into().unwrap();
                mat2_t.get_lane(k_).unwrap().into_iter().for_each(|j| {
                    row[[(*j).try_into().unwrap()]] += weights.map_or(1.0, |w| w[k_]);
                })
            });
        });
    }

    let sizes1: Vec<f64> = (0..n).into_par_iter().map(|i|
        mat1.get_lane(i).unwrap().iter().map(|j|
            weights.map_or(1.0, |w| w[(*j).try_into().unwrap()])
        ).sum()
    ).collect();
    let sizes2: Vec<f64> = (0..m).into_par_iter().map(|i|
        mat2.get_lane(i).unwrap().iter().map(|j|
            weights.map_or(1.0, |w| w[(*j).try_into().unwrap()])
        ).sum()
    ).collect();
    res.indexed_iter_mut().for_each(|((i,j), x)| {
        let u = sizes1[i] + sizes2[j];
        let v = if u == 0.0 { 1.0 } else { *x / (u - *x) };
        *x = v;
    });
    res
}

pub fn cosine(
    mut mat: CsrMatrix<f64>,
    weights: Option<&[f64]>,
) -> Array2<f64> {
    let n = mat.nrows();
    let mut res = Array2::zeros((n, n));

    {
        let norms: Vec<f64> = mat.row_iter_mut().map(|mut x| {
            if let Some(w) = weights {
                let (indices, values) = x.cols_and_values_mut();
                indices.into_iter().zip(values).for_each(|(i, v)| *v = *v * w[*i].sqrt());
            }
            l2_norm(x)
        }).collect();
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

pub fn cosine2(
    mut mat1: CsrMatrix<f64>,
    mut mat2: CsrMatrix<f64>,
    weights: Option<&[f64]>,
) -> Array2<f64> {
    let n = mat1.nrows();
    let m = mat2.nrows();
    let mut res = Array2::zeros((n, m));

    {
        mat1.row_iter_mut().for_each(|mut x| {
            if let Some(w) = weights {
                let (indices, values) = x.cols_and_values_mut();
                indices.into_iter().zip(values).for_each(|(i, v)| *v = *v * w[*i].sqrt());
            }
            l2_norm(x);
        });

        mat2.row_iter_mut().for_each(|mut x| {
            if let Some(w) = weights {
                let (indices, values) = x.cols_and_values_mut();
                indices.into_iter().zip(values).for_each(|(i, v)| *v = *v * w[*i].sqrt());
            }
            l2_norm(x);
        });
    }

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
    let max = row.values().into_iter().map(|x| x.abs()).reduce(f64::max).unwrap_or(0.0);
    let norm = if max != 0.0 {
        let mut norm = row.values().into_iter().map(|x| {
            let x_ = x / max;
            x_ * x_
        }).sum::<f64>().sqrt();
        norm *= max;
        norm
    } else {
        f64::NAN
    };
    row.values_mut().into_iter().for_each(|x| { *x /= norm; });
    norm
}

pub fn transpose_cs<I>(
    major_dim: usize,
    minor_dim: usize,
    source_major_offsets: &[I],
    source_minor_indices: &[I],
) -> (Vec<I>, Vec<I>)
where
    I: TryInto<usize> + Copy + TryFrom<usize> + num::Bounded,
    <I as TryInto<usize>>::Error: std::fmt::Debug,
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    assert_eq!(source_major_offsets.len(), major_dim + 1);
    let nnz = source_minor_indices.len();

    // Count the number of occurences of each minor index
    let mut minor_counts = vec![0; minor_dim];
    for minor_idx in source_minor_indices {
        minor_counts[(*minor_idx).try_into().unwrap()] += 1;
    }
    convert_counts_to_offsets(&mut minor_counts);
    let mut target_offsets = minor_counts;
    target_offsets.push(nnz);
    let mut target_indices = vec![num::Bounded::max_value(); nnz];

    // Keep track of how many entries we have placed in each target major lane
    let mut current_target_major_counts = vec![0; minor_dim];

    for source_major_idx in 0..major_dim {
        let source_lane_begin: usize = source_major_offsets[source_major_idx].try_into().unwrap();
        let source_lane_end: usize = source_major_offsets[source_major_idx + 1].try_into().unwrap();
        let source_lane_indices = &source_minor_indices[source_lane_begin..source_lane_end];

        for &source_minor_idx in source_lane_indices.iter() {
            // Compute the offset in the target data for this particular source entry
            let target_lane_count = &mut current_target_major_counts[source_minor_idx.try_into().unwrap()];
            let entry_offset = target_offsets[source_minor_idx.try_into().unwrap()] + *target_lane_count;
            target_indices[entry_offset] = source_major_idx.try_into().unwrap();
            *target_lane_count += 1;
        }
    }

    (target_offsets.into_iter().map(|x| x.try_into().unwrap()).collect(), target_indices)
}

fn convert_counts_to_offsets(counts: &mut [usize]) {
    // Convert the counts to an offset
    let mut offset = 0;
    for i_offset in counts.iter_mut() {
        let count = *i_offset;
        *i_offset = offset;
        offset += count;
    }
}

pub fn pearson2<T>(
    mut mat1: Array2<T>,
    mut mat2: Array2<T>,
) -> Array2<T>
where
    T: Float + ToPrimitive + NumCast + std::iter::Sum + One,
{
    fn sum_of_square_centered<T>(mat: &mut Array2<T>) -> Vec<T>
    where
        T: Float + ToPrimitive + NumCast + std::iter::Sum,
    {
        mat.rows_mut().into_iter().map(|mut row| {
            let mean = NumCast::from(row.iter().map(|x| x.to_f64().unwrap()).mean()).unwrap();
            row.iter_mut().map(|x| {
                let v = *x - mean;
                *x = v;
                v * v
            }).sum()
        }).collect()
    }

    if mat1.shape()[1] != mat2.shape()[1] {
        panic!("number of columns mismatch");
    }

    let ss1 = sum_of_square_centered(&mut mat1);
    let ss2 = sum_of_square_centered(&mut mat2);

    let n1 = mat1.shape()[0];
    let n2 = mat2.shape()[0];

    let cor: Vec<_> = mat1.rows().into_iter().enumerate().flat_map(|(i, x)| {
        mat2.rows().into_iter().enumerate().map(|(j, y)| {
            let s: T = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
            let r: T = s / (ss1[i] * ss2[j]).sqrt();
            if r.is_nan() {
                r
            } else {
                r.min(T::one()).max(-T::one())
            }
        }).collect::<Vec<_>>()
    }).collect();
    ArrayBase::from_vec(cor).into_shape((n1, n2)).unwrap()
}

pub fn spearman2<T1, T2>(
    mat1: Array2<T1>,
    mat2: Array2<T2>,
) -> Array2<f64>
where
    T1: ToPrimitive,
    T2: ToPrimitive,
{
    fn rank<T>(mat: Array2<T>) -> Array2<f64>
    where
        T: ToPrimitive,
    {
        let vec: Vec<_> = mat.rows().into_iter().flat_map(|row|
            Data::new(row.into_iter().map(|x| x.to_f64().unwrap()).collect::<Vec<_>>()).ranks(RankTieBreaker::Average)
        ).collect();
        ArrayBase::from_vec(vec).into_shape((mat.shape()[0], mat.shape()[1])).unwrap()
    }
    let mat1_ = rank(mat1);
    let mat2_ = rank(mat2);
    pearson2(mat1_, mat2_)
}

/// Compute the distance matrix from X and Y.
/// 
/// # Arguments
/// * `mat_x` - matrix of shape (n_samples_X, n_features).
/// * `mat_y` - matrix of shape (n_samples_Y, n_features).
/// * `dist_fn` - distance function.
pub fn pairwise_distances<F>(
    mat_x: CsrMatrix<f64>,
    mat_y: CsrMatrix<f64>,
    dist_fn: F,
) -> Array2<f64>
where
    F: Fn(CsrRow<'_, f64>, CsrRow<'_, f64>) -> f64 + std::marker::Sync,
{
    let n1 = mat_x.nrows();
    let n2 = mat_y.nrows();
    let dists = (0..n1).cartesian_product(0..n2).collect::<Vec<_>>().into_par_iter()
        .map(|(i, j)| dist_fn(mat_x.get_row(i).unwrap(), mat_y.get_row(j).unwrap())).collect();
    ArrayBase::from_vec(dists).into_shape((n1, n2)).unwrap()
}