use nalgebra_sparse::CsrMatrix;
use ndarray::{AsArray, Ix2};
use kdtree::kdtree::KdTree;
use smallvec::SmallVec;
use rayon::{iter::{IntoParallelIterator, IndexedParallelIterator}, prelude::ParallelIterator};
use hora::core::ann_index::ANNIndex;
use kdtree::distance::squared_euclidean;

pub fn nearest_neighbour_graph<'a, A>(
    points: A,
    k: usize,
) -> CsrMatrix<f64>
where
    A: AsArray<'a, f64, Ix2>
{
    let points = points.into();
    let dimensions = points.shape()[1];
    let mut kdtree = KdTree::new(dimensions);
    points.outer_iter().enumerate().for_each(|(i, point)| {
        kdtree.add(point.into_iter().cloned().collect::<SmallVec<[f64;64]>>(), i).unwrap();
    });

    let result = points.outer_iter().into_par_iter().enumerate().map(|(i, point)| {
        let point = point.into_iter().cloned().collect::<SmallVec<[f64;64]>>();
        kdtree.iter_nearest(point.as_slice(), &squared_euclidean)
            .unwrap()
            .filter_map(|(distance, index)| if *index == i { None } else { Some((*index, distance.sqrt())) })
            .take(k)
            .collect()
    }).collect::<Vec<_>>();
    to_csr_matrix(result)
}

pub fn approximate_nearest_neighbour_graph<'a, A>(
    points: A,
    k: usize,
) -> CsrMatrix<f32>
where
    A: AsArray<'a, f32, Ix2>
{
    let points = points.into();
    let shape = points.shape();
    let mut index = hora::index::hnsw_idx::HNSWIndex::<f32, usize>::new(
        shape[1],
        &hora::index::hnsw_params::HNSWParams::<f32>::default().max_item(shape[0].max(1000000)),
    );
    for (i, sample) in points.outer_iter().enumerate() {
        index.add(sample.to_vec().as_slice(), i).unwrap();
    }
    index.build(hora::core::metrics::Metric::Euclidean).unwrap();
    let result = points.outer_iter().into_par_iter().map(|row| {
        index.search_nodes(row.to_vec().as_slice(), k).into_iter()
            .map(|(n, d)| (n.idx().unwrap(), d)).collect::<Vec<_>>()
    }).collect::<Vec<_>>();
    to_csr_matrix(result)
}

fn to_csr_matrix<I, D>(iter: I) -> CsrMatrix<D>
where
    I: IntoIterator<Item = Vec<(usize, D)>>,
{
    let mut data: Vec<D> = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::new();

    let n = iter.into_iter().fold(0, |r_idx, mut row| {
        row.sort_by(|a, b| a.0.cmp(&b.0));
        indptr.push(r_idx.try_into().unwrap());
        let new_idx = r_idx + row.len();
        let (mut a, mut b) = row.into_iter().unzip();
        indices.append(&mut a);
        data.append(&mut b);
        new_idx
    });
    indptr.push(n.try_into().unwrap());
    CsrMatrix::try_from_csr_data(
        indptr.len() - 1, indptr.len() - 1, indptr, indices, data
    ).unwrap()
}