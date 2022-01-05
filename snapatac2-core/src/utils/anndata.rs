use crate::{ResizableVectorData, create_str_attr};

use hdf5::{H5Type, Result, Group};
use std::ops::Deref; 
use ndarray::{arr1, Array, Dimension};
use itertools::Itertools;

pub struct SparseRowIter<I> {
    iter: I,
    num_col: usize,
}

impl<I, D> SparseRowIter<I>
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    pub fn new(iter: I, num_col: usize) -> Self { SparseRowIter {iter, num_col} }
}

pub trait AnnData {
    type Container;

    const VERSION: &'static str;

    fn create<T>(self, location: &T, name: &str) -> Result<Self::Container>
    where T: Deref<Target = Group>;
}

impl<A, D> AnnData for Array<A, D>
where
    A: H5Type,
    D: Dimension,
{
    type Container = hdf5::Dataset;
    const VERSION: &'static str = "0.2.0";

    fn create<T>(self, location: &T, name: &str) -> Result<Self::Container>
    where T: Deref<Target = Group>,
    {
        let dataset = location.new_dataset_builder().deflate(9)
            .with_data(&self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", Self::VERSION)?;

        Ok(dataset)
    }

}

impl<I, D> AnnData for SparseRowIter<I>
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    type Container = Group;
    const VERSION: &'static str = "0.1.0";

    /// Compressed Row Storage (CRS) stores the sparse matrix in 3 vectors:
    /// one for floating-point numbers (data), and the other two for integers (indices, indptr).
    /// The `data` vector stores the values of the nonzero elements of the matrix,
    /// as they are traversed in a row-wise fashion.
    /// The `indices` vector stores the column indexes of the elements in the `data` vector.
    /// The `indptr` vector stores the locations in the `data` vector that start a row,
    /// The last element is NNZ.
    fn create<T>(self, location: &T, name: &str) -> Result<Self::Container>
    where T: Deref<Target = Group>,
    {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", Self::VERSION)?;

        let data: ResizableVectorData<D> = ResizableVectorData::new(&group, "data", 100000)?;
        let indices: ResizableVectorData<u32> = ResizableVectorData::new(&group, "indices", 100000)?;
        let mut indptr: Vec<u32> = vec![0];
        let iter = self.iter.scan(0, |state, x| {
            *state = *state + x.len();
            Some((*state, x))
        });
        for chunk in &iter.chunks(1000) {
            let (a, b): (Vec<u32>, Vec<D>) = chunk.map(|(x, vec)| {
                indptr.push(x.try_into().unwrap());
                vec
            }).flatten().map(|(x, y)| -> (u32, D) {
                (x.try_into().unwrap(), y) }).unzip();
            indices.extend(a.into_iter())?;
            data.extend(b.into_iter())?;
        }
        group.new_attr_builder()
            .with_data(&arr1(&[indptr.len() - 1, self.num_col]))
            .create("shape")?;

        group.new_dataset_builder().deflate(9)
            .with_data(&Array::from_vec(indptr)).create("indptr")?;
        Ok(group)
    }
}