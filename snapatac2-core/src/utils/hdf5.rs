use hdf5::{Location, Error, Selection, H5Type, Result, Extent, Group};
use hdf5::dataset::Dataset;
use std::marker::PhantomData;
use ndarray::{Dimension, Array, ArrayView};
use itertools::Itertools;

pub const COMPRESSION: u8 = 1;

pub fn create_str_attr(location: &Location, name: &str, value: &str) -> Result<()>
{
    let attr = location.new_attr::<hdf5::types::VarLenUnicode>().create(name)?;
    let value_: hdf5::types::VarLenUnicode = value.parse().unwrap();
    attr.write_scalar(&value_)
}

pub fn read_str_attr(location: &Location, name: &str) -> Result<String>
{
    let attr: hdf5::types::VarLenUnicode = location.attr(name)?.read_scalar()?;
    Ok(attr.parse().unwrap())
}

pub struct ResizableVectorData<T> {
    dataset: Dataset,
    dataset_type: PhantomData<T>,
}

impl<T: H5Type> ResizableVectorData<T> {
    pub fn new(group: &Group, name: &str, chunk_size: usize) -> Result<Self> {
        let dataset = group.new_dataset::<T>().deflate(COMPRESSION).chunk(chunk_size)
            .shape(Extent::resizable(0)).create(name)?;
        Ok(ResizableVectorData { dataset, dataset_type: PhantomData })
    }

    /// Returns the current size of the vector.
    pub fn size(&self) -> usize { self.dataset.shape()[0] }

    /// Resizes the dataset to a new length.
    pub fn resize(&self, size: usize) -> Result<()> {
        self.dataset.resize(size)
    }

    /// Returns the chunk size of the vector.
    pub fn chunk_size(&self) -> usize { self.dataset.chunk().unwrap()[0] }

    pub fn extend<I>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = T>,
    {
        let arr = Array::from_iter(iter);
        let n = arr.raw_dim().size();
        let old_size = self.size();
        let new_size = old_size + n;
        self.resize(new_size)?;
        self.write_slice(&arr, old_size..new_size)
    }

    pub fn extend_by<I>(&self, iter: I, step: usize) -> Result<()>
    where
        I: Iterator<Item = T>,
    {
        for chunk in &iter.chunks(step) {
            self.extend(chunk)?;
        }
        Ok(())
    }

    fn write_slice<'a, A, S, D>(&self, arr: A, selection: S) -> Result<()>
    where
        A: Into<ArrayView<'a, T, D>>,
        S: TryInto<Selection>,
        Error: From<S::Error>,
        D: Dimension,
    {
        self.dataset.write_slice(arr, selection)
    }

}

/*
impl Extend<T: H5Type> for ResizableVectorData<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = A>,
    {

    }

}
*/