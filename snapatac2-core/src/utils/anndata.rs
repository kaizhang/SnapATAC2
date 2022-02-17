use crate::utils::hdf5::{ResizableVectorData, create_str_attr, COMPRESSION};
use crate::qc::{CellBarcode, QualityControl};

use hdf5::{File, H5Type, Result, Group, types::VarLenUnicode};
use ndarray::{arr1, Array1, Array, Dimension};
use itertools::Itertools;

pub struct StrVec(pub Vec<String>);

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
    pub fn to_csr_matrix(self) -> (Vec<D>, Vec<i32>, Vec<i32>)
    {
        let mut data: Vec<D> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i32> = Vec::new();

        let n = self.iter.fold(0, |r_idx, row| {
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
}

pub trait AnnData {
    type Container;

    const VERSION: &'static str;

    fn create(self, location: &Group, name: &str) -> Result<Self::Container>;
}

impl AnnData for StrVec {
    type Container = hdf5::Dataset;
    const VERSION: &'static str = "0.2.0";

    fn create(self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let data: Array1<VarLenUnicode> = self.0.into_iter()
            .map(|x| x.parse::<VarLenUnicode>().unwrap()).collect();
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&data).create(name)?;
        create_str_attr(&*dataset, "encoding-type", "string-array")?;
        create_str_attr(&*dataset, "encoding-version", Self::VERSION)?;

        Ok(dataset)
    }
}

impl<A, D> AnnData for Array<A, D>
where
    A: H5Type,
    D: Dimension,
{
    type Container = hdf5::Dataset;
    const VERSION: &'static str = "0.2.0";

    fn create(self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
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
    fn create(self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", Self::VERSION)?;

        let data: ResizableVectorData<D> = ResizableVectorData::new(&group, "data", 10000)?;
        let indices: ResizableVectorData<i32> = ResizableVectorData::new(&group, "indices", 10000)?;
        let mut indptr: Vec<i32> = vec![0];
        let iter = self.iter.scan(0, |state, x| {
            *state = *state + x.len();
            Some((*state, x))
        });
        for chunk in &iter.chunks(10000) {
            let (a, b): (Vec<i32>, Vec<D>) = chunk.map(|(x, vec)| {
                indptr.push(x.try_into().unwrap());
                vec
            }).flatten().map(|(x, y)| -> (i32, D) {
                (x.try_into().unwrap(), y) }).unzip();
            indices.extend(a.into_iter())?;
            data.extend(b.into_iter())?;
        }
        group.new_attr_builder()
            .with_data(&arr1(&[indptr.len() - 1, self.num_col]))
            .create("shape")?;

        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&Array::from_vec(indptr)).create("indptr")?;
        Ok(group)
    }
}

pub fn create_obs(
    file: &File,
    cells: Vec<CellBarcode>,
    optional_qc: Option<Vec<QualityControl>>,
    ) -> Result<()>
{
    let group = file.create_group("obs")?;
    create_str_attr(&group, "encoding-type", "dataframe")?;
    create_str_attr(&group, "encoding-version", "0.2.0")?;
    create_str_attr(&group, "_index", "Cell")?;
    StrVec(cells).create(&group, "Cell")?;

    match optional_qc {
        Some(qc) => {
            let columns: Array1<hdf5::types::VarLenUnicode> =
                ["tsse", "n_fragment", "frac_dup", "frac_mito"]
                .into_iter().map(|x| x.parse().unwrap()).collect();
            group.new_attr_builder().with_data(&columns).create("column-order")?;
            qc.iter().map(|x| x.tss_enrichment).collect::<Array1<f64>>().create(&group, "tsse")?;
            qc.iter().map(|x| x.num_unique_fragment).collect::<Array1<u64>>().create(&group, "n_fragment")?;
            qc.iter().map(|x| x.frac_duplicated).collect::<Array1<f64>>().create(&group, "frac_dup")?;
            qc.iter().map(|x| x.frac_mitochondrial).collect::<Array1<f64>>().create(&group, "frac_mito")?;
        },
        _ => {
            let columns: Array1<hdf5::types::VarLenUnicode> = [].into_iter().collect();
            group.new_attr_builder().with_data(&columns).create("column-order")?;
        },
    }

    Ok(())
}

pub fn create_var(file: &File, features: Vec<String>) -> Result<()> {
    let group = file.create_group("var")?;
    create_str_attr(&group, "encoding-type", "dataframe")?;
    create_str_attr(&group, "encoding-version", "0.2.0")?;
    create_str_attr(&group, "_index", "Region")?;
    let columns: Array1<hdf5::types::VarLenUnicode> = [].into_iter().collect();
    group.new_attr_builder().with_data(&columns).create("column-order")?;
    StrVec(features).create(&group, "Region")?;
    Ok(())
}

#[cfg(test)]
mod sm_tests {
    use super::*;

    #[test]
    fn test_sparse_row() {
        let matrix =
            [ vec![(0, 1), (2, 2)]
            , vec![(2, 3)]
            , vec![(0, 4), (1, 5), (2, 6)] ];
        let iter = SparseRowIter::new(matrix.into_iter(), 3);

        let file = File::create("1.h5").unwrap();
        let grp = iter.create(&file, "X").unwrap();
        let data: Vec<u32> = grp.dataset("data").unwrap().read_raw().unwrap();
        let indicies: Vec<u32> = grp.dataset("indices").unwrap().read_raw().unwrap();
        let indptr: Vec<u32> = grp.dataset("indptr").unwrap().read_raw().unwrap();

        assert_eq!(vec![1,2,3,4,5,6], data);
        assert_eq!(vec![0,2,2,0,1,2], indicies);
        assert_eq!(vec![0,2,3,6], indptr);
    }

}