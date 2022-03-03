use crate::utils::hdf5::{ResizableVectorData, create_str_attr, read_str_attr, COMPRESSION};
use crate::qc::{CellBarcode, QualityControl};

use hdf5::{File, H5Type, Result, Group, Dataset, types::VarLenUnicode};
use ndarray::{arr1, Array1, Array, Dimension};
use itertools::Itertools;
use nalgebra_sparse::csr;

#[derive(Clone)]
pub struct StrVec(pub Vec<String>);

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
    StrVec(cells).write(&group, "Cell")?;

    match optional_qc {
        Some(qc) => {
            let columns: Array1<hdf5::types::VarLenUnicode> =
                ["tsse", "n_fragment", "frac_dup", "frac_mito"]
                .into_iter().map(|x| x.parse().unwrap()).collect();
            group.new_attr_builder().with_data(&columns).create("column-order")?;
            qc.iter().map(|x| x.tss_enrichment).collect::<Array1<f64>>().write(&group, "tsse")?;
            qc.iter().map(|x| x.num_unique_fragment).collect::<Array1<u64>>().write(&group, "n_fragment")?;
            qc.iter().map(|x| x.frac_duplicated).collect::<Array1<f64>>().write(&group, "frac_dup")?;
            qc.iter().map(|x| x.frac_mitochondrial).collect::<Array1<f64>>().write(&group, "frac_mito")?;
        },
        _ => {
            let columns: Array1<hdf5::types::VarLenUnicode> = [].into_iter().collect();
            group.new_attr_builder().with_data(&columns).create("column-order")?;
        },
    }

    Ok(())
}

pub fn create_var(
    file: &File,
    features: Vec<String>,
) -> Result<()> {
    let group = file.create_group("var")?;
    create_str_attr(&group, "encoding-type", "dataframe")?;
    create_str_attr(&group, "encoding-version", "0.2.0")?;
    create_str_attr(&group, "_index", "ID")?;
    let columns: Array1<hdf5::types::VarLenUnicode> = [].into_iter().collect();
    group.new_attr_builder().with_data(&columns).create("column-order")?;
    StrVec(features).write(&group, "ID")?;
    Ok(())
}

pub struct Ann<T> {
    file: File,
    pub x: AnnDataElement<csr::CsrMatrix<T>, Group>,
    pub var_names: AnnDataElement<StrVec, Dataset>,
    pub obs_names: AnnDataElement<StrVec, Dataset>,
}

impl<T: H5Type> Ann<T> {
    pub fn read(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let x = AnnDataElement::new(file.group("X")?);
        let obs_names = read_dataframe_index(&file.group("obs")?)?;
        let var_names = read_dataframe_index(&file.group("var")?)?;
        Ok(Ann { file, x, var_names, obs_names })
    }

    /*
    pub fn x(&mut self) -> &csr::CsrMatrix<T> { self.x.get() }

    pub fn var_names(&mut self) -> &[String] { self.var_names.get().0.as_slice() }

    pub fn obs_names(&mut self) -> &[String] { self.obs_names.get().0.as_slice() }
    */

    pub fn ann_row_iter(&self) -> AnnSparseRowIter<'_, T>
    where
        T: Copy,
    {
        AnnSparseRowIter {
            iter: self.x.row_iter().enumerate(),
            rownames: self.obs_names.get().0,
            colnames: self.var_names.get().0,
        }
    }
}

pub fn read_dataframe_index(group: &Group) -> Result<AnnDataElement<StrVec, Dataset>> {
    let index_name = read_str_attr(group, "_index")?;
    Ok(AnnDataElement::new(group.dataset(&index_name)?))
}

pub struct AnnDataElement<D, C> {
    data_memory: Option<D>,
    data_disk: C,
    data_changed: bool,
}

impl<D, C> AnnDataElement<D, C>
where
    D: AnnDataIO<Container = C>
{
    pub fn new(data: C) -> Self {
        Self {
            data_memory: None,
            data_disk: data,
            data_changed: false,
        }
    }

    pub fn load(&mut self) {
        if let None = self.data_memory {
            self.data_memory = Some(AnnDataIO::read(&self.data_disk).unwrap());
        }
    }

    pub fn unload(&mut self) {
        if let Some(data) = &self.data_memory {
            if self.data_changed {
                data.update(&self.data_disk).unwrap();
            }
            self.data_changed = false;
            self.data_memory = None;
        }
    }

    pub fn get(&self) -> D 
    where
        D: Clone,
    {
        match &self.data_memory {
            None => AnnDataIO::read(&self.data_disk).unwrap(),
            Some(x) => x.clone()
        }
    }

    pub fn get_ref(&self) -> Option<&D> { self.data_memory.as_ref() }

    pub fn get_mut(&mut self) -> &mut D {
        self.load();
        self.data_changed = true;
        self.data_memory.as_mut().unwrap()
    }
}

impl<T> AnnDataElement<csr::CsrMatrix<T>, Group> {
    pub fn row_iter(&self) -> SparseRowIter<T> {
        match &self.data_memory {
            Some(csr) => SparseRowIter::Memory(csr.row_iter()),
            None => {
                let data = self.data_disk.dataset("data").unwrap();
                let indices = self.data_disk.dataset("indices").unwrap();
                let indptr: Vec<usize> = self.data_disk.dataset("indptr").unwrap().read_raw().unwrap();
                SparseRowIter::Disk((data, indices, indptr, 0))
            },
        }
    }

}

pub enum SparseRowIter<'a, T> {
    Memory(csr::CsrRowIter<'a, T>),
    Disk((Dataset, Dataset, Vec<usize>, usize)),
}

impl<'a, D> Iterator for SparseRowIter<'a, D>
where
    D: H5Type + Copy,
{
    type Item = Vec<(usize, D)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SparseRowIter::Memory(iter) => iter.next().map(|r| r.col_indices().iter()
                .zip(r.values()).map(|(i, v)| (*i, *v)).collect()),
            SparseRowIter::Disk((data, indices, indptr, current_row)) => {
                if *current_row >= indptr.len() - 1 {
                    None
                } else {
                    let i = indptr[*current_row];
                    let j = indptr[*current_row + 1];
                    let data: Array1<D> = data.read_slice_1d(i..j).unwrap();
                    let indices: Array1<usize> = indices.read_slice_1d(i..j).unwrap();
                    let result = indices.into_iter().zip(data).collect();
                    *current_row += 1;
                    Some(result)
                }
            },
        }
    }
}

pub trait AnnDataIO {
    type Container;

    const VERSION: &'static str;

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>;

    fn read(container: &Self::Container) -> Result<Self> where Self: Sized;

    fn update(&self, container: &Self::Container) -> Result<()>;

}

impl<D: H5Type + Clone> AnnDataIO for Vec<D> {
    type Container = Dataset;
    const VERSION: &'static str = "0.1.0";

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&arr1(self)).create(name)?;
        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", Self::VERSION)?;
        Ok(dataset)
    }

    fn read(dataset: &Self::Container) -> Result<Self> {
        let data: Array1<D> = dataset.read_1d()?;
        Ok(data.to_vec())
    }

    fn update(&self, container: &Self::Container) -> Result<()> {
        container.resize(self.len())?;
        container.write(&arr1(self))
    }
}

impl<A, D> AnnDataIO for Array<A, D>
where
    A: H5Type,
    D: Dimension,
{
    type Container = Dataset;
    const VERSION: &'static str = "0.2.0";

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", Self::VERSION)?;

        Ok(dataset)
    }

    fn read(dataset: &Self::Container) -> Result<Self> { dataset.read() }

    fn update(&self, container: &Self::Container) -> Result<()> {
        todo!()
    }
}

impl AnnDataIO for StrVec {
    type Container = Dataset;
    const VERSION: &'static str = "0.2.0";

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let data: Array1<VarLenUnicode> = self.0.iter()
            .map(|x| x.parse::<VarLenUnicode>().unwrap()).collect();
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&data).create(name)?;
        create_str_attr(&*dataset, "encoding-type", "string-array")?;
        create_str_attr(&*dataset, "encoding-version", Self::VERSION)?;
        Ok(dataset)
    }

    fn read(dataset: &Self::Container) -> Result<Self> {
        let data: Array1<VarLenUnicode> = dataset.read_1d()?;
        Ok(StrVec(data.into_iter().map(|x| x.parse().unwrap()).collect()))
    }

    fn update(&self, container: &Self::Container) -> Result<()> {
        let data: Array1<VarLenUnicode> = self.0.iter()
            .map(|x| x.parse::<VarLenUnicode>().unwrap()).collect();
        container.resize(self.0.len())?;
        container.write(&data)
    }
}

impl<T> AnnDataIO for csr::CsrMatrix<T>
where
    T: H5Type,
{
    type Container = Group;
    const VERSION: &'static str = "0.1.0";

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", Self::VERSION)?;

        let (indptr_, indices_, data) = self.csr_data();
        let indptr: Array1<i32> = indptr_.iter().map(|x| *x as i32).collect();  // scipy compatibility
        let indices: Array1<i32> = indices_.iter().map(|x| *x as i32).collect(); // scipy compatibility

        group.new_attr_builder()
            .with_data(&[self.nrows(), self.ncols()]).create("shape")?;
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&indptr).create("indptr")?;
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&indices).create("indices")?;
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(data).create("data")?;
        Ok(group)
    }

    fn read(dataset: &Self::Container) -> Result<Self> {
        let shape: Vec<usize> = dataset.attr("shape")?.read_raw()?;
        let data = dataset.dataset("data")?.read_raw()?;
        let indices: Vec<usize> = dataset.dataset("indices")?.read_raw()?;
        let indptr: Vec<usize> = dataset.dataset("indptr")?.read_raw()?;

        Ok(csr::CsrMatrix::try_from_csr_data(shape[0], shape[1],
            indptr,
            indices,
            data,
        ).expect("CSR data must conform to format specifications"))
    }

    fn update(&self, container: &Self::Container) -> Result<()> {
        todo!()
    }
}

pub struct AnnSparseRowIter<'a, T> {
    iter: std::iter::Enumerate<SparseRowIter<'a, T>>,
    rownames: Vec<String>,
    colnames: Vec<String>,
}

pub struct AnnSparseRow<T> {
    pub row_name: String,
    pub data: Vec<(String, T)>,
}

impl<'a, T> Iterator for AnnSparseRowIter<'a, T>
where
    T: H5Type + Copy,
{
    type Item = AnnSparseRow<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(row_idx, item)|
            AnnSparseRow {
                row_name: self.rownames[row_idx].clone(),
                data: item.into_iter().map(|(i, v)| (self.colnames[i].clone(), v)).collect(),
            }
        )
    }
}


/// Compressed Row Storage (CRS) stores the sparse matrix in 3 vectors:
/// one for floating-point numbers (data), and the other two for integers (indices, indptr).
/// The `data` vector stores the values of the nonzero elements of the matrix,
/// as they are traversed in a row-wise fashion.
/// The `indices` vector stores the column indexes of the elements in the `data` vector.
/// The `indptr` vector stores the locations in the `data` vector that start a row,
/// The last element is NNZ.
pub fn write_csr_rows<I, D>(
    csr: I,
    num_col: usize,
    location: &Group,
    name: &str,
    encoding: &str,
    version: &str,
    ) -> Result<Group>
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    let group = location.create_group(name)?;
    create_str_attr(&group, "encoding-type", encoding)?;
    create_str_attr(&group, "encoding-version", version)?;
    create_str_attr(&group, "h5sparse_format", "csr")?;
    let data: ResizableVectorData<D> =
        ResizableVectorData::new(&group, "data", 10000)?;
    let mut indptr: Vec<usize> = vec![0];
    let iter = csr.scan(0, |state, x| {
        *state = *state + x.len();
        Some((*state, x))
    });

    if num_col <= (i32::MAX as usize) {
        let indices: ResizableVectorData<i32> =
            ResizableVectorData::new(&group, "indices", 10000)?;
        for chunk in &iter.chunks(10000) {
            let (a, b): (Vec<i32>, Vec<D>) = chunk.map(|(x, vec)| {
                indptr.push(x);
                vec
            }).flatten().map(|(x, y)| -> (i32, D) {(
                x.try_into().expect(&format!("cannot convert '{}' to i32", x)),
                y
            ) }).unzip();
            indices.extend(a.into_iter())?;
            data.extend(b.into_iter())?;
        }

        group.new_attr_builder()
            .with_data(&arr1(&[indptr.len() - 1, num_col]))
            .create("shape")?;

        let try_convert_indptr: Option<Vec<i32>> = indptr.iter()
            .map(|x| (*x).try_into().ok()).collect();
        match try_convert_indptr {
            Some(vec) => {
                group.new_dataset_builder().deflate(COMPRESSION)
                    .with_data(&Array::from_vec(vec)).create("indptr")?;
            },
            _ => {
                let vec: Vec<i64> = indptr.into_iter()
                    .map(|x| x.try_into().unwrap()).collect();
                group.new_dataset_builder().deflate(COMPRESSION)
                    .with_data(&Array::from_vec(vec)).create("indptr")?;
            },
        }
    } else {
        let indices: ResizableVectorData<i64> =
            ResizableVectorData::new(&group, "indices", 10000)?;
        for chunk in &iter.chunks(10000) {
            let (a, b): (Vec<i64>, Vec<D>) = chunk.map(|(x, vec)| {
                indptr.push(x);
                vec
            }).flatten().map(|(x, y)| -> (i64, D) {(
                x.try_into().expect(&format!("cannot convert '{}' to i64", x)),
                y
            ) }).unzip();
            indices.extend(a.into_iter())?;
            data.extend(b.into_iter())?;
        }

        group.new_attr_builder()
            .with_data(&arr1(&[indptr.len() - 1, num_col]))
            .create("shape")?;

        let vec: Vec<i64> = indptr.into_iter()
            .map(|x| x.try_into().unwrap()).collect();
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&Array::from_vec(vec)).create("indptr")?;
    }

    Ok(group)
}

pub fn to_csr_matrix<I, D>(iter: I, num_col: usize) -> (Vec<D>, Vec<i32>, Vec<i32>)
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    let mut data: Vec<D> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::new();

    let n = iter.fold(0, |r_idx, row| {
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

#[cfg(test)]
mod sm_tests {
    use super::*;

    #[test]
    fn test_sparse_row() {
        let matrix =
            [ vec![(0, 1), (2, 2)]
            , vec![(2, 3)]
            , vec![(0, 4), (1, 5), (2, 6)] ];
        let file = File::create("1.h5").unwrap();
        let grp = write_csr_rows(matrix.into_iter(), 3, &file, "X", "csr_matrix", "0.1.0").unwrap();
        let data: Vec<u32> = grp.dataset("data").unwrap().read_raw().unwrap();
        let indicies: Vec<u32> = grp.dataset("indices").unwrap().read_raw().unwrap();
        let indptr: Vec<u32> = grp.dataset("indptr").unwrap().read_raw().unwrap();

        assert_eq!(vec![1,2,3,4,5,6], data);
        assert_eq!(vec![0,2,2,0,1,2], indicies);
        assert_eq!(vec![0,2,3,6], indptr);
    }

}

