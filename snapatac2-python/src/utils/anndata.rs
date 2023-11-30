use anndata::{
    data::{ArrayChunk, DataFrameIndex},
    AnnDataOp, ArrayData, HasShape,
    WriteArrayData, AxisArraysOp,
};
use nalgebra_sparse::CsrMatrix;
use anyhow::{Result, bail};
use polars::prelude::DataFrame;
use pyanndata::anndata::memory;
use pyanndata::{AnnData, AnnDataSet};
use pyo3::prelude::*;

use snapatac2_core::preprocessing::{qc, SnapData, GenomeCount, ContactMap, count_data::FragmentType};

pub struct PyAnnData<'py>(memory::PyAnnData<'py>);

impl<'py> FromPyObject<'py> for PyAnnData<'py> {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        obj.extract().map(PyAnnData)
    }
}

impl ToPyObject for PyAnnData<'_> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}


impl IntoPy<PyObject> for PyAnnData<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

impl<'py> AnnDataOp for PyAnnData<'py> {
    type X = memory::ArrayElem<'py>;
    type ElemCollectionRef<'a> = memory::ElemCollection<'a> where Self: 'a;
    type AxisArraysRef<'a> = memory::AxisArrays<'a> where Self: 'a;

    fn x(&self) -> Self::X {
        self.0.x()
    }

    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>,
    {
        self.0.set_x_from_iter(iter)
    }

    fn set_x<D: WriteArrayData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()> {
        self.0.set_x(data)
    }

    /// Delete the 'X' element.
    fn del_x(&self) -> Result<()> {
        self.0.del_x()
    }

    /// Return the number of observations (rows).
    fn n_obs(&self) -> usize {
        self.0.n_obs()
    }
    /// Return the number of variables (columns).
    fn n_vars(&self) -> usize {
        self.0.n_vars()
    }

    /// Return the names of observations.
    fn obs_names(&self) -> DataFrameIndex {
        self.0.obs_names()
    }
    /// Return the names of variables.
    fn var_names(&self) -> DataFrameIndex {
        self.0.var_names()
    }

    /// Chagne the names of observations.
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.0.set_obs_names(index)
    }
    /// Chagne the names of variables.
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.0.set_var_names(index)
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        self.0.obs_ix(names)
    }
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        self.0.var_ix(names)
    }

    fn read_obs(&self) -> Result<DataFrame> {
        self.0.read_obs()
    }
    fn read_var(&self) -> Result<DataFrame> {
        self.0.read_var()
    }

    /// Change the observation annotations.
    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        self.0.set_obs(obs)
    }

    /// Change the variable annotations.
    fn set_var(&self, var: DataFrame) -> Result<()> {
        self.0.set_var(var)
    }

    /// Delete the observation annotations.
    fn del_obs(&self) -> Result<()> {
        self.0.del_obs()
    }

    /// Delete the variable annotations.
    fn del_var(&self) -> Result<()> {
        self.0.del_var()
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        self.0.uns()
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        self.0.obsm()
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        self.0.obsp()
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        self.0.varm()
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        self.0.varp()
    }

    fn del_uns(&self) -> Result<()> {
        self.0.del_uns()
    }
    fn del_obsm(&self) -> Result<()> {
        self.0.del_obsm()
    }
    fn del_obsp(&self) -> Result<()> {
        self.0.del_obsp()
    }
    fn del_varm(&self) -> Result<()> {
        self.0.del_varm()
    }
    fn del_varp(&self) -> Result<()> {
        self.0.del_varp()
    }

    fn layers(&self) -> Self::AxisArraysRef<'_> {
        self.0.layers()
    }
    fn del_layers(&self) -> Result<()> {
        self.0.del_layers()
    }
}

impl<'py> SnapData for PyAnnData<'py> {
    type CountIter = memory::PyArrayIterator<CsrMatrix<u8>>;

    fn get_count_iter(&self, chunk_size: usize) ->
        Result<GenomeCount<Box<dyn ExactSizeIterator<Item = (FragmentType, usize, usize)>>>>
    {
        let obsm = self.obsm();
        let matrices: Box<dyn ExactSizeIterator<Item = (FragmentType, usize, usize)>> =
            if let Some(insertion) = obsm.get_item_iter("fragment_single", chunk_size) {
                Box::new(insertion.map(|(x, a, b)| (FragmentType::FragmentSingle(x), a, b)))
            } else if let Some(fragment) = obsm.get_item_iter("fragment_paired", chunk_size) {
                Box::new(fragment.map(|(x, a, b)| (FragmentType::FragmentPaired(x), a, b)))
            } else {
                anyhow::bail!("neither 'fragment_single' nor 'fragment_paired' is present in the '.obsm'")
            };
        Ok(GenomeCount::new(self.read_chrom_sizes()?, matrices))
    }

    fn contact_count_iter(
        &self, chunk_size: usize
    ) -> Result<ContactMap<Self::CountIter>>
    {
        Ok(ContactMap::new(
            self.read_chrom_sizes()?,
            self.obsm().get_item_iter("contact", chunk_size).expect("'contact' not found in obsm"),
        ))
    }

    fn fragment_size_distribution(&self, max_size: usize) -> Result<Vec<usize>> {
        if let Some(fragment) = self.obsm().get_item_iter("fragment_paired", 500) {
            Ok(qc::fragment_size_distribution(fragment.map(|x| x.0), max_size))
        } else {
            bail!("key 'fragment_paired' is not present in the '.obsm'")
        }
    }
}


#[derive(FromPyObject)]
pub enum AnnDataLike<'py> {
    AnnData(AnnData),
    PyAnnData(PyAnnData<'py>),
    AnnDataSet(AnnDataSet),
}

impl IntoPy<PyObject> for AnnDataLike<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            AnnDataLike::AnnData(x) => x.into_py(py),
            AnnDataLike::PyAnnData(x) => x.into_py(py),
            AnnDataLike::AnnDataSet(x) => x.into_py(py),
        }
    }
}

impl From<AnnData> for AnnDataLike<'_> {
    fn from(value: AnnData) -> Self {
        AnnDataLike::AnnData(value)
    }
}

impl From<AnnDataSet> for AnnDataLike<'_> {
    fn from(x: AnnDataSet) -> Self {
        AnnDataLike::AnnDataSet(x)
    }
}

impl<'py> From<PyAnnData<'py>> for AnnDataLike<'py> {
    fn from(x: PyAnnData<'py>) -> Self {
        AnnDataLike::PyAnnData(x)
    }
}

#[macro_export]
macro_rules! with_anndata {
    ($anndata:expr, $fun:ident) => {
        match $anndata {
            AnnDataLike::AnnData(x) => match x.backend().as_str() {
                H5::NAME => {
                    $fun!(x.inner_ref::<H5>().deref())
                }
                x => panic!("Unsupported backend: {}", x),
            },
            AnnDataLike::AnnDataSet(x) => match x.backend().as_str() {
                H5::NAME => {
                    $fun!(x.inner_ref::<H5>().deref())
                }
                x => panic!("Unsupported backend: {}", x),
            },
            AnnDataLike::PyAnnData(x) => {
                $fun!(x)
            }
        }
    };
}


#[derive(FromPyObject)]
pub enum RustAnnDataLike {
    AnnData(AnnData),
    AnnDataSet(AnnDataSet),
}

impl IntoPy<PyObject> for RustAnnDataLike {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            RustAnnDataLike::AnnData(x) => x.into_py(py),
            RustAnnDataLike::AnnDataSet(x) => x.into_py(py),
        }
    }
}

impl From<AnnData> for RustAnnDataLike {
    fn from(value: AnnData) -> Self {
        RustAnnDataLike::AnnData(value)
    }
}

impl From<AnnDataSet> for RustAnnDataLike {
    fn from(x: AnnDataSet) -> Self {
        RustAnnDataLike::AnnDataSet(x)
    }
}

#[macro_export]
macro_rules! with_rs_anndata {
    ($anndata:expr, $fun:ident) => {
        match $anndata {
            RustAnnDataLike::AnnData(x) => match x.backend().as_str() {
                H5::NAME => {
                    $fun!(x.inner_ref::<H5>().deref())
                }
                x => panic!("Unsupported backend: {}", x),
            },
            RustAnnDataLike::AnnDataSet(x) => match x.backend().as_str() {
                H5::NAME => {
                    $fun!(x.inner_ref::<H5>().deref())
                }
                x => panic!("Unsupported backend: {}", x),
            },
        }
    };
}
