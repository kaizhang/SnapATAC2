mod matrix;
mod counter;
mod data_iter;

use std::str::FromStr;

use anyhow::{bail, Context, Result};
use anndata::{data::DynCsrMatrix, AnnData, AnnDataOp, AnnDataSet, ArrayElemOp, AxisArraysOp, Backend, ElemCollectionOp};
use bed_utils::bed::GenomicRange;
pub use data_iter::{ValueType, BaseValue, ChromValueIter, BaseData, FragmentData, ContactData, FragmentDataIter};
pub use counter::{FeatureCounter, CountingStrategy};
pub use matrix::{create_gene_matrix, create_tile_matrix, create_peak_matrix};
use nalgebra_sparse::CsrMatrix;
use num::integer::div_ceil;
use polars::frame::DataFrame;

use crate::genome::ChromSizes;

pub const FRAGMENT_SINGLE: &str = "fragment_single";
pub const FRAGMENT_PAIRED: &str = "fragment_paired";
pub const BASE_VALUE: &str = "_values";

/// The `SnapData` trait represents an interface for reading and
/// manipulating single-cell assay data. It extends the `AnnDataOp` trait,
/// adding methods for reading chromosome sizes and genome-wide base-resolution coverage.
pub trait SnapData: AnnDataOp {
    /// Read fragment data stored in the `.obsm` matrix.
    fn get_fragment_iter(&self, chunk_size: usize) -> Result<FragmentData>;

    /// Read base values stored in the `.obsm` matrix.
    fn get_base_iter(&self, chunk_size: usize) -> Result<BaseData<impl ExactSizeIterator<Item = (DynCsrMatrix, usize, usize)>>>;

    /// Read counts stored in the `X` matrix.
    fn read_chrom_values(
        &self,
        chunk_size: usize,
    ) -> Result<ChromValueIter<<<Self as AnnDataOp>::X as ArrayElemOp>::ArrayIter<CsrMatrix<u32>>>>
    {
        let regions = self
            .var_names()
            .into_vec()
            .into_iter()
            .map(|x| GenomicRange::from_str(x.as_str()).unwrap())
            .collect();
        Ok(ChromValueIter {
            regions,
            iter: self.x().iter(chunk_size),
            length: div_ceil(self.n_obs(), chunk_size),
        })
    }

    fn read_chrom_sizes(&self) -> Result<ChromSizes> {
        let df = self
            .uns()
            .get_item::<DataFrame>("reference_sequences")?
            .context("key 'reference_sequences' is not present in the '.uns'")?;
        let chrs = df.column("reference_seq_name").unwrap().str()?;
        let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
        let res = chrs
            .into_iter()
            .flatten()
            .map(|x| x.to_string())
            .zip(chr_sizes.into_iter().flatten())
            .collect();
        Ok(res)
    }

    fn genome_size(&self) -> Result<u64> {
        Ok(self.read_chrom_sizes()?.total_size())
    }
}

impl<B: Backend> SnapData for AnnData<B> {
    fn get_fragment_iter(&self, chunk_size: usize) -> Result<FragmentData> {
        let obsm = self.obsm();
        let matrices: FragmentDataIter = if let Some(insertion) =
            obsm.get_item_iter(FRAGMENT_SINGLE, chunk_size)
        {
            FragmentDataIter::FragmentSingle(Box::new(insertion))
        } else if let Some(fragment) = obsm.get_item_iter(FRAGMENT_PAIRED, chunk_size) {
            FragmentDataIter::FragmentPaired(Box::new(fragment))
        } else {
            bail!("one of the following keys must be present in the '.obsm': '{}', '{}'", FRAGMENT_SINGLE, FRAGMENT_PAIRED)
        };
        Ok(FragmentData::new(self.read_chrom_sizes()?, matrices))
    }

    fn get_base_iter(&self, chunk_size: usize) -> Result<BaseData<impl ExactSizeIterator<Item = (DynCsrMatrix, usize, usize)>>> {
        let obsm = self.obsm();
        if let Some(data) = obsm.get_item_iter(BASE_VALUE, chunk_size) {
            Ok(BaseData::new(self.read_chrom_sizes()?, data))
        } else {
            bail!("key '_values' is not present in the '.obsm'")
        }
    }
}

impl<B: Backend> SnapData for AnnDataSet<B> {
    fn get_fragment_iter(&self, chunk_size: usize) -> Result<FragmentData> {
        let adatas = self.adatas().inner();
        let obsm = adatas.get_obsm();
        let matrices: FragmentDataIter = if let Some(insertion) =
            obsm.get_item_iter(FRAGMENT_SINGLE, chunk_size)
        {
            FragmentDataIter::FragmentSingle(Box::new(insertion))
        } else if let Some(fragment) = obsm.get_item_iter(FRAGMENT_PAIRED, chunk_size) {
            FragmentDataIter::FragmentPaired(Box::new(fragment))
        } else {
            bail!("one of the following keys must be present in the '.obsm': '{}', '{}'", FRAGMENT_SINGLE, FRAGMENT_PAIRED)
        };
        Ok(FragmentData::new(self.read_chrom_sizes()?, matrices))
    }

    fn get_base_iter(&self, chunk_size: usize) -> Result<BaseData<impl ExactSizeIterator<Item = (DynCsrMatrix, usize, usize)>>>
    {
        let obsm = self.obsm();
        if let Some(data) = obsm.get_item_iter(BASE_VALUE, chunk_size) {
            Ok(BaseData::new(self.read_chrom_sizes()?, data))
        } else {
            bail!("key '_values' is not present in the '.obsm'")
        }
    }
}