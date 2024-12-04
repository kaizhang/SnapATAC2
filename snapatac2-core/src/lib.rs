pub mod genome;
pub mod preprocessing;
pub mod feature_count;
pub mod export;
pub mod motif;
pub mod network;
pub mod embedding;
pub mod utils;

use feature_count::{BaseData, FragmentData, FragmentDataIter};
use genome::{ChromSizes, ChromValueIter};

use anndata::{
    container::{ChunkedArrayElem, StackedChunkedArrayElem},
    ArrayElemOp,
};
use anndata::{AnnData, AnnDataOp, AnnDataSet, AxisArraysOp, Backend, ElemCollectionOp};
use anyhow::{bail, Context, Result};
use bed_utils::bed::{map::GIntervalMap, GenomicRange};
use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;
use num::integer::div_ceil;
use polars::frame::DataFrame;
use preprocessing::{fraction_of_reads_in_region, fragment_size_distribution, TSSe, TssRegions};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    str::FromStr,
    sync::{Arc, Mutex},
};

/// The `SnapData` trait represents an interface for reading and
/// manipulating single-cell assay data. It extends the `AnnDataOp` trait,
/// adding methods for reading chromosome sizes and genome-wide base-resolution coverage.
pub trait SnapData: AnnDataOp {
    type CountIter: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>;

    /// Return chromosome names and sizes.
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

    /// Read fragment data stored in the `.obsm` matrix.
    fn get_fragment_iter(&self, chunk_size: usize) -> Result<FragmentData>;

    /// Read base values stored in the `.obsm` matrix.
    fn get_base_iter(&self, chunk_size: usize) -> Result<BaseData<impl ExactSizeIterator<Item = (CsrMatrix<f32>, usize, usize)>>>;

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

    /// QC metrics for the data.

    /// Compute TSS enrichment.
    fn tss_enrichment<'a>(&self, promoter: &'a TssRegions) -> Result<(Vec<f64>, TSSe<'a>)> {
        let library_tsse = Arc::new(Mutex::new(TSSe::new(promoter)));
        let scores = self
            .get_fragment_iter(2000)?
            .into_fragments()
            .flat_map(|(list_of_fragments, _, _)| {
                list_of_fragments
                    .into_par_iter()
                    .map(|fragments| {
                        let mut tsse = TSSe::new(promoter);
                        fragments.into_iter().for_each(|x| tsse.add(&x));
                        library_tsse.lock().unwrap().add_from(&tsse);
                        tsse.result().0
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok((
            scores,
            Arc::into_inner(library_tsse).unwrap().into_inner().unwrap(),
        ))
    }

    /// Compute the fragment size distribution.
    fn fragment_size_distribution(&self, max_size: usize) -> Result<Vec<usize>>;

    /// Compute the fraction of reads in each region.
    fn frip<D>(
        &self,
        regions: &Vec<GIntervalMap<D>>,
        normalized: bool,
        count_as_insertion: bool,
    ) -> Result<Array2<f64>> {
        let vec = fraction_of_reads_in_region(
            self.get_fragment_iter(2000)?.into_fragments(),
            regions,
            normalized,
            count_as_insertion,
        )
        .map(|x| x.0)
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
        Array2::from_shape_vec((self.n_obs(), regions.len()), vec).map_err(Into::into)
    }

    fn genome_size(&self) -> Result<u64> {
        Ok(self.read_chrom_sizes()?.total_size())
    }
}

impl<B: Backend> SnapData for AnnData<B> {
    type CountIter = ChunkedArrayElem<B, CsrMatrix<u8>>;

    fn get_fragment_iter(&self, chunk_size: usize) -> Result<FragmentData> {
        let obsm = self.obsm();
        let matrices: FragmentDataIter = if let Some(insertion) =
            obsm.get_item_iter("fragment_single", chunk_size)
        {
            FragmentDataIter::FragmentSingle(Box::new(insertion))
        } else if let Some(fragment) = obsm.get_item_iter("fragment_paired", chunk_size) {
            FragmentDataIter::FragmentPaired(Box::new(fragment))
        } else {
            bail!("one of the following keys must be present in the '.obsm': 'fragment_single', 'fragment_paired'")
        };
        Ok(FragmentData::new(self.read_chrom_sizes()?, matrices))
    }

    fn get_base_iter(&self, chunk_size: usize) -> Result<BaseData<impl ExactSizeIterator<Item = (CsrMatrix<f32>, usize, usize)>>> {
        let obsm = self.obsm();
        if let Some(data) = obsm.get_item_iter::<CsrMatrix<f32>>("_values", chunk_size) {
            Ok(BaseData::new(self.read_chrom_sizes()?, data))
        } else {
            bail!("key '_values' is not present in the '.obsm'")
        }
    }

    fn fragment_size_distribution(&self, max_size: usize) -> Result<Vec<usize>> {
        if let Some(fragment) = self.obsm().get_item_iter("fragment_paired", 500) {
            Ok(fragment_size_distribution(
                fragment.map(|x| x.0),
                max_size,
            ))
        } else {
            bail!("key 'fragment_paired' is not present in the '.obsm'")
        }
    }
}

impl<B: Backend> SnapData for AnnDataSet<B> {
    type CountIter = StackedChunkedArrayElem<B, CsrMatrix<u8>>;

    fn get_fragment_iter(&self, chunk_size: usize) -> Result<FragmentData> {
        let adatas = self.adatas().inner();
        let obsm = adatas.get_obsm();
        let matrices: FragmentDataIter = if let Some(insertion) =
            obsm.get_item_iter("fragment_single", chunk_size)
        {
            FragmentDataIter::FragmentSingle(Box::new(insertion))
        } else if let Some(fragment) = obsm.get_item_iter("fragment_paired", chunk_size) {
            FragmentDataIter::FragmentPaired(Box::new(fragment))
        } else {
            bail!("one of the following keys must be present in the '.obsm': 'fragment_single', 'fragment_paired'")
        };
        Ok(FragmentData::new(self.read_chrom_sizes()?, matrices))
    }

    fn get_base_iter(&self, chunk_size: usize) -> Result<BaseData<impl ExactSizeIterator<Item = (CsrMatrix<f32>, usize, usize)>>>
    {
        let obsm = self.obsm();
        if let Some(data) = obsm.get_item_iter::<CsrMatrix<f32>>("_values", chunk_size) {
            Ok(BaseData::new(self.read_chrom_sizes()?, data))
        } else {
            bail!("key '_values' is not present in the '.obsm'")
        }
    }

    fn fragment_size_distribution(&self, max_size: usize) -> Result<Vec<usize>> {
        if let Some(fragment) = self
            .adatas()
            .inner()
            .get_obsm()
            .get_item_iter("fragment_paired", 500)
        {
            Ok(fragment_size_distribution(
                fragment.map(|x| x.0),
                max_size,
            ))
        } else {
            bail!("key 'fragment_paired' is not present in the '.obsm'")
        }
    }
}