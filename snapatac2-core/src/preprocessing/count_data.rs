mod import;
mod coverage;
mod genome;
mod matrix;

pub use import::{import_insertions, import_fragments, import_contacts};
pub use coverage::{GenomeCoverage, ContactMap, fragments_to_insertions};
pub use genome::{
    Transcript, Promoters, FeatureCounter, TranscriptCount, GeneCount,
    read_transcripts_from_gff, read_transcripts_from_gtf,
    ChromSizes, ChromValueIter, ChromValues, GenomeBaseIndex, 
};
pub use matrix::{create_gene_matrix, create_tile_matrix, create_peak_matrix};

use anndata::{container::{ChunkedArrayElem, StackedChunkedArrayElem}, ArrayElemOp};
use bed_utils::bed::{tree::BedTree, GenomicRange};
use anndata::{AnnDataOp, ElemCollectionOp, AxisArraysOp, AnnDataSet, Backend, AnnData};
use ndarray::Array2;
use polars::frame::DataFrame;
use nalgebra_sparse::CsrMatrix;
use anyhow::{Result, Context};
use num::integer::div_ceil;
use std::str::FromStr;

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
        let chrs = df.column("reference_seq_name").unwrap().utf8()?;
        let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
        let res = chrs.into_iter().flatten().map(|x| x.to_string())
            .zip(chr_sizes.into_iter().flatten()).collect();
        Ok(res)
    }

    /// Read insertion counts stored in the `.obsm` matrix.
    fn get_count_iter(&self, chunk_size: usize) ->
        Result<GenomeCoverage<Box<dyn ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>>>>;

    fn contact_count_iter(&self, chunk_size: usize) -> Result<ContactMap<Self::CountIter>>;

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

    /// Compute the fraction of reads in each region.
    fn frip<D>(&self, regions: &Vec<BedTree<D>>) -> Result<Array2<f64>> {
        let vec = fraction_in_regions(self.get_count_iter(2000)?.into_chrom_values(), regions)
            .map(|x| x.0).flatten().flatten().collect::<Vec<_>>();
        Array2::from_shape_vec((self.n_obs(), regions.len()), vec).map_err(Into::into)
    }
}

/// Count the fraction of the records in the given regions.
fn fraction_in_regions<'a, I, D>(
    iter: I, regions: &'a Vec<BedTree<D>>,
) -> impl Iterator<Item = (Vec<Vec<f64>>, usize, usize)> + 'a
where
    I: Iterator<Item = (Vec<ChromValues<f64>>, usize, usize)> + 'a,
{
    let k = regions.len();
    iter.map(move |(values, start, end)| {
        let frac = values.into_iter().map(|xs| {
            let sum = xs.iter().map(|x| x.value).sum::<f64>();
            let mut counts = vec![0.0; k];
            xs.into_iter().for_each(|x|
                regions.iter().enumerate().for_each(|(i, r)| {
                    if r.is_overlapped(&x) {
                        counts[i] += x.value;
                    }
                })
            );
            counts.iter_mut().for_each(|x| *x /= sum);
            counts
        }).collect::<Vec<_>>();
        (frac, start, end)
    })
}

impl<B: Backend> SnapData for AnnData<B> {
    type CountIter = ChunkedArrayElem<B, CsrMatrix<u8>>;

    fn get_count_iter(&self, chunk_size: usize) ->
        Result<GenomeCoverage<Box<dyn ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>>>>
    {
        let obsm = self.obsm();
        let matrices: Box<dyn ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>> =
            if let Some(insertion) = obsm.get_item_iter("insertion", chunk_size) {
                Box::new(insertion)
            } else if let Some(fragment) = obsm.get_item_iter("fragment", chunk_size) {
                Box::new(fragment.map(|(x, a, b)| (coverage::fragments_to_insertions(x), a, b)))
            } else {
                anyhow::bail!("neither 'insertion' nor 'fragment' is present in the '.obsm'")
            };
        Ok(GenomeCoverage::new(self.read_chrom_sizes()?, matrices))
    }

    fn contact_count_iter(&self, chunk_size: usize) -> Result<ContactMap<Self::CountIter>> {
        Ok(ContactMap::new(
            self.read_chrom_sizes()?,
            self.obsm().get_item_iter("contact", chunk_size).unwrap(),
        ))
    }
}

impl<B: Backend> SnapData for AnnDataSet<B> {
    type CountIter = StackedChunkedArrayElem<B, CsrMatrix<u8>>;

    fn get_count_iter(&self, chunk_size: usize) ->
        Result<GenomeCoverage<Box<dyn ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>>>>
    {
        let adatas = self.adatas().inner();
        let obsm = adatas.get_obsm();
        let matrices: Box<dyn ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>> =
            if let Some(insertion) = obsm.get_item_iter("insertion", chunk_size) {
                Box::new(insertion)
            } else if let Some(fragment) = obsm.get_item_iter("fragment", chunk_size) {
                Box::new(fragment.map(|(x, a, b)| (coverage::fragments_to_insertions(x), a, b)))
            } else {
                anyhow::bail!("neither 'insertion' nor 'fragment' is present in the '.obsm'")
            };
        Ok(GenomeCoverage::new(self.read_chrom_sizes()?, matrices))
    }

    fn contact_count_iter(&self, chunk_size: usize) -> Result<ContactMap<Self::CountIter>> {
        Ok(ContactMap::new(
            self.read_chrom_sizes()?,
            self.adatas()
                .inner()
                .get_obsm()
                .get_item_iter("contact", chunk_size)
                .unwrap(),
        ))
    }
}