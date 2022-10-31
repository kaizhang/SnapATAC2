pub mod gene;
pub mod similarity;

use bed_utils::bed::{
    BEDLike, GenomicRange, BedGraph, NarrowPeak, merge_bed_with,
    tree::{SparseCoverage,  SparseBinnedCoverage},
};
use anndata_rs::{anndata::{AnnData, AnnDataSet}, element::ElemCollection};
use anyhow::{Result, anyhow, bail};
use num::integer::div_ceil;
use polars::frame::DataFrame;
use std::{fmt::Debug, str::FromStr};
use nalgebra_sparse::CsrMatrix;
use itertools::Itertools;


/// Genomic interval associating with integer values
pub type ChromValues = Vec<BedGraph<u32>>;

/// A structure that stores the feature counts.
pub trait FeatureCounter {
    type Value;

    /// Reset the counter.
    fn reset(&mut self);

    /// Update counter according to the region and the assocated count.
    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32);

    fn inserts<B: Into<ChromValues>>(&mut self, data: B) {
        data.into().into_iter().for_each(|x| self.insert(&x, x.value));
    }

    /// Retrieve feature ids.
    fn get_feature_ids(&self) -> Vec<String>;

    /// Retrieve feature names.
    fn get_feature_names(&self) -> Option<Vec<String>> { None }

    /// Retrieve stored counts.
    fn get_counts(&self) -> Vec<(usize, Self::Value)>;
}

impl<D: BEDLike + Clone> FeatureCounter for SparseBinnedCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().flatten().map(|x| x.to_genomic_range().pretty_show()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

impl<D: BEDLike> FeatureCounter for SparseCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().map(|x| x.to_genomic_range().pretty_show()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

/// GenomeIndex stores genomic loci in a compact way. It maps
/// integers to genomic intervals.
pub(crate) trait GenomeIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange;
}

/// Base-resolution compact representation of genomic, stored as a vector of
/// chromosome names and their accumulated lengths.
/// The accumulated lengths start from 0.
pub struct GBaseIndex(Vec<(String, u64)>);

impl GBaseIndex {
    pub fn read_from_anndata(elems: &mut ElemCollection) -> Result<Self> {
        let (chrs, chr_sizes): (Vec<_>, Vec<_>) = get_reference_seq_info_(elems)?.into_iter().unzip();
        let chrom_index = chrs.into_iter().zip(
            std::iter::once(0).chain(chr_sizes.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            }))
        ).collect();
        Ok(Self(chrom_index))
    }

    pub(crate) fn index_downsampled(&self, ori_idx: usize, sample_size: usize) -> usize {
        if sample_size <= 1 {
            ori_idx
        } else { 
            match self.0.binary_search_by_key(&ori_idx, |s| s.1.try_into().unwrap()) {
                Ok(_) => ori_idx,
                Err(j) => {
                    let p: usize = self.0[j - 1].1.try_into().unwrap();
                    (ori_idx - p) / sample_size * sample_size + p
                },
            }
        }
    }
}

impl GenomeIndex for GBaseIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange {
        match self.0.binary_search_by_key(&i, |s| s.1.try_into().unwrap()) {
            Ok(j) => GenomicRange::new(self.0[j].0.clone(), 0, 1),
            Err(j) => {
                let (chr, p) = self.0[j - 1].clone();
                GenomicRange::new(chr, i as u64 - p, i as u64 - p + 1)
            },
        }
    }
}

/// A set of genomic loci.
pub struct GIntervalIndex(pub Vec<GenomicRange>);

impl GenomeIndex for GIntervalIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange { self.0[i].clone() }
}

pub struct ChromValueIter<I, G> {
    iter: I,
    genome_index: G,
    length: usize,
}

impl<I, G, N> Iterator for ChromValueIter<I, G>
where
    I: Iterator<Item = Vec<Vec<(usize, N)>>>,
    G: GenomeIndex,
    N: std::convert::TryInto<u32>,
    <N as TryInto<u32>>::Error: Debug,
{
    type Item = Vec<ChromValues>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|items| items.into_iter().map(|item|
            item.into_iter().map(|(i, x)|
                BedGraph::from_bed(
                    &self.genome_index.lookup_region(i),
                    x.try_into().unwrap(),
                )
            ).collect()
        ).collect())
    }
}

impl<I, G, N> ExactSizeIterator for ChromValueIter<I, G>
where
    I: Iterator<Item = Vec<Vec<(usize, N)>>>,
    G: GenomeIndex,
    N: std::convert::TryInto<u32>,
    <N as TryInto<u32>>::Error: Debug,
{
    fn len(&self) -> usize { self.length }
}

pub type BaseCountIter = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u8)>>>>, GBaseIndex>;
pub type ChromValueIterator = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u32)>>>>, GIntervalIndex>;

/// Read genomic region and its associated account
pub trait ChromValuesReader {
    /// Return values in .obsm['insertion']
    fn raw_count_iter(&self, chunk_size: usize) -> Result<BaseCountIter>;

    /// Return values in .X
    fn read_chrom_values(&self) -> Result<ChromValueIterator>;

    /// Return chromosome names and sizes.
    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>>;
}


impl ChromValuesReader for AnnData {
    fn raw_count_iter(&self, chunk_size: usize) -> Result<BaseCountIter> {
       Ok(ChromValueIter {
            iter: Box::new(self.get_obsm().inner().get("insertion")
                .expect("cannot find 'insertion' in .obsm")
                .chunked(chunk_size).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u8>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<(usize, u8)>>()
                    ).collect::<Vec<_>>()
                })
            ),
            genome_index: GBaseIndex::read_from_anndata(&mut self.get_uns().inner())?,
            length: div_ceil(self.n_obs(), chunk_size),
        })
    }

    fn read_chrom_values(&self) -> Result<ChromValueIterator>
    {
        let chunk_size = 500;
        Ok(ChromValueIter {
            genome_index: GIntervalIndex(
                self.var_names()?.into_iter()
                    .map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect()
            ),
            iter: Box::new(
                self.get_x().chunked(chunk_size).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u32>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                })
            ),
            length: div_ceil(self.n_obs(), chunk_size),
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.get_uns().inner())
    }
}

impl ChromValuesReader for AnnDataSet {
    fn raw_count_iter(&self, chunk_size: usize) -> Result<BaseCountIter> {
        let n = self.n_obs();
        let inner = self.anndatas.inner();
        let ref_seq_same = inner.iter().map(|(_, adata)|
            get_reference_seq_info_(&mut adata.get_uns().inner()).unwrap()
        ).all_equal();
        if !ref_seq_same {
            return Err(anyhow!("reference genome information mismatch"));
        }
        let genome_index = GBaseIndex::read_from_anndata(
            &mut inner.iter().next().unwrap().1.get_uns().inner()
        )?;

        Ok(ChromValueIter {
            iter: Box::new(inner.obsm.data.get("insertion").unwrap()
                .chunked(chunk_size).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u8>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect()
                    ).collect()
                })),
            genome_index,
            length: div_ceil(n, chunk_size),
        })
    }

    fn read_chrom_values(&self) -> Result<ChromValueIterator>
    {
        let n = self.n_obs();
        let chunk_size = 500;
        Ok(ChromValueIter {
            genome_index: GIntervalIndex(
                self.var_names()?.into_iter()
                    .map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect()
            ),
            iter: Box::new(
                self.anndatas.inner().x.chunked(chunk_size).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u32>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                })
            ),
            length: div_ceil(n, chunk_size),
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.anndatas.inner().iter().next().unwrap()
            .1.get_uns().inner())
    }
}

pub fn merge_peaks<I>(peaks: I, half_window_size: u64) -> impl Iterator<Item = Vec<NarrowPeak>>
where
    I: Iterator<Item = NarrowPeak>,
{
    fn iterative_merge(mut peaks: Vec<NarrowPeak>) -> Vec<NarrowPeak> {
        let mut result = Vec::new();
        while !peaks.is_empty() {
            let best_peak = peaks.iter()
                .max_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap()).unwrap()
                .clone();
            peaks = peaks.into_iter().filter(|x| x.n_overlap(&best_peak) == 0).collect();
            result.push(best_peak);
        }
        result
    }

    merge_bed_with(
        peaks.map(move |mut x| {
            let summit = x.start() + x.peak;
            x.start = summit.saturating_sub(half_window_size);
            x.end = summit + half_window_size + 1;
            x.peak = summit - x.start;
            x
        }),
        iterative_merge,
    )
}

/*
pub fn aggregate_X<A, I>(adata: A, groupby: Option<Either<&str, &Vec<&str>>>)
where
    A: AnnDataReadOp<MatrixIter = I>,
    I: Iterator<Item = Box<dyn DataPartialIO>>,
{
    match groupby {
        None => adata.iter_x().map()

    }
}
*/

fn get_reference_seq_info_(elems: &mut ElemCollection) -> Result<Vec<(String, u64)>> {
    match elems.get_mut("reference_sequences") {
        None => bail!("Cannot find key 'reference_sequences' in: {}", elems),
        Some(ref_seq) => {
            let df: Box<DataFrame> = ref_seq.read()?.into_any().downcast().unwrap();
            let chrs = df.column("reference_seq_name").unwrap().utf8()?;
            let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
            Ok(chrs.into_iter().flatten().map(|x| x.to_string()).zip(
                chr_sizes.into_iter().flatten()
            ).collect())
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bed_utils::bed::io::Reader;

    #[test]
    fn genome_index() {
        let gindex = GBaseIndex(vec![("1".to_owned(), 0), ("2".to_owned(), 13), ("3".to_owned(), 84)]);

        [
            (0, ("1", 0)),
            (12, ("1", 12)),
            (13, ("2", 0)),
            (100, ("3", 16)),
        ].into_iter().for_each(|(i, (chr, s))|
            assert_eq!(gindex.lookup_region(i), GenomicRange::new(chr, s, s+1))
        );

        [
            (0, 2, 0),
            (1, 2, 0),
            (2, 2, 2),
            (3, 2, 2),
            (10, 2, 10),
            (11, 2, 10),
            (12, 2, 12),
            (13, 2, 13),
            (14, 2, 13),
            (15, 2, 15),
            (16, 2, 15),
            (84, 2, 84),
            (85, 2, 84),
            (86, 2, 86),
            (87, 2, 86),
            (85, 1, 85),
        ].into_iter().for_each(|(i, s, i_)|
            assert_eq!(gindex.index_downsampled(i, s), i_)
        );
    }

    #[test]
    fn test_merge_peaks() {
        let input = "chr1\t9977\t16487\ta\t1000\t.\t74.611\t290.442\t293.049\t189
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295.33\t290.939\t425
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t425
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t625
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t925
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t625
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t325
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t525
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t725
chr3\t0\t1164\tb\t1000\t.\t74.1871\t290\t290.939\t100
";
        let output = "chr1\t10202\t10603\tb\t1000\t.\t74.1871\t295.33\t290.939\t200
chr1\t10702\t11103\tb\t1000\t.\t74.1871\t290\t290.939\t200
chr2\t10402\t10803\tb\t1000\t.\t74.1871\t295\t290.939\t200
chr3\t0\t301\tb\t1000\t.\t74.1871\t290\t290.939\t100
";

        let expected: Vec<NarrowPeak> = Reader::new(output.as_bytes(), None)
            .into_records().map(|x| x.unwrap()).collect();
        let result: Vec<NarrowPeak> = merge_peaks(
            Reader::new(input.as_bytes(), None).into_records().map(|x| x.unwrap()),
            200
        ).flatten().collect();

        assert_eq!(expected, result);
    }
}