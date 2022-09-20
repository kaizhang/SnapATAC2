pub mod gene;
pub mod similarity;

use bed_utils::bed::{
    BEDLike, GenomicRange, BedGraph, NarrowPeak,
    ParseError, merge_bed_with,
    tree::{SparseCoverage,  SparseBinnedCoverage}, Strand,
};
use anyhow::{Result, anyhow, bail};
use anndata_rs::{
    anndata::{AnnData, AnnDataSet},
    element::ElemCollection,
};
use polars::frame::DataFrame;
use std::{fmt::Debug, str::FromStr};
use nalgebra_sparse::CsrMatrix;
use itertools::Itertools;

pub type CellBarcode = String;

/// Fragments from single-cell ATAC-seq experiment. Each fragment is represented
/// by a genomic coordinate, cell barcode and a integer value.
pub struct Fragment {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub barcode: CellBarcode,
    pub count: u32,
    pub strand: Option<Strand>,
}

impl BEDLike for Fragment {
    fn chrom(&self) -> &str { &self.chrom }
    fn set_chrom(&mut self, chrom: &str) -> &mut Self {
        self.chrom = chrom.to_string();
        self
    }
    fn start(&self) -> u64 { self.start }
    fn set_start(&mut self, start: u64) -> &mut Self {
        self.start = start;
        self
    }
    fn end(&self) -> u64 { self.end }
    fn set_end(&mut self, end: u64) -> &mut Self {
        self.end = end;
        self
    }
    fn name(&self) -> Option<&str> { None }
    fn score(&self) -> Option<bed_utils::bed::Score> { None }
    fn strand(&self) -> Option<Strand> { None }
}

impl std::str::FromStr for Fragment {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut fields = s.split('\t');
        let chrom = fields.next().ok_or(ParseError::MissingReferenceSequenceName)?.to_string();
        let start = fields.next().ok_or(ParseError::MissingStartPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidStartPosition))?;
        let end = fields.next().ok_or(ParseError::MissingEndPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidEndPosition))?;
        let barcode = fields.next().ok_or(ParseError::MissingName)
            .map(|s| s.into())?;
        let count = fields.next().map_or(Ok(1), |s| if s == "." {
            Ok(1)
        } else {
            lexical::parse(s).map_err(ParseError::InvalidStartPosition)
        })?;
        let strand = fields.next().map_or(Ok(None), |s| if s == "." {
            Ok(None)
        } else {
            s.parse().map(Some).map_err(ParseError::InvalidStrand)
        })?;
        Ok(Fragment { chrom, start, end, barcode, count, strand })
    }
}


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
        self.get_regions().flatten().map(|x| x.pretty_show()).collect()
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
        self.get_regions().map(|x| x.pretty_show()).collect()
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

pub type TN5InsertionIter = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u8)>>>>, GBaseIndex>;
pub type ChromValueIterator = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u32)>>>>, GIntervalIndex>;

/// Read genomic region and its associated account
pub trait ChromValuesReader {
    /// Return values in .obsm['insertion']
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter>;

    /// Return values in .X
    fn read_chrom_values(&self) -> Result<ChromValueIterator>;

    /// Return chromosome names and sizes.
    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>>;
}


impl ChromValuesReader for AnnData {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter> {
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
        })
    }

    fn read_chrom_values(&self) -> Result<ChromValueIterator>
    {
        Ok(ChromValueIter {
            genome_index: GIntervalIndex(
                self.var_names()?.into_iter()
                    .map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect()
            ),
            iter: Box::new(
                self.get_x().chunked(500).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u32>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                })
            ),
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.get_uns().inner())
    }
}

impl ChromValuesReader for AnnDataSet {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter> {
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
        })
    }

    fn read_chrom_values(&self) -> Result<ChromValueIterator>
    {
        Ok(ChromValueIter {
            genome_index: GIntervalIndex(
                self.var_names()?.into_iter()
                    .map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect()
            ),
            iter: Box::new(
                self.anndatas.inner().x.chunked(500).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u32>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                })
            ),
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
    merge_bed_with(
        peaks.map(|mut x| {
            let summit = x.start() + x.peak;
            x.start = (summit - half_window_size).max(0);
            x.end = summit + half_window_size + 1;
            x.peak = half_window_size;
            x
        }),
        iterative_merge,
    )
}

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
}