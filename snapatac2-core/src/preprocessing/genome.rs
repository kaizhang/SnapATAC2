use anndata::{container::{ChunkedArrayElem, StackedChunkedArrayElem}, ArrayElemOp};
use bed_utils::bed::{tree::{GenomeRegions, BedTree}, GenomicRange, BedGraph, BEDLike};
use anndata::{AnnDataOp, ElemCollectionOp, AxisArraysOp, AnnDataSet, Backend, AnnData};
use indexmap::{IndexSet, IndexMap};
use ndarray::Array2;
use polars::frame::DataFrame;
use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;
use noodles::{core::Position, gff, gff::record::Strand, gtf};
use num::{
    integer::div_ceil,
    traits::{FromPrimitive, Zero},
};
use polars::prelude::{DataFrame, NamedFrom, Series};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    io::BufRead,
    ops::{AddAssign, Range},
    str::FromStr,
};

use super::counter::FeatureCounter;

/// Position is 1-based.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transcript {
    pub transcript_name: Option<String>,
    pub transcript_id: String,
    pub gene_name: String,
    pub gene_id: String,
    pub is_coding: Option<bool>,
    pub chrom: String,
    pub left: Position,
    pub right: Position,
    pub strand: Strand,
}

impl TryFrom<gtf::Record> for Transcript {
    type Error = anyhow::Error;

    fn try_from(record: gtf::Record) -> Result<Self, Self::Error> {
        if record.ty() != "transcript" {
            return Err(anyhow::anyhow!("record is not a transcript"));
        }

        let err_msg =
            |x: &str| -> String { format!("failed to find '{}' in record: {}", x, record) };

        let left = record.start();
        let right = record.end();
        let attributes: HashMap<&str, &str> = record
            .attributes()
            .iter()
            .map(|x| (x.key(), x.value()))
            .collect();
        Ok(Transcript {
            transcript_name: attributes.get("transcript_name").map(|x| x.to_string()),
            transcript_id: attributes
                .get("transcript_id")
                .expect(&err_msg("transcript_id"))
                .to_string(),
            gene_name: attributes
                .get("gene_name")
                .expect(&err_msg("gene_name"))
                .to_string(),
            gene_id: attributes
                .get("gene_id")
                .expect(&err_msg("gene_id"))
                .to_string(),
            is_coding: attributes
                .get("transcript_type")
                .map(|x| *x == "protein_coding"),
            chrom: record.reference_sequence_name().to_string(),
            left,
            right,
            strand: match record.strand() {
                None => Strand::None,
                Some(gtf::record::Strand::Forward) => Strand::Forward,
                Some(gtf::record::Strand::Reverse) => Strand::Reverse,
            },
        })
    }
}

impl TryFrom<gff::Record> for Transcript {
    type Error = anyhow::Error;

    fn try_from(record: gff::Record) -> Result<Self, Self::Error> {
        if record.ty() != "transcript" {
            return Err(anyhow::anyhow!("record is not a transcript"));
        }

        let err_msg =
            |x: &str| -> String { format!("failed to find '{}' in record: {}", x, record) };

        let left = record.start();
        let right = record.end();
        let attributes: HashMap<&str, &str> = record
            .attributes()
            .iter()
            .map(|x| (x.key(), x.value()))
            .collect();
        Ok(Transcript {
            transcript_name: attributes.get("transcript_name").map(|x| x.to_string()),
            transcript_id: attributes
                .get("transcript_id")
                .expect(&err_msg("transcript_id"))
                .to_string(),
            gene_name: attributes
                .get("gene_name")
                .expect(&err_msg("gene_name"))
                .to_string(),
            gene_id: attributes
                .get("gene_id")
                .expect(&err_msg("gene_id"))
                .to_string(),
            is_coding: attributes
                .get("transcript_type")
                .map(|x| *x == "protein_coding"),
            chrom: record.reference_sequence_name().to_string(),
            left,
            right,
            strand: record.strand(),
        })
    }
}

impl Transcript {
    pub fn get_tss(&self) -> Option<usize> {
        match self.strand {
            Strand::Forward => Some(<Position as TryInto<usize>>::try_into(self.left).unwrap() - 1),
            Strand::Reverse => {
                Some(<Position as TryInto<usize>>::try_into(self.right).unwrap() - 1)
            }
            _ => None,
        }
    }
}

pub fn read_transcripts_from_gtf<R>(input: R) -> Result<Vec<Transcript>>
where
    R: BufRead,
{
    gtf::Reader::new(input)
        .records()
        .try_fold(Vec::new(), |mut acc, rec| {
            if let Ok(transcript) = rec?.try_into() {
                acc.push(transcript);
            }
            Ok(acc)
        })
}

pub fn read_transcripts_from_gff<R>(input: R) -> Result<Vec<Transcript>>
where
    R: BufRead,
{
    gff::Reader::new(input)
        .records()
        .try_fold(Vec::new(), |mut acc, rec| {
            if let Ok(transcript) = rec?.try_into() {
                acc.push(transcript);
            }
            Ok(acc)
        })
}

pub struct Promoters {
    pub regions: GenomeRegions<GenomicRange>,
    pub transcripts: Vec<Transcript>,
}

impl Promoters {
    pub fn new(
        transcripts: Vec<Transcript>,
        upstream: u64,
        downstream: u64,
        include_gene_body: bool,
    ) -> Self {
        let regions = transcripts
            .iter()
            .map(|transcript| {
                let left =
                    (<Position as TryInto<usize>>::try_into(transcript.left).unwrap() - 1) as u64;
                let right =
                    (<Position as TryInto<usize>>::try_into(transcript.right).unwrap() - 1) as u64;
                let (start, end) = match transcript.strand {
                    Strand::Forward => (
                        left.saturating_sub(upstream),
                        downstream + (if include_gene_body { right } else { left }),
                    ),
                    Strand::Reverse => (
                        (if include_gene_body { left } else { right }).saturating_sub(downstream),
                        right + upstream,
                    ),
                    _ => panic!("Miss strand information for {}", transcript.transcript_id),
                };
                GenomicRange::new(transcript.chrom.clone(), start, end)
            })
            .collect();
        Promoters {
            regions,
            transcripts,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ChromSizes(IndexMap<String, u64>);

impl ChromSizes {
    pub fn get(&self, chrom: &str) -> Option<u64> {
        self.0.get(chrom).copied()
    }
}

impl<S> FromIterator<(S, u64)> for ChromSizes
where
    S: Into<String>,
{
    fn from_iter<T: IntoIterator<Item = (S, u64)>>(iter: T) -> Self {
        ChromSizes(iter.into_iter().map(|(s, l)| (s.into(), l)).collect())
    }
}

impl IntoIterator for ChromSizes {
    type Item = (String, u64);
    type IntoIter = indexmap::map::IntoIter<String, u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl ChromSizes {
    pub fn to_dataframe(&self) -> DataFrame {
        DataFrame::new(vec![
            Series::new(
                "reference_seq_name",
                self.0.iter().map(|x| x.0.clone()).collect::<Series>(),
            ),
            Series::new(
                "reference_seq_length",
                self.0.iter().map(|x| x.1).collect::<Series>(),
            ),
        ])
        .unwrap()
    }
}

/// 0-based index that maps genomic loci to integers.
#[derive(Debug, Clone)]
pub struct GenomeBaseIndex {
    chroms: IndexSet<String>,
    base_accum_len: Vec<u64>,
    binned_accum_len: Vec<u64>,
    step: usize,
}

impl GenomeBaseIndex {
    pub fn new(chrom_sizes: &ChromSizes) -> Self {
        let mut acc = 0;
        let base_accum_len = chrom_sizes
            .0
            .iter()
            .map(|(_, length)| {
                acc += length;
                acc
            })
            .collect::<Vec<_>>();
        Self {
            chroms: chrom_sizes.0.iter().map(|x| x.0.clone()).collect(),
            binned_accum_len: base_accum_len.clone(),
            base_accum_len,
            step: 1,
        }
    }

    /// Retreive the range of a chromosome.
    pub fn get_range(&self, chr: &str) -> Option<Range<usize>> {
        let i = self.chroms.get_index_of(chr)?;
        let end = self.binned_accum_len[i];
        let start = if i == 0 {
            0
        } else {
            self.binned_accum_len[i - 1]
        };
        Some(start as usize..end as usize)
    }

    pub fn to_index(&self) -> anndata::data::index::Index {
        self.chrom_sizes()
            .map(|(chrom, length)| {
                let i = anndata::data::index::Interval {
                    start: 0,
                    end: length as usize,
                    size: self.step,
                    step: self.step,
                };
                (chrom.to_owned(), i)
            })
            .collect()
    }

    /// Number of indices.
    pub fn len(&self) -> usize {
        self.binned_accum_len
            .last()
            .map(|x| *x as usize)
            .unwrap_or(0)
    }

    pub fn chrom_sizes(&self) -> impl Iterator<Item = (&String, u64)> + '_ {
        let mut prev = 0;
        self.chroms
            .iter()
            .zip(self.base_accum_len.iter())
            .map(move |(chrom, acc)| {
                let length = acc - prev;
                prev = *acc;
                (chrom, length)
            })
    }

    /// Check if the index contains the given chromosome.
    pub fn contain_chrom(&self, chrom: &str) -> bool {
        self.chroms.contains(chrom)
    }

    pub fn with_step(&self, s: usize) -> Self {
        let mut prev = 0;
        let mut acc_low_res = 0;
        let binned_accum_len = self.base_accum_len.iter().map(|acc| {
            let length = acc - prev;
            prev = *acc;
            acc_low_res += num::Integer::div_ceil(&length, &(s as u64));
            acc_low_res
        }).collect();
        Self {
            chroms: self.chroms.clone(),
            base_accum_len: self.base_accum_len.clone(),
            binned_accum_len,
            step: s,
        }
    }

    /// Given a genomic position, return the corresponding index.
    pub fn get_position(&self, chrom: &str, pos: u64) -> usize {
        let i = self.chroms.get_index_of(chrom).expect(format!("Chromosome {} not found", chrom).as_str());
        let size = if i == 0 {
            self.base_accum_len[i]
        } else {
            self.base_accum_len[i] - self.base_accum_len[i - 1]
        };
        if pos as u64 >= size {
            panic!("Position {} is out of range for chromosome {}", pos, chrom);
        }
        let pos = (pos as usize) / self.step;
        if i == 0 {
            pos
        } else {
            self.binned_accum_len[i - 1] as usize + pos
        }
    }

    /// O(log(N)). Given a index, find the corresponding chromosome.
    pub fn get_chrom(&self, pos: usize) -> &String {
        let i = pos as u64;
        let j = match self.binned_accum_len.binary_search(&i) {
            Ok(j) => j + 1,
            Err(j) => j,
        };
        self.chroms.get_index(j).unwrap()
    }

    /// O(log(N)). Given a index, find the corresponding chromosome and position.
    pub fn get_locus(&self, pos: usize) -> GenomicRange {
        let i = pos as u64;
        match self.binned_accum_len.binary_search(&i) {
            Ok(j) => {
                let chr = self.chroms.get_index(j + 1).unwrap();
                let acc = self.base_accum_len[j + 1];
                let size = acc - self.base_accum_len[j];
                let start = 0;
                let end = (start + self.step as u64).min(size);
                GenomicRange::new(chr, start, end)
            }
            Err(j) => {
                let chr = self.chroms.get_index(j).unwrap();
                let acc = self.base_accum_len[j];
                let size = if j == 0 {
                    acc
                } else {
                    acc - self.base_accum_len[j - 1]
                };
                let prev = if j == 0 {
                    0
                } else {
                    self.binned_accum_len[j - 1]
                };
                let start = (i - prev) * self.step as u64;
                let end = (start + self.step as u64).min(size);
                GenomicRange::new(chr, start, end)
            }
        }
    }

    // Given a base index, find the corresponding index in the downsampled matrix.
    fn get_coarsed_position(&self, pos: usize) -> usize {
        if self.step <= 1 {
            pos
        } else {
            let i = pos as u64;
            match self.base_accum_len.binary_search(&i) {
                Ok(j) => self.binned_accum_len[j] as usize,
                Err(j) => {
                    let (acc, acc_low_res) = if j == 0 {
                        (0, 0)
                    } else {
                        (self.base_accum_len[j - 1], self.binned_accum_len[j - 1])
                    };
                    (acc_low_res + (i - acc) / self.step as u64) as usize
                }
            }
        }
    }
}

/// `ChromValues` is a type alias for a vector of `BedGraph<N>` objects.
/// Each `BedGraph` instance represents a genomic region along with a
/// numerical value (like coverage or score).
pub type ChromValues<N> = Vec<BedGraph<N>>;

/// `ChromValueIter` represents an iterator over the chromosome values.
/// Each item in the iterator is a tuple of a vector of `ChromValues<N>` objects,
/// a start index, and an end index.
pub struct ChromValueIter<I> {
    iter: I,
    regions: Vec<GenomicRange>,
    length: usize,
}

impl<'a, I, T> ChromValueIter<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> + 'a,
    T: Copy,
{
    /// Aggregate the values in the iterator by the given `FeatureCounter`.
    pub fn aggregate_by<C>(
        self,
        mut counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>
    where
        C: FeatureCounter<Value = T> + Clone + Sync,
        T: Sync + Send + num::ToPrimitive,
    {
        let n_col = counter.num_features();
        counter.reset();
        self.iter.map(move |(mat, i, j)| {
            let n = j - i;
            let vec = (0..n)
                .into_par_iter()
                .map(|k| {
                    let row = mat.get_row(k).unwrap();
                    let mut coverage = counter.clone();
                    row.col_indices()
                        .into_iter()
                        .zip(row.values())
                        .for_each(|(idx, val)| {
                            coverage.insert(&self.regions[*idx], *val);
                        });
                    coverage.get_counts()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap(), i, j)
        })
    }
}

impl<I, T> Iterator for ChromValueIter<I>
where
    I: Iterator<Item = (CsrMatrix<T>, usize, usize)>,
    T: Copy,
{
    type Item = (Vec<ChromValues<T>>, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(x, start, end)| {
            let values = x
                .row_iter()
                .map(|row| {
                    row.col_indices()
                        .iter()
                        .zip(row.values())
                        .map(|(i, v)| BedGraph::from_bed(&self.regions[*i], *v))
                        .collect()
                })
                .collect();
            (values, start, end)
        })
    }
}

impl<I, T> ExactSizeIterator for ChromValueIter<I>
where
    I: Iterator<Item = (CsrMatrix<T>, usize, usize)>,
    T: Copy,
{
    fn len(&self) -> usize {
        self.length
    }
}

/// `GenomeCoverage` represents a genome's base-resolution coverage.
/// It stores the coverage as an iterator of tuples, each containing a
/// compressed sparse row matrix, a start index, and an end index.
/// It also keeps track of the resolution and excluded chromosomes.
pub struct GenomeCoverage<I> {
    index: GenomeBaseIndex,
    coverage: I,
    resolution: usize,
    exclude_chroms: HashSet<String>,
}

impl<I> GenomeCoverage<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>,
{
    pub fn new(chrom_sizes: ChromSizes, coverage: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            coverage,
            resolution: 1,
            exclude_chroms: HashSet::new(),
        }
    }

    pub fn get_gindex(&self) -> GenomeBaseIndex {
        if !self.exclude_chroms.is_empty() {
            let chr_sizes = ChromSizes(
                self.index
                    .chrom_sizes()
                    .filter_map(|(chr, size)| {
                        if self.exclude_chroms.contains(chr) {
                            None
                        } else {
                            Some((chr.clone(), size))
                        }
                    })
                    .collect(),
            );
            GenomeBaseIndex::new(&chr_sizes).with_step(self.resolution)
        } else {
            self.index.with_step(self.resolution)
        }
    }

    /// Set the resolution of the coverage matrix.
    pub fn with_resolution(mut self, s: usize) -> Self {
        self.resolution = s;
        self
    }

    pub fn exclude(mut self, chroms: &[&str]) -> Self {
        self.exclude_chroms = chroms
            .iter()
            .filter(|x| self.index.chroms.contains(**x))
            .map(|x| x.to_string())
            .collect();
        self
    }

    /// Convert the coverage matrix into a vector of `BedGraph` objects.
    pub fn into_chrom_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (Vec<ChromValues<T>>, usize, usize)> {
        if !self.exclude_chroms.is_empty() {
            todo!("Implement exclude_chroms")
        }
        let index = self.get_gindex();
        self.coverage.map(move |(mat, i, j)| {
            let n = j - i;
            let values = (0..n)
                .into_par_iter()
                .map(|k| {
                    let row = mat.get_row(k).unwrap();
                    let row_entry_iter = row.col_indices().into_iter().zip(row.values());
                    if index.step <= 1 {
                        row_entry_iter
                            .map(|(idx, val)| {
                                let region = index.get_locus(*idx);
                                BedGraph::from_bed(&region, T::from_u8(*val).unwrap())
                            })
                            .collect::<Vec<_>>()
                    } else {
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row_entry_iter.for_each(|(idx, val)| {
                            let i = index.get_coarsed_position(*idx);
                            let val = T::from_u8(*val).unwrap();
                            *count.entry(i).or_insert(Zero::zero()) += val;
                        });
                        count
                            .into_iter()
                            .map(|(i, val)| {
                                let region = index.get_locus(i);
                                BedGraph::from_bed(&region, val)
                            })
                            .collect::<Vec<_>>()
                    }
                })
                .collect();
            (values, i, j)
        })
    }

    /// Output the raw coverage matrix.
    pub fn into_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> {
        let index = self.get_gindex();
        let ori_index = self.index;
        self.coverage.map(move |(mat, i, j)| {
            let new_mat = if self.resolution <= 1 && self.exclude_chroms.is_empty() {
                let (pattern, data) = mat.into_pattern_and_values();
                let new_data = data
                    .into_iter()
                    .map(|x| T::from_u8(x).unwrap())
                    .collect::<Vec<_>>();
                CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
            } else {
                let n = j - i;
                let vec = (0..n)
                    .into_par_iter()
                    .map(|k| {
                        let row = mat.get_row(k).unwrap();
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row.col_indices()
                            .into_iter()
                            .zip(row.values())
                            .for_each(|(idx, val)| {
                                let locus = ori_index.get_locus(*idx);
                                if self.exclude_chroms.is_empty()
                                    || !self.exclude_chroms.contains(locus.chrom())
                                {
                                    let i = index.get_position(locus.chrom(), locus.start());
                                    let val = T::from_u8(*val).unwrap();
                                    *count.entry(i).or_insert(Zero::zero()) += val;
                                }
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let (r, c, offset, ind, data) = to_csr_data(vec, index.len());
                CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
            };
            (new_mat, i, j)
        })
    }

    /// Aggregate the coverage by a feature counter.
    pub fn aggregate_by<C, T>(
        self,
        mut counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>
    where
        C: FeatureCounter<Value = T> + Clone + Sync,
        T: Send,
    {
        if !self.exclude_chroms.is_empty() {
            todo!("Implement exclude_chroms")
        }
        let n_col = counter.num_features();
        counter.reset();
        let index = self.index;
        self.coverage.map(move |(mat, i, j)| {
            let n = j - i;
            let vec = (0..n)
                .into_par_iter()
                .map(|k| {
                    let row = mat.get_row(k).unwrap();
                    let mut coverage = counter.clone();
                    row.col_indices()
                        .into_iter()
                        .zip(row.values())
                        .for_each(|(idx, val)| {
                            coverage.insert(&index.get_locus(*idx), *val);
                        });
                    coverage.get_counts()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap(), i, j)
        })
    }
}

pub struct ContactMap<I> {
    index: GenomeBaseIndex,
    coverage: I,
    resolution: usize,
}

impl<I> ContactMap<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>,
{
    pub fn new(chrom_sizes: ChromSizes, coverage: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            coverage,
            resolution: 1,
        }
    }

    pub fn get_gindex(&self) -> GenomeBaseIndex {
        self.index.with_step(self.resolution)
    }

    /// Set the resolution of the coverage matrix.
    pub fn with_resolution(mut self, s: usize) -> Self {
        self.resolution = s;
        self
    }

    /// Output the raw coverage matrix.
    pub fn into_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> {
        let index = self.get_gindex();
        let ori_index = self.index;
        let genome_size = ori_index.len();
        let new_size = index.len();
        self.coverage.map(move |(mat, i, j)| {
            let new_mat = if self.resolution <= 1 {
                let (pattern, data) = mat.into_pattern_and_values();
                let new_data = data
                    .into_iter()
                    .map(|x| T::from_u8(x).unwrap())
                    .collect::<Vec<_>>();
                CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
            } else {
                let n = j - i;
                let vec = (0..n)
                    .into_par_iter()
                    .map(|k| {
                        let row = mat.get_row(k).unwrap();
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row.col_indices()
                            .into_iter()
                            .zip(row.values())
                            .for_each(|(idx, val)| {
                                let ridx = idx / genome_size;
                                let cidx = idx % genome_size;
                                let locus1 = ori_index.get_locus(ridx);
                                let locus2 = ori_index.get_locus(cidx);
                                let i1 = index.get_position(locus1.chrom(), locus1.start());
                                let i2 = index.get_position(locus2.chrom(), locus2.start());
                                let i = i1 * new_size + i2;
                                let val = T::from_u8(*val).unwrap();
                                *count.entry(i).or_insert(Zero::zero()) += val;
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let (r, c, offset, ind, data) = to_csr_data(vec, new_size * new_size);
                CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
            };
            (new_mat, i, j)
        })
    }
}

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

    /// Read insertion counts stored in the `.obsm['insertion']` matrix.
    fn insertion_count_iter(&self, chunk_size: usize) -> Result<GenomeCoverage<Self::CountIter>>;

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
        let vec = fraction_in_regions(self.raw_count_iter(2000)?.into_chrom_values(), regions)
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

    fn insertion_count_iter(&self, chunk_size: usize) -> Result<GenomeCoverage<Self::CountIter>> {
        Ok(GenomeCoverage::new(
            self.read_chrom_sizes()?,
            self.obsm().get_item_iter("insertion", chunk_size).unwrap(),
        ))
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

    fn insertion_count_iter(&self, chunk_size: usize) -> Result<GenomeCoverage<Self::CountIter>> {
        Ok(GenomeCoverage::new(
            self.read_chrom_sizes()?,
            self.adatas()
                .inner()
                .get_obsm()
                .get_item_iter("insertion", chunk_size)
                .unwrap(),
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use bed_utils::bed::BEDLike;

    #[test]
    fn test_index1() {
        let chrom_sizes = vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ].into_iter().collect();
        let mut index = GenomeBaseIndex::new(&chrom_sizes);

        assert_eq!(index.get_range("1").unwrap(), 0..13);
        assert_eq!(index.get_range("2").unwrap(), 13..84);
        assert_eq!(index.get_range("3").unwrap(), 84..184);

        assert_eq!(
            chrom_sizes.clone(),
            ChromSizes(
                index
                    .chrom_sizes()
                    .map(|(a, b)| (a.to_owned(), b))
                    .collect()
            ),
        );

        [
            (0, "1:0-1"),
            (12, "1:12-13"),
            (13, "2:0-1"),
            (100, "3:16-17"),
        ]
        .into_iter()
        .for_each(|(i, txt)| {
            let locus = GenomicRange::from_str(txt).unwrap();
            assert_eq!(index.get_locus(i), locus);
            assert_eq!(index.get_position(locus.chrom(), locus.start()), i);
        });

        index = index.with_step(2);
        [(0, "1:0-2"), (6, "1:12-13"), (7, "2:0-2"), (11, "2:8-10")]
            .into_iter()
            .for_each(|(i, txt)| {
                let locus = GenomicRange::from_str(txt).unwrap();
                assert_eq!(index.get_locus(i), locus);
                assert_eq!(index.get_position(locus.chrom(), locus.start()), i);
            });

        index = index.with_step(3);
        [
            (0, "1:0-3"),
            (2, "1:6-9"),
            (4, "1:12-13"),
            (5, "2:0-3"),
            (29, "3:0-3"),
            (62, "3:99-100"),
        ]
        .into_iter()
        .for_each(|(i, txt)| {
            let locus = GenomicRange::from_str(txt).unwrap();
            assert_eq!(index.get_locus(i), locus);
            assert_eq!(index.get_position(locus.chrom(), locus.start()), i);
        });
    }

    #[test]
    fn test_index2() {
        let chrom_sizes = vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ].into_iter().collect();

        let index = GenomeBaseIndex::new(&chrom_sizes);
        [(0, 0), (12, 12), (13, 13), (100, 100)]
            .into_iter()
            .for_each(|(i, i_)| assert_eq!(index.get_coarsed_position(i), i_));

        let index2 = index.with_step(2);
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 2),
            (6, 3),
            (7, 3),
            (8, 4),
            (9, 4),
            (10, 5),
            (11, 5),
            (12, 6),
            (13, 7),
            (14, 7),
            (15, 8),
        ]
        .into_iter()
        .for_each(|(i1, i2)| {
            assert_eq!(index2.get_coarsed_position(i1), i2);
            let locus = index.get_locus(i1);
            assert_eq!(index2.get_position(locus.chrom(), locus.start()), i2);
        });
    }

    #[test]
    fn test_read_transcript() {
        let gff = "chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;level=2;hgnc_id=HGNC:37102;havana_gene=OTTHUMG00000000961.2\n\
                     chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;level=2;transcript_support_level=1\n\
                     chr1\tHAVANA\texon\t11869\t12227\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=1\n\
                     chr1\tHAVANA\texon\t12613\t12721\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=2\n\
                     chr1\tHAVANA\texon\t13221\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=3";

        let gtf = "chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id \"ENSG00000223972.5\"; gene_type \"transcribed_unprocessed_pseudogene\"; gene_name \"DDX11L1\"; level 2; hgnc_id \"HGNC:37102\"; havana_gene \"OTTHUMG00000000961.2\";\n\
            chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id \"ENSG00000223972.5\"; transcript_id \"ENST00000456328.2\"; gene_type \"transcribed_unprocessed_pseudogene\"; gene_name \"DDX11L1\"; transcript_type \"processed_transcript\"; transcript_name \"DDX11L1-202\"; level 2; transcript_support_level \"1\"; hgnc_id \"HGNC:37102\"; tag \"basic\"; havana_gene \"OTTHUMG00000000961.2\"; havana_transcript \"OTTHUMT00000362751.1\";\n\
            chr1\tHAVANA\texon\t11869\t12227\t.\t+\t.\tgene_id \"ENSG00000223972.5\"; transcript_id \"ENST00000456328.2\"; gene_type \"transcribed_unprocessed_pseudogene\"; gene_name \"DDX11L1\"; transcript_type \"processed_transcript\"; transcript_name \"DDX11L1-202\"; exon_number 1; exon_id \"ENSE00002234944.1\"; level 2; transcript_support_level \"1\"; hgnc_id \"HGNC:37102\"; tag \"basic\"; havana_gene \"OTTHUMG00000000961.2\"; havana_transcript \"OTTHUMT00000362751.1\";\n\
            chr1\tHAVANA\texon\t12613\t12721\t.\t+\t.\tgene_id \"ENSG00000223972.5\"; transcript_id \"ENST00000456328.2\"; gene_type \"transcribed_unprocessed_pseudogene\"; gene_name \"DDX11L1\"; transcript_type \"processed_transcript\"; transcript_name \"DDX11L1-202\"; exon_number 2; exon_id \"ENSE00003582793.1\"; level 2; transcript_support_level \"1\"; hgnc_id \"HGNC:37102\"; tag \"basic\"; havana_gene \"OTTHUMG00000000961.2\"; havana_transcript \"OTTHUMT00000362751.1\";\n\
            chr1\tHAVANA\texon\t13221\t14409\t.\t+\t.\tgene_id \"ENSG00000223972.5\"; transcript_id \"ENST00000456328.2\"; gene_type \"transcribed_unprocessed_pseudogene\"; gene_name \"DDX11L1\"; transcript_type \"processed_transcript\"; transcript_name \"DDX11L1-202\"; exon_number 3; exon_id \"ENSE00002312635.1\"; level 2; transcript_support_level \"1\"; hgnc_id \"HGNC:37102\"; tag \"basic\"; havana_gene \"OTTHUMG00000000961.2\"; havana_transcript \"OTTHUMT00000362751.1\";";

        let expected = Transcript {
            transcript_name: Some("DDX11L1-202".to_string()),
            transcript_id: "ENST00000456328.2".to_string(),
            gene_name: "DDX11L1".to_string(),
            gene_id: "ENSG00000223972.5".to_string(),
            is_coding: Some(false),
            chrom: "chr1".to_string(),
            left: Position::try_from(11869).unwrap(),
            right: Position::try_from(14409).unwrap(),
            strand: Strand::Forward,
        };
        assert_eq!(
            read_transcripts_from_gff(gff.as_bytes()).unwrap()[0],
            expected
        );
        assert_eq!(
            read_transcripts_from_gtf(gtf.as_bytes()).unwrap()[0],
            expected
        );
    }
}
