use anndata::{container::{ChunkedArrayElem, StackedChunkedArrayElem}, ArrayElemOp};
use bed_utils::bed::{tree::GenomeRegions, GenomicRange, BedGraph, BEDLike};
use anndata::{AnnDataOp, ElemCollectionOp, AxisArraysOp, AnnDataSet, Backend, AnnData};
use indexmap::IndexSet;
use polars::frame::DataFrame;
use nalgebra_sparse::CsrMatrix;
use noodles::{
    core::Position,
    gff::{record::Strand, Reader},
};
use num::{traits::{FromPrimitive, Zero}, integer::div_ceil};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use anyhow::{Result, Context};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    io::BufRead,
    ops::{AddAssign, Range}, str::FromStr,
};

use crate::utils::from_csr_rows;

use super::counter::FeatureCounter;

/// Position is 0-based.
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

pub fn read_transcripts<R>(input: R) -> Vec<Transcript>
where
    R: BufRead,
{
    Reader::new(input)
        .records()
        .flat_map(|r| {
            let record = r.unwrap();
            if record.ty() == "transcript" {
                let err_msg =
                    |x: &str| -> String { format!("failed to find '{}' in record: {}", x, record) };
                let left = record.start();
                let right = record.end();
                let attributes: HashMap<&str, &str> = record
                    .attributes()
                    .iter()
                    .map(|x| (x.key(), x.value()))
                    .collect();
                Some(Transcript {
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
            } else {
                None
            }
        })
        .collect()
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
pub struct ChromSizes(Vec<(String, u64)>);

impl IntoIterator for ChromSizes {
    type Item = (String, u64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// 0-based index that maps genomic loci to integers.
#[derive(Debug, Clone)]
pub struct GenomeBaseIndex {
    chroms: IndexSet<String>,
    base_accum_len: Vec<u64>,
    binned_accum_len: Vec<u64>,
    resolution: usize,
}

impl GenomeBaseIndex {
    fn new(chrom_sizes: &ChromSizes) -> Self {
        let mut acc = 0;
        let base_accum_len = chrom_sizes
            .0
            .iter()
            .map(|(_, length)| {
                acc += length;
                acc
            }).collect::<Vec<_>>();
        Self {
            chroms: chrom_sizes.0.iter().map(|x| x.0.clone()).collect(),
            binned_accum_len: base_accum_len.clone(),
            base_accum_len,
            resolution: 1,
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
        Some(start as usize .. end as usize)
    }

    pub fn to_index(&self) -> anndata::data::index::Index {
        self.chrom_sizes()
            .map(|(chrom, length)| {
                let i = anndata::data::index::Interval {
                    start: 0,
                    end: length as usize,
                    size: self.resolution,
                    step: self.resolution,
                };
                (chrom.to_owned(), i)
            }).collect()
    }

    /// Number of indices.
    pub fn len(&self) -> usize {
        self.binned_accum_len.last().map(|x| *x as usize).unwrap_or(0)
    }

    pub fn chrom_sizes(&self) -> impl Iterator<Item = (&String, u64)> + '_ {
        let mut prev = 0;
        self.chroms.iter().zip(self.base_accum_len.iter()).map(move |(chrom, acc)| {
            let length = acc - prev;
            prev = *acc;
            (chrom, length)
        })
    }

    fn with_resolution(&self, s: usize) -> Self {
        let mut prev = 0;
        let mut acc_low_res = 0;
        let binned_accum_len = self.base_accum_len.iter().map(|acc| {
            let length = acc - prev;
            prev = *acc;
            acc_low_res += length.div_ceil(s as u64);
            acc_low_res
        }).collect();
        Self {
            chroms: self.chroms.clone(),
            base_accum_len: self.base_accum_len.clone(),
            binned_accum_len,
            resolution: s,
        }
    }

    /// Given a genomic position, return the corresponding index. 
    pub fn get_position(&self, chrom: &str, pos: u64) -> usize {
        let i = self.chroms.get_index_of(chrom).unwrap();
        let size = if i == 0 {
            self.base_accum_len[i]
        } else {
            self.base_accum_len[i] - self.base_accum_len[i - 1]
        };
        if pos as u64 >= size {
            panic!("Position {} is out of range for chromosome {}", pos, chrom);
        }
        let pos = (pos as usize) / self.resolution;
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
                let chr = self.chroms.get_index(j+1).unwrap();
                let acc = self.base_accum_len[j + 1];
                let size = acc - self.base_accum_len[j];
                let start = 0;
                let end = (start + self.resolution as u64).min(size);
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
                let start = (i - prev) * self.resolution as u64;
                let end = (start + self.resolution as u64).min(size);
                GenomicRange::new(chr, start, end)
            }
        }
    }

    // Given a base index, find the corresponding index in the downsampled matrix.
    fn get_coarsed_position(&self, pos: usize) -> usize {
        if self.resolution <= 1 {
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
                    (acc_low_res + (i - acc) / self.resolution as u64) as usize
                }
            }
        }
    }
}

pub type ChromValues<N> = Vec<BedGraph<N>>;

pub struct ChromValueIter<I> {
    iter: I,
    regions: Vec<GenomicRange>,
    length: usize,
}

impl<I, T> ChromValueIter<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>,
    T: Copy,
{
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
                .collect();
            (from_csr_rows(vec, n_col), i, j)
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
            let values = x.row_iter().map(|row|
                row.col_indices().iter().zip(row.values()).map(|(i, v)|
                    BedGraph::from_bed(&self.regions[*i], *v)).collect()
            ).collect();
            (values, start, end)
        })
    }
}

impl<I, T> ExactSizeIterator for ChromValueIter<I>
where
    I: Iterator<Item = (CsrMatrix<T>, usize, usize)>,
    T: Copy,
{
    fn len(&self) -> usize { self.length }
}

/// A struct to store the base-resolution coverage of a genome.
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
                self.index.chrom_sizes().filter_map(|(chr, size)| {
                    if self.exclude_chroms.contains(chr) {
                        None
                    } else {
                        Some((chr.clone(), size))
                    }
            }).collect());
            GenomeBaseIndex::new(&chr_sizes).with_resolution(self.resolution)
        } else {
            self.index.with_resolution(self.resolution)
        }
    }

    /// Set the resolution of the coverage matrix.
    pub fn with_resolution(mut self, s: usize) -> Self {
        self.resolution = s;
        self
    }

    pub fn exclude(mut self, chroms: &[&str]) -> Self {
        self.exclude_chroms = chroms.iter().filter(|x| self.index.chroms.contains(**x))
            .map(|x| x.to_string()).collect();
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
                    if index.resolution <= 1 {
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
                }).collect();
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
                                if self.exclude_chroms.is_empty() || !self.exclude_chroms.contains(locus.chrom()) {
                                    let i = index.get_position(locus.chrom(), locus.start());
                                    let val = T::from_u8(*val).unwrap();
                                    *count.entry(i).or_insert(Zero::zero()) += val;
                                }
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect();
                from_csr_rows(vec, index.len())
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
                .collect();
            (from_csr_rows(vec, n_col), i, j)
        })
    }
}

pub trait SnapData: AnnDataOp {
    type CountIter: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>;

    /// Return chromosome names and sizes.
    fn read_chrom_sizes(&self) -> Result<ChromSizes> {
        let df = self.uns().get_item::<DataFrame>("reference_sequences")?
            .context("key 'reference_sequences' is not present in the '.uns'")?;
        let chrs = df.column("reference_seq_name").unwrap().utf8()?;
        let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
        let res = chrs.into_iter().flatten().map(|x| x.to_string())
            .zip(chr_sizes.into_iter().flatten()).collect();
        Ok(ChromSizes(res))
    }

    /// Read genome-wide base-resolution coverage
    fn raw_count_iter(
        &self, chunk_size: usize
    ) -> Result<GenomeCoverage<Self::CountIter>>;

    fn read_chrom_values(
        &self, chunk_size: usize
    ) -> Result<ChromValueIter<
        <<Self as AnnDataOp>::X as ArrayElemOp>::ArrayIter<CsrMatrix<u32>>
    >>
    {
        let regions = self.var_names().into_vec().into_iter()
            .map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect();
        Ok(ChromValueIter {
            regions,
            iter: self.x().iter(chunk_size),
            length: div_ceil(self.n_obs(), chunk_size),
        })
    }
}

impl<B: Backend> SnapData for AnnData<B> {
    type CountIter = ChunkedArrayElem<B, CsrMatrix<u8>>;

    fn raw_count_iter(
        &self, chunk_size: usize
    ) -> Result<GenomeCoverage<Self::CountIter>>
    {
        Ok(GenomeCoverage::new(
            self.read_chrom_sizes()?,
            self.obsm().get_item_iter("insertion", chunk_size).unwrap(),
        ))
    }
}

impl<B: Backend> SnapData for AnnDataSet<B> {
    type CountIter = StackedChunkedArrayElem<B, CsrMatrix<u8>>;

    fn raw_count_iter(
        &self, chunk_size: usize
    ) -> Result<GenomeCoverage<Self::CountIter>>
    {
        Ok(GenomeCoverage::new(
            self.read_chrom_sizes()?,
            self.adatas().inner().get_obsm().get_item_iter("insertion", chunk_size).unwrap(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bed_utils::bed::BEDLike;

    #[test]
    fn test_index1() {
        let chrom_sizes = ChromSizes(vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ]);
        let mut index = GenomeBaseIndex::new(&chrom_sizes);

        assert_eq!(index.get_range("1").unwrap(), 0..13);
        assert_eq!(index.get_range("2").unwrap(), 13..84);
        assert_eq!(index.get_range("3").unwrap(), 84..184);

        assert_eq!(
            chrom_sizes.clone(),
            ChromSizes(index.chrom_sizes().map(|(a,b)| (a.to_owned(),b)).collect()),
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

        index = index.with_resolution(2);
        [(0, "1:0-2"), (6, "1:12-13"), (7, "2:0-2"), (11, "2:8-10")]
            .into_iter()
            .for_each(|(i, txt)| {
                let locus = GenomicRange::from_str(txt).unwrap();
                assert_eq!(index.get_locus(i), locus);
                assert_eq!(index.get_position(locus.chrom(), locus.start()), i);
            });

        index = index.with_resolution(3);
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
        let chrom_sizes = ChromSizes(vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ]);

        let index = GenomeBaseIndex::new(&chrom_sizes);
        [(0, 0), (12, 12), (13, 13), (100, 100)]
            .into_iter()
            .for_each(|(i, i_)| {
                assert_eq!(index.get_coarsed_position(i), i_)
            });

        let index2 = index.with_resolution(2);
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
        let input = "chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;level=2;hgnc_id=HGNC:37102;havana_gene=OTTHUMG00000000961.2\n\
                     chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;level=2;transcript_support_level=1\n\
                     chr1\tHAVANA\texon\t11869\t12227\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=1\n\
                     chr1\tHAVANA\texon\t12613\t12721\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=2\n\
                     chr1\tHAVANA\texon\t13221\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=3";
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
        assert_eq!(read_transcripts(input.as_bytes())[0], expected)
    }
}
