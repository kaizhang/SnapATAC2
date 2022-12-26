use bed_utils::bed::{tree::GenomeRegions, GenomicRange, BedGraph};
use anndata::{AnnDataOp, AnnDataIterator};
use polars::frame::DataFrame;
use nalgebra_sparse::CsrMatrix;
use noodles::{
    core::Position,
    gff::{record::Strand, Reader},
};
use num::traits::{FromPrimitive, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use anyhow::{Result, Context};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    io::BufRead,
    ops::AddAssign,
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

pub struct ChromSizes(Vec<(String, u64)>);

impl IntoIterator for ChromSizes {
    type Item = (String, u64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct GenomeBaseIndex {
    accum_lengths: Vec<(String, (u64, u64))>,
    resolution: usize,
}

impl GenomeBaseIndex {
    fn new(chrom_sizes: ChromSizes) -> Self {
        let mut accum_lengths = Vec::new();
        let mut acc = 0;
        for (chrom, length) in &chrom_sizes.0 {
            acc += length;
            accum_lengths.push((chrom.clone(), (acc, acc)));
        }
        Self {
            accum_lengths,
            resolution: 1,
        }
    }

    pub fn chrom_sizes(&self) -> impl Iterator<Item = (&String, u64)> {
        let iter = self.accum_lengths.as_slice().windows(2).map(|x| {
            let (_, (acc, _)) = &x[0];
            let (chrom, (acc_next, _)) = &x[1];
            (chrom, acc_next - acc)
        });
        std::iter::once((&self.accum_lengths[0].0, self.accum_lengths[0].1.0))
            .chain(iter)
    }

    fn with_resolution(&self, s: usize) -> Self {
        let mut accum_lengths = Vec::new();
        let mut prev = 0;
        let mut acc_low_res = 0;

        for (chrom, (acc, _)) in &self.accum_lengths {
            let length = acc - prev;
            prev = *acc;
            acc_low_res += length.div_ceil(s as u64);
            accum_lengths.push((chrom.clone(), (*acc, acc_low_res)));
        }

        Self {
            accum_lengths,
            resolution: s,
        }
    }

    // O(log(N)). Given a index, find the corresponding chromosome and position.
    pub fn index(&self, pos: usize) -> GenomicRange {
        let i = pos as u64;
        match self.accum_lengths.binary_search_by_key(&i, |s| s.1 .1) {
            Ok(j) => {
                let (chr, (acc, _)) = &self.accum_lengths[j + 1];
                let size = *acc - self.accum_lengths[j].1 .0;
                let start = 0;
                let end = (start + self.resolution as u64).min(size);
                GenomicRange::new(chr, start, end)
            }
            Err(j) => {
                let (chr, (acc, _)) = &self.accum_lengths[j];
                let size = if j == 0 {
                    *acc
                } else {
                    *acc - self.accum_lengths[j - 1].1 .0
                };
                let prev = if j == 0 {
                    0
                } else {
                    self.accum_lengths[j - 1].1 .1
                };
                let start = (i - prev) * self.resolution as u64;
                let end = (start + self.resolution as u64).min(size);
                GenomicRange::new(chr, start, end)
            }
        }
    }

    // Given a base index, find the corresponding index in the downsampled matrix.
    fn get_index(&self, pos: usize) -> usize {
        if self.resolution <= 1 {
            pos
        } else {
            let i = pos as u64;
            match self.accum_lengths.binary_search_by_key(&i, |s| s.1 .0) {
                Ok(j) => self.accum_lengths[j].1 .1 as usize,
                Err(j) => {
                    let (acc, acc_low_res) = if j == 0 {
                        (0, 0)
                    } else {
                        self.accum_lengths[j - 1].1
                    };
                    (acc_low_res + (i - acc) / self.resolution as u64) as usize
                }
            }
        }
    }
}

pub type ChromValues<N> = Vec<BedGraph<N>>;

pub struct GenomeCoverage<I> {
    pub index: GenomeBaseIndex,
    coverage: I,
}

impl<'a, I> GenomeCoverage<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)> + 'a,
{
    pub fn new(chrom_sizes: ChromSizes, coverage: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(chrom_sizes),
            coverage,
        }
    }

    pub fn with_resolution(mut self, s: usize) -> Self {
        self.index = self.index.with_resolution(s);
        self
    }

    pub fn into_chrom_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (Vec<ChromValues<T>>, usize, usize)> {
        let index = self.index;
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
                                let region = index.index(*idx);
                                BedGraph::from_bed(&region, T::from_u8(*val).unwrap())
                            })
                            .collect::<Vec<_>>()
                    } else {
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row_entry_iter.for_each(|(idx, val)| {
                            let i = index.get_index(*idx);
                            let val = T::from_u8(*val).unwrap();
                            *count.entry(i).or_insert(Zero::zero()) += val;
                        });
                        count
                            .into_iter()
                            .map(|(i, val)| {
                                let region = index.index(i);
                                BedGraph::from_bed(&region, val)
                            })
                            .collect::<Vec<_>>()
                    }
                }).collect();
            (values, i, j)
        })
    }

    pub fn into_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> {
        let index = self.index;
        self.coverage.map(move |(mat, i, j)| {
            let new_mat = if index.resolution <= 1 {
                let (pattern, data) = mat.into_pattern_and_values();
                let new_data = data
                    .into_iter()
                    .map(|x| T::from_u8(x).unwrap())
                    .collect::<Vec<_>>();
                CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
            } else {
                let n = j - i;
                let n_col = mat.ncols();
                let vec = (0..n)
                    .into_par_iter()
                    .map(|k| {
                        let row = mat.get_row(k).unwrap();
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row.col_indices()
                            .into_iter()
                            .zip(row.values())
                            .for_each(|(idx, val)| {
                                let i = index.get_index(*idx);
                                let val = T::from_u8(*val).unwrap();
                                *count.entry(i).or_insert(Zero::zero()) += val;
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect();
                from_csr_rows(vec, n_col)
            };
            (new_mat, i, j)
        })
    }

    pub fn aggregate_by<C, T>(
        self,
        counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> + 'a
    where
        C: FeatureCounter<Value = T> + Clone + Sync + 'a,
        T: Send,
    {
        let index = self.index.with_resolution(1);
        self.coverage.map(move |(mat, i, j)| {
            let n = j - i;
            let n_col = mat.ncols();
            let vec = (0..n)
                .into_par_iter()
                .map(|k| {
                    let row = mat.get_row(k).unwrap();
                    let mut coverage = counter.clone();
                    coverage.reset();
                    row.col_indices()
                        .into_iter()
                        .zip(row.values())
                        .for_each(|(idx, val)| {
                            coverage.insert(&index.index(*idx), *val);
                        });
                    coverage.get_counts()
                })
                .collect();
            (from_csr_rows(vec, n_col), i, j)
        })
    }
}

/*
impl<I> ChromValues<I>
where
    I: Iterator<Item = (CsrMatrix<u8>, usize, usize)>,
{
}
*/

pub trait SnapData: AnnDataOp + AnnDataIterator {
    /// Return chromosome names and sizes.
    fn read_chrom_sizes(&self) -> Result<ChromSizes>;

    /// Read genome-wide base-resolution coverage
    fn raw_count_iter<'a>(
        &'a self, chunk_size: usize
    ) -> Result<GenomeCoverage<Self::ArrayIter<'a, CsrMatrix<u8>>>>;
}

impl<T> SnapData for T where T: AnnDataOp + AnnDataIterator {
    default fn read_chrom_sizes(&self) -> Result<ChromSizes> {
        let df = self.fetch_uns::<DataFrame>("reference_sequences")?
            .context("key 'reference_sequences' is not present in the '.uns'")?;
        let chrs = df.column("reference_seq_name").unwrap().utf8()?;
        let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
        let res = chrs.into_iter().flatten().map(|x| x.to_string())
            .zip(chr_sizes.into_iter().flatten()).collect();
        Ok(ChromSizes(res))
    }

    default fn raw_count_iter<'a>(
        &'a self, chunk_size: usize
    ) -> Result<GenomeCoverage<Self::ArrayIter<'a, CsrMatrix<u8>>>>
    {
        Ok(GenomeCoverage::new(
            self.read_chrom_sizes()?,
            self.fetch_obsm_iter("insertion", chunk_size)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index1() {
        let chrom_sizes = ChromSizes(vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ]);

        let mut index = GenomeBaseIndex::new(chrom_sizes);

        [
            (0, "1:0-1"),
            (12, "1:12-13"),
            (13, "2:0-1"),
            (100, "3:16-17"),
        ]
        .into_iter()
        .for_each(|(i, expected)| assert_eq!(index.index(i).pretty_show(), expected));

        index = index.with_resolution(2);
        [(0, "1:0-2"), (6, "1:12-13"), (7, "2:0-2"), (11, "2:8-10")]
            .into_iter()
            .for_each(|(i, expected)| assert_eq!(index.index(i).pretty_show(), expected));

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
        .for_each(|(i, expected)| assert_eq!(index.index(i).pretty_show(), expected));
    }

    #[test]
    fn test_index2() {
        let chrom_sizes = ChromSizes(vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ]);

        let mut index = GenomeBaseIndex::new(chrom_sizes);
        [(0, 0), (12, 12), (13, 13), (100, 100)]
            .into_iter()
            .for_each(|(i, expected)| assert_eq!(index.get_index(i), expected));

        index = index.with_resolution(2);
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
        .for_each(|(i, expected)| assert_eq!(index.get_index(i), expected));
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