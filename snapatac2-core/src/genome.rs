//! # Genomic Feature Counter Module
//!
//! This module provides the functionality to count genomic features (such as genes or transcripts)
//! in genomic data. The primary structures in this module are `TranscriptCount` and `GeneCount`,
//! both of which implement the `FeatureCounter` trait. The `FeatureCounter` trait provides a
//! common interface for handling feature counts, including methods for resetting counts,
//! updating counts, and retrieving feature IDs, names, and counts.
//!
//! `SparseCoverage`, from the bed_utils crate, is used for maintaining counts of genomic features,
//! and this structure also implements the `FeatureCounter` trait in this module.
//!
//! `TranscriptCount` and `GeneCount` structures also hold a reference to `Promoters`, which
//! provides additional information about the genomic features being counted.
//!
//! To handle the mapping of gene names to indices, an `IndexMap` is used in the `GeneCount` structure.
//! This allows for efficient look-up of gene indices by name, which is useful when summarizing counts
//! at the gene level.
//!
//! The module aims to provide a comprehensive, efficient, and flexible way to handle and manipulate
//! genomic feature counts in Rust.
use anyhow::{bail, Context, Result};
use bed_utils::bed::map::GIntervalIndexSet;
use bed_utils::bed::GenomicRange;
use indexmap::map::IndexMap;
use indexmap::IndexSet;
use noodles::{core::Position, gff, gff::record::Strand, gtf};
use polars::frame::DataFrame;
use polars::prelude::{NamedFrom, Series};
use std::ops::Range;
use std::{collections::HashMap, fmt::Debug, io::BufRead, str::FromStr};

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

pub struct TranscriptParserOptions {
    pub transcript_name_key: String,
    pub transcript_id_key: String,
    pub gene_name_key: String,
    pub gene_id_key: String,
}

impl<'a> Default for TranscriptParserOptions {
    fn default() -> Self {
        Self {
            transcript_name_key: "transcript_name".to_string(),
            transcript_id_key: "transcript_id".to_string(),
            gene_name_key: "gene_name".to_string(),
            gene_id_key: "gene_id".to_string(),
        }
    }
}

fn from_gtf(record: &gtf::Record, options: &TranscriptParserOptions) -> Result<Transcript> {
    if record.ty() != "transcript" {
        bail!("record is not a transcript");
    }

    let left = record.start();
    let right = record.end();
    let attributes: HashMap<&str, &str> = record
        .attributes()
        .iter()
        .map(|x| (x.key(), x.value()))
        .collect();
    let get_attr = |key: &str| -> String {
        attributes
            .get(key)
            .expect(&format!("failed to find '{}' in record: {}", key, record))
            .to_string()
    };

    Ok(Transcript {
        transcript_name: attributes
            .get(options.transcript_name_key.as_str())
            .map(|x| x.to_string()),
        transcript_id: get_attr(options.transcript_id_key.as_str()),
        gene_name: get_attr(options.gene_name_key.as_str()),
        gene_id: get_attr(options.gene_id_key.as_str()),
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

fn from_gff(record: &gff::Record, options: &TranscriptParserOptions) -> Result<Transcript> {
    if record.ty() != "transcript" {
        bail!("record is not a transcript");
    }

    let left = record.start();
    let right = record.end();
    let attributes = record.attributes();
    let get_attr = |key: &str| -> String {
        attributes
            .get(key)
            .expect(&format!("failed to find '{}' in record: {}", key, record))
            .to_string()
    };

    Ok(Transcript {
        transcript_name: attributes
            .get(options.transcript_name_key.as_str())
            .map(|x| x.to_string()),
        transcript_id: get_attr(options.transcript_id_key.as_str()),
        gene_name: get_attr(options.gene_name_key.as_str()),
        gene_id: get_attr(options.gene_id_key.as_str()),
        is_coding: attributes
            .get("transcript_type")
            .map(|x| x.as_string() == Some("protein_coding")),
        chrom: record.reference_sequence_name().to_string(),
        left,
        right,
        strand: record.strand(),
    })
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

pub fn read_transcripts_from_gtf<R>(
    input: R,
    options: &TranscriptParserOptions,
) -> Result<Vec<Transcript>>
where
    R: BufRead,
{
    let mut results = Vec::new();
    input.lines().try_for_each(|line| {
        let line = line?;
        let line = gtf::Line::from_str(&line)
            .with_context(|| format!("failed to parse GTF line: {}", line))?;
        if let gtf::line::Line::Record(rec) = line {
            if rec.ty() == "transcript" {
                results.push(from_gtf(&rec, options)?);
            }
        }
        anyhow::Ok(())
    })?;
    Ok(results)
}

pub fn read_transcripts_from_gff<R>(
    input: R,
    options: &TranscriptParserOptions,
) -> Result<Vec<Transcript>>
where
    R: BufRead,
{
    let mut results = Vec::new();
    input.lines().try_for_each(|line| {
        let line = line?;
        let line = gff::Line::from_str(&line)
            .with_context(|| format!("failed to parse GFF line: {}", line))?;
        if let gff::line::Line::Record(rec) = line {
            if rec.ty() == "transcript" {
                results.push(from_gff(&rec, options)?);
            }
        }
        anyhow::Ok(())
    })?;
    Ok(results)
}

pub struct Promoters {
    pub regions: GIntervalIndexSet,
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
    pub fn total_size(&self) -> u64 {
        self.0.iter().map(|x| x.1).sum()
    }

    pub fn get(&self, chrom: &str) -> Option<u64> {
        self.0.get(chrom).copied()
    }

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

impl<S> FromIterator<(S, u64)> for ChromSizes
where
    S: Into<String>,
{
    fn from_iter<T: IntoIterator<Item = (S, u64)>>(iter: T) -> Self {
        ChromSizes(iter.into_iter().map(|(s, l)| (s.into(), l)).collect())
    }
}

impl<'a> IntoIterator for &'a ChromSizes {
    type Item = (&'a String, &'a u64);
    type IntoIter = indexmap::map::Iter<'a, String, u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl IntoIterator for ChromSizes {
    type Item = (String, u64);
    type IntoIter = indexmap::map::IntoIter<String, u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// 0-based index that maps genomic loci to integers.
#[derive(Debug, Clone)]
pub struct GenomeBaseIndex {
    pub(crate) chroms: IndexSet<String>,
    pub(crate) base_accum_len: Vec<u64>,
    pub(crate) binned_accum_len: Vec<u64>,
    pub(crate) step: usize,
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
        let binned_accum_len = self
            .base_accum_len
            .iter()
            .map(|acc| {
                let length = acc - prev;
                prev = *acc;
                acc_low_res += num::Integer::div_ceil(&length, &(s as u64));
                acc_low_res
            })
            .collect();
        Self {
            chroms: self.chroms.clone(),
            base_accum_len: self.base_accum_len.clone(),
            binned_accum_len,
            step: s,
        }
    }

    /// Given a genomic position, return the corresponding index.
    pub fn get_position_rev(&self, chrom: &str, pos: u64) -> usize {
        let i = self
            .chroms
            .get_index_of(chrom)
            .expect(format!("Chromosome {} not found", chrom).as_str());
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
    pub fn get_position(&self, pos: usize) -> (&String, u64) {
        let i = pos as u64;
        match self.binned_accum_len.binary_search(&i) {
            Ok(j) => (self.chroms.get_index(j + 1).unwrap(), 0),
            Err(j) => {
                let chr = self.chroms.get_index(j).unwrap();
                let prev = if j == 0 {
                    0
                } else {
                    self.binned_accum_len[j - 1]
                };
                let start = (i - prev) * self.step as u64;
                (chr, start)
            }
        }
    }

    /// O(log(N)). Given a index, find the corresponding chromosome and position.
    pub fn get_region(&self, pos: usize) -> GenomicRange {
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
    pub(crate) fn get_coarsed_position(&self, pos: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use bed_utils::bed::BEDLike;
    use std::str::FromStr;

    #[test]
    fn test_index1() {
        let chrom_sizes = vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ]
        .into_iter()
        .collect();
        let mut index = GenomeBaseIndex::new(&chrom_sizes);

        assert_eq!(index.get_range("1").unwrap(), 0..13);
        assert_eq!(index.get_range("2").unwrap(), 13..84);
        assert_eq!(index.get_range("3").unwrap(), 84..184);

        assert_eq!(
            chrom_sizes.clone(),
            index
                .chrom_sizes()
                .map(|(a, b)| (a.to_owned(), b))
                .collect()
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
            assert_eq!(index.get_region(i), locus);
            assert_eq!(index.get_position_rev(locus.chrom(), locus.start()), i);
        });

        index = index.with_step(2);
        [(0, "1:0-2"), (6, "1:12-13"), (7, "2:0-2"), (11, "2:8-10")]
            .into_iter()
            .for_each(|(i, txt)| {
                let locus = GenomicRange::from_str(txt).unwrap();
                assert_eq!(index.get_region(i), locus);
                assert_eq!(index.get_position_rev(locus.chrom(), locus.start()), i);
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
            assert_eq!(index.get_region(i), locus);
            assert_eq!(index.get_position_rev(locus.chrom(), locus.start()), i);
        });
    }

    #[test]
    fn test_index2() {
        let chrom_sizes = vec![
            ("1".to_owned(), 13),
            ("2".to_owned(), 71),
            ("3".to_owned(), 100),
        ]
        .into_iter()
        .collect();

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
            let locus = index.get_region(i1);
            assert_eq!(index2.get_position_rev(locus.chrom(), locus.start()), i2);
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
            read_transcripts_from_gff(gff.as_bytes(), &Default::default()).unwrap()[0],
            expected
        );
        assert_eq!(
            read_transcripts_from_gtf(gtf.as_bytes(), &Default::default()).unwrap()[0],
            expected
        );

        //let failed_line = "NC_051341.1\tGnomon\ttranscript\t26923605\t26924789\t.\t+\t.\tgene_id \"RGD1565766\"; transcript_id \"XM_039113095.1\"; db_xref \"GeneID:500622\"; gbkey \"mRNA\"; gene_name \"RGD1565766\"; model_evidence \"Supporting evidence includes similarity to: 1 EST, 4 Proteins, and 100% coverage of the annotated genomic feature by RNAseq alignments\"; product \"hypothetical gene supported by BC088468; NM_001009712\"; transcript_biotype \"mRNA\";";
        let worked = "chr1\tG\ttranscript\t26\t92\t.\t+\t.\tgene_id \"RGD\"; transcript_id \"XM_5\"; gene_name \"RGD\"; note \"note1;note2\";";
        assert!(read_transcripts_from_gtf(worked.as_bytes(), &Default::default()).is_ok());
    }
}