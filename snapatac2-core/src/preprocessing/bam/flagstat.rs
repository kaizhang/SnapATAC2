use noodles::{bam::Record, sam::alignment::record::{Cigar, Flags, cigar::op::Kind}};
use std::collections::{HashMap, HashSet};
use anyhow::{Result, Context};
use itertools::Itertools;
use serde::{Serialize, Deserialize};

use super::BarcodeLocation;

/// BAM record statistics.
#[derive(Debug, Default)]
pub struct FlagStat {
    pub read: u64,
    pub primary: u64,
    pub secondary: u64,
    pub supplementary: u64,
    pub duplicate: u64,
    pub primary_duplicate: u64,
    pub mapped: u64,
    pub primary_mapped: u64,
    pub paired: u64,
    pub read_1: u64,
    pub read_2: u64,
    pub proper_pair: u64,
    pub mate_mapped: u64,
    pub singleton: u64,
    pub mate_reference_sequence_id_mismatch: u64,
}

impl FlagStat {
    pub fn new(record: &Record) -> Self {
        let mut flagstat = FlagStat::default();
        flagstat.update(record);
        flagstat
    }

    pub fn add(&mut self, other: &FlagStat) {
        self.read += other.read;
        self.primary += other.primary;
        self.secondary += other.secondary;
        self.supplementary += other.supplementary;
        self.duplicate += other.duplicate;
        self.primary_duplicate += other.primary_duplicate;
        self.mapped += other.mapped;
        self.primary_mapped += other.primary_mapped;
        self.paired += other.paired;
        self.read_1 += other.read_1;
        self.read_2 += other.read_2;
        self.proper_pair += other.proper_pair;
        self.mate_mapped += other.mate_mapped;
        self.singleton += other.singleton;
        self.mate_reference_sequence_id_mismatch += other.mate_reference_sequence_id_mismatch;
    }

    pub fn update(&mut self, record: &Record) {
        self.read += 1;
        let flags = record.flags();

        if flags.is_duplicate() {
            self.duplicate += 1;
        }

        if !flags.is_unmapped() {
            self.mapped += 1;
        }

        if flags.is_secondary() {
            self.secondary += 1;
        } else if flags.is_supplementary() {
            self.supplementary += 1;
        } else {
            self.primary += 1;

            if !flags.is_unmapped() {
                self.primary_mapped += 1;
            }

            if flags.is_duplicate() {
                self.primary_duplicate += 1;
            }

            if flags.is_segmented() {
                self.paired += 1;

                if flags.is_first_segment() {
                    self.read_1 += 1;
                }

                if flags.is_last_segment() {
                    self.read_2 += 1;
                }

                if !flags.is_unmapped() {
                    if flags.is_properly_segmented() {
                        self.proper_pair += 1;
                    }

                    if flags.is_mate_unmapped() {
                        self.singleton += 1;
                    } else {
                        self.mate_mapped += 1;
                        let rec_id = record.mate_reference_sequence_id().unwrap().unwrap(); 
                        let mat_id = record.reference_sequence_id().unwrap().unwrap(); 

                        if mat_id != rec_id {
                            self.mate_reference_sequence_id_mismatch += 1;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct BamQC {
    mitochondrion: Option<HashSet<usize>>,
    all_reads_flagstat: FlagStat,
    barcoded_reads_flagstat: FlagStat,
    hq_flagstat: FlagStat,
    mito_flagstat: FlagStat,
    num_read1_bases: u64,
    num_read1_q30_bases: u64,
    num_read2_bases: u64,
    num_read2_q30_bases: u64,
}

impl BamQC {
    pub fn new(mitochondrion: Option<HashSet<usize>>) -> Self {
        let mut qc = BamQC::default();
        qc.mitochondrion = mitochondrion;
        qc
    }

    pub fn update(&mut self, record: &Record, barcode: &Option<String>) {
        let flagstat = FlagStat::new(record);
        if flagstat.paired == 1 && flagstat.read_2 == 1{
            self.num_read2_bases += record.sequence().len() as u64;
            self.num_read2_q30_bases += record.quality_scores().as_ref().iter().filter(|x| **x >= 30).count() as u64;
        } else {
            self.num_read1_bases += record.sequence().len() as u64;
            self.num_read1_q30_bases += record.quality_scores().as_ref().iter().filter(|x| **x >= 30).count() as u64;
        }

        self.all_reads_flagstat.add(&flagstat);
        let is_hq = record.mapping_quality().map_or(true, |x| x.get() >= 30);
        if is_hq {
            self.hq_flagstat.add(&flagstat);
        }
        if barcode.is_some() {
            self.barcoded_reads_flagstat.add(&flagstat);
            if let Some(rid) = record.reference_sequence_id() {
                if is_hq && self.mitochondrion.as_ref().map_or(false, |x| x.contains(&rid.unwrap())) {
                    self.mito_flagstat.add(&flagstat);
                }
            }
        }
    }

    /// Report the quality control metrics.
    /// The metrics are:
    /// - sequenced_reads: number of reads in the input BAM file.
    /// - sequenced_read_pairs: number of read pairs in the input BAM file.
    /// - frac_confidently_mapped: Fraction of sequenced reads or read pairs with mapping quality >= 30.
    /// - frac_unmapped: Fraction of sequenced reads or read pairs that have
    ///                      a valid barcode but could not be mapped to the genome.
    /// - frac_valid_barcode: Fraction of reads or read pairs with barcodes that match the whitelist after error correction.
    pub fn report(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        let flagstat_all = &self.all_reads_flagstat;
        let flagstat_barcoded = &self.barcoded_reads_flagstat;
        let num_reads = flagstat_all.read;
        let num_pairs = flagstat_all.paired / 2;
        let num_barcoded_reads = flagstat_barcoded.read;
        let num_barcoded_pairs = flagstat_barcoded.paired / 2;
        let mapped_pairs = flagstat_barcoded.mate_mapped / 2;
        let is_paired = num_pairs > 0;

        let fraction_unmapped = if is_paired {
            1.0 - mapped_pairs as f64 / num_barcoded_pairs as f64
        } else {
            1.0 - flagstat_barcoded.mapped as f64 / num_barcoded_reads as f64
        };
        let valid_barcode = if is_paired {
            num_barcoded_pairs as f64 / num_pairs as f64
        } else {
            num_barcoded_reads as f64 / num_reads as f64
        };
        let fraction_confidently_mapped = if is_paired {
            (self.hq_flagstat.paired / 2) as f64 / num_pairs as f64
        } else {
            self.hq_flagstat.read as f64 / num_reads as f64
        };
        let fraction_nonnuclear = if is_paired {
            (self.mito_flagstat.paired / 2) as f64 / num_pairs as f64
        } else {
            self.mito_flagstat.read as f64 / num_reads as f64
        };
        result.insert("sequenced_reads".to_string(), num_reads as f64);
        result.insert("sequenced_read_pairs".to_string(), num_pairs as f64);
        result.insert("frac_q30_bases_read1".to_string(), self.num_read1_q30_bases as f64 / self.num_read1_bases as f64);
        result.insert("frac_q30_bases_read2".to_string(), self.num_read2_q30_bases as f64 / self.num_read2_bases as f64);
        result.insert("frac_confidently_mapped".to_string(), fraction_confidently_mapped);
        result.insert("frac_unmapped".to_string(), fraction_unmapped);
        result.insert("frac_valid_barcode".to_string(), valid_barcode);
        result.insert("frac_nonnuclear".to_string(), fraction_nonnuclear);

        result
    }
}

/// Minimal information about an alignment extracted from the BAM record.
#[derive(Serialize, Deserialize, Debug)]
pub struct AlignmentInfo {
    pub name: String,
    pub reference_sequence_id: u16,
    flags: u16,
    pub alignment_start: u32,
    pub alignment_end: u32,
    pub unclipped_start: u32,
    pub unclipped_end: u32,
    pub sum_of_qual_scores: u32,
    pub barcode: Option<String>,
    pub umi: Option<String>,
}

impl AlignmentInfo {
    pub fn new(
        rec: &Record,
        barcode: Option<String>,
        umi: Option<String>,
    ) -> Result<Self> {
        let cigar = rec.cigar();
        let start: usize = rec.alignment_start().unwrap().unwrap().try_into()?;
        let alignment_start: u32 = start.try_into()?;
        let alignment_span: u32 = cigar.alignment_span()?.try_into()?;
        let alignment_end = alignment_start + alignment_span - 1;
        let clip_groups = cigar.iter().map(Result::unwrap).chunk_by(|op| {
            let kind = op.kind();
            kind == Kind::HardClip || kind == Kind::SoftClip
        });
        let mut clips = clip_groups.into_iter();
        let clipped_start: u32 = clips.next().map_or(0, |(is_clip, x)| if is_clip {
            x.map(|x| x.len() as u32).sum()
        } else {
            0
        });
        let clipped_end: u32 = clips.last().map_or(0, |(is_clip, x)| if is_clip {
            x.map(|x| x.len() as u32).sum()
        } else {
            0
        });
        Ok(Self {
            name: std::str::from_utf8(rec.name().context("no read name")?)?.to_string(),
            reference_sequence_id: rec.reference_sequence_id().context("no reference sequence id")??.try_into()?,
            flags: rec.flags().bits(),
            alignment_start,
            alignment_end,
            unclipped_start: alignment_start - clipped_start,
            unclipped_end: alignment_end + clipped_end,
            sum_of_qual_scores: sum_of_qual_score(rec),
            barcode,
            umi,
        })
    }

    pub(crate) fn flags(&self) -> Flags { Flags::from_bits_retain(self.flags) }

    pub(crate) fn alignment_5p(&self) -> u32 {
        if self.flags().is_reverse_complemented() {
            self.alignment_end
        } else {
            self.alignment_start
        }
    }
}



/// Filter Bam records.
pub fn filter_bam<'a, I>(
    reads: I,
    is_paired: bool,
    barcode_loc: &'a BarcodeLocation,
    umi_loc: Option<&'a BarcodeLocation>,
    mapq_filter: Option<u8>,
    qc: &'a mut BamQC,
) -> impl Iterator<Item = AlignmentInfo> + 'a
where
    I: Iterator<Item = Record> + 'a,
{
    // flag (1804) meaning:
    //   - read unmapped
    //   - mate unmapped
    //   - not primary alignment
    //   - read fails platform/vendor quality checks
    //   - read is PCR or optical duplicate
    let flag_failed = Flags::from_bits(1804).unwrap();
    reads.filter_map(move |r| {
        let barcode = barcode_loc.extract(&r).ok();
        let umi = umi_loc.and_then(|x| x.extract(&r).ok());
        let is_hq = mapq_filter.map_or(true, |min_q| {
            let q = r.mapping_quality().map_or(255, |x| x.get());
            q >= min_q
        });
        qc.update(&r, &barcode);

        let flag = r.flags();
        let is_properly_aligned = !flag.is_supplementary() &&
            (!is_paired || flag.is_properly_segmented());
        let flag_pass = !flag.intersects(flag_failed);
        if is_properly_aligned && flag_pass && is_hq && barcode.is_some() {
            let alignment = AlignmentInfo::new(&r, barcode, umi).unwrap();
            Some(alignment)
        } else {
            None
        }
    })
}


// The sum of all base qualities in the record above 15.
fn sum_of_qual_score(read: &Record) -> u32 {
    read.quality_scores().as_ref().iter().map(|x| u8::from(*x) as u32)
        .filter(|x| *x >= 15).sum()
}