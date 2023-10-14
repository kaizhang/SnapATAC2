// In Illumina sequencing, there are typically two types of duplicates:
// PCR duplicates and optical duplicates. Optical duplicates are sequences from
// one cluster in fact but identified by software from multiple adjacent clusters.
// They can be identified without alignment. You just check sequence and
// the coordinates on the image.
// PCR duplicates are usually identified after alignment.
// Dedup typically works by identifying read pairs having identical 5'-end
// coordinates (3'-end coordinates are not considered). The chance of
// two 5'-end coordinates being identical depends on the coverage, but usually small.
// You can calculate the theoretical false dedup rate using my formula
// (0.28*m/s/L, where m is the number of read pairs, s is the standard deviation
// of insert size distribution and L is the length of the genome;
// sharp insert size distribution (small s) increases false dedup rate unfortunately).
// If the empirical duplicate rate is much higher than this 0.28*m/s/L, they are true PCR duplicates.
// Dedup for single-end reads has high false positive rate given deep data.
// PCR duplicates do not necessarily have the same sequence OR the same length.
// The 5' end of the read is the most reliable position because
// if it's a PCR duplicate, it will be guaranteed to be the same at that end
// but not at the 3' end.

use noodles::{
    bam::lazy::Record,
    sam::{
        Header,
        record::{Cigar, Data, Flags, cigar::op::Kind, data::field::{Tag, Value}, mapping_quality},
    },
};
use bed_utils::bed::{BED, BEDLike, Score, Strand};
use std::collections::HashMap;
use itertools::Itertools;
use extsort::{sorter::Sortable, ExternalSorter};
use bincode;
use rayon::prelude::ParallelSliceMut;
use serde::{Serialize, Deserialize};
use anyhow::{Result, bail, anyhow, Context};
use regex::Regex;
use either::Either;

// Library type    orientation   Vizualization according to first strand
// FF_firststrand  matching      3' <==2==----<==1== 5'
//                               5' ---------------- 3'
//
// FF_secondstrand matching      3' ---------------- 5'
//                               5' ==1==>----==2==> 3'
//
// RR_firststrand  matching      3' <==1==----<==2== 5'
//                               5' ---------------- 3'
//
// RR_secondstrand matching      3' ---------------- 5'
//                               5' ==2==>----==1==> 3'
//
// FR_firststrand  inward        3' ----------<==1== 5'
//                               5' ==2==>---------- 3'
//
// FR_secondstrand inward        3' ----------<==2== 5'
//                               5' ==1==>---------- 3'
//
// RF_firststrand  outward       3' <==2==---------- 5'
//                               5' ----------==1==> 3'
//
// RF_secondstrand outward       3' <==1==---------- 5'
//                               5' ----------==2==> 3'
#[derive(Eq, PartialEq, Debug, Hash)]
pub enum Orientation { FR, FF, RR, RF }


/// Where to extract the barcode information from the BAM records.
#[derive(Debug)]
pub enum BarcodeLocation {
    InData(Tag),
    Regex(Regex),
}

impl BarcodeLocation {
    pub fn extract(&self, rec: &Record) -> Result<String> {
        match self {
            BarcodeLocation::InData(tag) => match Data::try_from(rec.data())?
                .get(tag).ok_or(anyhow!("No data: {}", tag))?
                {
                    Value::String(barcode) => Ok(barcode.to_string()),
                    _ => bail!("Not a String"),
                },
            BarcodeLocation::Regex(re) => {
                let read_name = rec.read_name().context("No read name")?;
                let mat = re.captures(std::str::from_utf8(read_name.as_bytes())?)
                    .and_then(|x| x.get(1))
                    .ok_or(anyhow!("The regex must contain exactly one capturing group matching the barcode"))?
                    .as_str().to_string();
                Ok(mat)
            },
        }
    }
}

/// Minimal information about an alignment extracted from the BAM record.
#[derive(Serialize, Deserialize, Debug)]
pub struct AlignmentInfo {
    name: String,
    reference_sequence_id: u16,
    flags: u16,
    alignment_start: u32,
    alignment_end: u32,
    unclipped_start: u32,
    unclipped_end: u32,
    sum_of_qual_scores: u32,
    barcode: Option<String>,
    umi: Option<String>,
}

impl AlignmentInfo {
    fn new(
        rec: &Record,
        barcode_loc: &BarcodeLocation,
        umi_loc: Option<&BarcodeLocation>,
    ) -> Result<Self> {
        let cigar = Cigar::try_from(rec.cigar())?;
        let start: usize = rec.alignment_start().unwrap().unwrap().try_into()?;
        let alignment_start: u32 = start.try_into()?;
        let alignment_span: u32 = cigar.alignment_span().try_into()?;
        let alignment_end = alignment_start + alignment_span - 1;
        let clipped_start: u32 = cigar.iter()
            .take_while(|op| op.kind() == Kind::HardClip || op.kind() == Kind::SoftClip)
            .map(|x| x.len() as u32).sum();
        let clipped_end: u32 = cigar.iter().rev()
            .take_while(|op| op.kind() == Kind::HardClip || op.kind() == Kind::SoftClip)
            .map(|x| x.len() as u32).sum();

        Ok(Self {
            name: std::str::from_utf8(rec.read_name().context("no read name")?.as_bytes())?.to_string(),
            reference_sequence_id: rec.reference_sequence_id()?.context("no reference sequence id")?.try_into()?,
            flags: rec.flags().bits(),
            alignment_start,
            alignment_end,
            unclipped_start: alignment_start - clipped_start,
            unclipped_end: alignment_end + clipped_end,
            sum_of_qual_scores: sum_of_qual_score(rec),
            barcode: barcode_loc.extract(rec).ok(),
            umi: umi_loc.and_then(|x| x.extract(rec).ok()),
        })
    }

    fn flags(&self) -> Flags { Flags::from_bits_retain(self.flags) }

    fn alignment_5p(&self) -> u32 {
        if self.flags().is_reverse_complemented() {
            self.alignment_end
        } else {
            self.alignment_start
        }
    }
}

impl Sortable for AlignmentInfo {
    fn encode<W: std::io::Write>(&self, writer: &mut W) {
        bincode::serialize_into(writer, self).unwrap();
    }

    fn decode<R: std::io::Read>(reader: &mut R) -> Option<Self> {
        bincode::deserialize_from(reader).ok()
    }
}


/// Reads are considered duplicates if and only if they have the same fingerprint.
#[derive(Eq, PartialEq, Debug, Hash)]
pub enum FingerPrint {
    SingleRead {
        reference_id: usize,
        coord_5p: u32,
        orientation: Orientation,
        barcode: Option<String>,
    },
    PairedRead {
        left_reference_id: usize,
        right_reference_id: usize,
        left_coord_5p: u32,
        right_coord_5p: u32,
        orientation: Orientation,
        barcode: Option<String>,
    },
}

impl FingerPrint {
    /// Extract the fingerprint from a single-end BAM record.
    pub fn from_single_read(read: &AlignmentInfo) -> FingerPrint {
        let orientation = if read.flags().is_reverse_complemented() {
            Orientation::RR
        } else {
            Orientation::FF
        };
        FingerPrint::SingleRead {
            reference_id: read.reference_sequence_id.try_into().unwrap(),
            coord_5p: if orientation == Orientation::FF {
                read.unclipped_start
            } else {
                read.unclipped_end
            },
            orientation,
            barcode: read.umi.clone(),
        }
    }

    /// Extract the fingerprint from a paired-end BAM record.
    pub fn from_paired_reads(
        this: &AlignmentInfo,
        other: &AlignmentInfo,
    ) -> FingerPrint {
        if this.umi != other.umi { panic!("UMI mismatch"); }

        let this_flags = this.flags();
        let other_flags = other.flags();
        let this_ref = this.reference_sequence_id.try_into().unwrap();
        let other_ref = other.reference_sequence_id.try_into().unwrap();
        let this_is_rev = this_flags.is_reverse_complemented();
        let other_is_rev = other_flags.is_reverse_complemented();
        let this_coord_5p = if this_is_rev {
            this.unclipped_end
        } else {
            this.unclipped_start
        };
        let other_coord_5p = if other_is_rev {
            other.unclipped_end
        } else {
            other.unclipped_start
        };
        let this_is_leftmost = if this_ref == other_ref {
            if this_coord_5p < other_coord_5p { true } else { false }
        } else {
            this_ref < other_ref
        };
        let read1;
        let read2;
        if this_is_leftmost {
            read1 = (this_flags, this_ref, this_coord_5p, this_is_rev);
            read2 = (other_flags, other_ref, other_coord_5p, other_is_rev);
        } else {
            read2 = (this_flags, this_ref, this_coord_5p, this_is_rev);
            read1 = (other_flags, other_ref, other_coord_5p, other_is_rev);
        }
        let orientation = if read1.3 == read2.3 {
            if read1.3 {
                if read1.0.is_first_segment() { Orientation::RR } else { Orientation::FF }
            } else {
                if read1.0.is_first_segment() { Orientation::FF } else { Orientation::RR }
            }
        } else {
            if read1.3 { Orientation::RF } else { Orientation::FR }
        };
        FingerPrint::PairedRead {
            left_reference_id: read1.1,
            right_reference_id: read2.1,
            left_coord_5p: read1.2,
            right_coord_5p: read2.2,
            orientation,
            barcode: this.umi.clone(),
        }
    }
}

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
    pub mate_reference_sequence_id_mismatch_hq: u64,
}

impl FlagStat {
    pub fn update(&mut self, record: &Record) {
        let flags = record.flags();

        self.read += 1;

        if !flags.is_unmapped() {
            self.mapped += 1;
        }

        if flags.is_duplicate() {
            self.duplicate += 1;
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
                    if flags.is_properly_aligned() {
                        self.proper_pair += 1;
                    }

                    if flags.is_mate_unmapped() {
                        self.singleton += 1;
                    } else {
                        self.mate_mapped += 1;

                        if record.mate_reference_sequence_id().unwrap() != record.reference_sequence_id().unwrap() {
                            self.mate_reference_sequence_id_mismatch += 1;

                            let mapq = record
                                .mapping_quality()
                                .map_or(mapping_quality::MISSING, |x| x.get());

                            if mapq >= 5 {
                                self.mate_reference_sequence_id_mismatch_hq += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Filter Bam records.
pub fn filter_bam<'a, I>(
    reads: I,
    is_paired: bool,
    mapq_filter: Option<u8>,
    flagstat: &'a mut FlagStat,
) -> impl Iterator<Item = Record> + 'a
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
    reads.filter(move |r| {
        flagstat.update(r);
        let flag = r.flags();
        let is_properly_aligned = !flag.is_supplementary() &&
            (!is_paired || flag.is_properly_aligned());
        let flag_pass = !flag.intersects(flag_failed);
        let mapq_pass = mapq_filter.map_or(true, |min_q| {
            let q = r.mapping_quality().map_or(mapping_quality::MISSING, |x| x.get());
            q >= min_q
        });
        is_properly_aligned && flag_pass && mapq_pass
    })
}

/// Sort and group BAM
pub fn group_bam_by_barcode<'a, I>(
    reads: I,
    barcode_loc: &'a BarcodeLocation,
    umi_loc: Option<&'a BarcodeLocation>,
    is_paired: bool,
    sort_dir: std::path::PathBuf,
    chunk_size: usize,
) -> RecordGroups< 
    impl Iterator<Item = AlignmentInfo> + 'a,
    impl FnMut(&AlignmentInfo) -> String + 'a
>
where
    I: Iterator<Item = Record> + 'a,
{
    fn sort_rec<I>(reads: I, tmp_dir: std::path::PathBuf, chunk: usize) -> impl Iterator<Item = AlignmentInfo>
    where
        I: Iterator<Item = AlignmentInfo>,
    {
        ExternalSorter::new()
            .with_segment_size(chunk)
            .with_sort_dir(tmp_dir)
            .with_parallel_sort()
            .sort_by(reads, |a, b| a.barcode.cmp(&b.barcode)
                .then_with(|| a.unclipped_start.cmp(&b.unclipped_start))
                .then_with(|| a.unclipped_end.cmp(&b.unclipped_end))
            ).unwrap()
    }

    RecordGroups {
        is_paired,
        groups: sort_rec(
            reads.map(move |x| AlignmentInfo::new(&x, barcode_loc, umi_loc).unwrap())
                .filter(|x| x.barcode.is_some()),
            sort_dir,
            chunk_size,
        ).group_by(|x| x.barcode.as_ref().unwrap().clone()),
    }
}

pub struct RecordGroups<I, F>
    where
        I: Iterator<Item = AlignmentInfo>,
        F: FnMut(&AlignmentInfo) -> String,
{
    is_paired:  bool,
    groups: itertools::GroupBy<String, I, F>,
}

impl<I, F> RecordGroups<I, F>
where
    I: Iterator<Item = AlignmentInfo>,
    F: FnMut(&AlignmentInfo) -> String,
{
    pub fn into_fragments<'a>(&'a self, header: &'a Header) -> impl Iterator<Item = Either<BED<5>, BED<6>>> + '_ {
        self.groups.into_iter().flat_map(|(_, rec)| get_unique_fragments(rec, header, self.is_paired))
    }
}

fn get_unique_fragments<I>(
    reads: I,
    header: &Header,
    is_paired: bool,
) -> Vec<Either<BED<5>, BED<6>>>
where
    I: Iterator<Item = AlignmentInfo>,
{
    if is_paired {
        let mut result: Vec<_> = rm_dup_pair(reads).flat_map(move |(rec1, rec2, c)| {
            let ref_id1: usize = rec1.reference_sequence_id.try_into().unwrap();
            let ref_id2: usize = rec2.reference_sequence_id.try_into().unwrap();
            if ref_id1 != ref_id2 { return None; }
            let rec1_5p = rec1.alignment_5p();
            let rec2_5p = rec2.alignment_5p();
            let (start, end) = if rec1_5p < rec2_5p {
                (rec1_5p, rec2_5p)
            } else {
                (rec2_5p, rec1_5p)
            };
            Some(Either::Left(BED::new(
                header.reference_sequences().get_index(ref_id1).unwrap().0.as_str(),
                start as u64 - 1,
                end as u64,
                Some(rec1.barcode.as_ref().unwrap().clone()),
                Some(Score::try_from(u16::try_from(c).unwrap()).unwrap()),
                None,
                Default::default(),
            )))
        }).collect();
        result.par_sort_unstable_by(|a, b| match (a, b) {
            (Either::Left(a_), Either::Left(b_)) => BEDLike::compare(a_, b_),
            _ => todo!(),
        });
        result
    } else {
        rm_dup_single(reads).map(move |(r, c)| {
            let ref_id: usize = r.reference_sequence_id.try_into().unwrap();
            Either::Right(BED::new(
                header.reference_sequences().get_index(ref_id).unwrap().0.as_str(),
                r.alignment_start as u64 - 1,
                r.alignment_end as u64,
                Some(r.barcode.as_ref().unwrap().clone()),
                Some(Score::try_from(u16::try_from(c).unwrap()).unwrap()),
                Some(if r.flags().is_reverse_complemented() {
                    Strand::Reverse
                } else {
                    Strand::Forward
                }),
                Default::default(),
            ))
        }).collect()
    }
}

/// Remove duplicate single-end reads.
fn rm_dup_single<I>(reads: I) -> impl Iterator<Item = (AlignmentInfo, usize)>
where
    I: Iterator<Item = AlignmentInfo>,
{
    let mut result = HashMap::new();
    reads.for_each(|read| {
        let score = read.sum_of_qual_scores;
        let key = FingerPrint::from_single_read(&read);
        match result.get_mut(&key) {
            None => { result.insert(key, (read, score, 1)); },
            Some(val) => {
                val.2 = val.2 + 1;
                if val.1 < score {
                    val.0 = read;
                    val.1 = score;
                }
            },
        }
    });
    result.into_values().map(|x| (x.0, x.2))
}

/// Remove duplicate paired-end reads.
fn rm_dup_pair<I>(reads: I) -> impl Iterator<Item = (AlignmentInfo, AlignmentInfo, usize)>
where
    I: Iterator<Item = AlignmentInfo>,
{
    // Sort the reads by name, so that paired reads are next to each other.
    let mut sorted_reads: Vec<_> = reads.collect();
    sorted_reads.par_sort_unstable_by(|a, b| a.name.cmp(&b.name));

    let mut result = HashMap::new();
    sorted_reads.into_iter().fold(None, |state: Option<AlignmentInfo>, cur_rec| match state {
        Some(prev_rec) => if prev_rec.name == cur_rec.name {
            let (read1, read2) = if prev_rec.flags().is_first_segment() {
                (prev_rec, cur_rec)
            } else {
                (cur_rec, prev_rec)
            };
            let score1 = read1.sum_of_qual_scores;
            let score2 = read2.sum_of_qual_scores;
            let key = FingerPrint::from_paired_reads(&read1, &read2);
            match result.get_mut(&key) {
                None => { result.insert(key, (read1, score1, read2, score2, 1)); },
                Some(val) => {
                    val.4 = val.4 + 1;
                    if val.1 < score1 {
                        val.0 = read1;
                        val.1 = score1;
                    }
                    if val.3 < score2 {
                        val.2 = read2;
                        val.3 = score2;
                    }
                },
            }
            None
        } else {
            Some(cur_rec)
        },
        None => Some(cur_rec),
    });
    
    result.into_values().map(|x| (x.0, x.2, x.4))
}

// The sum of all base qualities in the record above 15.
fn sum_of_qual_score(read: &Record) -> u32 {
    read.quality_scores().as_ref().iter().map(|x| u8::from(*x) as u32)
        .filter(|x| *x >= 15).sum()
}