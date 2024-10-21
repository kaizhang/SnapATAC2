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

use noodles::{bam::Record, sam::{alignment::record::data::field::{Tag, Value}, Header}};
use bed_utils::bed::{BEDLike, Strand};
use bed_utils::extsort::ExternalSorterBuilder;
use std::collections::HashMap;
use itertools::Itertools;
use log::warn;
use rayon::prelude::ParallelSliceMut;
use anyhow::{Result, bail, anyhow, Context};
use regex::Regex;

use crate::preprocessing::Fragment;
use crate::preprocessing::bam::flagstat::AlignmentInfo;

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
            BarcodeLocation::InData(tag) => match rec.data()
                .get(tag).ok_or(anyhow!("No data: {}", std::str::from_utf8(tag.as_ref())?))??
                {
                    Value::String(barcode) => Ok(barcode.to_string()),
                    _ => bail!("Not a String"),
                },
            BarcodeLocation::Regex(re) => {
                let read_name = std::str::from_utf8(rec.name().context("No read name")?)?;
                let mat = re.captures(read_name)
                    .and_then(|x| x.get(1))
                    .ok_or(anyhow!("The regex must contain exactly one capturing group matching the barcode"))?
                    .as_str().to_string();
                if mat.is_empty() {
                    warn!("regex match is empty for read name: {}", read_name);
                }
                Ok(mat)
            },
        }
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

/// Sort and group BAM
pub fn group_bam_by_barcode<I, P>(
    reads: I,
    is_paired: bool,
    temp_dir: Option<P>,
    chunk_size: usize,
) -> RecordGroups<impl Iterator<Item = AlignmentInfo>, impl FnMut(&AlignmentInfo) -> String>
where
    I: Iterator<Item = AlignmentInfo>,
    P: AsRef<std::path::Path>,
{
    let mut sorter = ExternalSorterBuilder::new()
        .with_chunk_size(chunk_size)
        .with_compression(2);
    if let Some(tmp) = temp_dir {
        sorter = sorter.with_tmp_dir(tmp);
    }
    let groups = sorter.build().unwrap()
        .sort_by(reads, |a, b| a.barcode.cmp(&b.barcode)
            .then_with(|| a.unclipped_start.cmp(&b.unclipped_start))
            .then_with(|| a.unclipped_end.cmp(&b.unclipped_end))
        ).unwrap()
        .map(|x| x.unwrap())
        .chunk_by(|x| x.barcode.as_ref().unwrap().clone());

    RecordGroups {is_paired, groups}
}

pub struct RecordGroups<I, F>
    where
        I: Iterator<Item = AlignmentInfo>,
        F: FnMut(&AlignmentInfo) -> String,
{
    is_paired:  bool,
    groups: itertools::ChunkBy<String, I, F>,
}

impl<I, F> RecordGroups<I, F>
where
    I: Iterator<Item = AlignmentInfo>,
    F: FnMut(&AlignmentInfo) -> String,
{
    pub fn into_fragments<'a>(&'a self, header: &'a Header) -> impl Iterator<Item = Vec<Fragment>> + '_ {
        self.groups.into_iter().map(|(_, rec)| get_unique_fragments(rec, header, self.is_paired))
    }
}

fn get_unique_fragments<I>(
    reads: I,
    header: &Header,
    is_paired: bool,
) -> Vec<Fragment>
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
            Some(Fragment {
                chrom: header.reference_sequences().get_index(ref_id1).unwrap().0.to_string(),
                start: start as u64 - 1,
                end: end as u64,
                barcode: Some(rec1.barcode.as_ref().unwrap().clone()),
                count: c.try_into().unwrap(),
                strand: None,
            })
        }).collect();
        result.par_sort_unstable_by(|a, b| BEDLike::compare(a, b));
        result
    } else {
        rm_dup_single(reads).map(move |(r, c)| {
            let ref_id: usize = r.reference_sequence_id.try_into().unwrap();
            Fragment {
                chrom: header.reference_sequences().get_index(ref_id).unwrap().0.to_string(),
                start: r.alignment_start as u64 - 1,
                end: r.alignment_end as u64,
                barcode: Some(r.barcode.as_ref().unwrap().clone()),
                count: c.try_into().unwrap(),
                strand: Some(if r.flags().is_reverse_complemented() {
                    Strand::Reverse
                } else {
                    Strand::Forward
                }),
            }
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