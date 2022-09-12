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

//use noodles::sam::alignment::Record;
use noodles::{
    bam::lazy::Record,
    sam::{
        Header,
        record::{Cigar, Data, ReadName, Flags, MappingQuality},
    },
};
use noodles::sam::record::cigar::op::Kind;
use noodles::sam::record::data::field::Tag;
use noodles::sam::record::data::field::Value;
use bed_utils::bed::{BED, Score, Strand};
use std::collections::HashMap;
use itertools::Itertools;
use extsort::sorter::Sortable;
use anyhow::{Result, bail, anyhow};
use regex::Regex;

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

// Reads are considered duplicates if and only if they have the same fingerprint.
#[derive(Eq, PartialEq, Debug, Hash)]
pub enum FingerPrint {
    SingleRead {
        reference_id: usize,
        coord_5p: usize,
        orientation: Orientation,
    },
    PairedRead {
        left_reference_id: usize,
        right_reference_id: usize,
        left_coord_5p: usize,
        right_coord_5p: usize,
        orientation: Orientation,
    }
}

impl FingerPrint {
    pub fn from_single_read(read: &Record) -> Result<FingerPrint> {
        let orientation = if read.flags()?.is_reverse_complemented() {
            Orientation::RR
        } else {
            Orientation::FF
        };
        Ok(FingerPrint::SingleRead {
            reference_id: read.reference_sequence_id()?.ok_or(anyhow!("No reference id"))?,
            coord_5p: if orientation == Orientation::FF {
                unclipped_start(read)?
            } else {
                unclipped_end(read)?
            },
            orientation,
        })
    }

    pub fn from_paired_reads(this: &Record, other: &Record) -> Result<FingerPrint> {
        let this_flags = this.flags()?;
        let other_flags = other.flags()?;
        let this_ref = this.reference_sequence_id()?.ok_or(anyhow!("No reference id"))?;
        let other_ref = other.reference_sequence_id()?.ok_or(anyhow!("No reference id"))?;
        let this_is_rev = this_flags.is_reverse_complemented();
        let other_is_rev = other_flags.is_reverse_complemented();
        let this_coord_5p = if this_is_rev {
            unclipped_end(this)?
        } else {
            unclipped_start(this)?
        };
        let other_coord_5p = if other_is_rev {
            unclipped_end(other)?
        } else {
            unclipped_start(other)?
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
        Ok(FingerPrint::PairedRead {
            left_reference_id: read1.1,
            right_reference_id: read2.1,
            left_coord_5p: read1.2,
            right_coord_5p: read2.2,
            orientation,
        })
    }
}

#[derive(Debug)]
pub enum BarcodeLocation {
    InData(Tag),
    Regex(Regex),
    InReadName,
}

pub struct RecordGroups<'a, I, F>
    where
        I: Iterator<Item = (String, Record)>,
        F: FnMut(&(String, Record)) -> String,
{
    header: &'a Header,
    is_paired:  bool,
    groups: itertools::GroupBy<String, I, F>,
}

impl<'a, I, F> RecordGroups<'a, I, F>
where
    I: Iterator<Item = (String, Record)>,
    F: FnMut(&(String, Record)) -> String,
{
    pub fn unique_records(&self) -> impl Iterator<Item = BED<6>> + '_ {
        self.groups.into_iter().flat_map(move |(bc, data)| {
            if self.is_paired {
                rm_dup_pair(data.map(|x| x.1)).flat_map(move |(k, _, _, c)| match k {
                    // FIXME: should not use unclipped start and end
                    FingerPrint::PairedRead { left_reference_id, right_reference_id, left_coord_5p, right_coord_5p, .. } => {
                        if left_reference_id != right_reference_id { return None; }
                        let bed: BED<6> = BED::new(
                            self.header.reference_sequences()[left_reference_id].name().as_str(),
                            left_coord_5p as u64 - 1,
                            right_coord_5p as u64,
                            Some(bc.clone()),
                            Some(Score::try_from(u16::try_from(c).unwrap()).unwrap()),
                            None,
                            Default::default(),
                        );
                        Some(bed)
                    },
                    _ => { None }
                }).collect::<Vec<_>>()
            } else {
                rm_dup_single(data.map(|x| x.1)).map(move |(r, c)| {
                    let cigar = Cigar::try_from(r.cigar()).unwrap();
                    let start = usize::from(r.alignment_start().unwrap().unwrap());
                    let alignment_span = cigar.alignment_span();
                    let end = start + alignment_span - 1;
                    let ref_id = r.reference_sequence_id().unwrap().unwrap();
                    BED::new(
                        self.header.reference_sequences()[ref_id].name().as_str(),
                        start as u64 - 1,
                        end as u64,
                        Some(bc.clone()),
                        Some(Score::try_from(u16::try_from(c).unwrap()).unwrap()),
                        Some(if r.flags().unwrap().is_reverse_complemented() {
                            Strand::Reverse
                        } else {
                            Strand::Forward
                        }),
                        Default::default(),
                    )
                }).collect::<Vec<_>>()
            }
        })
    }
}

// Convert Bam records to fragments (paired-end) or insertion sites (single-end).
pub fn convert_bam_to_fragment<'a, I>(
    header: &'a Header,
    reads: I,
    barcode: &'a BarcodeLocation,
    is_paired: bool,
) -> RecordGroups< 
    'a,
    impl Iterator<Item = (String, Record)> + 'a,
    impl FnMut(&(String, Record)) -> String + 'a
>
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

    let iter = reads.filter_map(move |r| {
        let flag = r.flags().unwrap();
        let pass_qc = (if is_paired { flag.is_properly_aligned() } else { true }) && 
            !flag.intersects(flag_failed);
            //r.mapping_quality().unwrap().unwrap() >= MappingQuality::new(30).unwrap();
        if pass_qc {
            Some((extract_barcode(&r, barcode).unwrap(), r))
        } else {
            None
        }
    });
    RecordGroups {
        header, is_paired, groups: sort_bam(iter).group_by(|x| x.0.clone()),
    }
}

fn sort_bam<I>(reads: I) -> impl Iterator<Item = (String, Record)>
where
    I: Iterator<Item = (String, Record)>,
{
    let mut vec: Vec<_> = reads.collect();
    vec.sort_by_key(|x| x.0.clone());
    vec.into_iter()
}

fn rm_dup_single<I>(reads: I) -> impl Iterator<Item = (Record, usize)>
where
    I: Iterator<Item = Record>,
{
    let mut result = HashMap::new();
    reads.for_each(|read| {
        let score = sum_of_qual_score(&read);
        let key = FingerPrint::from_single_read(&read).unwrap();
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

fn rm_dup_pair<I>(reads: I) -> impl Iterator<Item = (FingerPrint, Record, Record, usize)>
where
    I: Iterator<Item = Record>,
{
    let mut sorted_reads = reads.map(|x| (x.read_name().unwrap().unwrap(), x))
        .sorted_by(|a, b| Ord::cmp(&a.0, &b.0));
    let mut result = HashMap::new();
    while let Some((name1, read1)) = sorted_reads.next() {
        let (name2, read2) = sorted_reads.next().expect("Missing read2");
        if name1 != name2 {
            panic!("read1 and read2 name different");
        }
        let (read1, read2) = if read1.flags().unwrap().is_first_segment() {
            (read1, read2)
        } else {
            (read2, read1)
        };
        let score1 = sum_of_qual_score(&read1);
        let score2 = sum_of_qual_score(&read2);
        let key = FingerPrint::from_paired_reads(&read1, &read2).unwrap();
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
    }
    result.into_iter().map(|(k, x)| (k, x.0, x.2, x.4))
}

fn extract_barcode(
    record: &Record,
    location: &BarcodeLocation,
) -> Result<String>
{
    match location {
        BarcodeLocation::InData(tag) => match Data::try_from(record.data())?
            .get(*tag).ok_or(anyhow!("No data: {}", tag))?.value()
            {
                Value::String(barcode) => Ok(barcode.to_string()),
                _ => bail!("Not a String"),
            },
        BarcodeLocation::Regex(re) => {
            let read_name = record.read_name()?.ok_or(anyhow!("No read name"))?;
            let mat = re.captures(read_name.as_ref()).and_then(|x| x.get(1))
                .ok_or(anyhow!("The regex must contain exactly one capturing group matching the barcode"))?
                .as_str().to_string();
            Ok(mat)
        },
        BarcodeLocation::InReadName => {
            let read_name = record.read_name()?.ok_or(anyhow!("No read name"))?;
            Ok(<ReadName as AsRef<str>>::as_ref(&read_name).split_once(':')
                .unwrap().0.to_string()
            )
        },
    }
}

// The sum of all base qualities in the record above 15.
fn sum_of_qual_score(read: &Record) -> usize {
    read.quality_scores().as_ref().iter().map(|x| u8::from(*x) as usize)
        .filter(|x| *x >= 15).sum()
}

fn unclipped_start(read: &Record) -> Result<usize> {
    let cigar = Cigar::try_from(read.cigar())?;
    let start: usize = read.alignment_start()?
        .ok_or(anyhow!("Missing alignment information"))?.into();
    let clipped: usize = cigar.iter()
        .take_while(|op| op.kind() == Kind::HardClip || op.kind() == Kind::SoftClip)
        .map(|x| x.len()).sum();
    Ok(start - clipped)
}

fn unclipped_end(read: &Record) -> Result<usize> {
    let cigar = Cigar::try_from(read.cigar())?;
    let start: usize = read.alignment_start()?
        .ok_or(anyhow!("Missing alignment information"))?.into();
    let alignment_span = cigar.alignment_span();
    let end = start + alignment_span - 1;
    let clipped: usize = cigar.iter().rev()
        .take_while(|op| op.kind() == Kind::HardClip || op.kind() == Kind::SoftClip)
        .map(|x| x.len()).sum();
    Ok(end + clipped)
}