use noodles::sam::alignment::Record;
use noodles::sam::record::cigar::op::Kind;
use noodles::sam::record::data::field::Tag;
use noodles::sam::record::data::field::Value;
use std::collections::HashMap;
use anyhow::{Result, bail, anyhow};

// PCR duplicates do not necessarily have the same sequence OR the same length.
// The 5' end of the read is the most reliable position because
// if it's a PCR duplicate, it will be guaranteed to be the same at that end
// but not at the 3' end. That is, I think, the reasoning behind not considering the 3' end.

//In Illumina sequencing, there are typically two types of duplicates: PCR duplicates and optical duplicates. Optical duplicates are sequences from one cluster in fact but identified by software from multiple adjacent clusters. They can be identified without alignment. You just check sequence and the coordinates on the image.

//PCR duplicates are usually identified after alignment. Although there are alignment-free approaches, they are less powerful when alignment is available. Dedup typically works by identifying read pairs having identical 5'-end coordinates (3'-end coordinates are not considered). The chance of two 5'-end coordinates being identical depends on the coverage, but usually small. You can calculate the theoretical false dedup rate using my formula (0.28*m/s/L, where m is the number of read pairs, s is the standard deviation of insert size distribution and L is the length of the genome; sharp insert size distribution (small s) increases false dedup rate unfortunately). If the empirical duplicate rate is much higher than this 0.28*m/s/L, they are true PCR duplicates.

//Dedup for single-end reads has high false positive rate given deep data. This is fine for SNP calling because the 40X coverage is enough for SNP calling. However, it is dangerous to dedup SE reads for RNA-seq or ChIP-seq where read count matters. As mrawlins has suggested, it would be better to account for duplicates in your read counting model rather than run a dedup program.

//The rate of PCR duplicates is 0.5*m/N (m is the number of sequenced reads and N is the number of DNA molecules before amplification). It is not affected by the PCR cycles, at least not greatly. The key to reducing PCR duplicates is to get enough DNA (get large N). Also, the dup rate is proportional to M. The more you sequence, the higher dup rate you get.

// Library type    orientation   Vizualization according to first strand
// ff_firststrand  matching      3' <==2==----<==1== 5'
//                               5' ---------------- 3'
//
// ff_secondstrand matching      3' ---------------- 5'
//                               5' ==1==>----==2==> 3'
//
// fr_firststrand  inward        3' ----------<==1== 5'
//                               5' ==2==>---------- 3'
//
// fr_secondstrand inward        3' ----------<==2== 5'
//                               5' ==1==>---------- 3'
//
// rf_firststrand  outward       3' <==2==---------- 5'
//                               5' ----------==1==> 3'
//
// rf_secondstrand outward       3' <==1==---------- 5'
//                               5' ----------==2==> 3'
#[derive(Eq, PartialEq, Debug, Hash)]
pub enum Orientation { FR, FF, RR, RF }

// Reads are considered duplicates if and only if they have the same fingerprint.
#[derive(Eq, PartialEq, Debug, Hash)]
pub enum FingerPrint {
    SingleRead {
        reference_id: usize,
        location_5p: usize,
        orientation: Orientation,
    },
    PairedRead {
        first_segment_reference_id: usize,
        second_segment_reference_id: usize,
        first_segment_location_5p: usize,
        second_segment_location_5p: usize,
        first_segment_orientation: Orientation,
        second_segment_orientation: Orientation,
        is_leftmost: bool,
    }
}

fn unclipped_start(read: &Record) -> Result<usize> {
    let start = read.alignment_start()
        .ok_or(anyhow!("Missing alignment information"))?;
    let clipped: usize = read.cigar().iter()
        .take_while(|op| op.kind() == Kind::HardClip || op.kind() == Kind::SoftClip)
        .map(|x| x.len()).sum();
    Ok(usize::from(start) - clipped)
}

fn unclipped_end(read: &Record) -> Result<usize> {
    let end = read.alignment_end()
        .ok_or(anyhow!("Missing alignment information"))?;
    let clipped: usize = read.cigar().iter().rev()
        .take_while(|op| op.kind() == Kind::HardClip || op.kind() == Kind::SoftClip)
        .map(|x| x.len()).sum();
    Ok(usize::from(end) + clipped)
}

impl FingerPrint {
    pub fn from_single_read(read: &Record) -> Result<FingerPrint> {
        let orientation = if read.flags().is_reverse_complemented() {
            Orientation::RR
        } else {
            Orientation::FF
        };
        Ok(FingerPrint::SingleRead {
            reference_id: read.reference_sequence_id().ok_or(anyhow!("No reference id"))?,
            location_5p: if orientation == Orientation::FF {
                unclipped_start(read)?
            } else {
                unclipped_end(read)?
            },
            orientation,
        })
    }
}

// The sum of all base qualities in the record above 15.
fn sum_of_qual_score(read: &Record) -> usize {
    read.quality_scores().as_ref().iter().map(|x| u8::from(*x) as usize).filter(|x| *x >= 15).sum()
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

pub enum BarcodeLocation {
    Data(Tag),
    Name(String),
}

fn extract_barcode(
    record: &Record,
    location: BarcodeLocation,
) -> Result<String>
{
    match location {
        BarcodeLocation::Data(tag) => match record.data().get(tag)
            .ok_or(anyhow!("No data: {}", tag))?.value() {
                Value::String(barcode) => Ok(barcode.to_string()),
                _ => bail!("Not a String"),
        }
        _ => todo!(),
    }
}