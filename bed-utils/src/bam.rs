use noodles::bam;
use crate::bed::{BED, Score, Strand};
use indexmap::map::IndexMap;
use noodles::sam::header::ReferenceSequence;
use std::collections::hash_map::RandomState;
use noodles::sam::RecordExt;

pub fn bam_to_bed(refs: &IndexMap<String, ReferenceSequence, RandomState>,
                  record: &bam::Record)
                  -> Option<BED<6>> {
    let chr = record.reference_sequence(refs)?.unwrap().name().as_str().to_string();
    let start_loc: i32 = record.alignment_start()?.into();
    let end_loc: i32 = record.alignment_end()?.unwrap().into();
    let name = record.read_name().unwrap().to_str().unwrap().to_string();
    let score = record.mapping_quality().map(|x| Score::try_from(x as u16).unwrap());
    let strand = if record.flags().is_reverse_complemented()
        { Strand::Reverse } else { Strand::Forward };
    Some(BED::new(chr, (start_loc - 1) as u64, end_loc as u64,
                  Some(name), score, Some(strand), Default::default()))
}