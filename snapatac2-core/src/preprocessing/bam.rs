mod mark_duplicates;
pub use mark_duplicates::{filter_bam, group_bam_by_barcode, BarcodeLocation, FlagStat, LibraryQC};

use bed_utils::bed::BEDLike;
use noodles::{bam, sam::alignment::record::data::field::Tag};
use indicatif::{style::ProgressStyle, ProgressBar, ProgressDrawTarget, ProgressIterator};
use regex::Regex;
use anyhow::{Result, bail};
use std::{io::Write, path::Path};

use crate::utils::{open_file_for_write, Compression};

/// Convert a BAM file to a fragment file by performing the following steps:
///
/// 1. Filtering: remove reads that are unmapped, not primary alignment, mapq < 30,
///    fails platform/vendor quality checks, or optical duplicate.
///    For paired-end sequencing, it also removes reads that are not properly aligned.
/// 2. Deduplicate: Sort the reads by cell barcodes and remove duplicated reads
///    for each unique cell barcode.
/// 3. Output: Convert BAM records to fragments (if paired-end) or single-end reads.
///
/// Note the bam file needn't be sorted or filtered.
///
/// # Arguments
///
/// * `bam_file` - File name of the BAM file.
/// * `output_file` - File name of the output fragment file.
/// * `is_paired` - Indicate whether the BAM file contain paired-end reads.
/// * `barcode_tag` - Extract barcodes from TAG fields of BAM records, e.g., `barcode_tag = "CB"`.
/// * `barcode_regex` - Extract barcodes from read names of BAM records using regular expressions.
///     Reguler expressions should contain exactly one capturing group
///     (Parentheses group the regex between them) that matches
///     the barcodes. For example, `barcode_regex = "(..:..:..:..):\w+$"`
///     extracts `bd:69:Y6:10` from
///     `A01535:24:HW2MMDSX2:2:1359:8513:3458:bd:69:Y6:10:TGATAGGTTG`.
/// * `umi_tag` - Extract UMI from TAG fields of BAM records.
/// * `umi_regex` - Extract UMI from read names of BAM records using regular expressions.
///     See `barcode_regex` for more details.
/// * `shift_left` - Insertion site correction for the left end.
/// * `shift_right` - Insertion site correction for the right end.
/// * `chunk_size` - The size of data retained in memory when performing sorting. Larger chunk sizes
///     result in faster sorting and greater memory usage.
/// * `compression` - Compression algorithm to use for the output file. Valid values are `gzip` and `zstandard`.
/// * `compression_level` - Compression level to use for the output file. Valid values are 0-9 for `gzip` and 1-22 for `zstandard`.
pub fn make_fragment_file<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
    bam_file: P1,
    output_file: P2,
    is_paired: bool,
    barcode_tag: Option<[u8; 2]>,
    barcode_regex: Option<&str>,
    umi_tag: Option<[u8; 2]>,
    umi_regex: Option<&str>,
    shift_left: i64,
    shift_right: i64,
    mapq: Option<u8>,
    chunk_size: usize,
    compression: Option<Compression>,
    compression_level: Option<u32>,
    temp_dir: Option<P3>,
) -> Result<LibraryQC> {
    if barcode_regex.is_some() && barcode_tag.is_some() {
        bail!("Can only set barcode_tag or barcode_regex but not both");
    }
    if umi_regex.is_some() && umi_tag.is_some() {
        bail!("Can only set umi_tag or umi_regex but not both");
    }
    let barcode = match barcode_tag {
        Some(tag) => BarcodeLocation::InData(Tag::try_from(tag)?),
        None => match barcode_regex {
            Some(regex) => BarcodeLocation::Regex(Regex::new(regex)?),
            None => bail!("Either barcode_tag or barcode_regex must be set"),
        },
    };
    let umi = match umi_tag {
        Some(tag) => Some(BarcodeLocation::InData(Tag::try_from(tag)?)),
        None => match umi_regex {
            Some(regex) => Some(BarcodeLocation::Regex(Regex::new(regex)?)),
            None => None,
        },
    };

    let mut reader = bam::io::reader::Builder::default().build_from_path(bam_file)?;
    let header = reader.read_header()?;

    let mut output = open_file_for_write(output_file, compression, compression_level)?;

    let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
        .with_style(
            ProgressStyle::with_template(
                "{spinner} Wrote {human_pos} barcodes in {elapsed} ({per_sec}) ...",
            )
            .unwrap(),
        );
    let mut library_qc = LibraryQC::default();
    let mut num_pcr_duplicates = 0;
    let mut num_uniques = 0;
    let mut num_barcodes = 0;
    let filtered_records = filter_bam(
        reader.records().map(Result::unwrap),
        is_paired,
        &barcode,
        umi.as_ref(),
        mapq,
        &mut library_qc,
    );
    group_bam_by_barcode(
        filtered_records,
        is_paired,
        temp_dir,
        chunk_size,
    )
    .into_fragments(&header)
    .progress_with(spinner)
    .for_each(|barcode| {
        num_barcodes += 1;
        barcode.into_iter().for_each(|mut rec| {
            num_pcr_duplicates += rec.count as u64 - 1;
            num_uniques += 1;
            if rec.strand().is_none() {
                let new_start = rec.start().saturating_add_signed(shift_left);
                let new_end = rec.end().saturating_add_signed(shift_right);
                if new_start < new_end {
                    rec.set_start(new_start);
                    rec.set_end(new_end);
                    writeln!(output, "{}", rec).unwrap();
                }
            } else {
                writeln!(output, "{}", rec).unwrap();
            }
        });
    });
    library_qc.num_pcr_duplicates = num_pcr_duplicates;
    library_qc.num_uniques = num_uniques;
    library_qc.num_barcodes = num_barcodes;
    Ok(library_qc)
}