mod mark_duplicates;
pub use mark_duplicates::{filter_bam, group_bam_by_barcode, BarcodeLocation, FlagStat};

use bed_utils::bed::BEDLike;
use either::Either;
use flate2::{write::GzEncoder, Compression};
use noodles::{bam, sam::record::data::field::Tag};
use regex::Regex;
use anyhow::{Result, bail};
use std::{fs::File, io::{BufWriter, Write}, path::Path};
use tempfile::Builder;

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
pub fn make_fragment_file<P1: AsRef<Path>, P2: AsRef<Path>>(
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
) -> Result<FlagStat> {
    let tmp_dir = Builder::new()
        .tempdir_in("./")
        .expect("failed to create tmperorary directory");

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

    let mut reader = bam::reader::Builder::default().build_from_path(bam_file)?;
    let header = reader.read_header()?;

    let f = File::create(output_file.as_ref()).expect("cannot create the output file");
    let mut output: Box<dyn Write> =
        if output_file.as_ref().extension().and_then(|x| x.to_str()) == Some("gz") {
            Box::new(GzEncoder::new(BufWriter::new(f), Compression::default()))
        } else {
            Box::new(BufWriter::new(f))
        };

    let mut flagstat = FlagStat::default();
    let filtered_records = filter_bam(
        reader.lazy_records().map(|x| x.unwrap()),
        is_paired,
        mapq,
        &mut flagstat,
    );
    group_bam_by_barcode(
        filtered_records,
        &barcode,
        umi.as_ref(),
        is_paired,
        tmp_dir.path().to_path_buf(),
        chunk_size,
    )
    .into_fragments(&header)
    .for_each(|rec| match rec {
        Either::Left(mut x) => {
            let new_start = x.start().saturating_add_signed(shift_left);
            let new_end = x.end().saturating_add_signed(shift_right);
            if new_start < new_end {
                x.set_start(new_start);
                x.set_end(new_end);
                writeln!(output, "{}", x).unwrap();
            }
        }
        Either::Right(x) => writeln!(output, "{}", x).unwrap(),
    });
    Ok(flagstat)
}

/// This function is used to fix 10X bam headers, as the headers of 10X bam
/// files do not have the required `VN` field in the `@HD` record.
fn fix_header(header: String) -> String {
    fn fix_hd_rec(rec: String) -> String {
        if rec.starts_with("@HD") {
            let mut fields: Vec<_> = rec.split('\t').collect();
            if fields.len() == 1 || !fields[1].starts_with("VN") {
                fields.insert(1, "VN:1.0");
                fields.join("\t")
            } else {
                rec
            }
        } else {
            rec
        }
    }

    match header.split_once('\n') {
        None => fix_hd_rec(header),
        Some((line1, others)) => [&fix_hd_rec(line1.to_owned()), others].join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noodles::sam::Header;

    #[test]
    fn test_fix_header() {
        let input1 = "@HD\tVN:1.0\tSO:coordinate".to_owned();
        assert!(input1.parse::<Header>().is_ok());
        assert!(fix_header(input1.clone()).parse::<Header>().is_ok());

        let input2 = "@HD\tSO:coordinate".to_owned();
        assert_eq!(fix_header(input2), input1);

        let input3 = "@HD".to_owned();
        assert!(input3.parse::<Header>().is_err());
        assert!(fix_header(input3).parse::<Header>().is_ok());

        let input4 = "@HD\tSO:coordinate\n@SQ\tSN:chr1\tLN:195471971".to_owned();
        assert!(input4.parse::<Header>().is_err());
        assert!(fix_header(input4).parse::<Header>().is_ok());
    }
}