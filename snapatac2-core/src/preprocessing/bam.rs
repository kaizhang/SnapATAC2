mod mark_duplicates;
mod header;
mod flagstat;
pub use mark_duplicates::{group_bam_by_barcode, BarcodeLocation};
pub use flagstat::{filter_bam, FlagStat, BamQC};

use bstr::BString;
use bed_utils::bed::BEDLike;
use noodles::{bam, sam::alignment::record::data::field::Tag};
use indicatif::{style::ProgressStyle, ProgressBar, ProgressDrawTarget, ProgressIterator};
use regex::Regex;
use anyhow::{Result, bail};
use std::{collections::{HashMap, HashSet}, io::Write, path::Path};
use log::warn;

use crate::utils::{open_file_for_write, Compression};
use crate::preprocessing::Fragment;

#[derive(Debug, Clone, Default)]
pub struct FragmentQC {
    mitochondrion: Option<HashSet<String>>,
    num_pcr_duplicates: u64,
    num_unique_fragments: u64,
    num_frag_nfr: u64,
    num_frag_single: u64,
}

impl FragmentQC {
    pub fn new(mitochondrion: Option<HashSet<String>>) -> Self {
        Self {
            mitochondrion,
            ..Default::default()
        }
    }

    pub fn update(&mut self, fragment: &Fragment) {
        self.num_pcr_duplicates += fragment.count as u64 - 1;
        self.num_unique_fragments += 1;
        let size = fragment.len();
        if self.mitochondrion.as_ref().map_or(true, |mito| !mito.contains(fragment.chrom())) {
            if size < 147 {
                self.num_frag_nfr += 1;
            } else if size <= 294 {
                self.num_frag_single += 1;
            }
        }
    }

    /// Report the quality control metrics.
    /// The metrics are:
    /// - Fraction_duplicates: Fraction of high-quality read pairs that are deemed
    ///                        to be PCR duplicates. This metric is a measure of
    ///                        sequencing saturation and is a function of library
    ///                        complexity and sequencing depth. More specifically,
    ///                        this is the fraction of high-quality fragments with a
    ///                        valid barcode that align to the same genomic position
    ///                        as another read pair in the library.
    pub fn report(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        result.insert("frac_duplicates".to_string(), self.num_pcr_duplicates as f64 / (self.num_unique_fragments + self.num_pcr_duplicates) as f64);
        result.insert("frac_fragment_in_nucleosome_free_region".to_string(), self.num_frag_nfr as f64 / self.num_unique_fragments as f64);
        result.insert("frac_fragment_flanking_single_nucleosome".to_string(), self.num_frag_single as f64 / self.num_unique_fragments as f64);
        result
    }
}

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
/// * `source` - The source of the data, e.g., "10x", used for specific processing.
/// * `compression` - Compression algorithm to use for the output file. Valid values are `gzip` and `zstandard`.
/// * `compression_level` - Compression level to use for the output file. Valid values are 0-9 for `gzip` and 1-22 for `zstandard`.
/// * `temp_dir` - Location for temperary files.
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
    source: Option<&str>,
    mitochondrion: Option<HashSet<String>>,
    compression: Option<Compression>,
    compression_level: Option<u32>,
    temp_dir: Option<P3>,
) -> Result<(BamQC, FragmentQC)> {
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

    let header = match source {
        Some("10x") => {
            warn!("The number of PCR duplicates cannot be computed for 10X Genomics BAM files.");
            header::read_10x_header(reader.get_mut())?
        },
        _ => reader.read_header()?,
    };

    let mut output = open_file_for_write(output_file, compression, compression_level)?;

    let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
        .with_style(
            ProgressStyle::with_template(
                "{spinner} Wrote {human_pos} barcodes in {elapsed} ({per_sec}) ...",
            )
            .unwrap(),
        );
    let mut fragment_qc = FragmentQC::new(mitochondrion.clone());
    let mut library_qc = BamQC::new(
        mitochondrion.map(|mito| mito.into_iter().flat_map(
            |x| header.reference_sequences().get_index_of(&BString::from(x))).collect()
        )
    );
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
    .for_each(|barcode| barcode.into_iter().for_each(|mut rec| {
        if rec.strand().is_none() { // perform fragment length correction for paired-end reads
            rec.set_start(rec.start().saturating_add_signed(shift_left));
            rec.set_end(rec.end().saturating_add_signed(shift_right));
        }
        if rec.len() > 0 {
            fragment_qc.update(&rec);
            writeln!(output, "{}", rec).unwrap();
        }
    }));
    Ok((library_qc, fragment_qc))
}