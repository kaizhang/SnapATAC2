use crate::utils::*;

use std::{
    fs::File,
    io::{BufWriter, BufReader, Write},
    str::FromStr,
    collections::BTreeMap,
    ops::Deref,
    collections::HashSet,
};
use pyo3::{
    PyTypeInfo,
    prelude::*,
    PyResult, Python,
    exceptions::PyTypeError,
};
use flate2::{Compression, write::GzEncoder};
use bed_utils::{bed, bed::{BEDLike, GenomicRange}};
use rayon::ThreadPoolBuilder;
use polars::prelude::DataFrame;
use noodles::bam;
use noodles::sam::Header;
use noodles::sam::record::data::field::Tag;
use anndata_rs::anndata;
use pyanndata::{AnnData, AnnDataSet};
use regex::Regex;
use tempfile::Builder;

use snapatac2_core::{
    preprocessing::{
        mark_duplicates::{BarcodeLocation, BedN, filter_bam, group_bam_by_barcode},
        matrix::{create_tile_matrix, create_peak_matrix, create_gene_matrix},
        qc,
    },
    utils::{gene::read_transcripts, ChromValuesReader},
};

/// Convert a BAM file to a fragment file.
/// 
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
/// Parameters
/// ----------
///
/// bam_file 
///     File name of the BAM file.
/// output_file
///     File name of the output fragment file.
/// is_paired
///     Indicate whether the BAM file contain paired-end reads
/// barcode_tag
///     Extract barcodes from TAG fields of BAM records, e.g., `barcode_tag = "CB"`.
/// barcode_regex
///     Extract barcodes from read names of BAM records using regular expressions.
///     Reguler expressions should contain exactly one capturing group 
///     (Parentheses group the regex between them) that matches
///     the barcodes. For example, `barcode_regex = "(..:..:..:..):\w+$"`
///     extracts `bd:69:Y6:10` from
///     `A01535:24:HW2MMDSX2:2:1359:8513:3458:bd:69:Y6:10:TGATAGGTTG`.
/// umi_tag
///     Extract UMI from TAG fields of BAM records.
/// umi_regex
///     Extract UMI from read names of BAM records using regular expressions.
///     See `barcode_regex` for more details.
/// shift_left
///     Insertion site correction for the left end. default: 4.
/// shift_right
///     Insertion site correction for the right end. default: -5.
/// chunk_size
///     The size of data retained in memory when performing sorting. Larger chunk sizes
///     result in faster sorting and greater memory usage. default is 50000000.
#[pyfunction(
    is_paired = "true",
    barcode_tag = "None",
    barcode_regex = "None",
    umi_tag = "None",
    umi_regex = "None",
    shift_left = "4",
    shift_right = "-5",
    chunk_size = "50000000",
)]
#[pyo3(text_signature = "(bam_file, output_file, is_paired, barcode_tag, barcode_regex, umi_tag, umi_regex, shift_left, shift_right, chunk_size)")]
pub(crate) fn make_fragment_file(
    bam_file: &str,
    output_file: &str,
    is_paired: bool,
    barcode_tag: Option<[u8; 2]>,
    barcode_regex: Option<&str>,
    umi_tag: Option<[u8; 2]>,
    umi_regex: Option<&str>,
    shift_left: i64,
    shift_right: i64,
    chunk_size: usize,
)
{
    let tmp_dir = Builder::new().tempdir_in("./")
        .expect("failed to create tmperorary directory");

    if barcode_regex.is_some() && barcode_tag.is_some() {
        panic!("Can only set barcode_tag or barcode_regex but not both");
    }
    if umi_regex.is_some() && umi_tag.is_some() {
        panic!("Can only set umi_tag or umi_regex but not both");
    }
    let barcode = match barcode_tag {
        Some(tag) => BarcodeLocation::InData(Tag::try_from(tag).unwrap()),
        None => match barcode_regex {
            Some(regex) => BarcodeLocation::Regex(Regex::new(regex).unwrap()),
            None => BarcodeLocation::InReadName,
        }
    };
    let umi = match umi_tag {
        Some(tag) => Some(BarcodeLocation::InData(Tag::try_from(tag).unwrap())),
        None => match umi_regex {
            Some(regex) => Some(BarcodeLocation::Regex(Regex::new(regex).unwrap())),
            None => None,
        }
    };
 
    let mut reader = File::open(bam_file).map(bam::Reader::new)
        .expect(&format!("cannot open bam file: {}", bam_file));
    let header: Header = reader.read_header().unwrap().parse().unwrap();
    reader.read_reference_sequences().unwrap();

    let f = File::create(output_file)
        .expect(&format!("cannot create file: {}", output_file));
    let mut output: Box<dyn Write> = if output_file.ends_with(".gz") {
        Box::new(GzEncoder::new(BufWriter::new(f), Compression::default()))
    } else {
        Box::new(BufWriter::new(f))
    };

    let filtered_records = filter_bam(
        reader.lazy_records().map(|x| x.unwrap()), is_paired
    );
    group_bam_by_barcode(filtered_records, &barcode, umi.as_ref(), is_paired, tmp_dir.path().to_path_buf(), chunk_size)
        .into_fragments(&header)
        .for_each(|x| match x {
            BedN::Bed5(mut x_) => {
                // TODO: use checked_add_signed.
                x_.set_start((x_.start() as i64 + shift_left) as u64);
                x_.set_end((x_.end() as i64 + shift_right) as u64);
                writeln!(output, "{}", x_).unwrap();
            },
            BedN::Bed6(x_) => writeln!(output, "{}", x_).unwrap(),
        });
}
 

#[pyfunction]
pub(crate) fn import_fragments(
    output_file: &str,
    fragment_file: &str,
    gtf_file: &str,
    chrom_size: BTreeMap<&str, u64>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    white_list: Option<HashSet<String>>,
    chunk_size: usize,
    num_cpu: usize,
) -> PyResult<AnnData>
{
    let mut anndata = anndata::AnnData::new(output_file, 0, 0).unwrap();
    let promoters = qc::make_promoter_map(
        qc::read_tss(open_file(gtf_file))
    );

    let final_white_list = if fragment_is_sorted_by_name || min_num_fragment <= 0 {
        white_list
    } else {
        let mut barcode_count = qc::get_barcode_count(
            bed::io::Reader::new(
                open_file(fragment_file),
                Some("#".to_string()),
            ).into_records().map(Result::unwrap)
        );
        let list: HashSet<String> = barcode_count.drain().filter_map(|(k, v)|
            if v >= min_num_fragment { Some(k) } else { None }).collect();
        match white_list {
            None => Some(list),
            Some(x) => Some(list.intersection(&x).map(Clone::clone).collect()),
        }
    };

    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(||
        qc::import_fragments(
            &mut anndata,
            bed::io::Reader::new(
                open_file(fragment_file),
                Some("#".to_string())
            ).into_records().map(Result::unwrap),
            &promoters,
            &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
            final_white_list.as_ref(),
            min_num_fragment,
            min_tsse,
            fragment_is_sorted_by_name,
            chunk_size,
        ).unwrap()
    );
    Ok(AnnData::wrap(anndata))
} 

#[pyfunction]
pub(crate) fn mk_gene_matrix<'py>(
    py: Python<'py>,
    input: &PyAny,
    gff_file: &str,
    output_file: &str,
    chunk_size: usize,
    use_x: bool,
    id_type: &str,
) -> PyResult<AnnData>
{
    let transcripts = read_transcripts(BufReader::new(open_file(gff_file)));

    let result = if input.is_instance(AnnData::type_object(py))? {
        let data: AnnData = input.extract()?;
        if use_x {
            let x = data.0.inner().read_chrom_values().unwrap();
            create_gene_matrix(output_file, x, transcripts, id_type).unwrap()
        } else {
            let x = data.0.inner().read_insertions(chunk_size).unwrap();
            create_gene_matrix(output_file, x, transcripts, id_type).unwrap()
        }
    } else if input.is_instance(AnnDataSet::type_object(py))? {
        let data: AnnDataSet = input.extract()?;
        if use_x {
            let x = data.0.inner().read_chrom_values().unwrap();
            create_gene_matrix(output_file, x, transcripts, id_type).unwrap()
        } else {
            let x = data.0.inner().read_insertions(chunk_size).unwrap();
            create_gene_matrix(output_file, x, transcripts, id_type).unwrap()
        }
    } else {
        return Err(PyTypeError::new_err("expecting an AnnData or AnnDataSet object"));
    };
    Ok(AnnData::wrap(result))
}

#[pyfunction]
pub(crate) fn mk_tile_matrix(anndata: &AnnData, bin_size: u64, chunk_size: usize, num_cpu: usize) {
    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(||
        create_tile_matrix(anndata.0.inner().deref(), bin_size, chunk_size).unwrap()
    );
} 

#[derive(FromPyObject)]
pub(crate) enum PeakRep {
    #[pyo3(transparent, annotation = "str")]
    String(String),
    #[pyo3(transparent, annotation = "list[str]")]
    StringVec(Vec<String>), 
}

#[pyfunction]
pub(crate) fn mk_peak_matrix<'py>(
    py: Python<'py>,
    input: &PyAny,
    use_rep: PeakRep,
    peak_file: Option<&str>,
    output_file: &str,
) -> PyResult<AnnData>
{
    let peaks = match peak_file {
        None => match use_rep {
            PeakRep::String(str_rep) => {
                let df: Box<DataFrame> = if input.is_instance(AnnData::type_object(py))? {
                    let data: AnnData = input.extract()?;
                    let x = data.0.inner().get_uns().inner().get_mut(&str_rep).unwrap()
                        .read().unwrap().into_any().downcast().unwrap();
                    x
                } else if input.is_instance(AnnDataSet::type_object(py))? {
                    let data: AnnDataSet = input.extract()?;
                    let x = data.0.inner().get_uns().inner().get_mut(&str_rep).unwrap()
                        .read().unwrap().into_any().downcast().unwrap();
                    x
                } else {
                    panic!("expecting an AnnData or AnnDataSet object");
                };
                df[0].utf8().into_iter().flatten()
                    .map(|x| GenomicRange::from_str(x.unwrap()).unwrap()).collect()
            },
            PeakRep::StringVec(list_rep) => list_rep.into_iter()
                .map(|x| GenomicRange::from_str(&x).unwrap()).collect(),
        },
        Some(fl) => bed::io::Reader::new(open_file(fl), None).into_records()
            .map(|x| x.unwrap()).collect(),
    };
    let result = if input.is_instance(AnnData::type_object(py))? {
        let data: AnnData = input.extract()?;
        let x = data.0.inner().read_insertions(500).unwrap();
        create_peak_matrix(output_file, x, &peaks).unwrap()
    } else if input.is_instance(AnnDataSet::type_object(py))? {
        let data: AnnDataSet = input.extract()?;
        let x = data.0.inner().read_insertions(500).unwrap();
        create_peak_matrix(output_file, x, &peaks).unwrap()
    } else {
        panic!("expecting an AnnData or AnnDataSet object");
    };
    Ok(AnnData::wrap(result))
}