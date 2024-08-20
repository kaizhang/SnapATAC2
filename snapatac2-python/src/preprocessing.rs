use crate::utils::*;

use itertools::Itertools;
use pyo3::{prelude::*, pybacked::PyBackedStr};
use anndata::Backend;
use anndata_hdf5::H5;
use snapatac2_core::preprocessing::count_data::TranscriptParserOptions;
use snapatac2_core::preprocessing::count_data::CountingStrategy;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::{str::FromStr, collections::BTreeMap, ops::Deref, collections::HashSet};
use bed_utils::{bed, bed::GenomicRange};
use bed_utils::extsort::ExternalSorterBuilder;
use pyanndata::PyAnnData;
use anyhow::Result;

use snapatac2_core::{
    preprocessing::{Fragment, Contact, SnapData},
    preprocessing, utils,
};

#[pyfunction]
pub(crate) fn make_fragment_file(
    bam_file: PathBuf,
    output_file: PathBuf,
    is_paired: bool,
    shift_left: i64,
    shift_right: i64,
    chunk_size: usize,
    barcode_tag: Option<&str>,
    barcode_regex: Option<&str>,
    umi_tag: Option<&str>,
    umi_regex: Option<&str>,
    mapq: Option<u8>,
    mitochondrial_dna: Option<Vec<String>>,
    source: Option<&str>,
    compression: Option<&str>,
    compression_level: Option<u32>,
    temp_dir: Option<PathBuf>,
) -> Result<HashMap<String, f64>>
{
    fn parse_tag(tag: &str) -> [u8; 2] {
        let tag_b = tag.as_bytes();
        if tag_b.len() == 2 {
            [tag_b[0], tag_b[1]]
        } else {
            panic!("TAG name must contain exactly two characters");
        }
    }
    let (bam_qc, frag_qc) = preprocessing::make_fragment_file(
        bam_file, output_file, is_paired,
        barcode_tag.map(|x| parse_tag(x)), barcode_regex,
        umi_tag.map(|x| parse_tag(x)), umi_regex,
        shift_left, shift_right, mapq, chunk_size, source, mitochondrial_dna.map(|x| x.into_iter().collect()),
        compression.map(|x| utils::Compression::from_str(x).unwrap()), compression_level, temp_dir,
    )?;
    Ok(bam_qc.report().into_iter().chain(frag_qc.report()).collect())
}

#[pyfunction]
pub(crate) fn import_fragments(
    anndata: AnnDataLike,
    fragment_file: PathBuf,
    chrom_size: BTreeMap<String, u64>,
    mitochondrial_dna: Vec<String>,
    min_num_fragment: u64,
    fragment_is_sorted_by_name: bool,
    shift_left: i64,
    shift_right: i64,
    chunk_size: usize,
    white_list: Option<HashSet<String>>,
    tempdir: Option<PathBuf>,
) -> Result<()>
{
    let mitochondrial_dna: HashSet<String> = mitochondrial_dna.into_iter().collect();
    let final_white_list = if fragment_is_sorted_by_name || min_num_fragment <= 0 {
        white_list
    } else {
        let mut barcode_count = preprocessing::get_barcode_count(
            bed::io::Reader::new(
                utils::open_file_for_read(&fragment_file),
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
    let chrom_sizes = chrom_size.into_iter().collect();
    let fragments = bed::io::Reader::new(utils::open_file_for_read(&fragment_file), Some("#".to_string()))
        .into_records::<Fragment>().map(|x| x.map(|mut f| {
            shift_fragment(&mut f, shift_left, shift_right);
            f
    }));
    let sorted_fragments: Box<dyn Iterator<Item = Fragment>> = if !fragment_is_sorted_by_name {
        let mut sorter = ExternalSorterBuilder::new()
            .with_chunk_size(50000000)
            .with_compression(2);
        if let Some(tmp) = tempdir {
            sorter = sorter.with_tmp_dir(tmp);
        }
        Box::new(sorter
            .build().unwrap()
            .sort_by(fragments, |a, b| a.barcode.cmp(&b.barcode))
            .unwrap()
            .map(Result::unwrap)
        )
    } else {
        Box::new(fragments.map(Result::unwrap))
    };

    macro_rules! run {
        ($data:expr) => {
            preprocessing::import_fragments(
                $data, sorted_fragments, &mitochondrial_dna, &chrom_sizes,
                final_white_list.as_ref(), min_num_fragment, chunk_size,
            )?
        };
    }

    crate::with_anndata!(&anndata, run);
    Ok(())
} 

fn shift_fragment(fragment: &mut Fragment, shift_left: i64, shift_right: i64) {
    if shift_left != 0 {
        fragment.start = fragment.start.saturating_add_signed(shift_left);
        if fragment.strand.is_some() {
            fragment.end = fragment.end.saturating_add_signed(shift_left);
        }
    }
    if shift_right != 0 && fragment.strand.is_none() {
        fragment.end = fragment.end.saturating_add_signed(shift_right);
    }
}

#[pyfunction]
pub(crate) fn import_contacts(
    anndata: AnnDataLike,
    contact_file: PathBuf,
    chrom_size: BTreeMap<String, u64>,
    fragment_is_sorted_by_name: bool,
    chunk_size: usize,
    tempdir: Option<PathBuf>,
) -> Result<()>
{
    let chrom_sizes = chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect();

    let contacts = BufReader::new(utils::open_file_for_read(&contact_file)).lines()
        .map(|r| r.map(|x| Contact::from_str(&x).unwrap()));
    let sorted_contacts: Box<dyn Iterator<Item = Contact>> = if !fragment_is_sorted_by_name {
        let mut sorter = ExternalSorterBuilder::new()
            .with_chunk_size(50000000)
            .with_compression(2);
        if let Some(tmp) = tempdir {
            sorter = sorter.with_tmp_dir(tmp);
        }
        Box::new(sorter
            .build().unwrap()
            .sort_by(contacts, |a, b| a.barcode.cmp(&b.barcode))
            .unwrap()
            .map(Result::unwrap)
        )
    } else {
        Box::new(contacts.map(Result::unwrap))
    };

    macro_rules! run {
        ($data:expr) => {
            preprocessing::import_contacts($data, sorted_contacts, &chrom_sizes, chunk_size)?
        };
    }

    crate::with_anndata!(&anndata, run);
    Ok(())
} 


fn parse_strategy(strategy: &str) -> CountingStrategy {
    match strategy {
        "insertion" => CountingStrategy::Insertion,
        "fragment" => CountingStrategy::Fragment,
        "paired-insertion" => CountingStrategy::PIC,
        _ => panic!("Counting strategy must be one of 'insertion', 'fragment', or 'paired-insertion'"),
    }
}

#[pyfunction]
pub(crate) fn mk_tile_matrix(
    anndata: AnnDataLike,
    bin_size: usize,
    chunk_size: usize,
    strategy: &str,
    exclude_chroms: Option<Vec<PyBackedStr>>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<AnnDataLike>
) -> Result<()>
{
    let exclude_chroms = exclude_chroms.as_ref()
        .map(|s| s.iter().map(|x| x.as_ref()).collect::<Vec<_>>());
    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_tile_matrix(
                            $data,
                            bin_size,
                            chunk_size,
                            exclude_chroms.as_ref().map(|x| x.as_slice()),
                            min_fragment_size,
                            max_fragment_size,
                            parse_strategy(strategy),
                            Some($out_data)
                        )?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_tile_matrix(
                    $data,
                    bin_size,
                    chunk_size,
                    exclude_chroms.as_ref().map(|x| x.as_slice()),
                    min_fragment_size,
                    max_fragment_size,
                    parse_strategy(strategy),
                    None::<&PyAnnData>
                )?;
            }
        };
    }

    crate::with_anndata!(&anndata, run);
    Ok(())
}

#[pyfunction]
pub(crate) fn mk_peak_matrix(
    anndata: AnnDataLike,
    peaks: Bound<'_, PyAny>,
    chunk_size: usize,
    use_x: bool,
    strategy: &str,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<AnnDataLike>,
) -> Result<()>
{
    let peaks = peaks.iter()?
        .map(|x| GenomicRange::from_str(x.unwrap().extract().unwrap()).unwrap());
    let strategy = parse_strategy(strategy);

    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_peak_matrix($data, peaks, chunk_size, strategy,
                            min_fragment_size, max_fragment_size, Some($out_data), use_x)?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_peak_matrix($data, peaks, chunk_size, strategy,
                    min_fragment_size, max_fragment_size, None::<&PyAnnData>, use_x)?;
            }
        }
    }
    crate::with_anndata!(&anndata, run);
    Ok(())
}

#[pyfunction]
pub(crate) fn mk_gene_matrix(
    anndata: AnnDataLike,
    gff_file: PathBuf,
    chunk_size: usize,
    use_x: bool,
    id_type: &str,
    upstream: u64,
    downstream: u64,
    include_gene_body: bool,
    transcript_name_key: String,
    transcript_id_key: String,
    gene_name_key: String,
    gene_id_key: String,
    strategy: &str,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<AnnDataLike>,
) -> Result<()>
{
    let options = TranscriptParserOptions {
        transcript_name_key,
        transcript_id_key,
        gene_name_key,
        gene_id_key,
    };
    let transcripts = read_transcripts(gff_file, &options);
    let strategy = parse_strategy(strategy);
    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_gene_matrix($data, transcripts, id_type,
                            upstream, downstream, include_gene_body, chunk_size, strategy,
                            min_fragment_size, max_fragment_size, Some($out_data), use_x)?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_gene_matrix($data, transcripts, id_type,
                    upstream, downstream, include_gene_body, chunk_size, strategy,
                    min_fragment_size, max_fragment_size, None::<&PyAnnData>, use_x)?;
            }
        }
    }
    crate::with_anndata!(&anndata, run);
    Ok(())
}

/// QC metrics

#[pyfunction]
pub(crate) fn tss_enrichment<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    gtf_file: PathBuf,
    exclude_chroms: Option<Vec<String>>,
) -> Result<HashMap<&'py str, PyObject>>
{
    let exclude_chroms = match exclude_chroms {
        Some(chrs) => chrs.into_iter().collect(),
        None => HashSet::new(), 
    };
    let tss = preprocessing::read_tss(utils::open_file_for_read(gtf_file))
        .unique()
        .filter(|(chr, _, _)| !exclude_chroms.contains(chr));
    let promoters = preprocessing::TssRegions::new(tss, 2000);

    macro_rules! run {
        ($data:expr) => {
            $data.tss_enrichment(&promoters)
        }
    }
    let (scores, tsse) = crate::with_anndata!(&anndata, run)?;
    let library_tsse = tsse.result();
    let mut result = HashMap::new();
    result.insert("tsse", scores.to_object(py));
    result.insert("library_tsse", library_tsse.0.to_object(py));
    result.insert("fraction_overlap_TSS", library_tsse.1.to_object(py));
    result.insert("TSS_profile", tsse.get_counts().to_object(py));
    Ok(result)
}

#[pyfunction]
pub(crate) fn add_frip(
    anndata: AnnDataLike,
    regions: BTreeMap<String, Vec<String>>,
    normalized: bool,
    count_as_insertion: bool,
) -> Result<BTreeMap<String, Vec<f64>>>
{
    let trees: Vec<_> = regions.values().map(|x|
        x.into_iter().map(|y| (GenomicRange::from_str(y).unwrap(), ())).collect()
    ).collect();

    macro_rules! run {
        ($data:expr) => {
            $data.frip(&trees, normalized, count_as_insertion)
        }
    }

    let frip = crate::with_anndata!(&anndata, run)?;
    Ok(
        regions.keys().zip(frip.columns())
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect()
    )
}

#[pyfunction]
pub(crate) fn fragment_size_distribution(
    anndata: AnnDataLike,
    max_recorded_size: usize,
) -> Result<Vec<usize>>
{
    macro_rules! run {
        ($data:expr) => {
            $data.fragment_size_distribution(max_recorded_size)
        }
    }

    crate::with_anndata!(&anndata, run)
}
