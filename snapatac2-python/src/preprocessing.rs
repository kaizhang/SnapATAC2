use crate::utils::*;

use anndata::Backend;
use anndata_hdf5::H5;
use snapatac2_core::preprocessing::count_data::TranscriptParserOptions;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::{str::FromStr, collections::BTreeMap, ops::Deref, collections::HashSet};
use pyo3::prelude::*;
use bed_utils::{bed, bed::GenomicRange};
use pyanndata::PyAnnData;
use anyhow::Result;

use snapatac2_core::{
    preprocessing::{Fragment, Contact, FlagStat, SnapData},
    preprocessing, utils,
};

#[pyclass]
pub(crate) struct PyFlagStat(FlagStat);

#[pymethods]
impl PyFlagStat {
    #[getter]
    fn num_reads(&self) -> u64 { self.0.read }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String { self.__repr__() }
}

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
    compression: Option<&str>,
    compression_level: Option<u32>,
    temp_dir: Option<PathBuf>,
) -> Result<PyFlagStat>
{
    fn parse_tag(tag: &str) -> [u8; 2] {
        let tag_b = tag.as_bytes();
        if tag_b.len() == 2 {
            [tag_b[0], tag_b[1]]
        } else {
            panic!("TAG name must contain exactly two characters");
        }
    }
    let stat = preprocessing::make_fragment_file(
        bam_file, output_file, is_paired,
        barcode_tag.map(|x| parse_tag(x)), barcode_regex,
        umi_tag.map(|x| parse_tag(x)), umi_regex,
        shift_left, shift_right, mapq, chunk_size,
        compression.map(|x| utils::Compression::from_str(x).unwrap()), compression_level, temp_dir,
    )?;
    Ok(PyFlagStat(stat))
}

#[pyfunction]
pub(crate) fn import_fragments(
    anndata: AnnDataLike,
    fragment_file: PathBuf,
    chrom_size: BTreeMap<&str, u64>,
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
        .into_records::<Fragment>().map(|x| {
            let mut f = x.unwrap();
            shift_fragment(&mut f, shift_left, shift_right);
            f
    });
    let sorted_fragments: Box<dyn Iterator<Item = Fragment>> = if !fragment_is_sorted_by_name {
        Box::new(bed::sort_bed_by_key(fragments, |x| x.barcode.clone(), tempdir))
    } else {
        Box::new(fragments)
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
    chrom_size: BTreeMap<&str, u64>,
    fragment_is_sorted_by_name: bool,
    chunk_size: usize,
    tempdir: Option<PathBuf>,
) -> Result<()>
{
    let chrom_sizes = chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect();

    let contacts = BufReader::new(utils::open_file_for_read(&contact_file)).lines()
        .map(|x| Contact::from_str(&x.unwrap()).unwrap());
    let sorted_contacts: Box<dyn Iterator<Item = Contact>> = if !fragment_is_sorted_by_name {
        let tmp = if let Some(dir) = tempdir {
            tempfile::Builder::new().tempdir_in(dir)
        } else {
            tempfile::tempdir()
        }.expect("failed to create tmperorary directory");
        Box::new(extsort::ExternalSorter::new()
            .with_segment_size(50000000)
            .with_sort_dir(tmp.path().to_path_buf())
            .with_parallel_sort()
            .sort_by_key(contacts, |x| x.barcode.clone()).unwrap()
        )
    } else {
        Box::new(contacts)
    };

    macro_rules! run {
        ($data:expr) => {
            preprocessing::import_contacts($data, sorted_contacts, &chrom_sizes, chunk_size)?
        };
    }

    crate::with_anndata!(&anndata, run);
    Ok(())
} 



#[pyfunction]
pub(crate) fn mk_tile_matrix(
    anndata: AnnDataLike, bin_size: usize, chunk_size: usize, 
    exclude_chroms: Option<Vec<&str>>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<AnnDataLike>
) -> Result<()>
{
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
    peaks: &PyAny,
    chunk_size: usize,
    use_x: bool,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<AnnDataLike>,
) -> Result<()>
{
    let peaks = peaks.iter()?
        .map(|x| GenomicRange::from_str(x.unwrap().extract().unwrap()).unwrap());

    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_peak_matrix($data, peaks, chunk_size,
                            min_fragment_size, max_fragment_size, Some($out_data), use_x)?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_peak_matrix($data, peaks, chunk_size,
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
    transcript_name_key: String,
    transcript_id_key: String,
    gene_name_key: String,
    gene_id_key: String,
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
    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_gene_matrix($data, transcripts, id_type, chunk_size,
                            min_fragment_size, max_fragment_size, Some($out_data), use_x)?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_gene_matrix($data, transcripts, id_type, chunk_size,
                    min_fragment_size, max_fragment_size, None::<&PyAnnData>, use_x)?;
            }
        }
    }
    crate::with_anndata!(&anndata, run);
    Ok(())
}

/// QC metrics

#[pyfunction]
pub(crate) fn tss_enrichment(
    anndata: AnnDataLike,
    gtf_file: PathBuf,
) -> Result<Vec<f64>>
{
    let promoters = preprocessing::make_promoter_map(preprocessing::read_tss(utils::open_file_for_read(gtf_file)));

    macro_rules! run {
        ($data:expr) => {
            $data.tss_enrichment(&promoters)
        }
    }
    crate::with_anndata!(&anndata, run)
}

#[pyfunction]
pub(crate) fn add_frip(
    anndata: AnnDataLike,
    regions: BTreeMap<String, Vec<&str>>,
) -> Result<BTreeMap<String, Vec<f64>>>
{
    let trees: Vec<_> = regions.values().map(|x|
        x.into_iter().map(|y| (GenomicRange::from_str(y).unwrap(), ())).collect()
    ).collect();

    macro_rules! run {
        ($data:expr) => {
            $data.frip(&trees)
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