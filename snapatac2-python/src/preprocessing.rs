use crate::utils::*;

use anndata::Backend;
use anndata_hdf5::H5;
use std::path::PathBuf;
use std::{io::BufReader, str::FromStr, collections::BTreeMap, ops::Deref, collections::HashSet};
use pyo3::prelude::*;
use bed_utils::{bed, bed::GenomicRange};
use pyanndata::PyAnnData;
use anyhow::Result;

use snapatac2_core::{
    preprocessing::{Fragment, FlagStat, read_transcripts},
    preprocessing,
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
    barcode_tag: Option<&str>,
    barcode_regex: Option<&str>,
    umi_tag: Option<&str>,
    umi_regex: Option<&str>,
    shift_left: i64,
    shift_right: i64,
    mapq: Option<u8>,
    chunk_size: usize,
) -> PyFlagStat
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
    );
    PyFlagStat(stat)
}

#[pyfunction]
pub(crate) fn import_fragments(
    anndata: AnnDataLike,
    fragment_file: PathBuf,
    gtf_file: PathBuf,
    chrom_size: BTreeMap<&str, u64>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    low_memory: bool,
    white_list: Option<HashSet<String>>,
    chunk_size: usize,
    tempdir: Option<PathBuf>,
) -> Result<()>
{
    let promoters = preprocessing::make_promoter_map(preprocessing::read_tss(open_file(gtf_file)));
    let final_white_list = if fragment_is_sorted_by_name || low_memory || min_num_fragment <= 0 {
        white_list
    } else {
        let mut barcode_count = preprocessing::get_barcode_count(
            bed::io::Reader::new(
                open_file(&fragment_file),
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
    let chrom_sizes = chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect();
    let fragments = bed::io::Reader::new(open_file(&fragment_file), Some("#".to_string()))
        .into_records::<Fragment>().map(Result::unwrap);
    let sorted_fragments: Box<dyn Iterator<Item = Fragment>> = if !fragment_is_sorted_by_name && low_memory {
        Box::new(bed::sort_bed_by_key(fragments, |x| x.barcode.clone(), tempdir))
    } else {
        Box::new(fragments)
    };
    let is_sorted = if fragment_is_sorted_by_name || low_memory { true } else { false };

    macro_rules! run {
        ($data:expr) => {
            preprocessing::import_fragments(
                $data, sorted_fragments, &promoters, &chrom_sizes,
                final_white_list.as_ref(), min_num_fragment, min_tsse, is_sorted, chunk_size,
            )?
        };
    }

    crate::with_anndata!(&anndata, run);
    Ok(())
} 

#[pyfunction]
pub(crate) fn mk_tile_matrix(
    anndata: AnnDataLike, bin_size: usize, chunk_size: usize, out: Option<AnnDataLike>
) -> Result<()>
{
    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_tile_matrix($data, bin_size, chunk_size, Some($out_data))?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_tile_matrix($data, bin_size, chunk_size, None::<&PyAnnData>)?;
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
                        preprocessing::create_peak_matrix($data, peaks, chunk_size, Some($out_data))?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_peak_matrix($data, peaks, chunk_size, None::<&PyAnnData>)?;
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
    out: Option<AnnDataLike>,
) -> Result<()>
{
    let transcripts = read_transcripts(BufReader::new(open_file(gff_file)));
    macro_rules! run {
        ($data:expr) => {
            if let Some(out) = out {
                macro_rules! run2 {
                    ($out_data:expr) => {
                        preprocessing::create_gene_matrix($data, transcripts, id_type, chunk_size, Some($out_data))?
                    };
                }
                crate::with_anndata!(&out, run2);
            } else {
                preprocessing::create_gene_matrix($data, transcripts, id_type, chunk_size, None::<&PyAnnData>)?;
            }
        }
    }
    crate::with_anndata!(&anndata, run);
    Ok(())
}