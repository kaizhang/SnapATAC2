use crate::utils::*;

use std::{io::BufReader, str::FromStr, collections::BTreeMap, ops::Deref, collections::HashSet};
use pyo3::{prelude::*, Python};
use bed_utils::{bed, bed::GenomicRange};
use bed_utils::bed::tree::GenomeRegions;
use anndata_rs::{anndata, AnnDataOp, AnnDataIterator};
use pyanndata::{AnnData, PyAnnData};
use anyhow::Result;

use snapatac2_core::{
    preprocessing::{
        Transcript, Fragment, FlagStat, ReadGenomeCoverage, create_gene_matrix,
        create_peak_matrix, create_tile_matrix, read_transcripts,
    },
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
    bam_file: &str,
    output_file: &str,
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
pub(crate) fn import_fragments<'py>(
    py: Python<'py>,
    output_file: Option<&str>,
    fragment_file: &str,
    gtf_file: &str,
    chrom_size: BTreeMap<&str, u64>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    low_memory: bool,
    white_list: Option<HashSet<String>>,
    chunk_size: usize,
) -> Result<PyObject>
{
    let promoters = preprocessing::make_promoter_map(preprocessing::read_tss(open_file(gtf_file)));
    let final_white_list = if fragment_is_sorted_by_name || low_memory || min_num_fragment <= 0 {
        white_list
    } else {
        let mut barcode_count = preprocessing::get_barcode_count(
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
    let is_sorted = if !fragment_is_sorted_by_name && low_memory { true } else { false };
    let chrom_sizes = chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect();
    let fragments = bed::io::Reader::new(open_file(fragment_file), Some("#".to_string()))
        .into_records::<Fragment>().map(Result::unwrap);
    let sorted_fragments: Box<dyn Iterator<Item = Fragment>> = if !fragment_is_sorted_by_name && low_memory {
        Box::new(bed::sort_bed_by_key(fragments, |x| x.barcode.clone()))
    } else {
        Box::new(fragments)
    };

    match output_file {
        None => {
            let mut anndata = PyAnnData::new(py)?;
            preprocessing::import_fragments(
                &mut anndata, sorted_fragments, &promoters, &chrom_sizes,
                final_white_list.as_ref(), min_num_fragment, min_tsse, is_sorted, chunk_size,
            )?;
            Ok(anndata.to_object(py))
        },
        Some(fl) => {
            let mut anndata = anndata::AnnData::new(fl, 0, 0)?;
            preprocessing::import_fragments(
                &mut anndata, sorted_fragments, &promoters, &chrom_sizes,
                final_white_list.as_ref(), min_num_fragment, min_tsse, is_sorted, chunk_size,
            )?;
            Ok(AnnData::wrap(anndata).into_py(py))
        },
    }
} 

#[pyfunction]
pub(crate) fn mk_tile_matrix<'py>(
    py: Python<'py>, anndata: &'py PyAny, bin_size: u64, chunk_size: usize, num_cpu: usize
) -> PyResult<()>
{
    match extract_anndata(py, anndata)? {
        AnnDataObj::AnnData(x) => with_cpu(num_cpu, ||
            create_tile_matrix(x.inner().deref(), bin_size, chunk_size).unwrap()),
        AnnDataObj::PyAnnData(x) => create_tile_matrix(&x, bin_size, chunk_size).unwrap(),
        _ => todo!(),
    };
    Ok(())
}

#[pyfunction]
pub(crate) fn mk_peak_matrix<'py>(
    py: Python<'py>,
    input: &'py PyAny,
    peaks_str: &'py PyAny,
    output_file: Option<&str>,
    chunk_size: usize,
) -> PyResult<PyObject>
{
    fn action<A>(in_data: AnnDataObj, peaks: &GenomeRegions<GenomicRange>, out_data: &A, chunk_size: usize) -> Result<()>
    where
        A: AnnDataOp + AnnDataIterator,
    {
        match in_data {
            AnnDataObj::AnnData(data) => create_peak_matrix(
                out_data, data.inner().raw_count_iter(chunk_size)?.map(|x| x.0), &peaks),
            AnnDataObj::AnnDataSet(data) => create_peak_matrix(
                out_data, data.inner().raw_count_iter(chunk_size)?.map(|x| x.0), &peaks),
            AnnDataObj::PyAnnData(data) => create_peak_matrix(
                out_data, data.raw_count_iter(chunk_size)?.map(|x| x.0), &peaks),
        }
    }

    let peaks: Result<GenomeRegions<GenomicRange>> = peaks_str.iter()?
        .map(|x| Ok(GenomicRange::from_str(x?.extract()?).unwrap())).collect();
    let in_anndata = extract_anndata(py, input)?;
    if let Some(fl) = output_file {
        let anndata = anndata::AnnData::new(fl, 0, 0)?;
        action(in_anndata, &peaks?, &anndata, chunk_size)?;
        Ok(AnnData::wrap(anndata).into_py(py))
    } else {
        let anndata = PyAnnData::new(py)?;
        action(in_anndata, &peaks?, &anndata, chunk_size)?;
        Ok(anndata.to_object(py))
    }
}

#[pyfunction]
pub(crate) fn mk_gene_matrix<'py>(
    py: Python<'py>,
    input: &PyAny,
    gff_file: &str,
    output_file: Option<&str>,
    chunk_size: usize,
    use_x: bool,
    id_type: &str,
) -> PyResult<PyObject>
{
    fn action<A>(in_data: AnnDataObj, transcripts: Vec<Transcript>, out_data: &A,
        chunk_size: usize, use_x: bool, id_type: &str) -> Result<()>
    where
        A: AnnDataOp + AnnDataIterator,
    {
        match in_data {
            AnnDataObj::AnnData(data) => if use_x {
                create_gene_matrix(out_data, data.inner().read_chrom_values::<u32>(chunk_size)?.map(|x| x.0), transcripts, id_type)
            } else {
                create_gene_matrix(out_data, data.inner().raw_count_iter(chunk_size)?.map(|x| x.0), transcripts, id_type)
            },
            AnnDataObj::AnnDataSet(data) => if use_x {
                create_gene_matrix(out_data, data.inner().read_chrom_values::<u32>(chunk_size)?.map(|x| x.0), transcripts, id_type)
            } else {
                create_gene_matrix(out_data, data.inner().raw_count_iter(chunk_size)?.map(|x| x.0), transcripts, id_type)
            },
            AnnDataObj::PyAnnData(data) => if use_x {
                create_gene_matrix(out_data, data.read_chrom_values::<u32>(chunk_size)?.map(|x| x.0), transcripts, id_type)
            } else {
                create_gene_matrix(out_data, data.raw_count_iter(chunk_size)?.map(|x| x.0), transcripts, id_type)
            },
        }
    }
 
    let transcripts = read_transcripts(BufReader::new(open_file(gff_file)));
    let in_anndata = extract_anndata(py, input)?;
    if let Some(fl) = output_file {
        let anndata = anndata::AnnData::new(fl, 0, 0)?;
        action(in_anndata, transcripts, &anndata, chunk_size, use_x, id_type)?;
        Ok(AnnData::wrap(anndata).into_py(py))
    } else {
        let anndata = PyAnnData::new(py)?;
        action(in_anndata, transcripts, &anndata, chunk_size, use_x, id_type)?;
        Ok(anndata.to_object(py))
    }
}