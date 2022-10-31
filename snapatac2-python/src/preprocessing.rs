use crate::utils::*;

use std::{io::BufReader, str::FromStr, collections::BTreeMap, ops::Deref, collections::HashSet};
use pyo3::{PyTypeInfo, prelude::*, PyResult, Python, exceptions::PyTypeError};
use bed_utils::{bed, bed::GenomicRange};
use rayon::ThreadPoolBuilder;
use polars::prelude::DataFrame;
use anndata_rs::anndata;
use pyanndata::{AnnData, AnnDataSet};

use snapatac2_core::{
    preprocessing::{Fragment, FlagStat, create_gene_matrix, create_peak_matrix, create_tile_matrix},
    preprocessing,
    utils::{gene::read_transcripts, ChromValuesReader},
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
pub(crate) fn import_fragments(
    output_file: &str,
    fragment_file: &str,
    gtf_file: &str,
    chrom_size: BTreeMap<&str, u64>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    low_memory: bool,
    white_list: Option<HashSet<String>>,
    chunk_size: usize,
    num_cpu: usize,
) -> PyResult<AnnData>
{
    let mut anndata = anndata::AnnData::new(output_file, 0, 0).unwrap();
    let promoters = preprocessing::make_promoter_map(
        preprocessing::read_tss(open_file(gtf_file))
    );

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

    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(|| {
        let fragments = bed::io::Reader::new(open_file(fragment_file), Some("#".to_string()))
            .into_records::<Fragment>().map(Result::unwrap);
        if !fragment_is_sorted_by_name && low_memory {
            preprocessing::import_fragments(
                &mut anndata, bed::sort_bed_by_key(fragments, |x| x.barcode.clone()),
                &promoters, &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
                final_white_list.as_ref(), min_num_fragment, min_tsse, true, chunk_size,
            ).unwrap();
        } else {
            preprocessing::import_fragments(
                &mut anndata, fragments,
                &promoters, &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
                final_white_list.as_ref(), min_num_fragment, min_tsse, fragment_is_sorted_by_name, chunk_size,
            ).unwrap();
        }
    });
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
            let x = data.0.inner().raw_count_iter(chunk_size).unwrap();
            create_gene_matrix(output_file, x, transcripts, id_type).unwrap()
        }
    } else if input.is_instance(AnnDataSet::type_object(py))? {
        let data: AnnDataSet = input.extract()?;
        if use_x {
            let x = data.0.inner().read_chrom_values().unwrap();
            create_gene_matrix(output_file, x, transcripts, id_type).unwrap()
        } else {
            let x = data.0.inner().raw_count_iter(chunk_size).unwrap();
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
        let x = data.0.inner().raw_count_iter(500).unwrap();
        create_peak_matrix(output_file, x, &peaks).unwrap()
    } else if input.is_instance(AnnDataSet::type_object(py))? {
        let data: AnnDataSet = input.extract()?;
        let x = data.0.inner().raw_count_iter(500).unwrap();
        create_peak_matrix(output_file, x, &peaks).unwrap()
    } else {
        panic!("expecting an AnnData or AnnDataSet object");
    };
    Ok(AnnData::wrap(result))
}