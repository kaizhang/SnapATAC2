mod export;
mod utils;
mod call_peaks;

use utils::*;

use std::io::BufReader;
use pyo3::{
    prelude::*,
    PyResult, Python,
    type_object::PyTypeObject,
    exceptions::PyTypeError,
};
use bed_utils::{bed, bed::GenomicRange};
use std::collections::BTreeMap;
use std::ops::Deref;
use rayon::ThreadPoolBuilder;
use polars::prelude::DataFrame;
use std::collections::HashSet;

use anndata_rs::anndata;
use pyanndata::{
    AnnData, AnnDataSet,
    read, read_mtx, read_csv, create_dataset, read_dataset
};

use snapatac2_core::{
    matrix::{create_tile_matrix, create_peak_matrix, create_gene_matrix},
    qc,
    utils::{gene::read_transcripts, ChromValuesReader},
};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[pyfunction]
fn mk_gene_matrix<'py>(
    py: Python<'py>,
    input: &PyAny,
    gff_file: &str,
    output_file: &str,
    chunk_size: usize,
    use_x: bool,
) -> PyResult<AnnData>
{
    let transcripts = read_transcripts(BufReader::new(open_file(gff_file)))
        .into_values().collect();

    let result = if input.is_instance(AnnData::type_object(py))? {
        let data: AnnData = input.extract()?;
        if use_x {
            let x = data.0.inner().read_chrom_values().unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        } else {
            let x = data.0.inner().read_insertions(chunk_size).unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        }
    } else if input.is_instance(AnnDataSet::type_object(py))? {
        let data: AnnDataSet = input.extract()?;
        if use_x {
            let x = data.0.inner().read_chrom_values().unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        } else {
            let x = data.0.inner().read_insertions(chunk_size).unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        }
    } else {
        return Err(PyTypeError::new_err("expecting an AnnData or AnnDataSet object"));
    };
    Ok(AnnData::wrap(result))
}

#[pyfunction]
fn mk_tile_matrix(anndata: &AnnData, bin_size: u64, chunk_size: usize, num_cpu: usize) {
    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(||
        create_tile_matrix(anndata.0.inner().deref(), bin_size, chunk_size).unwrap()
    );
} 

#[pyfunction]
fn mk_peak_matrix<'py>(
    py: Python<'py>,
    input: &PyAny,
    use_rep: &str,
    peak_file: Option<&str>,
    output_file: &str,
) -> PyResult<AnnData>
{
    let peaks = match peak_file {
        None => {
            let df: Box<DataFrame> = if input.is_instance(AnnData::type_object(py))? {
                let data: AnnData = input.extract()?;
                let x = data.0.inner().get_uns().inner().get_mut(use_rep).unwrap()
                    .read().unwrap().into_any().downcast().unwrap();
                x
            } else if input.is_instance(AnnDataSet::type_object(py))? {
                let data: AnnDataSet = input.extract()?;
                let x = data.0.inner().get_uns().inner().get_mut(use_rep).unwrap()
                    .read().unwrap().into_any().downcast().unwrap();
                x
            } else {
                panic!("expecting an AnnData or AnnDataSet object");
            };
            df[0].utf8().into_iter().flatten()
                .map(|x| str_to_genomic_region(x.unwrap()).unwrap()).collect()
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

#[pyfunction]
fn import_fragments(
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

#[pymodule]
fn _snapatac2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnData>().unwrap();
    m.add_class::<AnnDataSet>().unwrap();

    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(create_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
 
    m.add_function(wrap_pyfunction!(import_fragments, m)?)?;
    m.add_function(wrap_pyfunction!(mk_tile_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mk_gene_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mk_peak_matrix, m)?)?;

    m.add_function(wrap_pyfunction!(export::export_bed, m)?)?;
    m.add_function(wrap_pyfunction!(call_peaks::call_peaks, m)?)?;

    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(simple_lin_reg, m)?)?;
    m.add_function(wrap_pyfunction!(jm_regress, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_bed, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_nearest_neighbors, m)?)?;

    Ok(())
}
