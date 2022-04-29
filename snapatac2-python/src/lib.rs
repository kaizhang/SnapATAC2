mod export;
mod utils;
mod call_peaks;

use utils::*;
//pub mod extension;

use std::io::BufReader;
use pyo3::{
    prelude::*,
    PyResult, Python,
    type_object::PyTypeObject,
    exceptions::PyTypeError,
};
use bed_utils::{bed, bed::{GenomicRange, BEDLike}};
use std::collections::BTreeMap;
use std::ops::Deref;
use rayon::ThreadPoolBuilder;
use polars::prelude::{NamedFrom, DataFrame, Series};
use std::collections::HashSet;

use anndata_rs::anndata;
use pyanndata::{
    AnnData, AnnDataSet,
    read, read_mtx, read_csv, create_dataset, read_dataset
};

use snapatac2_core::{
    tile_matrix::create_tile_matrix,
    peak_matrix::create_peak_matrix,
    gene_score::create_gene_matrix,
    qc,
    utils::{gene::read_transcripts, ChromValuesReader},
};

#[pyfunction]
fn mk_gene_matrix<'py>(
    py: Python<'py>,
    input: &PyAny,
    gff_file: &str,
    output_file: &str,
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
            let x = data.0.inner().read_insertions().unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        }
    } else if input.is_instance(AnnDataSet::type_object(py))? {
        let data: AnnDataSet = input.extract()?;
        if use_x {
            let x = data.0.inner().read_chrom_values().unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        } else {
            let x = data.0.inner().read_insertions().unwrap();
            create_gene_matrix(output_file, x, transcripts).unwrap()
        }
    } else {
        return Err(PyTypeError::new_err("expecting an AnnData or AnnDataSet object"));
    };
    Ok(AnnData::wrap(result))
}

#[pyfunction]
fn mk_tile_matrix(anndata: &AnnData, bin_size: u64, num_cpu: usize) {
    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(||
        create_tile_matrix(anndata.0.inner().deref(), bin_size).unwrap()
    );
} 

#[pyfunction]
fn mk_peak_matrix(anndata: &AnnData, peak_file: &str, num_cpu: usize) -> PyResult<()>
{
    let anndata_guard = anndata.0.inner();
    let peaks = bed::io::Reader::new(open_file(peak_file), None).into_records()
        .map(|x| x.unwrap()).collect();
    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(||
        create_peak_matrix(anndata_guard.deref(), &peaks).unwrap()
    );
    let var_names = Series::new(
        "Peaks",
        peaks.regions.into_iter()
            .map(|x| format!("{}:{}-{}", x.chrom(), x.start(), x.end()))
            .collect::<Series>(),
    );
    anndata_guard.set_var(Some(&DataFrame::new(vec![var_names]).unwrap())).unwrap();
    Ok(())
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

    m.add_function(wrap_pyfunction!(simple_lin_reg, m)?)?;
    m.add_function(wrap_pyfunction!(jm_regress, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_bed, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_nearest_neighbors, m)?)?;

    Ok(())
}
