mod utils;
use utils::*;
//pub mod extension;

use std::io::BufReader;
use pyo3::{
    prelude::*,
    PyResult, Python,
};
use bed_utils::{bed, bed::{GenomicRange, BEDLike}};
use std::collections::BTreeMap;
use std::ops::Deref;
use rayon::ThreadPoolBuilder;
use nalgebra_sparse::CsrMatrix;
use polars::prelude::{NamedFrom, DataFrame, Series};

use anndata_rs::{
    anndata,
    iterator::IntoRowsIterator,
};
use pyanndata::{
    AnnData, AnnDataSet,
    read, read_mtx, read_csv, create_dataset, read_dataset
};

use snapatac2_core::{
    tile_matrix::create_tile_matrix,
    peak_matrix::create_peak_matrix,
    gene_score::create_gene_matrix,
    qc,
    utils::{gene::read_transcripts, Insertions, GenomeRegions, GenomeBaseIndex, InsertionIter},
};

#[pyfunction]
fn mk_gene_matrix(
    input: &AnnData,
    gff_file: &str,
    output_file: &str,
    use_x: bool,
    num_cpu: usize,
) -> PyResult<AnnData>
{
    let transcripts = read_transcripts(BufReader::new(open_file(gff_file)))
        .into_values().collect();
    let inner = input.0.inner();
    let anndata = ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(|| {
        let x_guard;
        let obsm_guard;
        let insertion_guard;
        let insertions: Box<dyn Iterator<Item = Vec<Insertions>>> = if use_x {
            x_guard = inner.get_x().inner();
            Box::new(InsertionIter {
                genome_index: GenomeRegions(
                    inner.get_var().read().unwrap()[0].utf8().unwrap().into_iter()
                        .map(|x| str_to_genomic_region(x.unwrap()).unwrap()).collect()
                ),
                iter: x_guard.into_csr_u32_iter(500),
            })
        } else {
            obsm_guard = inner.get_obsm().inner();
            insertion_guard = obsm_guard.get("insertion").unwrap().inner();
            Box::new(InsertionIter {
                iter: insertion_guard.downcast::<CsrMatrix<u8>>().into_row_iter(500),
                genome_index: GenomeBaseIndex::read_from_anndata(inner.deref()).unwrap(),
            })
        };
        create_gene_matrix(output_file, insertions, transcripts).unwrap()
    });
    Ok(AnnData::wrap(anndata))
}

#[pyfunction]
fn mk_tile_matrix(anndata: &AnnData,
                  bin_size: u64,
                  num_cpu: usize,
                  ) -> PyResult<()>
{
    ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap().install(||
        create_tile_matrix(anndata.0.inner().deref(), bin_size).unwrap()
    );
    Ok(())
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
    num_cpu: usize,
    ) -> PyResult<AnnData>
{
    let mut anndata = anndata::AnnData::new(output_file, 0, 0).unwrap();
    let promoters = qc::make_promoter_map(
        qc::read_tss(open_file(gtf_file))
    );

    let white_list = if fragment_is_sorted_by_name {
        None
    } else if min_num_fragment > 0 {
        let mut barcode_count = qc::get_barcode_count(
            bed::io::Reader::new(
                open_file(fragment_file),
                Some("#".to_string()),
            ).into_records().map(Result::unwrap)
        );
        Some(barcode_count.drain().filter_map(|(k, v)|
            if v >= min_num_fragment { Some(k) } else { None }).collect()
        )
    } else {
        None
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
            white_list.as_ref(),
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

    m.add_function(wrap_pyfunction!(simple_lin_reg, m)?)?;
    m.add_function(wrap_pyfunction!(jm_regress, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_bed, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_nearest_neighbors, m)?)?;

    Ok(())
}
