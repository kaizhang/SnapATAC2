mod export;
mod utils;
mod call_peaks;
mod preprocessing;
mod embedding;
mod network;
mod motif;
mod knn;

use pyo3::{prelude::*, PyResult, Python};
use pyanndata;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[pymodule]
fn _snapatac2(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    // AnnData related functions
    m.add_class::<pyanndata::AnnData>().unwrap();
    m.add_class::<pyanndata::AnnDataSet>().unwrap();
    m.add_class::<pyanndata::PyArrayElem>().unwrap();
    m.add_function(wrap_pyfunction!(pyanndata::read, m)?)?;
    m.add_function(wrap_pyfunction!(pyanndata::read_mtx, m)?)?;
    //m.add_function(wrap_pyfunction!(pyanndata::read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(pyanndata::read_dataset, m)?)?;

    // Motif analysis related functions
    m.add_class::<motif::PyDNAMotif>().unwrap();
    m.add_class::<motif::PyDNAMotifScanner>().unwrap();
    m.add_class::<motif::PyDNAMotifTest>().unwrap();
    m.add_function(wrap_pyfunction!(motif::read_motifs, m)?)?;
 
    // Preprocessing related functions
    m.add_class::<preprocessing::PyFlagStat>().unwrap();
    m.add_function(wrap_pyfunction!(preprocessing::make_fragment_file, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::import_fragments, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::import_contacts, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::mk_tile_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::mk_gene_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::mk_peak_matrix, m)?)?;

    m.add_function(wrap_pyfunction!(preprocessing::tss_enrichment, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::add_frip, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::fragment_size_distribution, m)?)?;

    m.add_function(wrap_pyfunction!(export::export_bed, m)?)?;
    m.add_function(wrap_pyfunction!(export::export_bigwig, m)?)?;

    m.add_function(wrap_pyfunction!(call_peaks::export_tags, m)?)?;
    m.add_function(wrap_pyfunction!(call_peaks::create_fwtrack_obj, m)?)?;
    m.add_function(wrap_pyfunction!(call_peaks::fetch_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(call_peaks::py_merge_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(call_peaks::find_reproducible_peaks, m)?)?;

    m.add_function(wrap_pyfunction!(knn::nearest_neighbour_graph, m)?)?;
    m.add_function(wrap_pyfunction!(knn::approximate_nearest_neighbour_graph, m)?)?;

    m.add_function(wrap_pyfunction!(network::link_region_to_gene, m)?)?;

    m.add_function(wrap_pyfunction!(utils::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(utils::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(utils::pearson, m)?)?;
    m.add_function(wrap_pyfunction!(utils::spearman, m)?)?;
    m.add_function(wrap_pyfunction!(utils::simple_lin_reg, m)?)?;
    m.add_function(wrap_pyfunction!(utils::jm_regress, m)?)?;
    m.add_function(wrap_pyfunction!(utils::read_regions, m)?)?;
    m.add_function(wrap_pyfunction!(utils::intersect_bed, m)?)?;
    m.add_function(wrap_pyfunction!(utils::kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::spectral_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::multi_spectral_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::spectral_embedding_nystrom, m)?)?;

    Ok(())
}