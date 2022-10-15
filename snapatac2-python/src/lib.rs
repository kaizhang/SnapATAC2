mod export;
mod utils;
mod call_peaks;
mod matrix;
mod network;
mod motif;

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

    m.add_class::<pyanndata::AnnData>().unwrap();
    m.add_class::<pyanndata::AnnDataSet>().unwrap();
    m.add_function(wrap_pyfunction!(pyanndata::read, m)?)?;
    m.add_function(wrap_pyfunction!(pyanndata::read_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(pyanndata::read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(pyanndata::create_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(pyanndata::read_dataset, m)?)?;

    m.add_class::<motif::PyDNAMotif>().unwrap();
    m.add_class::<motif::PyDNAMotifScanner>().unwrap();
    m.add_class::<motif::PyDNAMotifTest>().unwrap();
    m.add_function(wrap_pyfunction!(motif::read_motifs, m)?)?;
 
    m.add_class::<matrix::PyFlagStat>().unwrap();
    m.add_function(wrap_pyfunction!(matrix::make_fragment_file, m)?)?;
    m.add_function(wrap_pyfunction!(matrix::import_fragments, m)?)?;
    m.add_function(wrap_pyfunction!(matrix::mk_tile_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(matrix::mk_gene_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(matrix::mk_peak_matrix, m)?)?;

    m.add_function(wrap_pyfunction!(export::export_bed, m)?)?;
    m.add_function(wrap_pyfunction!(export::export_bigwig, m)?)?;
    m.add_function(wrap_pyfunction!(call_peaks::call_peaks, m)?)?;

    m.add_class::<network::NodeData>().unwrap();
    m.add_class::<network::LinkData>().unwrap();
    m.add_function(wrap_pyfunction!(network::link_region_to_gene, m)?)?;

    m.add_function(wrap_pyfunction!(utils::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(utils::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(utils::simple_lin_reg, m)?)?;
    m.add_function(wrap_pyfunction!(utils::jm_regress, m)?)?;
    m.add_function(wrap_pyfunction!(utils::intersect_bed, m)?)?;
    m.add_function(wrap_pyfunction!(utils::kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(utils::approximate_nearest_neighbors, m)?)?;

    Ok(())
}
