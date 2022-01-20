use pyo3::prelude::*;

use bed_utils::bed::GenomicRange;
use std::fs::File;
use std::collections::BTreeMap;
use flate2::read::GzDecoder;
use hdf5;

use snapatac2_core::{create_tile_matrix, qc};

#[pyfunction]
fn mk_tile_matrix(output_file: &str,
                  fragment_file: &str,
                  gtf_file: &str,
                  chrom_size: BTreeMap<&str, u64>,
                  bin_size: u64,
                  min_num_fragment: u64,
                  min_tsse: f64,
                  num_cpu: usize,
                  ) -> PyResult<()> {
    let file = hdf5::File::create(output_file).unwrap();
    let fragment_file_reader = File::open(fragment_file).unwrap();
    let gtf_file_reader = File::open(gtf_file).unwrap();
    let promoters = if is_gzipped(gtf_file) {
        qc::make_promoter_map(qc::read_tss(GzDecoder::new(gtf_file_reader)))
    } else {
        qc::make_promoter_map(qc::read_tss(gtf_file_reader))
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_cpu).build().unwrap();
    let result = pool.install(|| {
        if is_gzipped(fragment_file) {
            create_tile_matrix(
            file,
            qc::read_fragments(GzDecoder::new(fragment_file_reader)),
            &promoters,
            &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
            bin_size,
            min_num_fragment,
            min_tsse,
            ).unwrap()
        } else {
            create_tile_matrix(
            file,
            qc::read_fragments(fragment_file_reader),
            &promoters,
            &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
            bin_size,
            min_num_fragment,
            min_tsse,
            ).unwrap()
        }
    });
    Ok(result)
} 

fn is_gzipped(file: &str) -> bool {
    GzDecoder::new(File::open(file).unwrap()).header().is_some()
}

#[pymodule]
fn snapatac2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mk_tile_matrix, m)?)?;

    Ok(())
}