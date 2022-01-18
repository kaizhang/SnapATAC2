use pyo3::prelude::*;

use bed_utils::bed::{GenomicRange, BEDLike, tree::BedTree, io::Reader};
use std::fs::File;
use std::collections::HashMap;
use flate2::read::GzDecoder;
use hdf5;

use snapatac2_core::{create_tile_matrix, qc};

#[pyfunction]
fn mk_tile_matrix(output_file: &str,
                  fragment_file: &str,
                  gtf_file: &str,
                  chrom_size: HashMap<&str, u64>,
                  bin_size: u64,
                  min_num_fragment: u64,
                  ) -> PyResult<()> {
    let file = hdf5::File::create(output_file).unwrap();
    let frag = GzDecoder::new(File::open(fragment_file)
        .expect("Unable to open fragment file"));
    let gtf = GzDecoder::new(File::open(gtf_file).expect("Fail to open gtf file"));
    Ok(create_tile_matrix(
        file,
        qc::read_fragments(frag),
        &qc::make_promoter_map(qc::read_tss(gtf)),
        &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
        bin_size,
        min_num_fragment,
    ).unwrap())
} 

#[pymodule]
fn snapatac2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mk_tile_matrix, m)?)?;

    Ok(())
}