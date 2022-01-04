use std::env;
use hdf5::{File, H5Type, Result, Extent};
use snapatac2_core::utils::hdf5::*;
use flate2::read::GzDecoder;
use snapatac2_core::qc::*;
use snapatac2_core::create_count_matrix;
use bed_utils::bed::{GenomicRange};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let frag = GzDecoder::new(std::fs::File::open(&args[1]).expect("xx"));
    let file = File::create(&args[2])?;
    let chr_size =
        [ GenomicRange::new("chr1", 0, 248956422)
        , GenomicRange::new("chr2", 0, 242193529)
        , GenomicRange::new("chr3", 0, 198295559) ];
    create_count_matrix(file, read_fragments(frag), &chr_size.into_iter().collect(), Some(5000))
}