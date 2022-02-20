#[cfg(feature = "blosc")]
use hdf5::filters::blosc_set_nthreads;
use std::env;
use hdf5::{File, H5Type, Result, Extent};
use ndarray::{arr1, s};
use snapatac2_core::utils::hdf5::*;
use snapatac2_core::utils::anndata::*;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut anndata: Ann<u32> = Ann::read(&args[1])?;
    //println!("{:?}", anndata.obs_names());
    /*
    for x in anndata.ann_row_iter().take(10) {
        println!("{:?}", x);
    }
    */

    Ok(())
}