pub mod qc;
pub mod utils;
use utils::hdf5::*;
use utils::anndata::*;

use ndarray::{Array, arr1};
use std::path::Path;
use qc::{CellBarcode, read_fragments};
use flate2::read::GzDecoder;

use hdf5::{File, Error, Selection, H5Type, Result, Extent, Group};
use bed_utils::bed::{BED, BEDLike, GenomicRange, split_by_len, tree::GenomeRegions};
use itertools::{Itertools, GroupBy};

///
pub fn create_count_matrix<B, I>(
    file: File,
    fragments: GroupBy<CellBarcode, I, impl FnMut(&BED<5>) -> CellBarcode>,
    regions: &GenomeRegions<B>,
    bin_size: Option<u64>) -> Result<()>
where
    B: BEDLike + Clone,
    I: Iterator<Item = BED<5>>,
{
    let features: Vec<String> = match bin_size {
        None => regions.get_regions().iter().map(BEDLike::to_string).collect(),
        Some(k) => regions.get_regions().iter()
            .flat_map(|x| split_by_len(x, k))
            .map(|x| x.to_string()).collect(),
    };
    let mut list_of_barcodes = Vec::new();
    let sp_row_iter = SparseRowIter::new(
        fragments.into_iter().map(|(barcode, iter)| {
            list_of_barcodes.push(barcode);
            get_insertion_counts(regions, bin_size, iter)
        }),
        features.len()
    );
    sp_row_iter.create(&file, "X")?;

    Ok(())
}

fn get_insertion_counts<B, I>(regions: &GenomeRegions<B>,
                             bin_size: Option<u64>,
                             fragments: I) -> Vec<(usize, u64)>
where
    B: BEDLike,
    I: Iterator<Item = BED<5>>,
{
    match bin_size {
        None => regions.get_coverage(to_insertions(fragments)).0.into_iter().enumerate()
            .filter(|(_, x)| *x != 0).collect(),
        Some(k) => regions.get_binned_coverage(k, to_insertions(fragments)).0
            .into_iter().flatten().enumerate().filter(|(_, x)| *x != 0).collect(),
    }
}

fn to_insertions<I>(fragments: I) -> impl Iterator<Item = GenomicRange>
where
    I: Iterator<Item = BED<5>>,
{
    fragments.flat_map(|x| {
        [ GenomicRange::new(x.chrom().to_string(), x.start(), x.start() + 1)
        , GenomicRange::new(x.chrom().to_string(), x.end() - 1, x.end()) ]
    })
}