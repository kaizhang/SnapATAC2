use crate::utils::anndata::{AnnData, SparseRowIter, create_obs, create_var};
use crate::qc::{CellBarcode, QualityControl, FragmentSummary, get_insertions};

use std::collections::HashSet;
use std::collections::HashMap;
use hdf5::{File, Result};
use bed_utils::bed::{
    BED, BEDLike, tree::GenomeRegions,
    tree::{BedTree, SparseCoverage},
};
use itertools::GroupBy;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

/// Create cell by peak matrix
pub fn create_peak_matrix_unsorted<B, I>(
    file: File,
    fragments: I,
    regions: &GenomeRegions<B>,
    white_list: Option<&HashSet<String>>,
    ) -> Result<()>
where
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = BED<5>>,
{
    let features: Vec<String> = regions.get_regions().iter().map(|x| x.to_string()).collect();
    let num_features = regions.len();
    let mut saved_barcodes = Vec::new();
    let mut barcodes = HashMap::new();
    fragments.for_each(|frag| {
        let key = frag.name().unwrap();
        if white_list.map_or(true, |x| x.contains(key)) {
            let ins = get_insertions(&frag);
            match barcodes.get_mut(key) {
                None => {
                    let mut counts = SparseCoverage::new(regions);
                    counts.add(&ins[0]);
                    counts.add(&ins[1]);
                    barcodes.insert(key.to_string(), counts);
                },
                Some(counts) => {
                    counts.add(&ins[0]);
                    counts.add(&ins[1]);
                }
            }
        }
    });
    let row_iter = barcodes.drain().map(|(barcode, coverage)| {
        saved_barcodes.push(barcode);
        let count: Vec<(usize, u32)> = coverage.get_coverage()
            .iter().map(|(k, v)| (*k, *v)).collect();
        count
    });

    SparseRowIter::new(row_iter, num_features).create(&file, "X")?;
    create_obs(&file, saved_barcodes, None)?;
    create_var(&file, features)?;
    Ok(())
}