use crate::utils::anndata::{AnnData, SparseRowIter, create_obs, create_var};
use crate::qc::{get_insertions};

use std::collections::HashSet;
use std::collections::HashMap;
use hdf5::{File, Result};
use bed_utils::bed::{
    BED, BEDLike, tree::GenomeRegions,
    tree::{SparseCoverage},
};
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

/// Create cell by peak matrix, and compute qc matrix.
/// 
/// # Arguments
/// 
/// * `file` - 
/// * `fragments` -
/// * `promoter` -
/// * `region` -
/// * `bin_size` -
/// * `min_num_fragment` -
/// * `min_tsse` -
pub fn create_peak_matrix<B, I>(
    file: File,
    fragments: I,
    regions: &GenomeRegions<B>,
    white_list: Option<&HashSet<String>>,
    fragment_is_sorted_by_name: bool,
    ) -> Result<()>
where
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = BED<5>>,
{
    let features: Vec<String> = regions.get_regions().iter().map(|x| x.to_string()).collect();
    let num_features = features.len();
    let mut saved_barcodes = Vec::new();

    if fragment_is_sorted_by_name {
        let mut scanned_barcodes = HashSet::new();
        SparseRowIter::new(fragments
            .group_by(|x| { x.name().unwrap().to_string() }).into_iter()
            .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
            .chunks(5000).into_iter().map(|chunk| {
                let (barcodes, counts): (Vec<_>, Vec<_>) = chunk
                    .map(|(barcode, x)| (barcode, x.collect()))
                    .collect::<Vec<(String, Vec<_>)>>()
                    .into_par_iter()
                    .map(|(barcode, frag)| {
                        let mut coverage = SparseCoverage::new(regions);
                        frag.iter().for_each(|f| {
                            let ins = get_insertions(f);
                            coverage.add(&ins[0]);
                            coverage.add(&ins[1]);
                        });
                        let count: Vec<(usize, u32)> = coverage.get_coverage()
                            .iter().map(|(k, v)| (*k, *v)).collect();
                        (barcode, count)
                    }).unzip();

                barcodes.into_iter().for_each(|barcode| {
                    if !scanned_barcodes.insert(barcode.clone()) {
                        panic!("Please sort fragment file by barcodes");
                    }
                    saved_barcodes.push(barcode);
                });
                counts
            }).flatten(),
            num_features
        ).create(&file, "X")?;
    } else {
        let mut scanned_barcodes = HashMap::new();
        fragments
        .filter(|frag| white_list.map_or(true, |x| x.contains(frag.name().unwrap())))
        .for_each(|frag| {
            let key = frag.name().unwrap();
            let ins = get_insertions(&frag);
            match scanned_barcodes.get_mut(key) {
                None => {
                    let mut counts = SparseCoverage::new(regions);
                    counts.add(&ins[0]);
                    counts.add(&ins[1]);
                    scanned_barcodes.insert(key.to_string(), counts);
                },
                Some(counts) => {
                    counts.add(&ins[0]);
                    counts.add(&ins[1]);
                }
            }
        });
        let row_iter = scanned_barcodes.drain()
            .map(|(barcode, coverage)| {
                saved_barcodes.push(barcode);
                let count: Vec<(usize, u32)> = coverage.get_coverage()
                    .iter().map(|(k, v)| (*k, *v)).collect();
                count
            });
        SparseRowIter::new(row_iter, num_features).create(&file, "X")?;
    }

    create_obs(&file, saved_barcodes, None)?;
    create_var(&file, features)?;
    Ok(())
}