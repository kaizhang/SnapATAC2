use crate::utils::anndata::{AnnData, SparseRowWriter, create_obs, create_var};
use crate::qc::{QualityControl, FragmentSummary, get_insertions};

use std::collections::HashSet;
use std::collections::HashMap;
use hdf5::{File, Result};
use bed_utils::bed::{
    BED, BEDLike, tree::GenomeRegions,
    tree::{BedTree, SparseBinnedCoverage},
};
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

/// Create cell by bin matrix, and compute qc matrix.
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
pub fn create_tile_matrix<B, I>(
    file: File,
    fragments: I,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    bin_size: u64,
    white_list: Option<&HashSet<String>>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    ) -> Result<()>
where
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = BED<5>>,
{
    let features: Vec<String> = SparseBinnedCoverage::<_,u32>::new(regions, bin_size)
        .get_regions().flatten().map(|x| x.to_string()).collect();
    let num_features = features.len();
    let mut saved_barcodes = Vec::new();
    let mut qc = Vec::new();

    if fragment_is_sorted_by_name {
        let mut scanned_barcodes = HashSet::new();
        SparseRowWriter::new(fragments
            .group_by(|x| { x.name().unwrap().to_string() }).into_iter()
            .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
            .chunks(2000).into_iter().map(|chunk| {
                let data: Vec<(String, Vec<BED<5>>)> = chunk
                    .map(|(barcode, x)| (barcode, x.collect())).collect();
                let result: Vec<_> = data.into_par_iter()
                    .map(|(barcode, x)| (barcode, compute_qc_count(x, promoter, regions, bin_size, min_num_fragment, min_tsse)))
                    .collect();
                result.into_iter().filter_map(|(barcode, r)| {
                    if !scanned_barcodes.insert(barcode.clone()) {
                        panic!("Please sort fragment file by barcodes");
                    }
                    match r {
                        None => None,
                        Some((q, count)) => {
                            saved_barcodes.push(barcode);
                            qc.push(q);
                            Some(count)
                        },
                    }
                }).collect::<Vec<_>>()
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
                    let mut summary= FragmentSummary::new(promoter);
                    let mut counts = SparseBinnedCoverage::new(regions, bin_size);
                    summary.update(&frag);
                    counts.insert(&ins[0], 1);
                    counts.insert(&ins[1], 1);
                    scanned_barcodes.insert(key.to_string(), (summary, counts));
                },
                Some((summary, counts)) => {
                    summary.update(&frag);
                    counts.insert(&ins[0], 1);
                    counts.insert(&ins[1], 1);
                }
            }
        });
        let row_iter = scanned_barcodes.drain()
            .filter_map(|(barcode, (summary, binned_coverage))| {
                let q = summary.get_qc();
                if q.num_unique_fragment < min_num_fragment || q.tss_enrichment < min_tsse {
                    None
                } else {
                    saved_barcodes.push(barcode);
                    qc.push(q);
                    let count: Vec<(usize, u32)> = binned_coverage.get_coverage()
                        .iter().map(|(k, v)| (*k, *v)).collect();
                    Some(count)
                }
            });
        SparseRowWriter::new(row_iter, num_features).create(&file, "X")?;
    }

    create_obs(&file, saved_barcodes, Some(qc))?;
    create_var(&file, features)?;
    Ok(())
}

fn compute_qc_count<B>(fragments: Vec<BED<5>>,
           promoter: &BedTree<bool>,
           regions: &GenomeRegions<B>,
           bin_size: u64,
           min_n_fragment: u64,
           min_tsse: f64,
          ) -> Option<(QualityControl, Vec<(usize, u32)>)>
where
    B: BEDLike,
{

    let mut summary = FragmentSummary::new(promoter);
    fragments.iter().for_each(|frag| summary.update(frag));
    let qc = summary.get_qc();
    if qc.num_unique_fragment < min_n_fragment || qc.tss_enrichment < min_tsse {
        None
    } else {
        let mut binned_coverage = SparseBinnedCoverage::new(regions, bin_size);
        fragments.iter().for_each(|fragment| {
            let ins = get_insertions(fragment);
            binned_coverage.insert(&ins[0], 1);
            binned_coverage.insert(&ins[1], 1);
        });
        let count: Vec<(usize, u32)> = binned_coverage.get_coverage()
            .iter().map(|(k, v)| (*k, *v)).collect();
        Some((qc, count))
    }
}

/// barcode counting.
pub fn get_barcode_count<I>(fragments: I) -> HashMap<String, u64>
where
    I: Iterator<Item = BED<5>>,
{
    let mut barcodes = HashMap::new();
    fragments.for_each(|frag| {
        let key = frag.name().unwrap().to_string();
        *barcodes.entry(key).or_insert(0) += 1;
    });
    barcodes
}