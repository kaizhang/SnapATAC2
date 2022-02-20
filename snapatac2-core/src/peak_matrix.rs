use crate::utils::{
    Fragment,
    Insertions,
    Barcoded,
    FeatureCounter,
    anndata::{AnnData, SparseRowWriter, create_obs, create_var},
};

use std::collections::HashSet;
use std::collections::HashMap;
use hdf5::{File, Result};
use bed_utils::bed::{
    BEDLike, tree::GenomeRegions,
    tree::{SparseCoverage},
};
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

pub fn create_peak_matrix<'a, B, I>(
    file: File,
    fragments: I,
    regions: &GenomeRegions<B>,
    white_list: Option<&HashSet<String>>,
    fragment_is_sorted_by_name: bool,
    ) -> Result<()>
where
    I: Iterator<Item = Fragment>,
    B: BEDLike + Clone + std::marker::Sync,
{
    let feature_counter: SparseCoverage<'_, _, u32> = SparseCoverage::new(regions);
    create_feat_matrix(file, fragments, feature_counter, white_list, fragment_is_sorted_by_name)
}


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
pub fn create_feat_matrix<C, I, D>(
    file: File,
    fragments: I,
    feature_counter: C,
    white_list: Option<&HashSet<String>>,
    fragment_is_sorted_by_name: bool,
    ) -> Result<()>
where
    C: FeatureCounter<Value = u32> + Clone + std::marker::Sync,
    I: Iterator<Item = D>,
    D: Into<Insertions> + Barcoded + Send,
{
    let features: Vec<String> = feature_counter.get_feature_ids();
    let num_features = features.len();
    let mut saved_barcodes: Vec<String> = Vec::new();

    if fragment_is_sorted_by_name {
        let mut scanned_barcodes = HashSet::new();
        SparseRowWriter::new(fragments
            .group_by(|x| { x.get_barcode().to_string() }).into_iter()
            .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
            .chunks(5000).into_iter().map(|chunk| {
                let (barcodes, counts): (Vec<String>, Vec<_>) = chunk
                    .map(|(barcode, x)| (barcode, x.collect()))
                    .collect::<Vec<(String, Vec<_>)>>()
                    .into_par_iter()
                    .map(|(barcode, frag)| {
                        let mut counter = feature_counter.clone();
                        frag.into_iter().for_each(|f| counter.inserts(f));
                        let count: Vec<(usize, u32)> = counter.get_counts();
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
        .filter(|frag| white_list.map_or(true, |x| x.contains(frag.get_barcode())))
        .for_each(|frag| {
            let key = frag.get_barcode().to_string();
            match scanned_barcodes.get_mut(&key) {
                None => {
                    let mut counter = feature_counter.clone();
                    counter.inserts(frag);
                    scanned_barcodes.insert(key, counter);
                },
                Some(counter) => counter.inserts(frag),
            }
        });
        let row_iter = scanned_barcodes.drain()
            .map(|(barcode, counter)| {
                saved_barcodes.push(barcode);
                let count: Vec<(usize, u32)> = counter.get_counts();
                count
            });
        SparseRowWriter::new(row_iter, num_features).create(&file, "X")?;
    }

    create_obs(&file, saved_barcodes, None)?;
    create_var(&file, features)?;
    Ok(())
}