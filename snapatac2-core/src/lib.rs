pub mod qc;
pub mod utils;
use utils::SparseBinnedCoverage;
use utils::hdf5::*;
use utils::anndata::*;
use std::collections::HashSet;

use qc::{CellBarcode, FragmentSummary, QualityControl, get_insertions};
use ndarray::{Array1, arr1};

use hdf5::{File, Result};
use bed_utils::bed::{BED, BEDLike, GenomicRange, split_by_len,
                     tree::GenomeRegions, tree::BinnedCoverage, tree::BedTree};
use itertools::GroupBy;

/// Create cell by bin matrix, and compute qc matrix.
pub fn create_tile_matrix<B, I>(
    file: File,
    fragments: GroupBy<CellBarcode, I, impl FnMut(&BED<5>) -> CellBarcode>,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    bin_size: u64,
    min_num_fragment: u64,
    ) -> Result<()>
where
    B: BEDLike + Clone,
    I: Iterator<Item = BED<5>>,
{
    let mut binned_coverage = SparseBinnedCoverage::new(regions, bin_size);
    let num_features = binned_coverage.len;
    let features: Vec<String> = binned_coverage.get_regions().flatten()
        .map(|x| x.to_string()).collect();
    let mut feature_counts: Vec<u64> = vec![0; num_features];
    let mut list_of_barcodes = Vec::new();
    let mut qc = Vec::new();
    let sp_row_iter = SparseRowIter::new(
        fragments.into_iter().filter_map(|(barcode, iter)| {
            let mut summary = FragmentSummary::new();
            binned_coverage.reset();
            iter.for_each(|fragment| {
                summary.update(promoter, &fragment);
                let ins = get_insertions(&fragment);
                binned_coverage.add(&ins[0]);
                binned_coverage.add(&ins[1]);
            });
            if summary.num_unique_fragment < min_num_fragment {
                None
            } else {
                list_of_barcodes.push(barcode);
                let count: Vec<(usize, u32)> = binned_coverage.get_coverage()
                    .iter().map(|(k, v)| (*k, *v as u32)).collect();
                count.iter().for_each(|&(i, x)| feature_counts[i] += x as u64);
                qc.push(summary.get_qc());
                Some(count)
            }
        }),
        num_features
    );
    sp_row_iter.create(&file, "X")?;

    create_obs(&file, list_of_barcodes, qc)?;
    create_var(&file, features, feature_counts)?;
    Ok(())
}

fn create_obs(file: &File, cells: Vec<CellBarcode>, qc: Vec<QualityControl>) -> Result<()> {
    let group = file.create_group("obs")?;
    create_str_attr(&group, "encoding-type", "dataframe")?;
    create_str_attr(&group, "encoding-version", "0.2.0")?;
    create_str_attr(&group, "_index", "Cell")?;
    let columns: Array1<hdf5::types::VarLenUnicode> =
        ["tsse", "n_fragment", "frac_dup", "frac_mito"]
        .into_iter().map(|x| x.parse().unwrap()).collect();
    group.new_attr_builder().with_data(&columns).create("column-order")?;
    StrVec(cells).create(&group, "Cell")?;
    qc.iter().map(|x| x.tss_enrichment).collect::<Array1<f64>>().create(&group, "tsse")?;
    qc.iter().map(|x| x.num_unique_fragment).collect::<Array1<u64>>().create(&group, "n_fragment")?;
    qc.iter().map(|x| x.frac_duplicated).collect::<Array1<f64>>().create(&group, "frac_dup")?;
    qc.iter().map(|x| x.frac_mitochondrial).collect::<Array1<f64>>().create(&group, "frac_mito")?;
    Ok(())
}

fn create_var(file: &File, features: Vec<String>, feature_counts: Vec<u64>) -> Result<()> {
    let group = file.create_group("var")?;
    create_str_attr(&group, "encoding-type", "dataframe")?;
    create_str_attr(&group, "encoding-version", "0.2.0")?;
    create_str_attr(&group, "_index", "Region")?;
    let columns: Array1<hdf5::types::VarLenUnicode> = ["counts"].into_iter()
        .map(|x| x.parse().unwrap()).collect();
    group.new_attr_builder().with_data(&columns).create("column-order")?;
    StrVec(features).create(&group, "Region")?;
    Array1::from(feature_counts).create(&group, "counts")?;
    Ok(())
}

pub fn create_count_matrix<B, I>(
    file: File,
    fragments: GroupBy<CellBarcode, I, impl FnMut(&BED<5>) -> CellBarcode>,
    regions: &GenomeRegions<B>,
    bin_size: Option<u64>,
    selected_cells: Option<&HashSet<CellBarcode>>
    ) -> Result<(Vec<CellBarcode>, Vec<String>, Vec<u64>)>
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
    let mut feature_counts: Vec<u64> = vec![0; features.len()];
    let mut list_of_barcodes = Vec::new();
    let sp_row_iter = SparseRowIter::new(
        fragments.into_iter().filter_map(|(barcode, iter)| {
            if selected_cells.map_or(true, |x| x.contains(&barcode)) {
                list_of_barcodes.push(barcode);
                let count = get_insertion_counts(regions, bin_size, iter);
                count.iter().for_each(|&(i, x)| feature_counts[i] += x as u64);
                Some(count)
            } else {
                None
            }
        }),
        features.len()
    );
    sp_row_iter.create(&file, "X")?;

    Ok((list_of_barcodes, features, feature_counts))
}

fn get_insertion_counts<B, I>(regions: &GenomeRegions<B>,
                             bin_size: Option<u64>,
                             fragments: I) -> Vec<(usize, u32)>
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