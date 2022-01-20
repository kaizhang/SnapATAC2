pub mod qc;
pub mod utils;
use utils::SparseBinnedCoverage;
use utils::hdf5::*;
use utils::anndata::*;

use qc::{CellBarcode, QualityControl, get_insertions};
use ndarray::{Array1};
use std::collections::HashSet;
use hdf5::{File, Result};
use bed_utils::bed::{BED, BEDLike, tree::GenomeRegions, tree::BedTree};
use itertools::GroupBy;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

/// Create cell by bin matrix, and compute qc matrix.
pub fn create_tile_matrix<B, I>(
    file: File,
    fragments: GroupBy<CellBarcode, I, impl FnMut(&BED<5>) -> CellBarcode>,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    bin_size: u64,
    min_num_fragment: u64,
    min_tsse: f64,
    ) -> Result<()>
where
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = BED<5>>,
{
    let features: Vec<String> = SparseBinnedCoverage::new(regions, bin_size)
        .get_regions().flatten().map(|x| x.to_string()).collect();
    let num_features = features.len();
    let mut saved_barcodes = Vec::new();
    let mut scanned_barcodes = HashSet::new();
    let mut qc = Vec::new();
    let chunked_fragments = fragments.into_iter().chunks(5000);
    let sp_row_iter = SparseRowIter::new(
        chunked_fragments.into_iter().map(|chunk| {
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
    );
    sp_row_iter.create(&file, "X")?;

    create_obs(&file, saved_barcodes, qc)?;
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
    let qc = qc::get_qc(promoter, fragments.iter());
    if qc.num_unique_fragment < min_n_fragment || qc.tss_enrichment < min_tsse {
        None
    } else {
        let mut binned_coverage = SparseBinnedCoverage::new(regions, bin_size);
        fragments.iter().for_each(|fragment| {
            let ins = get_insertions(fragment);
            binned_coverage.add(&ins[0]);
            binned_coverage.add(&ins[1]);
        });
        let count: Vec<(usize, u32)> = binned_coverage.get_coverage()
            .iter().map(|(k, v)| (*k, *v as u32)).collect();
        Some((qc, count))
    }
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

fn create_var(file: &File, features: Vec<String>) -> Result<()> {
    let group = file.create_group("var")?;
    create_str_attr(&group, "encoding-type", "dataframe")?;
    create_str_attr(&group, "encoding-version", "0.2.0")?;
    create_str_attr(&group, "_index", "Region")?;
    let columns: Array1<hdf5::types::VarLenUnicode> = [].into_iter().collect();
    group.new_attr_builder().with_data(&columns).create("column-order")?;
    StrVec(features).create(&group, "Region")?;
    Ok(())
}