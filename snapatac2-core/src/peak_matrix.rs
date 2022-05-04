use crate::utils::{ChromValues, FeatureCounter, ChromValuesReader};

use anndata_rs::{
    anndata::AnnData,
    iterator::CsrIterator,
};
use polars::prelude::{NamedFrom, DataFrame, Series};
use anyhow::Result;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

use bed_utils::bed::{
    GenomicRange,
    tree::{GenomeRegions, SparseCoverage},
};

/// Create cell by feature matrix, and compute qc matrix.
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
pub fn create_feat_matrix<C, I>(
    anndata: &AnnData,
    insertions: I,
    feature_counter: C,
    ) -> Result<()>
where
    C: FeatureCounter<Value = u32> + Clone + std::marker::Sync,
    I: Iterator<Item = Vec<ChromValues>>,
{
    let features = feature_counter.get_feature_ids();
    anndata.set_x_from_row_iter(CsrIterator {
        iterator: insertions.map(|chunk|
            chunk.into_par_iter().map(|ins| {
                let mut counter = feature_counter.clone();
                counter.inserts(ins);
                counter.get_counts()
            }).collect::<Vec<_>>()
        ),
        num_cols: features.len(),
    })?;

    anndata.set_var(Some(
        &DataFrame::new(vec![Series::new("Feature_ID", features)]).unwrap()
    ))?;

    Ok(())
}

pub fn create_peak_matrix(
    anndata: &AnnData,
    peaks: &GenomeRegions<GenomicRange>,
    ) -> Result<()>
where
{
    let feature_counter: SparseCoverage<'_, _, u32> = SparseCoverage::new(&peaks);
    create_feat_matrix(
        anndata,
        anndata.read_insertions(500)?,
        feature_counter,
    )?;
    Ok(())
}