use crate::utils::{Insertions, FeatureCounter};

use anndata_rs::{
    anndata::AnnData,
    iterator::CsrIterator,
};
use polars::prelude::{NamedFrom, DataFrame, Series};
use hdf5::Result;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

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
    I: Iterator<Item = Vec<Insertions>>,
{
    let features = feature_counter.get_feature_ids();
    anndata.set_x_from_row_iter(CsrIterator {
        iterator: insertions.map(|chunk|
            chunk.into_par_iter().map(|ins| {
                let mut counter = feature_counter.clone();
                counter.inserts(ins);
                counter.get_counts()
            }).collect::<Vec<_>>()
        ).flatten(),
        num_cols: features.len(),
    })?;

    anndata.set_var(&DataFrame::new(vec![
        Series::new("Feature_ID", features)
    ]).unwrap())?;

    Ok(())
}