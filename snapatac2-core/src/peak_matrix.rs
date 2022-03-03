use crate::utils::{
    Insertions,
    FeatureCounter,
    anndata::{write_csr_rows, create_var},
};

use hdf5::{File, Result};
use itertools::Itertools;
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
pub fn create_feat_matrix<C, I, D>(
    file: &File,
    insertions: I,
    feature_counter: C,
    ) -> Result<()>
where
    C: FeatureCounter<Value = u32> + Clone + std::marker::Sync,
    I: Iterator<Item = D>,
    D: Into<Insertions> + Send,
{
    let features: Vec<String> = feature_counter.get_feature_ids();
    let num_features = features.len();

    if file.link_exists("X") { file.unlink("X")?; }
    write_csr_rows(
        insertions.chunks(2500).into_iter().map(|chunk|
            chunk.collect::<Vec<_>>().into_par_iter().map(|ins| {
                let mut counter = feature_counter.clone();
                counter.inserts(ins);
                counter.get_counts()
            }).collect::<Vec<_>>()
        ).flatten(),
        num_features,
        file,
        "X",
        "csr_matrix",
        "0.1.0"
    )?;

    if file.link_exists("var") { file.unlink("var")?; }
    create_var(file, features)?;
    Ok(())
}