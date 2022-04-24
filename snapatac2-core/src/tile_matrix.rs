use crate::{
    peak_matrix::create_feat_matrix,
    utils::ChromValuesReader,
};

use anndata_rs::anndata::AnnData;
use polars::prelude::DataFrame;
use anyhow::Result;
use bed_utils::bed::{
    GenomicRange,
    tree::{SparseBinnedCoverage},
};

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
pub fn create_tile_matrix(
    anndata: &AnnData,
    bin_size: u64,
    ) -> Result<()>
where
{
    let df: Box<DataFrame> = {
        anndata.get_uns().inner().get_mut("reference_sequences").unwrap()
            .read()?.into_any().downcast().unwrap()
    };
    let regions = df.column("reference_seq_length")
        .unwrap().u64().unwrap().into_iter()
        .zip(df.column("reference_seq_name").unwrap().utf8().unwrap())
        .map(|(s, chr)| GenomicRange::new(chr.unwrap(), 0, s.unwrap())).collect();
    let feature_counter: SparseBinnedCoverage<'_, _, u32> =
        SparseBinnedCoverage::new(&regions, bin_size);
    create_feat_matrix(
        anndata,
        anndata.read_insertions()?,
        feature_counter,
    )?;
    Ok(())
}