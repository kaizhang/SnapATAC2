use crate::{
    peak_matrix::create_feat_matrix,
    utils::{get_chrom_index, InsertionIter},
};

use anndata_rs::{
    anndata::AnnData,
    iterator::IntoRowsIterator,
    element::ElemTrait,
};
use polars::prelude::DataFrame;
use hdf5::Result;
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
    let df: Box<DataFrame> = anndata.uns.data.lock().unwrap()
        .get("reference_sequences").unwrap().read()?.into_any().downcast().unwrap();
    let regions = df.column("reference_seq_length")
        .unwrap().u64().unwrap().into_iter()
        .zip(df.column("reference_seq_name").unwrap().utf8().unwrap())
        .map(|(s, chr)| GenomicRange::new(chr.unwrap(), 0, s.unwrap())).collect();
    let feature_counter: SparseBinnedCoverage<'_, _, u32> =
        SparseBinnedCoverage::new(&regions, bin_size);
    let chrom_index = get_chrom_index(anndata)?;
    create_feat_matrix(
        anndata,
        InsertionIter {
            iter: anndata.obsm.data.lock().unwrap().get("insertion").unwrap()
                .0.lock().unwrap().downcast().into_row_iter(500),
            chrom_index,
        },
        feature_counter,
    )?;
    Ok(())
}