use crate::utils::anndata::{AnnDataIO, StrVec};
use crate::qc::{read_insertions};
use crate::peak_matrix::create_feat_matrix;

use hdf5::{File, Result};
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
    file: &File,
    bin_size: u64,
    ) -> Result<()>
where
{
    let group = file.group("obsm")?.group("base_count")?;
    let chrs: StrVec = AnnDataIO::read(&group.dataset("reference_seq_name")?)?;
    let chr_sizes: Vec<u64> = AnnDataIO::read(&group.dataset("reference_seq_length")?)?;
    let regions = chrs.0.into_iter().zip(chr_sizes).map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect();
    let feature_counter: SparseBinnedCoverage<'_, _, u32> = SparseBinnedCoverage::new(&regions, bin_size);
    let insertions = read_insertions(group)?;
    create_feat_matrix(file, insertions.iter(), feature_counter)
}